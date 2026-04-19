import os, json, math, random, argparse, hashlib
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (accuracy_score, f1_score,
                             classification_report, confusion_matrix)
from tqdm import tqdm
import torchaudio

CLASS_NAMES = ["RealVideo-RealAudio", "RealVideo-FakeAudio",
               "FakeVideo-RealAudio", "FakeVideo-FakeAudio"]

MAX_SYLLABLES       = 40
MAX_WORDS           = 12     # reduced from 20 — saves ~40% word_face_crops memory
MAX_FRAMES_PER_SYL  = 5
MAX_FRAMES_PER_WORD = 4      # reduced from 10 — CLIP doesn't need 10 frames/word
MAX_SENTENCE_FRAMES = 8
MAX_AUDIO_SAMPLES   = 16000 * 10
MIN_CONTEXT_FRAMES  = 3


def set_seed(s=42):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


# ═══════════════════════════════════════════════════════════════
# ARCFACE CACHE
# Keyed on video_id (stable) + split name so train/val/test never
# share a cache file even when they come from the same video directory.
# ═══════════════════════════════════════════════════════════════

def _arcface_cache_path(video_id: str, split: str, cache_dir: Path) -> Path:
    """
    Cache is keyed on (video_id, split) so two splits that happen to
    share the same underlying video directory cannot cross-contaminate.
    """
    key = f"{video_id}__{split}"
    h   = hashlib.md5(key.encode()).hexdigest()[:20]
    return cache_dir / f"arcface_{h}.pt"


def extract_arcface(frame_dir: Path, cache_path: Path, device="cpu"):
    """Extract ArcFace embeddings. Cache to writable location."""
    if cache_path.exists():
        return torch.load(cache_path, map_location="cpu", weights_only=True)

    if frame_dir is None or not frame_dir.exists():
        result = torch.zeros(1, 512)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(result, cache_path)
        return result

    frames = sorted(frame_dir.glob("*.jpg"))
    if not frames:
        result = torch.zeros(1, 512)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(result, cache_path)
        return result

    try:
        from insightface.app import FaceAnalysis
        import cv2
        prov = ["CUDAExecutionProvider"] if "cuda" in device else ["CPUExecutionProvider"]
        app  = FaceAnalysis(name="buffalo_l", providers=prov)
        app.prepare(ctx_id=0 if "cuda" in device else -1, det_size=(224, 224))
        embs = []
        for fp in frames:
            img   = cv2.imread(str(fp))
            faces = app.get(img) if img is not None else []
            if faces:
                face = sorted(faces,
                              key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]),
                              reverse=True)[0]
                embs.append(torch.tensor(face.embedding, dtype=torch.float32))
            else:
                embs.append(torch.zeros(512))
        result = F.normalize(torch.stack(embs), dim=-1)
    except ImportError:
        result = F.normalize(torch.randn(len(frames), 512), dim=-1)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, cache_path)
    return result


# ═══════════════════════════════════════════════════════════════
# BOUNDARY CONTEXT PADDING
# ═══════════════════════════════════════════════════════════════

def pad_boundary(sf, ef, fps, total, mn=MIN_CONTEXT_FRAMES):
    n = ef - sf
    if n >= mn:
        return sf, ef, sf/fps, ef/fps
    deficit = mn - n
    ns = max(0, sf - deficit//2)
    ne = min(total, ef + deficit - deficit//2)
    if ne - ns < mn:
        if ns == 0: ne = min(total, mn)
        else:       ns = max(0, ne - mn)
    return ns, ne, ns/fps, ne/fps


# ═══════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════

class FakeAVCelebDataset(Dataset):
    def __init__(self, data_root, split_indices=None, augment=False, device="cpu",
                 ethnicity=None, gender=None, category=None, cache_dir=None,
                 split_name="train"):
        """
        split_name: one of 'train', 'val', 'test'.  Used to namespace the
                    ArcFace cache so identities cannot leak across splits.
        """
        self.root       = Path(data_root)
        self.augment    = augment
        self.device     = device
        self.split_name = split_name
        self.cache_dir  = Path(cache_dir) if cache_dir else Path("checkpoints/arcface_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.samples    = self._scan()

        if ethnicity:
            el = [ethnicity] if isinstance(ethnicity, str) else ethnicity
            self.samples = [s for s in self.samples if s["ethnicity"] in el]
        if gender:
            gl = [gender] if isinstance(gender, str) else gender
            self.samples = [s for s in self.samples if s["gender"] in gl]
        if category:
            cl = [category] if isinstance(category, str) else category
            self.samples = [s for s in self.samples if s["label_str"] in cl]
        if split_indices is not None:
            self.samples = [self.samples[i] for i in split_indices if i < len(self.samples)]

        self.lip_tfm = transforms.Compose([
            transforms.Resize((48, 96)), transforms.ToTensor(),
            transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
        self.face_tfm = transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
        self.lip_aug = transforms.Compose([
            transforms.RandomHorizontalFlip(.5), transforms.ColorJitter(.2,.2),
            transforms.Resize((48, 96)), transforms.ToTensor(),
            transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
        self.face_aug = transforms.Compose([
            transforms.RandomHorizontalFlip(.5), transforms.ColorJitter(.2,.2),
            transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize([.485,.456,.406],[.229,.224,.225])])

    def _scan(self):
        samples = []
        for mp in sorted(self.root.rglob("meta.json")):
            vd   = mp.parent
            with open(mp) as f:
                meta = json.load(f)
            if not (vd/"audio"/"alignment.json").exists(): continue
            if not (vd/"audio"/"audio.wav").exists(): continue
            if meta.get("n_words", 0) < 1 and meta.get("n_syllables", 0) < 1: continue

            lip_dir = face_dir = full_dir = None
            for c in ["scale1/lip_crops", "scale1_phoneme/lip_crops"]:
                p = vd / c
                if p.exists(): lip_dir = p; break
            for c in ["scale2/face_crops", "scale2_word/face_crops"]:
                p = vd / c
                if p.exists(): face_dir = p; break
            for c in ["scale3/full_frames", "scale3_phrase/full_frames"]:
                p = vd / c
                if p.exists(): full_dir = p; break

            if lip_dir is None or face_dir is None: continue
            if len(list(lip_dir.glob("*.jpg"))) < 2: continue

            samples.append({
                "video_dir":    vd,
                "label":        meta["label"],
                "video_id":     meta.get("video_id", vd.name),
                "identity":     meta.get("identity", "unknown"),
                "ethnicity":    meta.get("ethnicity", "unknown"),
                "gender":       meta.get("gender", "unknown"),
                "label_str":    meta.get("label_str", "unknown"),
                "fps":          meta.get("fps", 25.0),
                "duration_sec": meta.get("duration_sec", 1.0),
                "total_frames": meta.get("total_frames", 100),
                "lip_dir":      str(lip_dir),
                "face_dir":     str(face_dir),
                "full_dir":     str(full_dir) if full_dir else None,
            })
        return samples

    def __len__(self): return len(self.samples)

    def get_label_counts(self):
        c = Counter(s["label"] for s in self.samples)
        return [c.get(i, 0) for i in range(4)]

    def get_demographics(self):
        return {
            "ethnicities":  dict(Counter(s["ethnicity"] for s in self.samples)),
            "genders":      dict(Counter(s["gender"]    for s in self.samples)),
            "categories":   dict(Counter(s["label_str"] for s in self.samples)),
            "n_identities": len(set(s["identity"]       for s in self.samples))}

    def get_identities(self):
        return [s["identity"] for s in self.samples]

    def _load_frames(self, fdir, sf, ef, tfm, mx):
        frames = []
        for i in range(sf, min(ef, sf+mx)):
            fp = Path(fdir) / f"frame_{i:06d}.jpg"
            if fp.exists():
                frames.append(tfm(Image.open(fp).convert("RGB")))
        if not frames:
            d = tfm(Image.new("RGB", (96, 48)))
            return d.unsqueeze(0).repeat(mx, 1, 1, 1) * 0
        while len(frames) < mx: frames.append(frames[-1])
        return torch.stack(frames[:mx])

    def __getitem__(self, idx):
        s   = self.samples[idx]
        vd, label, fps = s["video_dir"], s["label"], s["fps"]
        tf  = s["total_frames"]
        lip_dir, face_dir = s["lip_dir"], s["face_dir"]
        full_dir = s["full_dir"]
        aug = self.augment and label in (1, 2)
        lt  = self.lip_aug if aug else self.lip_tfm
        ft  = self.face_aug if aug else self.face_tfm

        with open(Path(vd)/"audio"/"alignment.json") as f:
            align = json.load(f)
        syls  = align.get("syllables", [])
        words = align.get("words", [])

        # SYLLABLE
        sc, sb = [], []
        for seg in syls[:MAX_SYLLABLES]:
            sf, ef = seg.get("start_frame", 0), seg.get("end_frame", 0)
            sf, ef, ss, se = pad_boundary(sf, ef, fps, tf)
            sc.append(self._load_frames(lip_dir, sf, ef, lt, MAX_FRAMES_PER_SYL))
            sb.append((ss, se))
        d_s = torch.zeros(MAX_FRAMES_PER_SYL, 3, 48, 96)
        while len(sc) < MAX_SYLLABLES: sc.append(d_s); sb.append((0., 0.))
        syl_crops = torch.stack(sc[:MAX_SYLLABLES])

        # WORD
        wc, wb = [], []
        for seg in words[:MAX_WORDS]:
            sf, ef = seg.get("start_frame", 0), seg.get("end_frame", 0)
            sf, ef, ss, se = pad_boundary(sf, ef, fps, tf)
            wc.append(self._load_frames(face_dir, sf, ef, ft, MAX_FRAMES_PER_WORD))
            wb.append((ss, se))
        d_w = torch.zeros(MAX_FRAMES_PER_WORD, 3, 224, 224)
        while len(wc) < MAX_WORDS: wc.append(d_w); wb.append((0., 0.))
        word_crops = torch.stack(wc[:MAX_WORDS])

        # SENTENCE — ArcFace cached with split-namespaced key
        full_path = Path(full_dir) if full_dir else None
        cp = _arcface_cache_path(s["video_id"], self.split_name, self.cache_dir)
        sa = extract_arcface(full_path, cp, self.device)
        if sa.shape[0] < MAX_SENTENCE_FRAMES:
            sa = torch.cat([sa, torch.zeros(MAX_SENTENCE_FRAMES - sa.shape[0], 512)])
        sa = sa[:MAX_SENTENCE_FRAMES]

        # WAVEFORM
        wav, sr = torchaudio.load(str(Path(vd)/"audio"/"audio.wav"))
        if sr != 16000: wav = torchaudio.functional.resample(wav, sr, 16000)
        wav = wav.mean(0)
        if wav.shape[0] > MAX_AUDIO_SAMPLES:
            wav = wav[:MAX_AUDIO_SAMPLES]
        elif wav.shape[0] < MAX_AUDIO_SAMPLES:
            wav = F.pad(wav, (0, MAX_AUDIO_SAMPLES - wav.shape[0]))

        # AUDIO BOUNDARIES
        asb = []
        for seg in syls[:MAX_SYLLABLES]:
            sf, ef = seg.get("start_frame", 0), seg.get("end_frame", 0)
            _, _, ss, se = pad_boundary(sf, ef, fps, tf)
            asb.append((ss, se))
        while len(asb) < MAX_SYLLABLES: asb.append((0., 0.))

        awb = []
        for seg in words[:MAX_WORDS]:
            sf, ef = seg.get("start_frame", 0), seg.get("end_frame", 0)
            _, _, ss, se = pad_boundary(sf, ef, fps, tf)
            awb.append((ss, se))
        while len(awb) < MAX_WORDS: awb.append((0., 0.))

        return {
            "syllable_lip_crops": syl_crops,
            "word_face_crops":    word_crops,
            "sentence_arcface":   sa,
            "waveform":           wav,
            "boundaries": {
                "syllable": asb,
                "word":     awb,
                "sentence": [(0.0, s["duration_sec"])]},
            "label":    label,
            "video_id": s["video_id"],
        }


# ═══════════════════════════════════════════════════════════════
# COLLATE + SAMPLER
# ═══════════════════════════════════════════════════════════════

def collate_fn(batch):
    bd = {
        "syllable_lip_crops": torch.stack([b["syllable_lip_crops"] for b in batch]),
        "word_face_crops":    torch.stack([b["word_face_crops"]    for b in batch]),
        "sentence_arcface":   torch.stack([b["sentence_arcface"]   for b in batch]),
        "waveform":           torch.stack([b["waveform"]           for b in batch]),
        "boundaries": {
            "syllable": [b["boundaries"]["syllable"] for b in batch],
            "word":     [b["boundaries"]["word"]     for b in batch],
            "sentence": [b["boundaries"]["sentence"] for b in batch]}}
    return (bd,
            torch.tensor([b["label"]    for b in batch], dtype=torch.long),
            [b["video_id"] for b in batch])


def build_sampler(ds):
    """Balanced WeightedRandomSampler. Build once and reuse across epochs."""
    labels = [s["label"] for s in ds.samples]
    c      = Counter(labels)
    mx     = max(c.values())
    cw     = {k: mx / v for k, v in c.items()}
    return WeightedRandomSampler([cw[l] for l in labels], mx * len(c), replacement=True)


# ═══════════════════════════════════════════════════════════════
# IDENTITY-GROUPED SPLITTING
# ═══════════════════════════════════════════════════════════════

def grouped_split(samples, test_size, val_size, seed):
    """
    Split sample indices so that no speaker identity appears in more than
    one of train / val / test.  Uses GroupShuffleSplit from sklearn.

    Returns: train_idx, val_idx, test_idx  (lists of integer indices into samples)
    """
    identities = [s["identity"] for s in samples]
    n          = len(samples)
    idx        = list(range(n))

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tv_idx, test_idx = next(gss.split(idx, groups=identities))

    tv_identities = [identities[i] for i in tv_idx]
    # val_size is relative to the train+val pool
    val_frac = val_size / (1.0 - test_size)
    gss2     = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    rel_train, rel_val = next(gss2.split(tv_idx, groups=tv_identities))

    train_idx = [tv_idx[i] for i in rel_train]
    val_idx   = [tv_idx[i] for i in rel_val]
    return train_idx, val_idx, list(test_idx)


def grouped_split_cross(train_samples, test_samples, val_frac, seed):
    """
    Cross-demographic variant: train/val drawn from train_samples,
    test is a separate demographic pool.  Still splits train/val by identity.
    """
    identities = [s["identity"] for s in train_samples]
    idx        = list(range(len(train_samples)))
    gss        = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    rel_train, rel_val = next(gss.split(idx, groups=identities))
    return rel_train, rel_val


# ═══════════════════════════════════════════════════════════════
# TEMPORAL SHIFT AUGMENTATION
# ═══════════════════════════════════════════════════════════════

def temporal_shift(batch, ms_range=(200, 500)):
    shifted = batch['waveform'].clone()
    n = random.randint(int(ms_range[0]/1000*16000), int(ms_range[1]/1000*16000))
    if random.random() > 0.5:
        shifted = F.pad(shifted[:, n:],  (0, n))
    else:
        shifted = F.pad(shifted[:, :-n], (n, 0))
    return shifted


# ═══════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════

def train_epoch(model, dl, loss_fn, opt, sched, dev, ep, gc=1.0):
    model.train()
    run = {k: 0. for k in ["total","ce","contrastive","sync_reg","consistency","shift"]}
    n   = 0
    pa, la = [], []

    for bd, labels, _ in tqdm(dl, desc=f"  Train E{ep:02d}", leave=False):
        for k, v in bd.items():
            if isinstance(v, torch.Tensor): bd[k] = v.to(dev)
        labels = labels.to(dev)

        # Forward — pass labels so ScaleGate can use class embeddings
        out = model(bd, labels=labels)

        ss_shift = None
        if (labels == 3).any() and ep >= 5:
            sw = temporal_shift(bd).to(dev)
            with torch.no_grad():
                so = model({**bd, 'waveform': sw}, labels=labels)
            ss_shift = so['sync_scores'].detach()

        losses = loss_fn(out, labels, ep, ss_shift)
        opt.zero_grad()
        losses['total'].backward()
        nn.utils.clip_grad_norm_(model.parameters(), gc)
        opt.step()

        for k in run:
            v = losses.get(k, 0)
            run[k] += v if isinstance(v, (int, float)) else v.item()
        n += 1
        pa.extend(out['logits'].argmax(-1).cpu().tolist())
        la.extend(labels.cpu().tolist())

    if sched: sched.step()
    return {k: v / max(n, 1) for k, v in run.items()} | {"acc": accuracy_score(la, pa)}


@torch.no_grad()
def evaluate(model, dl, loss_fn, dev, ep, name="Val"):
    """
    Evaluation loop.  Labels are NOT passed to model.forward() — the ScaleGate
    uses sync scores (the inference path).  Labels are only used for loss and
    metric computation after the forward pass.
    """
    model.eval()
    run = {k: 0. for k in ["total","ce","contrastive","sync_reg","consistency","shift"]}
    n   = 0
    pa, la, aa = [], [], []

    for bd, labels, _ in tqdm(dl, desc=f"  {name} E{ep:02d}", leave=False):
        for k, v in bd.items():
            if isinstance(v, torch.Tensor): bd[k] = v.to(dev)
        labels_dev = labels.to(dev)

        # No labels → ScaleGate uses sync scores (inference path)
        out    = model(bd)
        losses = loss_fn(out, labels_dev, ep)

        for k in run:
            v = losses.get(k, 0)
            run[k] += v if isinstance(v, (int, float)) else v.item()
        n += 1
        pa.extend(out['logits'].argmax(-1).cpu().tolist())
        la.extend(labels.tolist())
        aa.append(out['alphas'].cpu())

    avg = {k: v / max(n, 1) for k, v in run.items()}
    am  = torch.cat(aa).mean(0).tolist() if aa else [.33, .33, .33]
    return avg | {
        "acc":         accuracy_score(la, pa),
        "f1_macro":    f1_score(la, pa, average="macro",    zero_division=0),
        "f1_weighted": f1_score(la, pa, average="weighted", zero_division=0),
        "mean_alphas": am,
        "preds":       pa,
        "labels":      la}


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="HAVDNet-W Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_havdnet_w.py --data_root ./processed_multiscale

  # Cross-demographic: train African men, test African women
  python train_havdnet_w.py --data_root ./processed_multiscale \\
      --ethnicity African --gender men \\
      --test_ethnicity African --test_gender women

  python train_havdnet_w.py --data_root ./processed_multiscale --list
""")
    p.add_argument("--data_root",       required=True)
    p.add_argument("--epochs",          type=int,   default=30)
    p.add_argument("--batch_size",      type=int,   default=8)
    p.add_argument("--lr",              type=float, default=3e-4)
    p.add_argument("--grad_clip",       type=float, default=1.0)
    p.add_argument("--device",          default=None)
    p.add_argument("--num_workers",     type=int,   default=2)
    p.add_argument("--save_dir",        default="checkpoints")
    p.add_argument("--cache_dir",       default="checkpoints/arcface_cache")
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--ethnicity",       default=None)
    p.add_argument("--gender",          default=None, choices=["men","women"])
    p.add_argument("--category",        default=None)
    p.add_argument("--test_ethnicity",  default=None)
    p.add_argument("--test_gender",     default=None, choices=["men","women"])
    p.add_argument("--list",            action="store_true")
    args = p.parse_args()

    set_seed(args.seed)
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    sd  = Path(args.save_dir);  sd.mkdir(parents=True, exist_ok=True)
    cd  = Path(args.cache_dir); cd.mkdir(parents=True, exist_ok=True)

    if args.list:
        ds = FakeAVCelebDataset(args.data_root, cache_dir=args.cache_dir, split_name="scan")
        dm = ds.get_demographics()
        print(f"\n{'='*60}\nAVAILABLE ({len(ds)} samples, {dm['n_identities']} identities)\n{'='*60}")
        for k in ["ethnicities","genders","categories"]:
            print(f"\n  {k}:")
            for nm, c in sorted(dm[k].items()): print(f"    {nm:30s}: {c:5d}")
        return

    print("="*70)
    print("HAVDNet-W — Syllable / Word / Sentence")
    print("Splits: identity-grouped (no speaker leakage across train/val/test)")
    print("ArcFace cache: split-namespaced (no cross-split contamination)")
    print("="*70)

    cross = args.test_ethnicity is not None or args.test_gender is not None

    if cross:
        te, tg = args.test_ethnicity or args.ethnicity, args.test_gender or args.gender
        print(f"  Mode: CROSS-DEMOGRAPHIC")
        print(f"  Train: eth={args.ethnicity or 'all'} gen={args.gender or 'all'}")
        print(f"  Test:  eth={te or 'all'} gen={tg or 'all'}")

        # Full train pool and separate test demographic
        train_pool = FakeAVCelebDataset(
            args.data_root, ethnicity=args.ethnicity, gender=args.gender,
            category=args.category, cache_dir=args.cache_dir, split_name="pool")
        test_full  = FakeAVCelebDataset(
            args.data_root, ethnicity=te, gender=tg,
            category=args.category, cache_dir=args.cache_dir, split_name="test")

        rel_train, rel_val = grouped_split_cross(
            train_pool.samples, test_full.samples, val_frac=0.15, seed=args.seed)

        train_ds = FakeAVCelebDataset(
            args.data_root, rel_train, augment=True, device=dev,
            ethnicity=args.ethnicity, gender=args.gender, category=args.category,
            cache_dir=args.cache_dir, split_name="train")
        val_ds   = FakeAVCelebDataset(
            args.data_root, rel_val, augment=False, device=dev,
            ethnicity=args.ethnicity, gender=args.gender, category=args.category,
            cache_dir=args.cache_dir, split_name="val")
        test_ds  = test_full   # already filtered; split_name="test"

    else:
        full = FakeAVCelebDataset(
            args.data_root, ethnicity=args.ethnicity, gender=args.gender,
            category=args.category, cache_dir=args.cache_dir, split_name="pool")

        if args.ethnicity: print(f"  Ethnicity: {args.ethnicity}")
        if args.gender:    print(f"  Gender:    {args.gender}")
        print(f"  Total: {len(full)}")

        train_idx, val_idx, test_idx = grouped_split(
            full.samples, test_size=0.10, val_size=0.15, seed=args.seed)

        train_ds = FakeAVCelebDataset(
            args.data_root, train_idx, augment=True, device=dev,
            ethnicity=args.ethnicity, gender=args.gender, category=args.category,
            cache_dir=args.cache_dir, split_name="train")
        val_ds   = FakeAVCelebDataset(
            args.data_root, val_idx, augment=False, device=dev,
            ethnicity=args.ethnicity, gender=args.gender, category=args.category,
            cache_dir=args.cache_dir, split_name="val")
        test_ds  = FakeAVCelebDataset(
            args.data_root, test_idx, augment=False, device=dev,
            ethnicity=args.ethnicity, gender=args.gender, category=args.category,
            cache_dir=args.cache_dir, split_name="test")

    lc = train_ds.get_label_counts()
    print(f"  Train:{len(train_ds)}  Val:{len(val_ds)}  Test:{len(test_ds)}")
    for i, (nm, c) in enumerate(zip(CLASS_NAMES, lc)):
        print(f"    [{i}] {nm:28s}: {c:5d}")

    # Build sampler once — reused every epoch
    sampler = build_sampler(train_ds)

    tdl  = DataLoader(train_ds, args.batch_size, sampler=sampler,
                      num_workers=args.num_workers, pin_memory=True,
                      collate_fn=collate_fn, drop_last=True,
                      persistent_workers=(args.num_workers > 0))
    vdl  = DataLoader(val_ds,  args.batch_size, shuffle=False,
                      num_workers=args.num_workers, pin_memory=True,
                      collate_fn=collate_fn,
                      persistent_workers=(args.num_workers > 0))
    tedl = DataLoader(test_ds, args.batch_size, shuffle=False,
                      num_workers=args.num_workers, pin_memory=True,
                      collate_fn=collate_fn,
                      persistent_workers=(args.num_workers > 0))

    import clip
    from transformers import Wav2Vec2Model
    from havdnet_w import HAVDNetW, HAVDNetLoss

    cm, _  = clip.load("ViT-B/32", device=dev)
    w2v    = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53").to(dev)
    model  = HAVDNetW(cm, w2v, ew=3).to(dev)

    tp = sum(x.numel() for x in model.parameters())
    tr = sum(x.numel() for x in model.parameters() if x.requires_grad)
    print(f"  Params: {tp:,} total, {tr:,} trainable")

    pid = set(id(x) for x in model.prompt_refiner.parameters())
    opt = optim.AdamW([
        {"params": [x for x in model.parameters() if id(x) in pid],
         "lr": args.lr * 3},
        {"params": [x for x in model.parameters()
                    if id(x) not in pid and x.requires_grad],
         "lr": args.lr}],
        weight_decay=1e-4)
    sched   = optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs, 1e-6)
    loss_fn = HAVDNetLoss(lc).to(dev)

    best, be, hist = 0., 0, []
    for ep in range(1, args.epochs + 1):
        print(f"\nEpoch {ep}/{args.epochs}")
        tm = train_epoch(model, tdl, loss_fn, opt, sched, dev, ep, args.grad_clip)
        print(f"  Train — loss:{tm['total']:.4f} CE:{tm['ce']:.4f} acc:{tm['acc']:.4f}")
        vm = evaluate(model, vdl, loss_fn, dev, ep)
        print(f"  Val   — loss:{vm['total']:.4f} acc:{vm['acc']:.4f} "
              f"F1m:{vm['f1_macro']:.4f} "
              f"α=[{vm['mean_alphas'][0]:.3f},"
              f"{vm['mean_alphas'][1]:.3f},"
              f"{vm['mean_alphas'][2]:.3f}]")
        if vm['f1_macro'] > best:
            best, be = vm['f1_macro'], ep
            torch.save({"epoch": ep, "state": model.state_dict(),
                        "f1": best, "args": vars(args)}, sd/"best.pt")
            print(f"  ★ Best (F1={best:.4f})")
        hist.append({"epoch": ep, "train": tm,
                     "val": {k: v for k, v in vm.items()
                             if k not in ("preds","labels")}})

    print(f"\nLoading best checkpoint (epoch {be})...")
    model.load_state_dict(torch.load(sd/"best.pt", weights_only=False)["state"])
    tm = evaluate(model, tedl, loss_fn, dev, be, "Test")
    print(f"\n  Test — Acc:{tm['acc']:.4f}  F1m:{tm['f1_macro']:.4f}  "
          f"F1w:{tm['f1_weighted']:.4f}")
    print(f"         α=[{tm['mean_alphas'][0]:.3f},"
          f"{tm['mean_alphas'][1]:.3f},"
          f"{tm['mean_alphas'][2]:.3f}]")

    cm_ = confusion_matrix(tm["labels"], tm["preds"])
    print(f"\n  Confusion matrix:")
    for i, nm in enumerate(CLASS_NAMES):
        print(f"    {nm:28s}  {' '.join(f'{cm_[i][j]:5d}' for j in range(4))}")

    rpt = classification_report(tm["labels"], tm["preds"],
                                target_names=CLASS_NAMES, digits=4, zero_division=0)
    (sd/"test_report.txt").write_text(rpt)
    (sd/"history.json").write_text(json.dumps(hist, indent=2, default=str))
    (sd/"experiment.json").write_text(json.dumps({
        "mode":       "cross" if cross else "standard",
        "train_eth":  args.ethnicity,
        "train_gen":  args.gender,
        "test_eth":   args.test_ethnicity or args.ethnicity,
        "test_gen":   args.test_gender    or args.gender,
        "best_epoch": be,
        "best_f1":    best,
        "test_acc":   tm['acc'],
        "test_f1":    tm['f1_macro']}, indent=2))

    print(f"\n  Results saved to {sd}/")


if __name__ == "__main__":
    main()
