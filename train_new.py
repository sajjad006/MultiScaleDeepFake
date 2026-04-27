import os, json, math, random, argparse, hashlib
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
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
MAX_WORDS           = 12
MAX_FRAMES_PER_SYL  = 5
MAX_FRAMES_PER_WORD = 4
MAX_SENTENCE_FRAMES = 8
MAX_AUDIO_SAMPLES   = 16000 * 10
MIN_CONTEXT_FRAMES  = 3
PHRASE_FPS          = 1    # scale3 full_frames at 1 fps (every 25th source frame)


def set_seed(s=42):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


# ═══════════════════════════════════════════════════════════════
# ARCFACE CACHE
# ═══════════════════════════════════════════════════════════════

def _arcface_cache_path(video_id: str, split: str, cache_dir: Path) -> Path:
    key = f"{video_id}__{split}"
    h   = hashlib.md5(key.encode()).hexdigest()[:20]
    return cache_dir / f"arcface_{h}.pt"


def extract_arcface(frame_dir: Path, cache_path: Path, device="cpu"):
    """Extract ArcFace embeddings and cache.  Always CPU — safe for workers."""
    if cache_path.exists():
        return torch.load(cache_path, map_location="cpu", weights_only=True)

    if frame_dir is None or not frame_dir.exists():
        result = torch.zeros(1, 512)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(result, cache_path); return result

    frames = sorted(frame_dir.glob("*.jpg"))
    if not frames:
        result = torch.zeros(1, 512)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(result, cache_path); return result

    try:
        from insightface.app import FaceAnalysis
        import cv2
        app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(224, 224))
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
            vd = mp.parent
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

        # ── SYLLABLE ─────────────────────────────────────────────
        sc, sb = [], []
        for seg in syls[:MAX_SYLLABLES]:
            sf, ef = seg.get("start_frame", 0), seg.get("end_frame", 0)
            sf, ef, ss, se = pad_boundary(sf, ef, fps, tf)
            sc.append(self._load_frames(lip_dir, sf, ef, lt, MAX_FRAMES_PER_SYL))
            sb.append((ss, se))
        d_s = torch.zeros(MAX_FRAMES_PER_SYL, 3, 48, 96)
        while len(sc) < MAX_SYLLABLES: sc.append(d_s); sb.append((0., 0.))
        syl_crops = torch.stack(sc[:MAX_SYLLABLES])

        # ── WORD ─────────────────────────────────────────────────
        wc, wb = [], []
        for seg in words[:MAX_WORDS]:
            sf, ef = seg.get("start_frame", 0), seg.get("end_frame", 0)
            sf, ef, ss, se = pad_boundary(sf, ef, fps, tf)
            wc.append(self._load_frames(face_dir, sf, ef, ft, MAX_FRAMES_PER_WORD))
            wb.append((ss, se))
        d_w = torch.zeros(MAX_FRAMES_PER_WORD, 3, 224, 224)
        while len(wc) < MAX_WORDS: wc.append(d_w); wb.append((0., 0.))
        word_crops = torch.stack(wc[:MAX_WORDS])

        # ── SENTENCE ─────────────────────────────────────────────
        # ArcFace embeddings cached, split-namespaced to prevent cross-split leakage.
        full_path = Path(full_dir) if full_dir else None
        cp        = _arcface_cache_path(s["video_id"], self.split_name, self.cache_dir)
        sa_raw    = extract_arcface(full_path, cp, self.device)

        # Record valid frame count BEFORE zero-padding so audio boundaries align.
        valid_frames = min(sa_raw.shape[0], MAX_SENTENCE_FRAMES)
        if sa_raw.shape[0] < MAX_SENTENCE_FRAMES:
            sa = torch.cat([sa_raw,
                            torch.zeros(MAX_SENTENCE_FRAMES - sa_raw.shape[0], 512)])
        else:
            sa = sa_raw[:MAX_SENTENCE_FRAMES]

        # Audio boundaries locked to visual frame timestamps (scale3 = 1 fps).
        # Frame i covers [i/PHRASE_FPS, (i+1)/PHRASE_FPS) seconds.
        # Padded slots get (0.0, 0.0) so _pool skips them, mirroring zero visual rows.
        sent_bounds = []
        for i in range(MAX_SENTENCE_FRAMES):
            if i < valid_frames:
                sent_bounds.append((i / PHRASE_FPS, (i + 1) / PHRASE_FPS))
            else:
                sent_bounds.append((0.0, 0.0))

        # ── WAVEFORM ─────────────────────────────────────────────
        wav, sr = torchaudio.load(str(Path(vd)/"audio"/"audio.wav"))
        if sr != 16000: wav = torchaudio.functional.resample(wav, sr, 16000)
        wav = wav.mean(0)
        if wav.shape[0] > MAX_AUDIO_SAMPLES:
            wav = wav[:MAX_AUDIO_SAMPLES]
        elif wav.shape[0] < MAX_AUDIO_SAMPLES:
            wav = F.pad(wav, (0, MAX_AUDIO_SAMPLES - wav.shape[0]))

        # ── AUDIO BOUNDARIES ─────────────────────────────────────
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
                "sentence": sent_bounds},
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
    labels = [s["label"] for s in ds.samples]
    c      = Counter(labels)
    mx     = max(c.values())
    cw     = {k: mx / v for k, v in c.items()}
    return WeightedRandomSampler([cw[l] for l in labels], mx * len(c), replacement=True)


# ═══════════════════════════════════════════════════════════════
# SPLITTING
# ═══════════════════════════════════════════════════════════════

def grouped_split(samples, test_size, val_size, seed):
    """Identity-grouped split: no speaker appears in more than one partition."""
    identities = [s["identity"] for s in samples]
    idx        = list(range(len(samples)))

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tv_idx, test_idx = next(gss.split(idx, groups=identities))

    tv_identities = [identities[i] for i in tv_idx]
    val_frac      = val_size / (1.0 - test_size)
    gss2          = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    rel_train, rel_val = next(gss2.split(tv_idx, groups=tv_identities))

    return [tv_idx[i] for i in rel_train], [tv_idx[i] for i in rel_val], list(test_idx)


def grouped_split_cross(train_samples, test_samples, val_frac, seed):
    """Cross-demographic split: filter test identities from train pool first."""
    test_ids     = {s["identity"] for s in test_samples}
    safe_indices = [i for i, s in enumerate(train_samples)
                    if s["identity"] not in test_ids]

    if not safe_indices:
        raise ValueError(
            "After filtering test identities the training pool is empty. "
            "Ensure --ethnicity/--gender selects a population distinct from "
            "--test_ethnicity/--test_gender.")

    safe_ids  = [train_samples[i]["identity"] for i in safe_indices]
    local_idx = list(range(len(safe_indices)))

    gss = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    rel_tr, rel_val = next(gss.split(local_idx, groups=safe_ids))

    return [safe_indices[i] for i in rel_tr], [safe_indices[i] for i in rel_val]


# ═══════════════════════════════════════════════════════════════
# ARCFACE PRE-CACHING
# Run: python train_havdnet_w.py --data_root ... --precache
# Do this once before using num_workers > 0.
# ═══════════════════════════════════════════════════════════════

def precache_arcface(data_root, cache_dir, split_name="precache",
                     ethnicity=None, gender=None, category=None):
    ds = FakeAVCelebDataset(
        data_root, ethnicity=ethnicity, gender=gender,
        category=category, cache_dir=cache_dir, split_name=split_name)
    print(f"\nPre-caching ArcFace for {len(ds)} videos → {cache_dir}")
    cached, skipped = 0, 0
    for s in tqdm(ds.samples, desc="  ArcFace cache"):
        cp = _arcface_cache_path(s["video_id"], split_name, Path(cache_dir))
        if cp.exists(): skipped += 1; continue
        extract_arcface(Path(s["full_dir"]) if s["full_dir"] else None, cp)
        cached += 1
    print(f"  Done — {cached} extracted, {skipped} already cached.")


# ═══════════════════════════════════════════════════════════════
# TEMPORAL SHIFT AUGMENTATION
# ═══════════════════════════════════════════════════════════════

def temporal_shift(waveform, ms_range=(200, 500)):
    n = random.randint(int(ms_range[0]/1000*16000), int(ms_range[1]/1000*16000))
    if random.random() > 0.5:
        return F.pad(waveform[:, n:],  (0, n))
    return F.pad(waveform[:, :-n], (n, 0))


# ═══════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════════════════════════════

def train_epoch(model, dl, loss_fn, opt, sched, scaler, dev, ep, gc=1.0):
    model.train()
    run = {k: 0. for k in ["total", "ce", "contrastive", "sync_reg",
                            "consistency", "shift", "load_balance"]}
    n      = 0
    pa, la = [], []

    for bd, labels, _ in tqdm(dl, desc=f"  Train E{ep:02d}", leave=False):
        for k, v in bd.items():
            if isinstance(v, torch.Tensor): bd[k] = v.to(dev)
        labels = labels.to(dev)

        # Mixed-precision forward
        with autocast(enabled=(scaler is not None)):
            out = model(bd, labels=labels)

        # Temporal shift contrastive (epoch >= 5, real samples only)
        ss_shift = None
        if (labels == 3).any() and ep >= 5:
            sw = temporal_shift(bd['waveform']).to(dev)
            with torch.no_grad(), autocast(enabled=(scaler is not None)):
                so = model({**bd, 'waveform': sw}, labels=labels)
            ss_shift = so['sync_scores'].detach()

        with autocast(enabled=(scaler is not None)):
            losses = loss_fn(out, labels, ep, ss_shift)

        opt.zero_grad()
        if scaler is not None:
            scaler.scale(losses['total']).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), gc)
            scaler.step(opt)
            scaler.update()
        else:
            losses['total'].backward()
            nn.utils.clip_grad_norm_(model.parameters(), gc)
            opt.step()

        for k in run:
            v = losses.get(k, 0)
            run[k] += float(v) if isinstance(v, (int, float)) else v.item()
        n += 1
        pa.extend(out['logits'].argmax(-1).cpu().tolist())
        la.extend(labels.cpu().tolist())

    if sched: sched.step()
    return {k: v / max(n, 1) for k, v in run.items()} | {"acc": accuracy_score(la, pa)}


@torch.no_grad()
def evaluate(model, dl, loss_fn, dev, ep, name="Val", collect_routing=False):
    """
    Inference-mode evaluation.  Labels NOT passed to model.forward() so the
    router+ScaleGate use sync scores (inference path).

    collect_routing=True: also returns per-sample router_weights averaged by
    ground-truth label — used for MoE interpretability analysis at test time.
    """
    model.eval()
    run = {k: 0. for k in ["total", "ce", "contrastive", "sync_reg",
                            "consistency", "shift", "load_balance"]}
    n   = 0
    pa, la, aa = [], [], []
    rw_all, la_all = [], []   # for routing analysis

    for bd, labels, _ in tqdm(dl, desc=f"  {name} E{ep:02d}", leave=False):
        for k, v in bd.items():
            if isinstance(v, torch.Tensor): bd[k] = v.to(dev)
        labels_dev = labels.to(dev)

        out    = model(bd)
        losses = loss_fn(out, labels_dev, ep)

        for k in run:
            v = losses.get(k, 0)
            run[k] += float(v) if isinstance(v, (int, float)) else v.item()
        n += 1
        pa.extend(out['logits'].argmax(-1).cpu().tolist())
        la.extend(labels.tolist())
        aa.append(out['alphas'].cpu())

        if collect_routing:
            rw_all.append(out['router_weights'].cpu())
            la_all.extend(labels.tolist())

    avg = {k: v / max(n, 1) for k, v in run.items()}
    am  = torch.cat(aa).mean(0).tolist() if aa else [.33, .33, .33]

    result = avg | {
        "acc":         accuracy_score(la, pa),
        "f1_macro":    f1_score(la, pa, average="macro",    zero_division=0),
        "f1_weighted": f1_score(la, pa, average="weighted", zero_division=0),
        "mean_alphas": am,
        "preds":       pa,
        "labels":      la}

    if collect_routing and rw_all:
        rw   = torch.cat(rw_all)               # (N, n_experts)
        labs = torch.tensor(la_all)
        # Mean routing weight per expert, grouped by ground-truth class
        routing_by_class = {}
        for c in range(4):
            mask = labs == c
            routing_by_class[CLASS_NAMES[c]] = (
                rw[mask].mean(0).tolist() if mask.any() else [0.] * rw.shape[1])
        result["routing_by_class"] = routing_by_class

    return result


def _save_checkpoint(path, epoch, model, opt, scaler, best_f1, args):
    torch.save({
        "epoch":     epoch,
        "state":     model.state_dict(),
        "opt":       opt.state_dict(),
        "scaler":    scaler.state_dict() if scaler else None,
        "f1":        best_f1,
        "args":      vars(args),
    }, path)


def _load_checkpoint(path, model, opt, scaler, dev):
    ck = torch.load(path, map_location=dev, weights_only=False)
    model.load_state_dict(ck["state"])
    opt.load_state_dict(ck["opt"])
    if scaler and ck.get("scaler"):
        scaler.load_state_dict(ck["scaler"])
    return ck["epoch"], ck["f1"]


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="HAVDNet-W — Multi-scale AV deepfake detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard run
  python train_havdnet_w.py --data_root ./processed_multiscale

  # Resume interrupted run
  python train_havdnet_w.py --data_root ./processed_multiscale --resume

  # Cross-demographic
  python train_havdnet_w.py --data_root ./processed_multiscale \\
      --ethnicity African --gender men \\
      --test_ethnicity African --test_gender women

  # Warm ArcFace cache before using num_workers > 0
  python train_havdnet_w.py --data_root ./processed_multiscale --precache

  # List available data subsets
  python train_havdnet_w.py --data_root ./processed_multiscale --list
""")
    p.add_argument("--data_root",     required=True)
    p.add_argument("--epochs",        type=int,   default=30)
    p.add_argument("--batch_size",    type=int,   default=8)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--grad_clip",     type=float, default=1.0)
    p.add_argument("--device",        default=None)
    p.add_argument("--num_workers",   type=int,   default=0,
                   help="0 = main thread (safe). Run --precache first if using >0.")
    p.add_argument("--persistent_workers", action="store_true", default=False,
                   help="Keep workers alive between epochs. Needs --num_workers > 0 "
                        "and ≥16 GB system RAM.")
    p.add_argument("--no_amp",        action="store_true",
                   help="Disable automatic mixed precision (use if AMP causes NaN).")
    p.add_argument("--save_dir",      default="checkpoints")
    p.add_argument("--cache_dir",     default="checkpoints/arcface_cache")
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--resume",        action="store_true",
                   help="Resume from checkpoints/best.pt if it exists.")
    p.add_argument("--ethnicity",     default=None)
    p.add_argument("--gender",        default=None, choices=["men", "women"])
    p.add_argument("--category",      default=None)
    p.add_argument("--test_ethnicity",default=None)
    p.add_argument("--test_gender",   default=None, choices=["men", "women"])
    p.add_argument("--list",          action="store_true")
    p.add_argument("--precache",      action="store_true",
                   help="Warm ArcFace cache then exit. Run once before num_workers > 0.")
    args = p.parse_args()

    set_seed(args.seed)
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    sd  = Path(args.save_dir);  sd.mkdir(parents=True, exist_ok=True)
    cd  = Path(args.cache_dir); cd.mkdir(parents=True, exist_ok=True)
    use_amp = (not args.no_amp) and ("cuda" in dev)

    # ── Quick-exit modes ──────────────────────────────────────
    if args.list:
        ds = FakeAVCelebDataset(args.data_root, cache_dir=args.cache_dir, split_name="scan")
        dm = ds.get_demographics()
        print(f"\n{'='*60}\nAVAILABLE ({len(ds)} samples, {dm['n_identities']} identities)\n{'='*60}")
        for k in ["ethnicities", "genders", "categories"]:
            print(f"\n  {k}:")
            for nm, c in sorted(dm[k].items()): print(f"    {nm:30s}: {c:5d}")
        return

    if args.precache:
        precache_arcface(
            args.data_root, args.cache_dir,
            ethnicity=args.ethnicity, gender=args.gender, category=args.category)
        return

    # ── Dataset setup ─────────────────────────────────────────
    print("="*70)
    print("HAVDNet-W — Phoneme / Word / Phrase Deepfake Detector")
    print(f"AMP: {'enabled' if use_amp else 'disabled'}  "
          f"workers: {args.num_workers}  device: {dev}")
    print("="*70)

    cross = args.test_ethnicity is not None or args.test_gender is not None

    if cross:
        te = args.test_ethnicity or args.ethnicity
        tg = args.test_gender    or args.gender
        print(f"  Mode: CROSS-DEMOGRAPHIC")
        print(f"  Train: eth={args.ethnicity or 'all'}  gen={args.gender or 'all'}")
        print(f"  Test:  eth={te or 'all'}  gen={tg or 'all'}")

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
        test_ds  = test_full

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

    # ── DataLoaders ───────────────────────────────────────────
    sampler = build_sampler(train_ds)
    pw = args.persistent_workers and args.num_workers > 0
    pm = args.num_workers > 0

    tdl  = DataLoader(train_ds, args.batch_size, sampler=sampler,
                      num_workers=args.num_workers, pin_memory=pm,
                      collate_fn=collate_fn, drop_last=True, persistent_workers=pw)
    vdl  = DataLoader(val_ds,  args.batch_size, shuffle=False,
                      num_workers=args.num_workers, pin_memory=pm,
                      collate_fn=collate_fn, persistent_workers=pw)
    tedl = DataLoader(test_ds, args.batch_size, shuffle=False,
                      num_workers=args.num_workers, pin_memory=pm,
                      collate_fn=collate_fn, persistent_workers=pw)

    # ── Model, optimiser, scaler ──────────────────────────────
    import clip
    from transformers import Wav2Vec2Model
    from havdnet_w import HAVDNetW, HAVDNetLoss

    cm, _ = clip.load("ViT-B/32", device=dev)
    w2v   = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53").to(dev)
    model = HAVDNetW(cm, w2v).to(dev)          # ew=5 default; do NOT override

    tp = sum(x.numel() for x in model.parameters())
    tr = sum(x.numel() for x in model.parameters() if x.requires_grad)
    print(f"  Params: {tp:,} total  {tr:,} trainable")

    pid = set(id(x) for x in model.prompt_refiner.parameters())
    opt = optim.AdamW([
        {"params": [x for x in model.parameters() if id(x) in pid],
         "lr": args.lr * 3},
        {"params": [x for x in model.parameters()
                    if id(x) not in pid and x.requires_grad],
         "lr": args.lr}],
        weight_decay=1e-4)

    sched   = optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs, 1e-6)
    scaler  = GradScaler() if use_amp else None
    loss_fn = HAVDNetLoss(lc).to(dev)

    # ── Resume ────────────────────────────────────────────────
    start_ep = 1
    best, be = 0., 0
    ckpt_path = sd / "best.pt"

    if args.resume and ckpt_path.exists():
        start_ep, best = _load_checkpoint(ckpt_path, model, opt, scaler, dev)
        be = start_ep
        start_ep += 1
        print(f"  Resumed from epoch {start_ep-1}, best F1={best:.4f}")

    # ── Training loop ─────────────────────────────────────────
    hist = []
    for ep in range(start_ep, args.epochs + 1):
        print(f"\nEpoch {ep}/{args.epochs}")

        tm = train_epoch(model, tdl, loss_fn, opt, sched, scaler, dev, ep, args.grad_clip)
        print(f"  Train — loss:{tm['total']:.4f}  CE:{tm['ce']:.4f}  "
              f"sync:{tm['sync_reg']:.4f}  lb:{tm['load_balance']:.4f}  "
              f"acc:{tm['acc']:.4f}")

        vm = evaluate(model, vdl, loss_fn, dev, ep)
        print(f"  Val   — loss:{vm['total']:.4f}  acc:{vm['acc']:.4f}  "
              f"F1m:{vm['f1_macro']:.4f}  lb:{vm['load_balance']:.4f}  "
              f"α=[{vm['mean_alphas'][0]:.3f},"
              f"{vm['mean_alphas'][1]:.3f},"
              f"{vm['mean_alphas'][2]:.3f}]")

        if vm['f1_macro'] > best:
            best, be = vm['f1_macro'], ep
            _save_checkpoint(ckpt_path, ep, model, opt, scaler, best, args)
            print(f"  ★ Best (F1={best:.4f}) — checkpoint saved")

        hist.append({"epoch": ep, "train": tm,
                     "val": {k: v for k, v in vm.items()
                             if k not in ("preds", "labels")}})

    # ── Test evaluation + MoE routing analysis ────────────────
    print(f"\nLoading best checkpoint (epoch {be})...")
    _load_checkpoint(ckpt_path, model, opt, scaler, dev)

    tm = evaluate(model, tedl, loss_fn, dev, be, "Test", collect_routing=True)

    print(f"\n  Test — Acc:{tm['acc']:.4f}  F1m:{tm['f1_macro']:.4f}  "
          f"F1w:{tm['f1_weighted']:.4f}")
    print(f"  Scale weights α=[{tm['mean_alphas'][0]:.3f},"
          f"{tm['mean_alphas'][1]:.3f},"
          f"{tm['mean_alphas'][2]:.3f}]")

    # MoE routing table: rows = ground-truth class, cols = expert index
    if "routing_by_class" in tm:
        n_experts = len(next(iter(tm["routing_by_class"].values())))
        print(f"\n  MoE routing weights (mean per ground-truth class):")
        header = "  " + " " * 30 + "".join(f"  E{i}" for i in range(n_experts))
        print(header)
        for cls_name, weights in tm["routing_by_class"].items():
            row = f"  {cls_name:30s}" + "".join(f"  {w:.2f}" for w in weights)
            print(row)

    cm_ = confusion_matrix(tm["labels"], tm["preds"])
    print(f"\n  Confusion matrix:")
    for i, nm in enumerate(CLASS_NAMES):
        print(f"    {nm:28s}  {' '.join(f'{cm_[i][j]:5d}' for j in range(4))}")

    rpt = classification_report(tm["labels"], tm["preds"],
                                target_names=CLASS_NAMES, digits=4, zero_division=0)
    (sd/"test_report.txt").write_text(rpt)
    (sd/"history.json").write_text(json.dumps(hist, indent=2, default=str))
    (sd/"experiment.json").write_text(json.dumps({
        "mode":              "cross" if cross else "standard",
        "train_eth":         args.ethnicity,
        "train_gen":         args.gender,
        "test_eth":          args.test_ethnicity or args.ethnicity,
        "test_gen":          args.test_gender    or args.gender,
        "best_epoch":        be,
        "best_f1":           best,
        "test_acc":          tm["acc"],
        "test_f1":           tm["f1_macro"],
        "routing_by_class":  tm.get("routing_by_class", {}),
    }, indent=2))

    print(f"\n  Results saved to {sd}/")


if __name__ == "__main__":
    main()
