import os, json, math, random, argparse, hashlib
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.amp import GradScaler, autocast
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
# CLIP SENTENCE EMBEDDING
# Replaces ArcFace entirely — CLIP is already loaded for the model,
# zero extra cost, no preprocessing, no cache files needed.
# We load scale3 full_frames, run them through CLIP's image encoder
# (batched), mean-pool → (N, 512) per video.
# ═══════════════════════════════════════════════════════════════

# Face transform for CLIP (224×224, ImageNet normalisation)
_CLIP_TFM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.48145466, 0.4578275,  0.40821073],
                         [0.26862954, 0.26130258, 0.27577711]),
])

# Module-level CLIP reference — set once in main() after clip.load()
_CLIP_MODEL = None


def extract_clip_sentence(frame_dir, n_frames=MAX_SENTENCE_FRAMES, device="cpu"):
    """
    Load up to n_frames jpgs from frame_dir, encode with CLIP image encoder,
    return (N, 512) float32 tensor.  Falls back to zeros if no frames found.

    Caches result as clip_sentence.pt next to the frames directory so
    subsequent calls are a single torch.load (~1ms) instead of CLIP forward.
    """
    global _CLIP_MODEL
    if frame_dir is None:
        return torch.zeros(1, 512)

    frame_path = Path(frame_dir)
    cache_pt   = frame_path.parent / "clip_sentence.pt"   # sidecar next to scale3/

    if cache_pt.exists():
        return torch.load(cache_pt, weights_only=True)

    if not frame_path.exists():
        return torch.zeros(1, 512)

    jpgs = sorted(frame_path.glob("*.jpg"))[:n_frames]
    if not jpgs:
        result = torch.zeros(1, 512)
        torch.save(result, cache_pt)
        return result

    imgs = torch.stack([_CLIP_TFM(Image.open(p).convert("RGB")) for p in jpgs])
    with torch.no_grad():
        embs = _CLIP_MODEL.encode_image(imgs.to(device)).float().cpu()
    result = F.normalize(embs, dim=-1)
    torch.save(result, cache_pt)
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
                 ethnicity=None, gender=None, category=None,
                 split_name="train"):
        self.root       = Path(data_root)
        self.augment    = augment
        self.device     = device
        self.split_name = split_name
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
        # ── Scan cache ───────────────────────────────────────────
        # rglob on 20k+ videos is slow (many stat calls). Cache the result
        # as a JSON file next to the data root. Delete it to force rescan.
        cache_file = self.root / ".scan_cache.json"
        if cache_file.exists():
            print(f"  Loading scan cache from {cache_file} ...")
            with open(cache_file) as f:
                samples = json.load(f)
            # Convert video_dir back to Path
            for s in samples:
                s["video_dir"] = Path(s["video_dir"])
            print(f"  {len(samples)} videos loaded from cache")
            return samples

        print(f"  Scanning {self.root} for videos (first run — will cache result)...")
        meta_files = sorted(self.root.rglob("meta.json"))
        samples = []

        for mp in tqdm(meta_files, desc="  Scanning", unit="video"):
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
            # Use meta frame count instead of glob (avoids 20k extra filesystem calls)
            if meta.get("n_lip_frames", 2) < 2: continue

            # Pre-load alignment data here so __getitem__ never opens a file
            try:
                with open(vd/"audio"/"alignment.json") as f:
                    align = json.load(f)
            except Exception:
                continue

            samples.append({
                "video_dir":    str(vd),   # str for JSON serialisation
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
                # Pre-loaded — __getitem__ reads these directly, zero file I/O
                "syllables":    align.get("syllables", []),
                "words":        align.get("words", []),
            })

        # Save cache — delete .scan_cache.json to force rescan
        with open(cache_file, "w") as f:
            json.dump(samples, f)
        print(f"  Scan complete: {len(samples)} videos. Cache saved to {cache_file}")

        # Convert video_dir back to Path for consistency
        for s in samples:
            s["video_dir"] = Path(s["video_dir"])
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
        aug = self.augment and label in (1, 2)
        lt  = self.lip_aug if aug else self.lip_tfm
        ft  = self.face_aug if aug else self.face_tfm

        # Alignment pre-loaded during _scan — no file I/O here
        syls  = s["syllables"]
        words = s["words"]

        # ── SYLLABLE ─────────────────────────────────────────────
        # Load from cached .pt if available (single torch.load ~2ms)
        # else fall back to JPEG loading (~200ms) and save cache.
        syl_pt = Path(lip_dir).parent / "syl_crops.pt"
        if syl_pt.exists():
            raw = torch.load(syl_pt, weights_only=True)
            syl_crops = raw.float()  # handles both fp16 (new) and fp32 (old) caches
            # Build audio boundaries from pre-loaded alignment (no file I/O)
            sb = []
            for seg in syls[:MAX_SYLLABLES]:
                sf2, ef2 = seg.get("start_frame", 0), seg.get("end_frame", 0)
                _, _, ss2, se2 = pad_boundary(sf2, ef2, fps, tf)
                sb.append((ss2, se2))
            while len(sb) < MAX_SYLLABLES: sb.append((0., 0.))
        else:
            sc, sb = [], []
            for seg in syls[:MAX_SYLLABLES]:
                sf2, ef2 = seg.get("start_frame", 0), seg.get("end_frame", 0)
                sf2, ef2, ss2, se2 = pad_boundary(sf2, ef2, fps, tf)
                sc.append(self._load_frames(lip_dir, sf2, ef2, lt, MAX_FRAMES_PER_SYL))
                sb.append((ss2, se2))
            d_s = torch.zeros(MAX_FRAMES_PER_SYL, 3, 48, 96)
            while len(sc) < MAX_SYLLABLES: sc.append(d_s); sb.append((0., 0.))
            syl_crops = torch.stack(sc[:MAX_SYLLABLES])
            # Save as fp16 — halves disk I/O (10.5MB → 5.3MB per video)
            if not aug:
                try: torch.save(syl_crops.half(), syl_pt)
                except Exception: pass

        # ── WORD — cached CLIP embeddings ────────────────────────
        # clip_word.pt: (N_word, T_word, 512) pre-extracted CLIP embeddings.
        # Falls back to zero tensor if cache missing (will be slow until precached).
        clip_word_pt = Path(face_dir).parent / "clip_word.pt"
        if clip_word_pt.exists():
            word_embs = torch.load(clip_word_pt, weights_only=True)  # (N, T, 512)
        else:
            word_embs = torch.zeros(MAX_WORDS, MAX_FRAMES_PER_WORD, 512)

        # ── SENTENCE — CLIP face embedding ───────────────────────
        # extract_clip_sentence loads scale3 frames and encodes with CLIP.
        # ~5ms on GPU per video. No cache, no preprocessing step needed.
        sa_raw = extract_clip_sentence(
            s["full_dir"], n_frames=MAX_SENTENCE_FRAMES, device=self.device)

        valid_frames = min(sa_raw.shape[0], MAX_SENTENCE_FRAMES)
        if sa_raw.shape[0] < MAX_SENTENCE_FRAMES:
            sa = torch.cat([sa_raw,
                            torch.zeros(MAX_SENTENCE_FRAMES - sa_raw.shape[0], 512)])
        else:
            sa = sa_raw[:MAX_SENTENCE_FRAMES]

        sent_bounds = []
        for i in range(MAX_SENTENCE_FRAMES):
            if i < valid_frames:
                sent_bounds.append((i / PHRASE_FPS, (i + 1) / PHRASE_FPS))
            else:
                sent_bounds.append((0.0, 0.0))

        # ── WAVEFORM ─────────────────────────────────────────────
        # Cache resampled waveform as a .pt sidecar next to audio.wav.
        # First call: loads wav, resamples to 16kHz, saves .pt — takes ~0.1s.
        # All subsequent calls: torch.load — takes ~1ms.
        wav_path = Path(vd) / "audio" / "audio.wav"
        pt_path  = wav_path.with_suffix(".pt")
        if pt_path.exists():
            wav = torch.load(pt_path, weights_only=True)
        else:
            wav, sr = torchaudio.load(str(wav_path))
            if sr != 16000:
                wav = torchaudio.functional.resample(wav, sr, 16000)
            wav = wav.mean(0)
            if wav.shape[0] > MAX_AUDIO_SAMPLES:
                wav = wav[:MAX_AUDIO_SAMPLES]
            elif wav.shape[0] < MAX_AUDIO_SAMPLES:
                wav = F.pad(wav, (0, MAX_AUDIO_SAMPLES - wav.shape[0]))
            torch.save(wav, pt_path)

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
            "word_face_crops":    word_embs,
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


def build_sampler(ds, max_oversample=10.0):
    """
    Weighted sampler with a capped oversampling ratio.

    Without capping: minority class (37 samples) gets weight 834/37 = 22.5×
    → same 37 videos repeat 22× per epoch → model memorises them → 100% train acc.

    With max_oversample=10: minority gets at most 10× the majority weight.
    This keeps classes reasonably balanced without causing memorisation.
    Adjust max_oversample down if still overfitting, up if minority recall is poor.
    """
    labels = [s["label"] for s in ds.samples]
    c      = Counter(labels)
    mx     = max(c.values())
    # Raw weight = mx / count, capped at max_oversample
    cw     = {k: min(mx / v, max_oversample) for k, v in c.items()}
    weights = [cw[l] for l in labels]
    # num_samples = actual dataset size (not inflated)
    return WeightedRandomSampler(weights, len(labels), replacement=True)


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


# (No ArcFace precaching needed — sentence embeddings use CLIP, loaded at train time)

def precache_wav(samples, desc="Caching wav"):
    """
    Pre-convert all audio.wav files to .pt tensors in one pass with a progress
    bar, so the first training epoch isn't silently slow.
    Safe to call multiple times — skips files that already have a .pt sidecar.
    """
    import torchaudio as _ta
    skipped = 0
    for s in tqdm(samples, desc=f"  {desc}", unit="video"):
        wav_path = Path(s["video_dir"]) / "audio" / "audio.wav"
        pt_path  = wav_path.with_suffix(".pt")
        if pt_path.exists():
            skipped += 1
            continue
        try:
            wav, sr = _ta.load(str(wav_path))
            if sr != 16000:
                wav = _ta.functional.resample(wav, sr, 16000)
            wav = wav.mean(0)
            if wav.shape[0] > MAX_AUDIO_SAMPLES:
                wav = wav[:MAX_AUDIO_SAMPLES]
            elif wav.shape[0] < MAX_AUDIO_SAMPLES:
                wav = F.pad(wav, (0, MAX_AUDIO_SAMPLES - wav.shape[0]))
            torch.save(wav, pt_path)
        except Exception as e:
            pass   # skip corrupt files silently; __getitem__ will handle them
    cached = len(samples) - skipped
    print(f"  Wav cache: {cached} converted, {skipped} already cached")


def _cache_one_video_lip(args):
    """Worker function for parallel lip caching. Runs in a process pool."""
    s, lip_tfm_params = args
    lip_dir = s.get("lip_dir")
    if not lip_dir:
        return 0
    syl_pt = Path(lip_dir).parent / "syl_crops.pt"
    if syl_pt.exists():
        return -1  # already cached

    lip_tfm = transforms.Compose([
        transforms.Resize((48, 96)), transforms.ToTensor(),
        transforms.Normalize([.485,.456,.406],[.229,.224,.225])])

    try:
        fps = s["fps"]; tf = s["total_frames"]
        syls = s["syllables"]
        sc = []
        for seg in syls[:MAX_SYLLABLES]:
            sf2 = seg.get("start_frame", 0)
            ef2 = seg.get("end_frame", 0)
            # inline pad_boundary
            n = ef2 - sf2
            if n < MIN_CONTEXT_FRAMES:
                deficit = MIN_CONTEXT_FRAMES - n
                sf2 = max(0, sf2 - deficit // 2)
                ef2 = min(tf, ef2 + deficit - deficit // 2)
            frames = []
            for i in range(sf2, min(ef2, sf2 + MAX_FRAMES_PER_SYL)):
                fp = Path(lip_dir) / f"frame_{i:06d}.jpg"
                if fp.exists():
                    frames.append(lip_tfm(Image.open(fp).convert("RGB")))
            if not frames:
                frames = [torch.zeros(3, 48, 96)]
            while len(frames) < MAX_FRAMES_PER_SYL:
                frames.append(frames[-1])
            sc.append(torch.stack(frames[:MAX_FRAMES_PER_SYL]))
        d_s = torch.zeros(MAX_FRAMES_PER_SYL, 3, 48, 96)
        while len(sc) < MAX_SYLLABLES:
            sc.append(d_s)
        torch.save(torch.stack(sc[:MAX_SYLLABLES]).half(), syl_pt)
        return 1
    except Exception:
        return 0


def precache_lip(samples, desc="Caching lip crops", num_workers=16):
    """
    Pre-render all syllable lip crops to syl_crops.pt (fp16) in parallel.
    Uses ThreadPoolExecutor (not mp.Pool) to avoid CUDA fork deadlock —
    mp.Pool with spawn/fork after CUDA init hangs on Linux.
    Threads share memory safely here since each writes to a different file.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    to_cache = [s for s in samples
                if s.get("lip_dir") and
                not (Path(s["lip_dir"]).parent / "syl_crops.pt").exists()]
    skipped = len(samples) - len(to_cache)

    if not to_cache:
        print(f"  Lip cache: 0 saved, {skipped} already cached")
        return

    print(f"  Caching {len(to_cache)} videos across {num_workers} threads...")

    args = [(s, None) for s in to_cache]
    saved = 0; failed = 0

    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futures = {ex.submit(_cache_one_video_lip, a): a for a in args}
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc=f"  {desc}", unit="video"):
            r = fut.result()
            if r == 1:  saved  += 1
            elif r == 0: failed += 1

    print(f"  Lip cache: {saved} saved, {skipped} already cached"
          + (f", {failed} failed" if failed else ""))


def precache_clip_all(samples, clip_model, device="cuda",
                      batch_size=256, num_workers=4):
    """
    GPU-batched CLIP caching for BOTH sentence and word scales in one pass.

    Design for maximum GPU utilisation:
      - PyTorch DataLoader with num_workers CPU workers pre-loads and
        transforms images while the GPU runs CLIP on the previous batch
        (double-buffered pipeline via pin_memory + non_blocking transfers).
      - One large batch (default 256 images) fills the GPU fully rather
        than drip-feeding one video at a time.
      - Both sentence (scale3) and word (scale2) frames are queued into
        the same DataLoader so one CLIP model handles both in one sweep.
      - Already-cached files are skipped before image loading (zero cost).

    Saves:
      <video_dir>/clip_sentence.pt  — (N_sent, 512)  sentence embeddings
      <video_dir>/clip_word.pt      — (N_word, T, 512) word embeddings
    """
    from torch.utils.data import Dataset as _DS, DataLoader as _DL

    CLIP_TFM = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.48145466, 0.4578275,  0.40821073],
                             [0.26862954, 0.26130258, 0.27577711]),
    ])

    # ── Build work list (skip already-cached) ─────────────────────────────
    # Each item: (img_path, cache_pt, slot_key)
    # slot_key groups images that belong to the same output tensor.
    # We collect all JPEGs, run them through CLIP in one big batched pass,
    # then scatter results back into per-video .pt files.

    work = []   # (img_path_str, cache_pt_str, video_idx, slot_type, pos_n, pos_t)
    video_meta = {}   # video_idx → {sent_cache, sent_n, word_cache, word_shape}

    vidx = 0
    for s in samples:
        full_dir  = s.get("full_dir")
        face_dir  = s.get("face_dir")
        need_sent = full_dir  and not (Path(full_dir).parent  / "clip_sentence.pt").exists()
        need_word = face_dir  and not (Path(face_dir).parent  / "clip_word.pt").exists()
        if not need_sent and not need_word:
            continue

        meta = {}
        if need_sent:
            sent_cache = str(Path(full_dir).parent / "clip_sentence.pt")
            jpgs = sorted(Path(full_dir).glob("*.jpg"))[:MAX_SENTENCE_FRAMES]
            meta["sent_cache"] = sent_cache
            meta["sent_n"]     = len(jpgs)
            for n_i, jp in enumerate(jpgs):
                work.append((str(jp), sent_cache, vidx, "sent", n_i, 0))

        if need_word:
            word_cache = str(Path(face_dir).parent / "clip_word.pt")
            meta["word_cache"] = word_cache
            # collect all word frame paths in order
            fps = s["fps"]; tf = s["total_frames"]
            word_jpgs = []   # list of (n_idx, t_idx, path_str)
            for n_i, seg in enumerate(s["words"][:MAX_WORDS]):
                sf = seg.get("start_frame", 0)
                ef = seg.get("end_frame", 0)
                # inline pad_boundary
                deficit = max(0, MIN_CONTEXT_FRAMES - (ef - sf))
                sf = max(0, sf - deficit // 2)
                ef = min(tf, ef + deficit - deficit // 2)
                t_i = 0
                for fi in range(sf, min(ef, sf + MAX_FRAMES_PER_WORD)):
                    fp = Path(face_dir) / f"frame_{fi:06d}.jpg"
                    if fp.exists():
                        word_jpgs.append((n_i, t_i, str(fp)))
                        t_i += 1
                        if t_i >= MAX_FRAMES_PER_WORD:
                            break
            meta["word_jpgs"]  = word_jpgs
            meta["word_cache"] = word_cache
            for n_i, t_i, jp in word_jpgs:
                work.append((jp, word_cache, vidx, "word", n_i, t_i))

        if meta:
            video_meta[vidx] = meta
            vidx += 1

    if not work:
        print("  CLIP cache: all files already cached")
        return

    total_imgs = len(work)
    print(f"  CLIP cache: {total_imgs:,} images across {vidx} videos — batching on GPU...")

    # ── Dataset that loads one image + its metadata ───────────────────────
    class ImgDataset(_DS):
        def __init__(self, items, tfm):
            self.items = items
            self.tfm   = tfm
        def __len__(self):
            return len(self.items)
        def __getitem__(self, i):
            path, cache_pt, vidx, slot, pos_n, pos_t = self.items[i]
            try:
                img = self.tfm(Image.open(path).convert("RGB"))
            except Exception:
                img = torch.zeros(3, 224, 224)
            return img, vidx, slot == "word", pos_n, pos_t

    ds     = ImgDataset(work, CLIP_TFM)
    loader = _DL(ds, batch_size=batch_size, num_workers=num_workers,
                 pin_memory=True, shuffle=False, drop_last=False)

    # Accumulators: vidx → {sent: list[(n, emb)], word: list[(n,t,emb)]}
    acc = {v: {"sent": [], "word": []} for v in video_meta}

    clip_model.eval()
    with torch.no_grad():
        for imgs, vidxs, is_word, pos_ns, pos_ts in tqdm(
                loader, desc="  CLIP GPU", unit="batch"):
            imgs = imgs.to(device, non_blocking=True)
            embs = clip_model.encode_image(imgs).float().cpu()
            embs = F.normalize(embs, dim=-1)

            for i in range(len(vidxs)):
                v  = vidxs[i].item()
                e  = embs[i]
                n  = pos_ns[i].item()
                t  = pos_ts[i].item()
                if is_word[i].item():
                    acc[v]["word"].append((n, t, e))
                else:
                    acc[v]["sent"].append((n, e))

    # ── Scatter results into per-video .pt files ──────────────────────────
    saved_sent = saved_word = 0
    for v, meta in video_meta.items():
        if "sent_cache" in meta and acc[v]["sent"]:
            n_frames = meta["sent_n"]
            sent_t   = torch.zeros(n_frames, 512)
            for n_i, e in acc[v]["sent"]:
                if n_i < n_frames:
                    sent_t[n_i] = e
            torch.save(sent_t, meta["sent_cache"])
            saved_sent += 1

        if "word_cache" in meta and acc[v]["word"]:
            word_t = torch.zeros(MAX_WORDS, MAX_FRAMES_PER_WORD, 512)
            for n_i, t_i, e in acc[v]["word"]:
                if n_i < MAX_WORDS and t_i < MAX_FRAMES_PER_WORD:
                    word_t[n_i, t_i] = e
            torch.save(word_t, meta["word_cache"])
            saved_word += 1

    skipped = len(samples) - vidx
    print(f"  CLIP cache: {saved_sent} sentence, {saved_word} word files saved "
          f"({skipped} already cached)")


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
        with autocast('cuda', enabled=(scaler is not None)):
            out = model(bd)

        # Temporal shift contrastive (epoch >= 5, real samples only)
        ss_shift = None
        if (labels == 3).any() and ep >= 5:
            sw = temporal_shift(bd['waveform']).to(dev)
            with torch.no_grad(), autocast('cuda', enabled=(scaler is not None)):
                so = model({**bd, 'waveform': sw})
            ss_shift = so['sync_scores'].detach()

        with autocast('cuda', enabled=(scaler is not None)):
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
            rw_all.append(out['game_info']['final_weights'].cpu())
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
        rw   = torch.cat(rw_all)               # (N, 3) — GT fusion scale weights
        labs = torch.tensor(la_all)
        # Mean scale weight per scale (syl/word/sent), grouped by ground-truth class
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


  # List available data subsets
  python train_havdnet_w.py --data_root ./processed_multiscale --list
""")
    p.add_argument("--data_root",     required=True)
    p.add_argument("--epochs",        type=int,   default=30)
    p.add_argument("--batch_size",    type=int,   default=32)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--grad_clip",     type=float, default=1.0)
    p.add_argument("--device",        default=None)
    p.add_argument("--num_workers",   type=int,   default=8,
                   help="DataLoader workers. 8 recommended for A5000 after caches built.")
    p.add_argument("--persistent_workers", action="store_true", default=True,
                   help="Keep workers alive between epochs (default True, needs num_workers>0).")
    p.add_argument("--no_amp",        action="store_true",
                   help="Disable automatic mixed precision (use if AMP causes NaN).")
    p.add_argument("--save_dir",      default="checkpoints")
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--resume",        action="store_true",
                   help="Resume from checkpoints/best.pt if it exists.")
    p.add_argument("--ethnicity",     default=None)
    p.add_argument("--gender",        default=None, choices=["men", "women"])
    p.add_argument("--category",      default=None)
    p.add_argument("--test_ethnicity",default=None)
    p.add_argument("--test_gender",   default=None, choices=["men", "women"])
    p.add_argument("--list",          action="store_true")
    p.add_argument("--max_oversample", type=float, default=10.0,
                   help="Max oversampling ratio for minority classes (default 10). "
                        "Lower if train acc hits 100% quickly, raise if minority recall is poor.")
    args = p.parse_args()

    set_seed(args.seed)
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    sd  = Path(args.save_dir);  sd.mkdir(parents=True, exist_ok=True)
    use_amp = (not args.no_amp) and ("cuda" in dev)

    # ── Quick-exit modes ──────────────────────────────────────
    if args.list:
        ds = FakeAVCelebDataset(args.data_root, split_name="scan")
        dm = ds.get_demographics()
        print(f"\n{'='*60}\nAVAILABLE ({len(ds)} samples, {dm['n_identities']} identities)\n{'='*60}")
        for k in ["ethnicities", "genders", "categories"]:
            print(f"\n  {k}:")
            for nm, c in sorted(dm[k].items()): print(f"    {nm:30s}: {c:5d}")
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
            category=args.category, split_name="pool")
        test_full  = FakeAVCelebDataset(
            args.data_root, ethnicity=te, gender=tg,
            category=args.category, split_name="test")

        rel_train, rel_val = grouped_split_cross(
            train_pool.samples, test_full.samples, val_frac=0.15, seed=args.seed)

        train_ds = FakeAVCelebDataset(
            args.data_root, rel_train, augment=True, device=dev,
            ethnicity=args.ethnicity, gender=args.gender, category=args.category,
            split_name="train")
        val_ds   = FakeAVCelebDataset(
            args.data_root, rel_val, augment=False, device=dev,
            ethnicity=args.ethnicity, gender=args.gender, category=args.category,
            split_name="val")
        test_ds  = test_full

    else:
        full = FakeAVCelebDataset(
            args.data_root, ethnicity=args.ethnicity, gender=args.gender,
            category=args.category, split_name="pool")

        if args.ethnicity: print(f"  Ethnicity: {args.ethnicity}")
        if args.gender:    print(f"  Gender:    {args.gender}")
        print(f"  Total: {len(full)}")

        train_idx, val_idx, test_idx = grouped_split(
            full.samples, test_size=0.10, val_size=0.15, seed=args.seed)

        train_ds = FakeAVCelebDataset(
            args.data_root, train_idx, augment=True, device=dev,
            ethnicity=args.ethnicity, gender=args.gender, category=args.category,
            split_name="train")
        val_ds   = FakeAVCelebDataset(
            args.data_root, val_idx, augment=False, device=dev,
            ethnicity=args.ethnicity, gender=args.gender, category=args.category,
            split_name="val")
        test_ds  = FakeAVCelebDataset(
            args.data_root, test_idx, augment=False, device=dev,
            ethnicity=args.ethnicity, gender=args.gender, category=args.category,
            split_name="test")

    lc = train_ds.get_label_counts()
    print(f"  Train:{len(train_ds)}  Val:{len(val_ds)}  Test:{len(test_ds)}")
    for i, (nm, c) in enumerate(zip(CLASS_NAMES, lc)):
        print(f"    [{i}] {nm:28s}: {c:5d}")

    # Pre-convert all wav files to .pt sidecars with a progress bar.
    # Skips files already cached. After this, __getitem__ loads in ~1ms not ~100ms.
    all_samples = train_ds.samples + val_ds.samples + test_ds.samples
    print(f"\nPre-caching {len(all_samples)} wav files (skips already-done)...")
    precache_wav(all_samples, desc="Wav → .pt")
    print(f"Pre-caching {len(all_samples)} lip crop files (skips already-done)...")
    precache_lip(all_samples, desc="Lip → .pt")
    print()

    # ── DataLoaders ───────────────────────────────────────────
    sampler = build_sampler(train_ds, max_oversample=args.max_oversample)
    pw = args.persistent_workers and args.num_workers > 0
    pm = args.num_workers > 0
    pf = 2 if args.num_workers > 0 else None   # prefetch 2 batches ahead

    tdl  = DataLoader(train_ds, args.batch_size, sampler=sampler,
                      num_workers=args.num_workers, pin_memory=pm,
                      collate_fn=collate_fn, drop_last=True,
                      persistent_workers=pw, prefetch_factor=pf)
    vdl  = DataLoader(val_ds,  args.batch_size, shuffle=False,
                      num_workers=args.num_workers, pin_memory=pm,
                      collate_fn=collate_fn, persistent_workers=pw,
                      prefetch_factor=pf)
    tedl = DataLoader(test_ds, args.batch_size, shuffle=False,
                      num_workers=args.num_workers, pin_memory=pm,
                      collate_fn=collate_fn, persistent_workers=pw,
                      prefetch_factor=pf)

    # ── Model, optimiser, scaler ──────────────────────────────
    # Load models from local cache (no internet required after setup_havdnet.sh).
    # load_models.py is written by the setup script next to this file.
    try:
        from load_models import load_clip, load_wav2vec2
        cm, _ = load_clip(dev)
        w2v   = load_wav2vec2(dev)
    except ImportError:
        # Fallback: load normally (requires internet on first run)
        import clip
        from transformers import Wav2Vec2Model
        cm, _ = clip.load("ViT-B/32", device=dev)
        w2v   = Wav2Vec2Model.from_pretrained(
                    "facebook/wav2vec2-large-xlsr-53").to(dev)

    # Make CLIP available to the dataset's extract_clip_sentence()
    import sys
    sys.modules[__name__]._CLIP_MODEL = cm

    from havdnet_w import HAVDNetW, HAVDNetLoss
    model = HAVDNetW(cm, w2v).to(dev)

    tp = sum(x.numel() for x in model.parameters())
    tr = sum(x.numel() for x in model.parameters() if x.requires_grad)
    print(f"  Params: {tp:,} total  {tr:,} trainable")

    # GPU-batched CLIP caching — sentence + word in one pass, max GPU util
    print(f"\nPre-caching CLIP embeddings ({len(all_samples)} videos)...")
    precache_clip_all(all_samples, cm, device=dev,
                      batch_size=512, num_workers=4)
    print()

    pid = set(id(x) for x in model.prompt_refiner.parameters())
    opt = optim.AdamW([
        {"params": [x for x in model.parameters() if id(x) in pid],
         "lr": args.lr * 3},
        {"params": [x for x in model.parameters()
                    if id(x) not in pid and x.requires_grad],
         "lr": args.lr}],
        weight_decay=1e-4)

    sched   = optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs, 1e-6)
    scaler  = GradScaler('cuda') if use_amp else None
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

    # Game-theoretic scale weights per ground-truth class
    if "routing_by_class" in tm:
        print(f"\n  GT fusion scale weights (mean per ground-truth class):")
        header = "  " + " " * 30 + "  Syllable   Word   Sentence"
        print(header)
        for cls_name, weights in tm["routing_by_class"].items():
            row = f"  {cls_name:30s}" + "".join(f"  {w:.3f}  " for w in weights)
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
