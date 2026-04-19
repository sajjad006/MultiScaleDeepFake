"""
FakeAVCeleb Multi-Scale Preprocessing Pipeline  —  Final Version (4-Scale)
===========================================================================
Produces per-video output:

  processed/
  └── {Category}/{Ethnicity}/{Gender}/{identity_id}/{video_id}/
      ├── scale1/lip_crops/
      │   └── frame_XXXXXX.jpg          96×48 JPEG, every frame at native fps
      │                                 Scale 1 — Syllable visual input
      ├── scale2/face_crops/
      │   └── frame_XXXXXX.jpg          96×96 JPEG, every frame at native fps
      │                                 Scale 2 — Word visual input
      ├── scale3/full_frames/
      │   └── frame_XXXXXX.jpg          224×224 JPEG, 1 fps subset
      │                                 Scale 3 — Phrase visual input
      ├── audio/
      │   ├── audio.wav                 16 kHz mono
      │   └── alignment.json
      └── meta.json

Three Visual Scales  +  Phoneme Alignment Metadata
───────────────────────────────────────────────────
  Scale 1 — Syllable (~150ms)  WhisperX words + pyphen      → lip_crops
  Scale 2 — Word     (~400ms)  WhisperX word boundaries     → face_crops
  Scale 3 — Phrase   (full)    full utterance, 1fps          → full_frames

  Phonemes (wav2vec2 CTC IPA, ~20ms) are stored in alignment.json for
  use by the DataLoader as fine-grained audio token boundaries when
  slicing XLSR features — they do not drive a separate visual scale.

alignment.json schema
─────────────────────
{
  "transcript":        str,
  "language":          str,
  "silence_threshold": float,
  "phonemes":  [ {"ph_idx",  "phoneme",  "start", "end",
                  "duration_ms", "start_frame", "end_frame"} ],
  "syllables": [ {"syl_idx", "syllable", "word",  "word_pos",
                  "start",   "end",      "duration_ms",
                  "start_frame", "end_frame"} ],
  "words":     [ {"word", "start", "end", "start_frame", "end_frame"} ],
  "gaps":      [ {"gap_idx", "start", "end", "duration_ms",
                  "type", "rms_energy", "start_frame", "end_frame"} ]

  phonemes  → fine-grained audio token boundaries for XLSR slicing (no visual scale)
  syllables → Scale 1 lip_crops/ frame windows
  words     → Scale 2 face_crops/ frame windows
  (phrase)  → Scale 3 full_frames/ — no alignment entry needed (full clip)
}

Fixes carried over
──────────────────
  FIX 1  MIN_PHONEME_MS 30 → 20ms (1 CTC frame = true minimum)
  FIX 2  Amplitude normalisation before wav2vec2 (quiet-clip conditioning)
  FIX 3  Word boundary | and noise tokens excluded from phoneme output
  FIX 4  Post-cap duration re-check after clamping to audio length

Usage
─────
  python preprocess_fakeavceleb.py --input /data/FakeAVCeleb_v1.2 --list

  python preprocess_fakeavceleb.py \\
      --input    /data/FakeAVCeleb_v1.2 \\
      --output   /data/processed_multiscale \\
      --category FakeVideo-FakeAudio \\
      --ethnicity African --gender women

Dependencies
────────────
  pip install torch torchaudio transformers whisperx pyphen \\
              opencv-python mediapipe pillow tqdm scipy numpy
  ffmpeg must be on PATH
"""

import os
import sys
import json
import argparse
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch
import torchaudio
from PIL import Image
from scipy.io import wavfile
from tqdm import tqdm


# ── lazy import helper ────────────────────────────────────────────────────────

def _require(pkg, pip_name=None):
    import importlib
    try:
        return importlib.import_module(pkg)
    except ImportError:
        sys.exit(f"[ERROR] '{pkg}' not installed.  pip install {pip_name or pkg}")


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

CATEGORIES = [
    "FakeVideo-FakeAudio",
    "FakeVideo-RealAudio",
    "RealVideo-FakeAudio",
    "RealVideo-RealAudio",
]

LABEL_MAP = {
    "RealVideo-RealAudio": 0,
    "RealVideo-FakeAudio": 1,
    "FakeVideo-RealAudio": 2,
    "FakeVideo-FakeAudio": 3,
}

MANIP_MAP = {
    "FakeVideo-FakeAudio": "FSGAN+SV2TTS",
    "FakeVideo-RealAudio": "FSGAN",
    "RealVideo-FakeAudio": "SV2TTS",
    "RealVideo-RealAudio": "real",
}

LIP_W,  LIP_H  = 96,  48
FACE_W, FACE_H = 96,  96
FULL_W, FULL_H = 224, 224
PHRASE_FPS     = 1
AUDIO_SR       = 16000

WAV2VEC2_STRIDE = 320
STRIDE_SEC      = WAV2VEC2_STRIDE / AUDIO_SR   # 0.020 s = 20 ms

# FIX 1: was 30ms — diagnostic showed 95/106 phonemes discarded at 30ms
MIN_PHONEME_MS     = 20.0
MAX_PHONEME_MS     = 800.0
NON_PHONEME_TOKENS = {"|", "spn", "<unk>", "<s>", "</s>", "<pad>"}

MIN_GAP_SEC = 0.08

LIP_LANDMARKS = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78,
]

_PHONEME_CACHE:   dict = {}
_FACE_MESH_CACHE: dict = {}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1  —  MediaPipe face crops
# ══════════════════════════════════════════════════════════════════════════════

def _get_face_mesh():
    if "mesh" not in _FACE_MESH_CACHE:
        mp = _require("mediapipe")
        try:
            mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.4,
            )
            _FACE_MESH_CACHE["mesh"] = ("legacy", mesh)
        except AttributeError:
            import urllib.request
            model_path = os.path.expanduser(
                "~/.cache/mediapipe/face_landmarker.task")
            if not os.path.exists(model_path):
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                url = (
                    "https://storage.googleapis.com/mediapipe-models/"
                    "face_landmarker/face_landmarker/float16/1/"
                    "face_landmarker.task"
                )
                print(f"    [MediaPipe] Downloading -> {model_path}")
                urllib.request.urlretrieve(url, model_path)
            from mediapipe.tasks import python as mpt
            from mediapipe.tasks.python import vision as mpv
            opts = mpv.FaceLandmarkerOptions(
                base_options=mpt.BaseOptions(model_asset_path=model_path),
                num_faces=1,
                min_face_detection_confidence=0.4,
                min_face_presence_confidence=0.4,
            )
            _FACE_MESH_CACHE["mesh"] = (
                "tasks", mpv.FaceLandmarker.create_from_options(opts))

    return _FACE_MESH_CACHE["mesh"]


def _landmarks(frame_rgb: np.ndarray):
    api, mesh = _get_face_mesh()
    h, w = frame_rgb.shape[:2]
    if api == "legacy":
        result = mesh.process(frame_rgb)
        if not result.multi_face_landmarks:
            return None
        lm = result.multi_face_landmarks[0].landmark
        return [(int(l.x * w), int(l.y * h)) for l in lm]
    else:
        mp  = _require("mediapipe")
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = mesh.detect(img)
        if not result.face_landmarks:
            return None
        lm = result.face_landmarks[0]
        return [(int(l.x * w), int(l.y * h)) for l in lm]


def crop_lip(frame_rgb: np.ndarray, padding: int = 8):
    """96x48 lip crop — shared by Scale 1 (phoneme) and Scale 2 (syllable)."""
    pts_all = _landmarks(frame_rgb)
    if pts_all is None:
        return None
    h, w = frame_rgb.shape[:2]
    pts = [pts_all[i] for i in LIP_LANDMARKS]
    x1  = max(0, min(p[0] for p in pts) - padding)
    y1  = max(0, min(p[1] for p in pts) - padding)
    x2  = min(w, max(p[0] for p in pts) + padding)
    y2  = min(h, max(p[1] for p in pts) + padding)
    if x2 <= x1 or y2 <= y1:
        return None
    return cv2.resize(frame_rgb[y1:y2, x1:x2], (LIP_W, LIP_H))


def crop_face(frame_rgb: np.ndarray, padding: int = 30):
    """96x96 face crop — used by Scale 3 (word)."""
    pts_all = _landmarks(frame_rgb)
    if pts_all is None:
        return None
    h, w = frame_rgb.shape[:2]
    xs = [p[0] for p in pts_all]
    ys = [p[1] for p in pts_all]
    x1 = max(0, min(xs) - padding)
    y1 = max(0, min(ys) - padding)
    x2 = min(w, max(xs) + padding)
    y2 = min(h, max(ys) + padding)
    if x2 <= x1 or y2 <= y1:
        return None
    return cv2.resize(frame_rgb[y1:y2, x1:x2], (FACE_W, FACE_H))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2  —  wav2vec2 IPA phoneme extraction  (Scale 1)
# ══════════════════════════════════════════════════════════════════════════════

def _load_phoneme_model(device: str):
    if "model" not in _PHONEME_CACHE:
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        model_id  = "facebook/wav2vec2-lv-60-espeak-cv-ft"
        print(f"    [Phoneme] Loading {model_id} on {device} ...")
        processor = Wav2Vec2Processor.from_pretrained(model_id)
        model     = Wav2Vec2ForCTC.from_pretrained(model_id).to(device)
        model.eval()
        vocab_size = len(processor.tokenizer)
        _PHONEME_CACHE.update({
            "model":     model,
            "processor": processor,
            "vocab":     processor.tokenizer.convert_ids_to_tokens(
                             range(vocab_size)),
            "blank_id":  processor.tokenizer.pad_token_id,
        })
        print(f"    [Phoneme] Ready — vocab {vocab_size}, "
              f"blank_id {_PHONEME_CACHE['blank_id']}")
    return (_PHONEME_CACHE["model"], _PHONEME_CACHE["processor"],
            _PHONEME_CACHE["vocab"],  _PHONEME_CACHE["blank_id"])


def _decode_ctc(ids: list, vocab: list, blank_id: int,
                max_audio_sec: float, fps: float) -> list:
    """Greedy CTC blank-collapse -> per-phoneme timestamps."""
    raw = []
    prev = seg_start = 0

    for t, tid in enumerate(ids):
        if tid != prev:
            if prev != blank_id:
                label  = vocab[prev]
                dur_ms = (t - seg_start) * STRIDE_SEC * 1000
                if label not in NON_PHONEME_TOKENS and dur_ms >= MIN_PHONEME_MS:
                    raw.append({"label":      label,
                                "start_step": seg_start,
                                "end_step":   t})
            seg_start = t
            prev      = tid

    if prev != blank_id:
        label  = vocab[prev]
        dur_ms = (len(ids) - seg_start) * STRIDE_SEC * 1000
        if label not in NON_PHONEME_TOKENS and dur_ms >= MIN_PHONEME_MS:
            raw.append({"label":      label,
                        "start_step": seg_start,
                        "end_step":   len(ids)})

    out = []
    for s in raw:
        start_sec = s["start_step"] * STRIDE_SEC
        end_sec   = min(s["end_step"] * STRIDE_SEC, max_audio_sec)
        end_sec   = min(end_sec, start_sec + MAX_PHONEME_MS / 1000.0)
        dur_ms    = (end_sec - start_sec) * 1000.0
        if dur_ms < MIN_PHONEME_MS:       # FIX 4: post-cap re-check
            continue
        out.append({
            "ph_idx":      len(out),
            "phoneme":     s["label"],
            "start":       round(start_sec, 4),
            "end":         round(end_sec,   4),
            "duration_ms": round(dur_ms,    1),
            "start_frame": int(round(start_sec * fps)),
            "end_frame":   int(round(end_sec   * fps)),
        })
    return out


def extract_phonemes(wav_np: np.ndarray, sr: int,
                     fps: float, device: str) -> list:
    """Run wav2vec2 CTC and return IPA phoneme list with frame indices."""
    model, processor, vocab, blank_id = _load_phoneme_model(device)

    if sr != AUDIO_SR:
        t      = torch.from_numpy(wav_np).unsqueeze(0)
        t      = torchaudio.functional.resample(t, sr, AUDIO_SR)
        wav_np = t.squeeze(0).numpy()

    # FIX 2: normalise amplitude before processor
    peak = np.abs(wav_np).max()
    if peak > 1e-6:
        wav_np = wav_np / peak

    max_audio_sec = len(wav_np) / AUDIO_SR

    inputs = processor(wav_np, sampling_rate=AUDIO_SR,
                       return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(inputs.input_values.to(device)).logits

    ids = torch.argmax(logits, dim=-1)[0].cpu().tolist()
    return _decode_ctc(ids, vocab, blank_id, max_audio_sec, fps)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3  —  Syllable extraction  (Scale 2)
# ══════════════════════════════════════════════════════════════════════════════

def _split_syllables(word: str, dic) -> list:
    """pyphen syllable split with punctuation stripping."""
    clean = word.lower().strip(".,!?;:\"'")
    if not clean:
        return [word]
    return dic.inserted(clean, hyphen="|").split("|")


def extract_syllables(words: list, fps: float) -> list:
    """
    Derive syllable-level timing from WhisperX word alignment using pyphen.

    Each word's duration is distributed across its syllables proportionally
    by character length. These timestamps tell the DataLoader which frames
    in scale1/lip_crops/ to mean-pool into a syllable-level visual feature
    for Scale 2 — no extra frames are stored on disk.

    Parameters
    ----------
    words : list of {"word", "start", "end"} dicts from WhisperX
    fps   : video frame rate for start_frame / end_frame

    Returns
    -------
    list of syllable dicts with full timing and frame indices
    """
    pyphen = _require("pyphen")
    dic    = pyphen.Pyphen(lang="en_US")

    syllables = []
    syl_idx   = 0

    for w in words:
        word    = w["word"]
        w_start = w["start"]
        w_end   = w["end"]
        w_dur   = max(w_end - w_start, 0.01)

        parts     = _split_syllables(word, dic)
        n_syl     = len(parts)
        char_lens = [max(len(s), 1) for s in parts]
        total     = sum(char_lens)
        durations = [w_dur * (c / total) for c in char_lens]

        t = w_start
        for i, (text, dur) in enumerate(zip(parts, durations)):
            s_end = t + dur
            syllables.append({
                "syl_idx":     syl_idx,
                "syllable":    text,
                "word":        word,
                "word_pos":    f"{i + 1}/{n_syl}",
                "start":       round(t,     4),
                "end":         round(s_end, 4),
                "duration_ms": round(dur * 1000, 1),
                "start_frame": int(round(t     * fps)),
                "end_frame":   int(round(s_end * fps)),
            })
            t = s_end
            syl_idx += 1

    return syllables


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4  —  Gap classification
# ══════════════════════════════════════════════════════════════════════════════

def _adaptive_silence_threshold(wav: np.ndarray, factor: float = 0.5) -> float:
    rms = float(np.sqrt(np.mean(wav.astype(np.float64) ** 2)))
    return rms * factor


def _classify_gap(wav: np.ndarray, sr: int, start_sec: float,
                  end_sec: float, threshold: float) -> tuple:
    seg = wav[int(start_sec * sr):int(end_sec * sr)]
    if len(seg) == 0:
        return "silence", 0.0
    rms = float(np.sqrt(np.mean(seg.astype(np.float64) ** 2)))
    return ("vocalization" if rms >= threshold else "silence"), rms


def extract_gaps(words: list, wav: np.ndarray, sr: int,
                 audio_dur: float, fps: float, threshold: float) -> list:
    if not words:
        return []

    bounds = []
    if words[0]["start"] > MIN_GAP_SEC:
        bounds.append((0.0, words[0]["start"]))
    for i in range(len(words) - 1):
        g0, g1 = words[i]["end"], words[i + 1]["start"]
        if g1 - g0 >= MIN_GAP_SEC:
            bounds.append((g0, g1))
    if audio_dur - words[-1]["end"] >= MIN_GAP_SEC:
        bounds.append((words[-1]["end"], audio_dur))

    gaps = []
    for idx, (g0, g1) in enumerate(bounds):
        gtype, rms = _classify_gap(wav, sr, g0, g1, threshold)
        gaps.append({
            "gap_idx":     idx,
            "start":       round(g0,  4),
            "end":         round(g1,  4),
            "duration_ms": round((g1 - g0) * 1000, 1),
            "type":        gtype,
            "rms_energy":  round(rms, 6),
            "start_frame": int(round(g0 * fps)),
            "end_frame":   int(round(g1 * fps)),
        })
    return gaps


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5  —  Audio extraction & WhisperX
# ══════════════════════════════════════════════════════════════════════════════

def extract_audio(video_path: Path, out_wav: Path) -> bool:
    cmd = (f'ffmpeg -y -i "{video_path}" '
           f'-ac 1 -ar {AUDIO_SR} -vn "{out_wav}" -loglevel error')
    ret = os.system(cmd)
    return ret == 0 and out_wav.exists() and out_wav.stat().st_size > 0


def load_wav(path: Path) -> tuple:
    sr, data = wavfile.read(str(path))
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        data = (data.astype(np.float32) - 128) / 128.0
    else:
        data = data.astype(np.float32)
    if data.ndim == 2:
        data = data.mean(axis=1)
    return data, sr


def run_whisperx(wav_path: Path, device: str) -> dict:
    try:
        wx     = _require("whisperx")
        model  = wx.load_model("large-v3", device=device,
                               compute_type="float16")
        audio  = wx.load_audio(str(wav_path))
        result = model.transcribe(audio, batch_size=8)
        am, meta = wx.load_align_model(
            language_code=result["language"], device=device)
        aligned = wx.align(result["segments"], am, meta, audio,
                           device=device)
        words = [
            {"word":  s.get("word", "").strip(),
             "start": round(s.get("start", 0.0), 4),
             "end":   round(s.get("end",   0.0), 4)}
            for s in aligned.get("word_segments", [])
        ]
        return {"transcript": " ".join(w["word"] for w in words),
                "language":   result.get("language", "en"),
                "words":      words}
    except Exception as e:
        print(f"    [WhisperX] failed: {e}")
        return {"transcript": "", "language": "unknown", "words": []}


def _attach_frames(words: list, fps: float) -> list:
    for w in words:
        w["start_frame"] = int(round(w["start"] * fps))
        w["end_frame"]   = int(round(w["end"]   * fps))
    return words


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6  —  Per-video processing
# ══════════════════════════════════════════════════════════════════════════════

def video_is_done(video_out: Path) -> bool:
    return (video_out / "meta.json").exists()


def process_video(video_path: Path, video_out: Path,
                  category: str, device: str,
                  csv_row: dict = None) -> bool:
    """
    Full three-scale preprocessing for one video.

    Directories written
    ───────────────────
      scale1/lip_crops/    96x48  every frame  <- Scale 1 (syllable)
      scale2/face_crops/   96x96  every frame  <- Scale 2 (word)
      scale3/full_frames/  224x224  1 fps       <- Scale 3 (phrase)

    Phonemes are stored in alignment.json only — they provide fine-grained
    audio token boundaries for XLSR feature slicing in the DataLoader but
    do not drive a separate visual scale on disk.
    """

    lip_dir  = video_out / "scale1" / "lip_crops"
    face_dir = video_out / "scale2" / "face_crops"
    full_dir = video_out / "scale3" / "full_frames"
    aud_dir  = video_out / "audio"

    for d in [lip_dir, face_dir, full_dir, aud_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Audio
    wav_path = aud_dir / "audio.wav"
    if not wav_path.exists():
        if not extract_audio(video_path, wav_path):
            print(f"    [SKIP] audio extraction failed: {video_path.name}")
            return False

    # 2. Load WAV
    wav_np, wav_sr = load_wav(wav_path)
    audio_dur      = len(wav_np) / wav_sr

    # 3. Video metadata
    cap          = cv2.VideoCapture(str(video_path))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 4. alignment.json — idempotent
    align_path = aud_dir / "alignment.json"
    if align_path.exists():
        with open(align_path) as f:
            alignment = json.load(f)
    else:
        # 4a. WhisperX word boundaries (Scale 3 + base for Scale 2)
        raw   = run_whisperx(wav_path, device)
        words = _attach_frames(raw["words"], fps)

        # 4b. Scale 1 audio metadata — wav2vec2 CTC IPA phonemes at ~20ms
        #     Stored in alignment.json for XLSR feature slicing in the
        #     DataLoader. No visual scale on disk for phonemes.
        try:
            phonemes = extract_phonemes(wav_np, wav_sr, fps, device)
        except Exception as e:
            print(f"    [Phoneme] failed ({e}) — empty list")
            phonemes = []

        # 4c. Scale 1 visual boundaries — pyphen syllables from WhisperX words
        #     Timestamps tell the DataLoader which scale1/lip_crops/ frames
        #     to use for syllable-level visual features (~150ms windows).
        try:
            syllables = extract_syllables(words, fps)
        except Exception as e:
            print(f"    [Syllable] failed ({e}) — empty list")
            syllables = []

        # 4d. Gap classification (silence vs vocalization)
        sil_thresh = _adaptive_silence_threshold(wav_np)
        gaps = extract_gaps(words, wav_np, wav_sr, audio_dur,
                            fps, sil_thresh)

        alignment = {
            "transcript":        raw["transcript"],
            "language":          raw["language"],
            "silence_threshold": round(sil_thresh, 6),
            "phonemes":          phonemes,   # audio metadata — XLSR slicing
            "syllables":         syllables,  # Scale 1 visual boundaries
            "words":             words,      # Scale 2 visual boundaries
            "gaps":              gaps,
        }
        with open(align_path, "w") as f:
            json.dump(alignment, f, indent=2)

    # 5. Frame extraction — single pass
    phrase_step   = max(1, round(fps / PHRASE_FPS))
    lip_saved = face_saved = phrase_saved = failed_frames = frame_idx = 0

    while True:
        ret, bgr = cap.read()
        if not ret:
            break

        name = f"frame_{frame_idx:06d}.jpg"
        rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # Scale 1: lip crop — syllable visual input
        lp = lip_dir / name
        if not lp.exists():
            lip = crop_lip(rgb)
            if lip is not None:
                Image.fromarray(lip).save(lp, quality=92)
                lip_saved += 1
            else:
                failed_frames += 1

        # Scale 2: face crop — word visual input
        fp = face_dir / name
        if not fp.exists():
            face = crop_face(rgb)
            if face is not None:
                Image.fromarray(face).save(fp, quality=92)
                face_saved += 1

        # Scale 3: full frame at 1 fps — phrase visual input
        if frame_idx % phrase_step == 0:
            pp = full_dir / name
            if not pp.exists():
                full = cv2.resize(rgb, (FULL_W, FULL_H))
                Image.fromarray(full).save(pp, quality=92)
                phrase_saved += 1

        frame_idx += 1

    cap.release()

    # 6. Resolve path components
    parts     = video_path.parts
    cat_idx   = next((i for i, p in enumerate(parts) if p == category), None)
    ethnicity = parts[cat_idx + 1] if cat_idx is not None else "unknown"
    gender    = parts[cat_idx + 2] if cat_idx is not None else "unknown"
    identity  = parts[cat_idx + 3] if cat_idx is not None else "unknown"

    gaps_list = alignment.get("gaps", [])

    # 7. meta.json
    # csv_row fields (None if --meta not provided or video not in CSV)
    csv = csv_row or {}
    meta = {
        "video_id":          video_path.stem,
        "identity":          identity,
        "label":             LABEL_MAP.get(category, -1),
        "label_str":         category,
        # ── deepfake provenance from meta_data.csv ──────────────────────────
        # source   : identity whose face/voice is the base  (e.g. "id00076")
        # target1  : face-swap target identity              ("-" if not swapped)
        # target2  : voice-clone target identity            ("-" if not cloned)
        # method   : generation method name or hash         ("wavtolip", "real", ...)
        # csv_type : dataset-internal type label            ("A", "B", "C", ...)
        "source":            csv.get("source",   ""),
        "target1":           csv.get("target1",  "-"),
        "target2":           csv.get("target2",  "-"),
        "method":            csv.get("method",   MANIP_MAP.get(category, "unknown")),
        "csv_type":          csv.get("csv_type", ""),
        # ── spatial / demographic ───────────────────────────────────────────
        "ethnicity":         ethnicity,
        "gender":            gender,
        # ── video stats ─────────────────────────────────────────────────────
        "duration_sec":      round(total_frames / fps, 3),
        "fps":               fps,
        "total_frames":      total_frames,
        "transcript":        alignment.get("transcript", ""),
        "language":          alignment.get("language",   "unknown"),
        "silence_threshold": alignment.get("silence_threshold", 0.0),
        "n_phonemes":        len(alignment.get("phonemes",  [])),
        "n_syllables":       len(alignment.get("syllables", [])),
        "n_words":           len(alignment.get("words",     [])),
        "n_gaps":            len(gaps_list),
        "n_vocalizations":   sum(1 for g in gaps_list
                                 if g["type"] == "vocalization"),
        "n_silences":        sum(1 for g in gaps_list
                                 if g["type"] == "silence"),
        "n_lip_frames":      lip_saved,
        "n_face_frames":     face_saved,
        "n_phrase_frames":   phrase_saved,
        "failed_frames":     failed_frames,
    }
    with open(video_out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return True


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7  —  Dataset scanner
# ══════════════════════════════════════════════════════════════════════════════

def scan_dataset(input_dir: Path) -> dict:
    tree     = {}
    vid_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

    for cat in CATEGORIES:
        cat_path = input_dir / cat
        if not cat_path.exists():
            continue
        tree[cat] = {}
        for eth_path in sorted(cat_path.iterdir()):
            if not eth_path.is_dir():
                continue
            tree[cat][eth_path.name] = {}
            for gen_path in sorted(eth_path.iterdir()):
                if not gen_path.is_dir():
                    continue
                videos = [
                    Path(root) / f
                    for root, _, files in os.walk(gen_path)
                    for f in files
                    if Path(f).suffix.lower() in vid_exts
                ]
                if videos:
                    tree[cat][eth_path.name][gen_path.name] = sorted(videos)
    return tree


def list_dataset(input_dir: Path):
    tree  = scan_dataset(input_dir)
    total = 0
    print(f"\n{'='*62}")
    print("FAKEAVCELEB DATASET")
    print(f"{'='*62}")
    for cat, ethnicities in tree.items():
        cat_n = sum(len(v) for g in ethnicities.values() for v in g.values())
        print(f"\n  [{LABEL_MAP[cat]}] {cat}  ({cat_n} videos)")
        for eth, genders in ethnicities.items():
            for gen, vids in genders.items():
                print(f"      --ethnicity {eth:20s} "
                      f"--gender {gen:6s}  -> {len(vids):4d} videos")
        total += cat_n
    print(f"\n  TOTAL: {total} videos")
    print(f"{'='*62}")
    print("\nExample:")
    print("  python preprocess_fakeavceleb.py \\")
    print("      --input  <path> --output <path> \\")
    print("      --category FakeVideo-FakeAudio \\")
    print("      --ethnicity African --gender women\n")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8  —  Metadata CSV loader
# ══════════════════════════════════════════════════════════════════════════════

def load_metadata(csv_path: Path) -> dict:
    """
    Parse meta_data.csv and return a lookup dict keyed by video stem.

    CSV columns (FakeAVCeleb v1.2):
        source, target1, target2, method, category, type, race, gender, path

    'path' is the relative path to the video inside the dataset root, e.g.:
        FakeAVCeleb/FakeVideo-RealAudio/African/men/id00076/00109_id00166_wavtolip.mp4

    We key by Path(path).stem so we can look up any video by its filename stem:
        "00109_id00166_wavtolip"  ->  {source, target1, target2, method, ...}

    One identity folder contains many videos — each is its own CSV row, so
    the stem key is unique across the whole dataset.

    Parameters
    ----------
    csv_path : path to meta_data.csv

    Returns
    -------
    dict: { video_stem (str) -> row_dict }
    """
    import csv

    lookup = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        # Normalise header names: strip whitespace + lowercase
        if reader.fieldnames:
            reader.fieldnames = [h.strip().lower() for h in reader.fieldnames]

        for row in reader:
            row       = {k: (v.strip() if v else "") for k, v in row.items()}
            path_val  = row.get("path", "")
            if not path_val:
                continue
            stem = Path(path_val).stem
            lookup[stem] = {
                "source":   row.get("source",   ""),
                "target1":  row.get("target1",  "-"),
                "target2":  row.get("target2",  "-"),
                "method":   row.get("method",   "unknown"),
                "csv_type": row.get("type",     ""),  # A / B / C etc.
                "category": row.get("category", ""),
                "race":     row.get("race",     ""),
                "gender":   row.get("gender",   ""),
                "path":     path_val,
            }

    print(f"[INFO] Metadata loaded: {len(lookup)} entries from {csv_path.name}")
    return lookup


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9  —  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="FakeAVCeleb 4-scale preprocessor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input",     required=True)
    parser.add_argument("--output",    default=None)
    parser.add_argument("--meta",      default=None,
                        help="Path to meta_data.csv for deepfake provenance fields")
    parser.add_argument("--list",      action="store_true")
    parser.add_argument("--category",  default=None, choices=CATEGORIES)
    parser.add_argument("--ethnicity", default=None)
    parser.add_argument("--gender",    default=None, choices=["men", "women"])
    parser.add_argument("--device",    default=None)
    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.exists():
        sys.exit(f"[ERROR] Not found: {input_dir}")

    if args.list:
        list_dataset(input_dir)
        return

    if not args.output:
        sys.exit("[ERROR] --output required")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INFO] Device         : {device}")
    print(f"[INFO] Input          : {input_dir}")
    print(f"[INFO] Output         : {output_dir}")
    print(f"[INFO] MIN_PHONEME_MS : {MIN_PHONEME_MS} ms  (FIX 1)")
    print(f"[INFO] Scales         : syllable | word | phrase  (phonemes in alignment.json)")

    print("[INFO] Pre-loading wav2vec2 phoneme model ...")
    _load_phoneme_model(device)

    # Load CSV metadata if provided — optional but recommended
    meta_lookup = {}
    if args.meta:
        meta_csv = Path(args.meta)
        if not meta_csv.exists():
            sys.exit(f"[ERROR] --meta file not found: {meta_csv}")
        meta_lookup = load_metadata(meta_csv)
    else:
        print("[INFO] --meta not provided — provenance fields will use MANIP_MAP fallback")

    tree     = scan_dataset(input_dir)
    sel_cats = [args.category]  if args.category  else list(tree.keys())
    sel_eths = [args.ethnicity] if args.ethnicity else None
    sel_gens = [args.gender]    if args.gender    else None

    queue = []
    missing_meta = 0
    for cat in sel_cats:
        if cat not in tree:
            print(f"[WARN] '{cat}' not in dataset.")
            continue
        for eth, genders in tree[cat].items():
            if sel_eths and eth not in sel_eths:
                continue
            for gen, videos in genders.items():
                if sel_gens and gen not in sel_gens:
                    continue
                for vp in videos:
                    identity  = vp.parent.name
                    video_out = (output_dir / cat / eth / gen
                                 / identity / vp.stem)
                    csv_row   = meta_lookup.get(vp.stem)
                    if meta_lookup and csv_row is None:
                        missing_meta += 1
                    queue.append((vp, video_out, cat, csv_row))

    if not queue:
        sys.exit("[ERROR] No videos matched. Use --list.")

    done = sum(1 for _, vo, _, _ in queue if video_is_done(vo))
    todo = len(queue) - done

    print(f"\n[INFO] Matched   : {len(queue)}")
    print(f"[INFO] Done      : {done}  (skipped)")
    print(f"[INFO] To process: {todo}")
    if meta_lookup:
        print(f"[INFO] Meta CSV  : {len(meta_lookup)} entries loaded, "
              f"{missing_meta} videos unmatched")
    if args.category:  print(f"[INFO] Category  : {args.category}")
    if args.ethnicity: print(f"[INFO] Ethnicity : {args.ethnicity}")
    if args.gender:    print(f"[INFO] Gender    : {args.gender}")
    print()

    success = failed = 0

    for vp, vo, cat, csv_row in tqdm(queue, desc="Videos", unit="vid"):
        if video_is_done(vo):
            continue

        tqdm.write(f"  -> {vp.parent.name}/{vp.name}")

        try:
            ok = process_video(vp, vo, cat, device, csv_row)
            if ok:
                success += 1
                m = json.loads((vo / "meta.json").read_text())
                tqdm.write(
                    f"    OK  ph={m['n_phonemes']:3d}  "
                    f"syl={m['n_syllables']:3d}  "
                    f"w={m['n_words']:2d}  "
                    f"gaps={m['n_gaps']}(voc={m['n_vocalizations']})  "
                    f"method={m.get('method','?'):12s}  "
                    f"type={m.get('csv_type','?')}"
                )
            else:
                failed += 1
                tqdm.write("    FAILED")
        except Exception as e:
            failed += 1
            tqdm.write(f"    Exception: {e}")
            tqdm.write(traceback.format_exc())

    print(f"\n{'='*50}")
    print("DONE")
    print(f"{'='*50}")
    print(f"  Success : {success}")
    print(f"  Failed  : {failed}")
    print(f"  Skipped : {done}")
    print(f"  Total   : {len(queue)}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
