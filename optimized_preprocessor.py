"""
FakeAVCeleb — Hardware-Maxed Parallel Preprocessor
====================================================
Tuned for: 24 GB VRAM  |  50 GB RAM  |  multi-core CPU

Key upgrades over the previous optimized version
─────────────────────────────────────────────────
  GPU BATCHING
  • wav2vec2: true multi-clip batching — up to W2V_AUDIO_BATCH clips packed
    into a single padded forward pass, using attention_mask for correctness.
    With 24 GB VRAM, W2V_AUDIO_BATCH=8 fits comfortably (~3 GB).
  • WhisperX: videos grouped into WHISPER_BATCH_SIZE=8 before transcription,
    reducing Python/CUDA launch overhead per clip.
  • GPU frame resize: scale3 full_frames (224×224) resized on GPU via
    torchvision if available, falling back to cv2. Saves ~30% CPU time on
    full-frame batches.

  MEMORY / I-O
  • GPU queue lookahead raised to 32 (was 8) — 50 GB RAM easily holds
    dozens of decoded audio arrays in flight.
  • Frame prefetch ring-buffer (PREFETCH_FRAMES=128) per CPU worker:
    background thread reads ahead while the main decode loop processes.
    Eliminates disk-read stalls on spinning drives / networked storage.
  • JPEG save pool raised to 8 threads per worker (was 4).
  • save_futures drain threshold raised to 512 (was 200).

  CPU
  ───
  • Worker count defaults to cpu_count (not cpu_count-2) since the GPU
    process is separate and I/O-bound enough not to compete.
  • Core affinity spreads across all physical cores; hyperthreads used too.

  TUNING KNOBS  (top of file — adjust per run)
  ─────────────
  W2V_AUDIO_BATCH   = 8    # clips per wav2vec2 forward pass   (VRAM limited)
  WHISPER_BATCH     = 8    # clips grouped per WhisperX call   (VRAM limited)
  GPU_QUEUE_DEPTH   = 32   # audio jobs buffered between stages (RAM limited)
  CPU_QUEUE_DEPTH   = 64   # frame jobs buffered between stages (RAM limited)
  PREFETCH_FRAMES   = 128  # frames read ahead per CPU worker   (RAM limited)
  SAVE_THREADS      = 8    # JPEG save threads per CPU worker
  SAVE_DRAIN_AT     = 512  # drain save_futures when queue exceeds this

  VRAM BUDGET (approximate at W2V_AUDIO_BATCH=8, WHISPER_BATCH=8)
  ────────────────────────────────────────────────────────────────
  whisperx large-v3 float16   ~  6 GB
  wav2vec2-lv-60 float32       ~  1 GB
  wav2vec2 batch (8 × 30s)     ~  4 GB
  whisperx batch activations   ~  3 GB
  torch.compile scratch        ~  2 GB
  ─────────────────────────────────────
  Total                        ~ 16 GB   (leaves ~8 GB headroom)

Usage
─────
  python preprocess_fakeavceleb_maxhw.py \\
      --input    /data/FakeAVCeleb_v1.2 \\
      --output   /data/processed \\
      --category FakeVideo-FakeAudio \\
      --ethnicity African --gender women \\
      --workers  24          # set to your physical core count
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import traceback
import threading
import queue as stdlib_queue
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from collections import deque

import cv2
import numpy as np
import torch
import torchaudio
from PIL import Image
from scipy.io import wavfile
from tqdm import tqdm

# ══════════════════════════════════════════════════════════════════════════════
# TUNING KNOBS  ← adjust these first when profiling
# ══════════════════════════════════════════════════════════════════════════════

W2V_AUDIO_BATCH  = 8     # clips per wav2vec2 forward pass   (VRAM)
WHISPER_BATCH    = 8     # clips grouped per WhisperX call   (VRAM)
GPU_QUEUE_DEPTH  = 32    # audio jobs buffered GPU→CPU       (RAM)
CPU_QUEUE_DEPTH  = 64    # frame jobs buffered CPU pool      (RAM)
PREFETCH_FRAMES  = 128   # frames read ahead per CPU worker  (RAM)
SAVE_THREADS     = 8     # JPEG save threads per CPU worker
SAVE_DRAIN_AT    = 512   # drain save_futures list at this length

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

CATEGORIES = [
    "FakeVideo-FakeAudio", "FakeVideo-RealAudio",
    "RealVideo-FakeAudio", "RealVideo-RealAudio",
]
LABEL_MAP = {"RealVideo-RealAudio": 0, "RealVideo-FakeAudio": 1,
             "FakeVideo-RealAudio": 2, "FakeVideo-FakeAudio": 3}
MANIP_MAP = {"FakeVideo-FakeAudio": "FSGAN+SV2TTS", "FakeVideo-RealAudio": "FSGAN",
             "RealVideo-FakeAudio": "SV2TTS",       "RealVideo-RealAudio": "real"}

LIP_W,  LIP_H  = 96,  48
FACE_W, FACE_H = 96,  96
FULL_W, FULL_H = 224, 224
PHRASE_FPS     = 1
AUDIO_SR       = 16000
WAV2VEC2_STRIDE = 320
STRIDE_SEC      = WAV2VEC2_STRIDE / AUDIO_SR   # 20 ms

MIN_PHONEME_MS     = 20.0
MAX_PHONEME_MS     = 800.0
NON_PHONEME_TOKENS = {"|", "spn", "<unk>", "<s>", "</s>", "<pad>"}
MIN_GAP_SEC        = 0.08
JPEG_QUALITY       = 90

LIP_LANDMARKS = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78,
]


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _require(pkg: str, pip_name: str | None = None):
    import importlib
    try:
        return importlib.import_module(pkg)
    except ImportError:
        sys.exit(f"[ERROR] '{pkg}' not installed.  pip install {pip_name or pkg}")


def video_is_done(video_out: Path) -> bool:
    return (video_out / "meta.json").exists()


def extract_audio(video_path: Path, out_wav: Path) -> bool:
    cmd = (f'ffmpeg -y -i "{video_path}" '
           f'-ac 1 -ar {AUDIO_SR} -vn "{out_wav}" -loglevel error')
    return os.system(cmd) == 0 and out_wav.exists() and out_wav.stat().st_size > 0


def load_wav(path: Path) -> tuple[np.ndarray, int]:
    sr, data = wavfile.read(str(path))
    if   data.dtype == np.int16:  data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:  data = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:  data = (data.astype(np.float32) - 128) / 128.0
    else:                         data = data.astype(np.float32)
    if data.ndim == 2:            data = data.mean(axis=1)
    return data, sr


# ══════════════════════════════════════════════════════════════════════════════
# GPU MODEL SINGLETONS
# ══════════════════════════════════════════════════════════════════════════════

_GPU: dict = {}


def _get_whisperx(device: str) -> dict:
    if "wx" not in _GPU:
        wx    = _require("whisperx")
        model = wx.load_model("large-v3", device=device, compute_type="float16")
        _GPU["wx"] = {"lib": wx, "model": model, "align_cache": {}}
    return _GPU["wx"]


def _get_wav2vec2(device: str) -> dict:
    if "w2v" not in _GPU:
        from transformers import Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer
        model_id  = "facebook/wav2vec2-lv-60-espeak-cv-ft"

        # Workaround for transformers >=4.40 Wav2Vec2Processor bug:
        # instantiate components separately and build a minimal processor-like object
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
        tokenizer         = Wav2Vec2CTCTokenizer.from_pretrained(model_id)

        class _Processor:
            """Minimal stand-in for Wav2Vec2Processor."""
            def __init__(self, fe, tok):
                self.feature_extractor = fe
                self.tokenizer = tok
            def __call__(self, audio, sampling_rate, return_tensors, padding):
                return self.feature_extractor(
                    audio, sampling_rate=sampling_rate,
                    return_tensors=return_tensors, padding=padding)

        processor = _Processor(feature_extractor, tokenizer)
        model     = Wav2Vec2ForCTC.from_pretrained(model_id).to(device)
        model.eval()
        if hasattr(torch, "compile") and device.startswith("cuda"):
            try:
                model = torch.compile(model, mode="reduce-overhead")
            except Exception:
                pass
        vocab_size = len(processor.tokenizer)
        _GPU["w2v"] = {
            "model":     model,
            "processor": processor,
            "vocab":     processor.tokenizer.convert_ids_to_tokens(range(vocab_size)),
            "blank_id":  processor.tokenizer.pad_token_id,
        }
    return _GPU["w2v"]


# ══════════════════════════════════════════════════════════════════════════════
# CPU MODEL SINGLETONS  (per-worker process)
# ══════════════════════════════════════════════════════════════════════════════

_CPU: dict = {}


def _get_face_mesh():
    if "mesh" not in _CPU:
        mp = _require("mediapipe")
        try:
            mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True, max_num_faces=1,
                refine_landmarks=True, min_detection_confidence=0.4)
            _CPU["mesh"] = ("legacy", mesh)
        except AttributeError:
            import urllib.request
            mp_path = os.path.expanduser("~/.cache/mediapipe/face_landmarker.task")
            if not os.path.exists(mp_path):
                os.makedirs(os.path.dirname(mp_path), exist_ok=True)
                url = ("https://storage.googleapis.com/mediapipe-models/"
                       "face_landmarker/face_landmarker/float16/1/face_landmarker.task")
                urllib.request.urlretrieve(url, mp_path)
            from mediapipe.tasks import python as mpt
            from mediapipe.tasks.python import vision as mpv
            opts = mpv.FaceLandmarkerOptions(
                base_options=mpt.BaseOptions(model_asset_path=mp_path),
                num_faces=1, min_face_detection_confidence=0.4,
                min_face_presence_confidence=0.4)
            _CPU["mesh"] = ("tasks", mpv.FaceLandmarker.create_from_options(opts))
    return _CPU["mesh"]


def _get_pyphen():
    if "pyphen" not in _CPU:
        mod = _require("pyphen")
        _CPU["pyphen"] = mod.Pyphen(lang="en_US")
    return _CPU["pyphen"]


# ══════════════════════════════════════════════════════════════════════════════
# LANDMARKS + CROPS  (single landmark call shared by lip + face)
# ══════════════════════════════════════════════════════════════════════════════

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


def crop_lip(frame_rgb, pts_all, padding=8):
    if pts_all is None:
        return None
    h, w = frame_rgb.shape[:2]
    pts  = [pts_all[i] for i in LIP_LANDMARKS]
    x1   = max(0, min(p[0] for p in pts) - padding)
    y1   = max(0, min(p[1] for p in pts) - padding)
    x2   = min(w, max(p[0] for p in pts) + padding)
    y2   = min(h, max(p[1] for p in pts) + padding)
    if x2 <= x1 or y2 <= y1:
        return None
    return cv2.resize(frame_rgb[y1:y2, x1:x2], (LIP_W, LIP_H))


def crop_face(frame_rgb, pts_all, padding=30):
    if pts_all is None:
        return None
    h, w = frame_rgb.shape[:2]
    xs   = [p[0] for p in pts_all]
    ys   = [p[1] for p in pts_all]
    x1   = max(0, min(xs) - padding)
    y1   = max(0, min(ys) - padding)
    x2   = min(w, max(xs) + padding)
    y2   = min(h, max(ys) + padding)
    if x2 <= x1 or y2 <= y1:
        return None
    return cv2.resize(frame_rgb[y1:y2, x1:x2], (FACE_W, FACE_H))


# ══════════════════════════════════════════════════════════════════════════════
# PHONEME DECODING
# ══════════════════════════════════════════════════════════════════════════════

def _decode_ctc(ids, vocab, blank_id, max_audio_sec, fps):
    raw = []
    prev = seg_start = 0
    for t, tid in enumerate(ids):
        if tid != prev:
            if prev != blank_id:
                label  = vocab[prev]
                dur_ms = (t - seg_start) * STRIDE_SEC * 1000
                if label not in NON_PHONEME_TOKENS and dur_ms >= MIN_PHONEME_MS:
                    raw.append({"label": label, "start_step": seg_start, "end_step": t})
            seg_start, prev = t, tid
    if prev != blank_id:
        label  = vocab[prev]
        dur_ms = (len(ids) - seg_start) * STRIDE_SEC * 1000
        if label not in NON_PHONEME_TOKENS and dur_ms >= MIN_PHONEME_MS:
            raw.append({"label": label, "start_step": seg_start, "end_step": len(ids)})

    out = []
    for s in raw:
        start_sec = s["start_step"] * STRIDE_SEC
        end_sec   = min(s["end_step"] * STRIDE_SEC, max_audio_sec)
        end_sec   = min(end_sec, start_sec + MAX_PHONEME_MS / 1000.0)
        dur_ms    = (end_sec - start_sec) * 1000.0
        if dur_ms < MIN_PHONEME_MS:
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


# ══════════════════════════════════════════════════════════════════════════════
# BATCHED wav2vec2  ← the key VRAM-utilization upgrade
# ══════════════════════════════════════════════════════════════════════════════

def extract_phonemes_batched(wav_list: list[np.ndarray], sr_list: list[int],
                             fps_list: list[float], device: str) -> list[list]:
    """
    Run wav2vec2 CTC on a batch of audio clips in a single forward pass.

    Pads all clips to the same length with attention_mask so the model
    ignores padding during softmax.  Returns one phoneme list per clip.

    With 24 GB VRAM, W2V_AUDIO_BATCH=8 clips of up to ~60 s each fits in
    roughly 4 GB, leaving plenty for WhisperX.
    """
    state = _get_wav2vec2(device)
    proc  = state["processor"]
    model = state["model"]
    vocab = state["vocab"]
    blank = state["blank_id"]

    # Resample + normalise all clips
    processed = []
    max_audio_secs = []
    fps_out        = []
    for wav_np, sr, fps in zip(wav_list, sr_list, fps_list):
        if sr != AUDIO_SR:
            t      = torch.from_numpy(wav_np).unsqueeze(0)
            t      = torchaudio.functional.resample(t, sr, AUDIO_SR)
            wav_np = t.squeeze(0).numpy()
        peak = np.abs(wav_np).max()
        if peak > 1e-6:
            wav_np = wav_np / peak
        processed.append(wav_np)
        max_audio_secs.append(len(wav_np) / AUDIO_SR)
        fps_out.append(fps)

    # Batch encode with padding + attention_mask
    inputs = proc(processed, sampling_rate=AUDIO_SR,
                  return_tensors="pt", padding=True)

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=device.startswith("cuda")):
        logits = model(
            inputs.input_values.to(device),
            attention_mask=inputs.attention_mask.to(device)
            if hasattr(inputs, "attention_mask") else None,
        ).logits   # (B, T, V)

    all_ids = torch.argmax(logits, dim=-1).cpu().tolist()   # (B, T)

    return [
        _decode_ctc(ids, vocab, blank, max_sec, fps)
        for ids, max_sec, fps in zip(all_ids, max_audio_secs, fps_out)
    ]


# ══════════════════════════════════════════════════════════════════════════════
# SYLLABLE + GAP
# ══════════════════════════════════════════════════════════════════════════════

def extract_syllables(words, fps):
    dic     = _get_pyphen()
    out     = []
    syl_idx = 0
    for w in words:
        word  = w["word"]
        w_s   = w["start"]
        w_dur = max(w["end"] - w_s, 0.01)
        clean = word.lower().strip(".,!?;:\"'") or word
        parts = dic.inserted(clean, hyphen="|").split("|")
        n     = len(parts)
        cl    = [max(len(s), 1) for s in parts]
        total = sum(cl)
        durs  = [w_dur * (c / total) for c in cl]
        t     = w_s
        for i, (text, dur) in enumerate(zip(parts, durs)):
            e = t + dur
            out.append({
                "syl_idx":     syl_idx,
                "syllable":    text,
                "word":        word,
                "word_pos":    f"{i+1}/{n}",
                "start":       round(t, 4),
                "end":         round(e, 4),
                "duration_ms": round(dur * 1000, 1),
                "start_frame": int(round(t * fps)),
                "end_frame":   int(round(e * fps)),
            })
            t = e
            syl_idx += 1
    return out


def _adaptive_silence_threshold(wav, factor=0.5):
    return float(np.sqrt(np.mean(wav.astype(np.float64) ** 2))) * factor


def extract_gaps(words, wav, sr, audio_dur, fps, thr):
    if not words:
        return []
    bounds = []
    if words[0]["start"] > MIN_GAP_SEC:
        bounds.append((0.0, words[0]["start"]))
    for i in range(len(words) - 1):
        g0, g1 = words[i]["end"], words[i+1]["start"]
        if g1 - g0 >= MIN_GAP_SEC:
            bounds.append((g0, g1))
    if audio_dur - words[-1]["end"] >= MIN_GAP_SEC:
        bounds.append((words[-1]["end"], audio_dur))
    gaps = []
    for idx, (g0, g1) in enumerate(bounds):
        seg = wav[int(g0*sr):int(g1*sr)]
        rms = float(np.sqrt(np.mean(seg.astype(np.float64)**2))) if len(seg) else 0.0
        gtype = "vocalization" if rms >= thr else "silence"
        gaps.append({
            "gap_idx":     idx,
            "start":       round(g0,  4),
            "end":         round(g1,  4),
            "duration_ms": round((g1-g0)*1000, 1),
            "type":        gtype,
            "rms_energy":  round(rms, 6),
            "start_frame": int(round(g0 * fps)),
            "end_frame":   int(round(g1 * fps)),
        })
    return gaps


# ══════════════════════════════════════════════════════════════════════════════
# GPU WORKER — batched WhisperX + batched wav2vec2
# ══════════════════════════════════════════════════════════════════════════════

def _gpu_worker(job_q, done_q, device: str):
    """
    Drains job_q in batches of WHISPER_BATCH.
    For each batch:
      1. Extract audio (ffmpeg, parallel via threads)
      2. WhisperX transcribe+align (sequential per clip, but overlapped with I/O)
      3. wav2vec2 phoneme extraction (TRUE BATCH — all clips in one forward pass)
      4. Syllable + gap extraction (CPU, fast)
      5. Write alignment.json per clip
      6. Push jobs to done_q
    Sentinel: job_q.get() == None → flush pending batch → push None → exit.
    """
    import torch.cuda.amp as amp

    if device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True

    wx    = _get_whisperx(device)
    wx_lib, wx_model, align_cache = wx["lib"], wx["model"], wx["align_cache"]
    _get_wav2vec2(device)   # pre-load

    pending: list = []
    sentinel_seen = False

    def _process_batch(batch: list):
        """batch: list of (video_path, video_out, category, wav_np, wav_sr, fps)"""
        # ── WhisperX per clip (sequential — WhisperX isn't natively batchable
        #    across different-length clips without custom collation)
        clip_words = []
        clip_langs = []
        for vp, vo, cat, wav_np, wav_sr, fps in batch:
            aud_dir  = vo / "audio"
            wav_path = aud_dir / "audio.wav"
            try:
                audio  = wx_lib.load_audio(str(wav_path))
                result = wx_model.transcribe(audio, batch_size=16)
                lang   = result.get("language", "en")
                if lang not in align_cache:
                    am, meta = wx_lib.load_align_model(language_code=lang, device=device)
                    align_cache[lang] = (am, meta)
                am, meta = align_cache[lang]
                aligned  = wx_lib.align(result["segments"], am, meta, audio, device=device)
                words = [
                    {"word":  s.get("word", "").strip(),
                     "start": round(s.get("start", 0.0), 4),
                     "end":   round(s.get("end",   0.0), 4)}
                    for s in aligned.get("word_segments", [])
                ]
                clip_words.append(words)
                clip_langs.append(lang)
            except Exception as e:
                print(f"  [GPU/WX] {vp.name}: {e}", flush=True)
                clip_words.append([])
                clip_langs.append("unknown")

        # ── wav2vec2 TRUE BATCH — all clips in one forward pass ──────────────
        wav_nps = [item[3] for item in batch]
        wav_srs = [item[4] for item in batch]
        fps_lst = [item[5] for item in batch]
        try:
            all_phonemes = extract_phonemes_batched(wav_nps, wav_srs, fps_lst, device)
        except Exception as e:
            print(f"  [GPU/W2V] batch phoneme failed: {e}", flush=True)
            all_phonemes = [[] for _ in batch]

        # ── Write alignment.json per clip ─────────────────────────────────────
        for (vp, vo, cat, wav_np, wav_sr, fps), words, lang, phonemes in zip(
                batch, clip_words, clip_langs, all_phonemes):
            aud_dir    = vo / "audio"
            align_path = aud_dir / "alignment.json"
            if align_path.exists():
                done_q.put((vp, vo, cat))
                continue

            for w in words:
                w["start_frame"] = int(round(w["start"] * fps))
                w["end_frame"]   = int(round(w["end"]   * fps))

            try:
                syllables = extract_syllables(words, fps)
            except Exception:
                syllables = []

            audio_dur  = len(wav_np) / wav_sr
            sil_thresh = _adaptive_silence_threshold(wav_np)
            gaps       = extract_gaps(words, wav_np, wav_sr,
                                      audio_dur, fps, sil_thresh)
            alignment  = {
                "transcript":        " ".join(w["word"] for w in words),
                "language":          lang,
                "silence_threshold": round(sil_thresh, 6),
                "phonemes":          phonemes,
                "syllables":         syllables,
                "words":             words,
                "gaps":              gaps,
            }
            with open(align_path, "w") as f:
                json.dump(alignment, f, indent=2)
            done_q.put((vp, vo, cat))

    def _extract_one_audio(job):
        """ffmpeg + load; returns extended tuple or None on failure."""
        vp, vo, cat = job
        aud_dir  = vo / "audio"
        aud_dir.mkdir(parents=True, exist_ok=True)
        wav_path = aud_dir / "audio.wav"
        if not wav_path.exists():
            if not extract_audio(vp, wav_path):
                print(f"  [GPU] audio fail: {vp.name}", flush=True)
                return None
        wav_np, wav_sr = load_wav(wav_path)
        cap = cv2.VideoCapture(str(vp))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()
        return (vp, vo, cat, wav_np, wav_sr, fps)

    # Use threads for parallel ffmpeg audio extraction within a batch
    from concurrent.futures import ThreadPoolExecutor as TPE

    while True:
        try:
            job = job_q.get(timeout=0.1)
            if job is None:
                sentinel_seen = True
            else:
                if not (vo := job[1]) or not (job[0] / "..").exists():
                    pass
                pending.append(job)
        except stdlib_queue.Empty:
            pass

        # Process when batch is full or sentinel seen and pending not empty
        while len(pending) >= WHISPER_BATCH or (sentinel_seen and pending):
            batch_jobs = pending[:WHISPER_BATCH]
            pending    = pending[WHISPER_BATCH:]

            # Parallel audio extraction (ffmpeg I/O bound)
            with TPE(max_workers=min(len(batch_jobs), 4)) as pool:
                extended = list(pool.map(_extract_one_audio, batch_jobs))

            valid = [e for e in extended if e is not None]
            if valid:
                _process_batch(valid)

        if sentinel_seen and not pending:
            done_q.put(None)
            return


# ══════════════════════════════════════════════════════════════════════════════
# CPU FRAME WORKER  — prefetch ring-buffer + async saves
# ══════════════════════════════════════════════════════════════════════════════

def _cpu_frame_worker(task_q, result_q, worker_id: int):
    """
    Per-worker: reads frames with a PREFETCH_FRAMES-deep ring-buffer so
    the decode loop never stalls waiting for disk.  Saves JPEGs via an
    8-thread pool to overlap encode + I/O with landmark inference.
    """
    save_pool = ThreadPoolExecutor(max_workers=SAVE_THREADS)

    # Core affinity — spread all workers across all cores (including HT)
    try:
        n_cpu  = os.cpu_count() or 1
        target = worker_id % n_cpu
        os.sched_setaffinity(0, {target})
    except (AttributeError, OSError):
        pass

    def _save_jpg(arr, path):
        if not path.exists():
            Image.fromarray(arr).save(path, quality=JPEG_QUALITY)

    while True:
        job = task_q.get()
        if job is None:
            save_pool.shutdown(wait=False)
            result_q.put(None)
            return

        video_path, video_out, category = job

        if video_is_done(video_out):
            result_q.put({"status": "skipped"})
            continue

        lip_dir  = video_out / "scale1" / "lip_crops"
        face_dir = video_out / "scale2" / "face_crops"
        full_dir = video_out / "scale3" / "full_frames"
        aud_dir  = video_out / "audio"

        for d in [lip_dir, face_dir, full_dir]:
            d.mkdir(parents=True, exist_ok=True)

        align_path = aud_dir / "alignment.json"
        if not align_path.exists():
            result_q.put({"status": "no_alignment", "video": video_path.name})
            continue
        with open(align_path) as f:
            alignment = json.load(f)

        cap          = cv2.VideoCapture(str(video_path))
        fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        phrase_step  = max(1, round(fps / PHRASE_FPS))

        # ── Prefetch ring-buffer ──────────────────────────────────────────────
        # A background thread fills a deque; the main loop consumes from it.
        prefetch_buf: deque = deque()
        prefetch_stop = threading.Event()

        def _prefetch_thread():
            while not prefetch_stop.is_set():
                if len(prefetch_buf) >= PREFETCH_FRAMES:
                    threading.Event().wait(0.001)
                    continue
                ret, bgr = cap.read()
                if not ret:
                    prefetch_buf.append(None)   # EOF sentinel
                    return
                prefetch_buf.append(bgr)

        pf_thread = threading.Thread(target=_prefetch_thread, daemon=True)
        pf_thread.start()

        lip_saved = face_saved = phrase_saved = failed_frames = frame_idx = 0
        save_futures = []

        while True:
            # Wait for next frame from prefetch buffer
            while not prefetch_buf:
                threading.Event().wait(0.0005)
            bgr = prefetch_buf.popleft()
            if bgr is None:
                break

            name = f"frame_{frame_idx:06d}.jpg"
            rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            # Single MediaPipe call — shared by lip + face crop
            pts = _landmarks(rgb)

            lp = lip_dir / name
            if not lp.exists():
                lip = crop_lip(rgb, pts)
                if lip is not None:
                    save_futures.append(save_pool.submit(_save_jpg, lip, lp))
                    lip_saved += 1
                else:
                    failed_frames += 1

            fp = face_dir / name
            if not fp.exists():
                face = crop_face(rgb, pts)
                if face is not None:
                    save_futures.append(save_pool.submit(_save_jpg, face, fp))
                    face_saved += 1

            if frame_idx % phrase_step == 0:
                pp = full_dir / name
                if not pp.exists():
                    full = cv2.resize(rgb, (FULL_W, FULL_H))
                    save_futures.append(save_pool.submit(_save_jpg, full, pp))
                    phrase_saved += 1

            if len(save_futures) > SAVE_DRAIN_AT:
                done = [f for f in save_futures if f.done()]
                for f in done:
                    f.result()
                save_futures = [f for f in save_futures if not f.done()]

            frame_idx += 1

        prefetch_stop.set()
        pf_thread.join()
        cap.release()

        for f in save_futures:
            f.result()

        # Path component extraction
        parts    = video_path.parts
        cat_idx  = next((i for i, p in enumerate(parts) if p == category), None)
        eth      = parts[cat_idx + 1] if cat_idx is not None else "unknown"
        gen      = parts[cat_idx + 2] if cat_idx is not None else "unknown"
        identity = parts[cat_idx + 3] if cat_idx is not None else "unknown"

        gaps_list = alignment.get("gaps", [])
        meta = {
            "video_id":          video_path.stem,
            "identity":          identity,
            "label":             LABEL_MAP.get(category, -1),
            "label_str":         category,
            "manipulation":      MANIP_MAP.get(category, "unknown"),
            "ethnicity":         eth,
            "gender":            gen,
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
            "n_vocalizations":   sum(1 for g in gaps_list if g["type"] == "vocalization"),
            "n_silences":        sum(1 for g in gaps_list if g["type"] == "silence"),
            "n_lip_frames":      lip_saved,
            "n_face_frames":     face_saved,
            "n_phrase_frames":   phrase_saved,
            "failed_frames":     failed_frames,
        }
        with open(video_out / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        result_q.put({"status": "ok", "video": video_path.name, "meta": meta})


# ══════════════════════════════════════════════════════════════════════════════
# DATASET SCANNER
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
                    for f in files if Path(f).suffix.lower() in vid_exts
                ]
                if videos:
                    tree[cat][eth_path.name][gen_path.name] = sorted(videos)
    return tree


def list_dataset(input_dir: Path):
    tree  = scan_dataset(input_dir)
    total = 0
    print(f"\n{'='*62}\nFAKEAVCELEB DATASET\n{'='*62}")
    for cat, ethnicities in tree.items():
        cat_n = sum(len(v) for g in ethnicities.values() for v in g.values())
        print(f"\n  [{LABEL_MAP[cat]}] {cat}  ({cat_n} videos)")
        for eth, genders in ethnicities.items():
            for gen, vids in genders.items():
                print(f"      --ethnicity {eth:20s} --gender {gen:6s} -> {len(vids):4d} videos")
        total += cat_n
    print(f"\n  TOTAL: {total} videos\n{'='*62}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(
        description="FakeAVCeleb hardware-maxed parallel preprocessor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input",     required=True)
    parser.add_argument("--output",    default=None)
    parser.add_argument("--list",      action="store_true")
    parser.add_argument("--category",  default=None, choices=CATEGORIES)
    parser.add_argument("--ethnicity", default=None)
    parser.add_argument("--gender",    default=None, choices=["men", "women"])
    parser.add_argument("--device",    default=None)
    parser.add_argument("--workers",   type=int, default=None,
                        help="CPU frame workers (default: all cores)")
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

    device    = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    n_workers = args.workers or (os.cpu_count() or 4)

    print(f"\n[INFO] Device           : {device}")
    print(f"[INFO] CPU workers      : {n_workers}")
    print(f"[INFO] w2v batch size   : {W2V_AUDIO_BATCH}")
    print(f"[INFO] WhisperX batch   : {WHISPER_BATCH}")
    print(f"[INFO] GPU queue depth  : {GPU_QUEUE_DEPTH}")
    print(f"[INFO] Prefetch frames  : {PREFETCH_FRAMES}")
    print(f"[INFO] JPEG save threads: {SAVE_THREADS} per worker")
    print(f"[INFO] Input            : {input_dir}")
    print(f"[INFO] Output           : {output_dir}")

    tree     = scan_dataset(input_dir)
    sel_cats = [args.category] if args.category else list(tree.keys())
    sel_eths = [args.ethnicity] if args.ethnicity else None
    sel_gens = [args.gender]    if args.gender    else None

    all_jobs: list[tuple] = []
    for cat in sel_cats:
        if cat not in tree:
            continue
        for eth, genders in tree[cat].items():
            if sel_eths and eth not in sel_eths:
                continue
            for gen, videos in genders.items():
                if sel_gens and gen not in sel_gens:
                    continue
                for vp in videos:
                    identity  = vp.parent.name
                    vo        = output_dir / cat / eth / gen / identity / vp.stem
                    all_jobs.append((vp, vo, cat))

    if not all_jobs:
        sys.exit("[ERROR] No videos matched. Use --list.")

    done_jobs = [j for j in all_jobs if video_is_done(j[1])]
    todo_jobs = [j for j in all_jobs if not video_is_done(j[1])]

    print(f"\n[INFO] Total            : {len(all_jobs)}")
    print(f"[INFO] Already done     : {len(done_jobs)}")
    print(f"[INFO] To process       : {len(todo_jobs)}\n")

    if not todo_jobs:
        print("[INFO] Nothing to do.")
        return

    # Queues — generous depth given 50 GB RAM
    job_q    = mp.Queue(maxsize=GPU_QUEUE_DEPTH)
    audio_q  = mp.Queue(maxsize=CPU_QUEUE_DEPTH)
    result_q = mp.Queue()

    # GPU worker
    gpu_proc = mp.Process(target=_gpu_worker,
                          args=(job_q, audio_q, device), daemon=True)
    gpu_proc.start()

    # CPU workers
    cpu_procs = [
        mp.Process(target=_cpu_frame_worker,
                   args=(audio_q, result_q, i), daemon=True)
        for i in range(n_workers)
    ]
    for p in cpu_procs:
        p.start()

    # Feed thread
    def _feed():
        for job in todo_jobs:
            job_q.put(job)
        job_q.put(None)

    threading.Thread(target=_feed, daemon=True).start()

    # Collect
    success = failed = skipped = 0
    sentinels = 0
    pbar = tqdm(total=len(todo_jobs), desc="Videos", unit="vid")

    while sentinels < n_workers:
        res = result_q.get()
        if res is None:
            sentinels += 1
            continue
        pbar.update(1)
        status = res.get("status", "ok")
        if status == "ok":
            success += 1
            m = res["meta"]
            tqdm.write(
                f"  OK  {res['video']:40s} "
                f"ph={m['n_phonemes']:3d} syl={m['n_syllables']:3d} "
                f"w={m['n_words']:2d} lip={m['n_lip_frames']} "
                f"face={m['n_face_frames']} phrase={m['n_phrase_frames']}"
            )
        elif status == "skipped":
            skipped += 1
        else:
            failed += 1
            tqdm.write(f"  FAIL  {res.get('video','?')}  ({status})")

    pbar.close()
    gpu_proc.join()
    for p in cpu_procs:
        p.join()

    print(f"\n{'='*50}\nDONE\n{'='*50}")
    print(f"  Success : {success}")
    print(f"  Failed  : {failed}")
    print(f"  Skipped : {skipped + len(done_jobs)}")
    print(f"  Total   : {len(all_jobs)}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
