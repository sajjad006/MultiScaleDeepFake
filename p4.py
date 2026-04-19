"""
FakeAVCeleb Multi-Scale Preprocessing Pipeline
===============================================
Produces the following structure per video:

processed_FakeAVCeleb_multiscale/
└── {Category}/
    └── {Ethnicity}/
        └── {Gender}/
            └── {identity_id}/
                └── {video_id}/
                    ├── scale1_phoneme/
                    │   └── lip_crops/
                    │       ├── frame_000006.jpg   (96×48, native fps)
                    │       └── ...
                    ├── scale2_word/
                    │   └── face_crops/
                    │       ├── frame_000006.jpg   (96×96, native fps)
                    │       └── ...
                    ├── scale3_phrase/
                    │   └── full_frames/
                    │       ├── frame_000000.jpg   (224×224, 1fps)
                    │       └── ...
                    ├── audio/
                    │   ├── audio.wav              (16kHz mono)
                    │   └── alignment.json
                    └── meta.json

alignment.json schema
---------------------
{
  "transcript":        str,
  "language":          str,
  "silence_threshold": float,   // adaptive per-video RMS threshold
  "words": [
    {
      "word":        str,
      "start":       float,     // seconds
      "end":         float,
      "start_frame": int,
      "end_frame":   int
    }
  ],
  "phonemes": [
    {
      "ph_idx":      int,
      "phoneme":     str,       // IPA symbol from wav2vec2-lv-60-espeak-cv-ft
      "start":       float,     // seconds  (20ms resolution, CNN stride=320)
      "end":         float,
      "duration_ms": float,
      "start_frame": int,
      "end_frame":   int
    }
  ],
  "gaps": [
    {
      "gap_idx":     int,
      "start":       float,
      "end":         float,
      "duration_ms": float,
      "type":        str,       // "silence" | "vocalization"
      "rms_energy":  float,
      "start_frame": int,
      "end_frame":   int
    }
  ]
}

Phoneme model
-------------
facebook/wav2vec2-lv-60-espeak-cv-ft
  - Trained on 60 languages → language-agnostic IPA output
  - CNN encoder downsamples by 320 samples → 20ms per output frame at 16kHz
  - Greedy CTC decode with blank collapsing gives per-phoneme timestamps
  - No pronunciation dictionary or transcript required

Usage
-----
  # List available subsets
  python preprocess_fakeavceleb_final.py --input /path/to/FakeAVCeleb_v1.2 --list

  # Process a specific subset
  python preprocess_fakeavceleb_final.py \\
      --input  /path/to/FakeAVCeleb_v1.2 \\
      --output /path/to/processed_multiscale \\
      --category FakeVideo-FakeAudio \\
      --ethnicity African \\
      --gender women

  # Process an entire category
  python preprocess_fakeavceleb_final.py \\
      --input  /path/to/FakeAVCeleb_v1.2 \\
      --output /path/to/processed_multiscale \\
      --category RealVideo-RealAudio

  # Process everything
  python preprocess_fakeavceleb_final.py \\
      --input  /path/to/FakeAVCeleb_v1.2 \\
      --output /path/to/processed_multiscale

  # Resume-safe: already-completed videos are skipped automatically.

Install
-------
  pip install torch torchaudio transformers whisperx opencv-python \\
              mediapipe pillow tqdm scipy numpy
"""

import os
os.environ["TORCHCODEC_DISABLE"] = "1"

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


# ── lazy heavy imports ───────────────────────────────────────────────────────

def _require(pkg, pip_name=None):
    import importlib
    try:
        return importlib.import_module(pkg)
    except ImportError:
        name = pip_name or pkg
        sys.exit(f"[ERROR] '{pkg}' not installed.  Run:  pip install {name}")


# ── constants ────────────────────────────────────────────────────────────────

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

# Minimum gap duration to analyse — shorter gaps are co-articulation noise
MIN_GAP_SEC = 0.08

# wav2vec2 CNN encoder downsampling factor (fixed across all wav2vec2 variants)
# 320 samples @ 16kHz = 20ms per output frame → matches ~25ms phoneme scale
WAV2VEC2_STRIDE = 320

# MediaPipe landmark indices that define the lip region
LIP_LANDMARKS = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78,
]


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — MediaPipe face mesh
# ═══════════════════════════════════════════════════════════════════════════════

def load_face_mesh():
    """Return (api_type, face_mesh_object), supporting both legacy and tasks API."""
    mp = _require("mediapipe")
    try:
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.4,
        )
        return ("legacy", face_mesh)
    except AttributeError:
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision as mp_vision
        import urllib.request

        model_path = os.path.expanduser("~/.cache/mediapipe/face_landmarker.task")
        if not os.path.exists(model_path):
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            url = (
                "https://storage.googleapis.com/mediapipe-models/"
                "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            )
            print(f"    [MediaPipe] Downloading model → {model_path} ...")
            urllib.request.urlretrieve(url, model_path)

        base_options = mp_tasks.BaseOptions(model_asset_path=model_path)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1,
            min_face_detection_confidence=0.4,
            min_face_presence_confidence=0.4,
        )
        return ("tasks", mp_vision.FaceLandmarker.create_from_options(options))


def get_landmarks(frame_rgb, face_mesh_tuple):
    """Return list of (x_px, y_px) for all 478 landmarks, or None."""
    api, face_mesh = face_mesh_tuple
    h, w = frame_rgb.shape[:2]

    if api == "legacy":
        result = face_mesh.process(frame_rgb)
        if not result.multi_face_landmarks:
            return None
        lm = result.multi_face_landmarks[0].landmark
        return [(int(l.x * w), int(l.y * h)) for l in lm]
    else:
        mp = _require("mediapipe")
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result   = face_mesh.detect(mp_image)
        if not result.face_landmarks:
            return None
        lm = result.face_landmarks[0]
        return [(int(l.x * w), int(l.y * h)) for l in lm]


def crop_lip(frame_rgb, face_mesh_tuple, padding=8):
    """Return (LIP_W × LIP_H) lip crop or None."""
    h, w    = frame_rgb.shape[:2]
    pts_all = get_landmarks(frame_rgb, face_mesh_tuple)
    if pts_all is None:
        return None
    pts = [pts_all[i] for i in LIP_LANDMARKS]
    x1  = max(0, min(p[0] for p in pts) - padding)
    y1  = max(0, min(p[1] for p in pts) - padding)
    x2  = min(w, max(p[0] for p in pts) + padding)
    y2  = min(h, max(p[1] for p in pts) + padding)
    if x2 <= x1 or y2 <= y1:
        return None
    return cv2.resize(frame_rgb[y1:y2, x1:x2], (LIP_W, LIP_H))


def crop_face(frame_rgb, face_mesh_tuple, padding=30):
    """Return (FACE_W × FACE_H) face crop or None."""
    h, w    = frame_rgb.shape[:2]
    pts_all = get_landmarks(frame_rgb, face_mesh_tuple)
    if pts_all is None:
        return None
    xs = [p[0] for p in pts_all]
    ys = [p[1] for p in pts_all]
    x1 = max(0, min(xs) - padding)
    y1 = max(0, min(ys) - padding)
    x2 = min(w, max(xs) + padding)
    y2 = min(h, max(ys) + padding)
    if x2 <= x1 or y2 <= y1:
        return None
    return cv2.resize(frame_rgb[y1:y2, x1:x2], (FACE_W, FACE_H))


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — wav2vec2 IPA phoneme extraction
# ═══════════════════════════════════════════════════════════════════════════════

# Module-level cache — model is loaded once per process, not once per video
_phoneme_model_cache = {}


def load_phoneme_model(device: str):
    """
    Load facebook/wav2vec2-lv-60-espeak-cv-ft once and cache it.

    Why this model over alternatives:
      - 60-language training → IPA output is truly language-agnostic.
        Critical for PolyGlotFake cross-dataset evaluation (6 languages).
      - No pronunciation dictionary or transcript required — runs directly
        on the waveform, so it works even when WhisperX fails or returns
        an empty transcript.
      - CNN stride of 320 samples → 20ms resolution at 16kHz, which
        matches the ~25ms phoneme scale in the HAVDNet architecture.
      - IPA symbols are consistent across languages, meaning the DataLoader
        can use the same phoneme vocabulary for all videos.
    """
    if "model" not in _phoneme_model_cache:
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        model_id  = "facebook/wav2vec2-lv-60-espeak-cv-ft"
        print(f"    [Phoneme] Loading {model_id} ...")
        processor = Wav2Vec2Processor.from_pretrained(model_id)
        model     = Wav2Vec2ForCTC.from_pretrained(model_id).to(device)
        model.eval()
        _phoneme_model_cache["model"]     = model
        _phoneme_model_cache["processor"] = processor
        _phoneme_model_cache["device"]    = device
        print(f"    [Phoneme] Model ready on {device}")

    return _phoneme_model_cache["model"], _phoneme_model_cache["processor"]


def extract_phonemes(wav_np: np.ndarray, sr: int, fps: float, device: str) -> list:
    """
    Run wav2vec2 CTC phoneme decoding on a 16kHz mono waveform.

    The model outputs IPA phoneme logits at 20ms resolution (CNN stride =
    320 samples at 16kHz). Greedy CTC decoding with blank collapsing gives
    the phoneme sequence; we then convert model output step indices to
    wall-clock seconds and video frame indices.

    This is entirely transcript-free — the model infers phonemes directly
    from the waveform, making it robust to WhisperX failures and language
    variation.

    Parameters
    ----------
    wav_np : float32 mono waveform at AUDIO_SR (16kHz)
    sr     : sample rate — must be AUDIO_SR; resampled automatically if not
    fps    : video frame rate, used for start_frame / end_frame
    device : "cuda" or "cpu"

    Returns
    -------
    list of dicts:
        ph_idx (int), phoneme (IPA str), start (sec), end (sec),
        duration_ms (float), start_frame (int), end_frame (int)
    """
    model, processor = load_phoneme_model(device)

    # Defensive resample (our pipeline always writes 16kHz, but just in case)
    if sr != AUDIO_SR:
        wav_tensor = torch.from_numpy(wav_np).unsqueeze(0)
        wav_tensor = torchaudio.functional.resample(wav_tensor, sr, AUDIO_SR)
        wav_np     = wav_tensor.squeeze(0).numpy()

    # Processor: normalise to zero-mean unit-variance, return pt tensor
    inputs = processor(
        wav_np,
        sampling_rate=AUDIO_SR,
        return_tensors="pt",
        padding=True,
    )
    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        logits = model(input_values).logits   # (1, T, vocab_size)

    # Greedy argmax over vocab dimension
    ids      = torch.argmax(logits, dim=-1)[0].cpu().tolist()   # (T,)
    vocab    = processor.tokenizer.convert_ids_to_tokens(range(len(processor.tokenizer)))
    blank_id = processor.tokenizer.pad_token_id   # CTC blank token

    # Seconds per model output frame
    # wav2vec2 CNN always downsamples by exactly WAV2VEC2_STRIDE=320 samples
    stride_sec = WAV2VEC2_STRIDE / AUDIO_SR   # 0.02s = 20ms

    # ── Greedy CTC blank collapsing ──────────────────────────────────────────
    # Walk the token sequence; whenever the token changes, flush the previous
    # segment (unless it was a blank). Each surviving segment becomes one
    # phoneme entry with its start/end step indices.
    phonemes_raw = []
    prev         = None
    seg_start    = 0

    for t, token_id in enumerate(ids):
        if token_id != prev:
            if prev is not None and prev != blank_id:
                phonemes_raw.append({
                    "label":      vocab[prev],
                    "start_step": seg_start,
                    "end_step":   t,
                })
            seg_start = t
            prev      = token_id

    # Flush the final segment
    if prev is not None and prev != blank_id:
        phonemes_raw.append({
            "label":      vocab[prev],
            "start_step": seg_start,
            "end_step":   len(ids),
        })

    # ── Convert model steps → seconds → video frame indices ─────────────────
    max_duration = len(wav_np) / AUDIO_SR
    phonemes_out = []

    for i, p in enumerate(phonemes_raw):
        start_sec = p["start_step"] * stride_sec
        end_sec   = min(p["end_step"] * stride_sec, max_duration)

        phonemes_out.append({
            "ph_idx":      i,
            "phoneme":     p["label"],
            "start":       round(start_sec, 4),
            "end":         round(end_sec,   4),
            "duration_ms": round((end_sec - start_sec) * 1000, 1),
            "start_frame": int(round(start_sec * fps)),
            "end_frame":   int(round(end_sec   * fps)),
        })

    return phonemes_out


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Gap classification  (silence vs vocalization)
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_adaptive_threshold(wav: np.ndarray, factor: float = 0.5) -> float:
    """
    Per-video silence threshold = factor × mean RMS of the full signal.

    A fixed global threshold (e.g. 0.01) is unreliable across FakeAVCeleb
    because recording conditions, microphone gains, and speaker volumes vary
    significantly between identities and synthesis methods.  Adaptive
    thresholding normalises for those factors: a quiet "umm" in a
    low-volume video still classifies as vocalization rather than silence.

    factor=0.5 — a gap needs at least half the clip's average energy to be
    called a vocalization.  Conservative enough to ignore breath noise,
    sensitive enough to catch filled pauses ("umm", "ahhh"), which are a
    meaningful discriminative signal because voice-clone fakes typically
    produce silence in those positions rather than synthesising a natural
    filled pause.
    """
    rms_full = float(np.sqrt(np.mean(wav.astype(np.float64) ** 2)))
    return rms_full * factor


def _classify_gap(
    wav: np.ndarray,
    sr: int,
    start_sec: float,
    end_sec: float,
    silence_threshold: float,
) -> tuple:
    """
    Classify the audio segment [start_sec, end_sec].

    Returns
    -------
    (type_str, rms_value)
        type_str  : "silence" | "vocalization"
        rms_value : float, useful for downstream weighting
    """
    start_sample = int(start_sec * sr)
    end_sample   = int(end_sec   * sr)
    segment      = wav[start_sample:end_sample]

    if len(segment) == 0:
        return "silence", 0.0

    rms      = float(np.sqrt(np.mean(segment.astype(np.float64) ** 2)))
    gap_type = "silence" if rms < silence_threshold else "vocalization"
    return gap_type, rms


def extract_gaps(
    word_alignment: list,
    wav: np.ndarray,
    sr: int,
    audio_duration_sec: float,
    fps: float,
    silence_threshold: float,
) -> list:
    """
    Find and classify all inter-word gaps from the WhisperX word alignment.

    Checks leading gap (before first word), all inter-word gaps, and trailing
    gap (after last word).  Gaps shorter than MIN_GAP_SEC are discarded as
    co-articulation artifacts from the forced aligner.

    Also checks leading and trailing gaps because synthesis models sometimes
    produce unnatural silence at audio boundaries even when the speech content
    itself is convincing.

    Parameters
    ----------
    word_alignment     : list of {"word", "start", "end"} from WhisperX
    wav                : float32 mono waveform at AUDIO_SR
    sr                 : sample rate
    audio_duration_sec : total audio length in seconds
    fps                : video frame rate
    silence_threshold  : from _compute_adaptive_threshold()

    Returns
    -------
    list of gap dicts with full metadata
    """
    if not word_alignment:
        return []

    boundaries = []

    # Leading gap
    first_start = word_alignment[0]["start"]
    if first_start > MIN_GAP_SEC:
        boundaries.append((0.0, first_start))

    # Inter-word gaps
    for i in range(len(word_alignment) - 1):
        gap_start = word_alignment[i]["end"]
        gap_end   = word_alignment[i + 1]["start"]
        if gap_end - gap_start >= MIN_GAP_SEC:
            boundaries.append((gap_start, gap_end))

    # Trailing gap
    last_end = word_alignment[-1]["end"]
    if audio_duration_sec - last_end >= MIN_GAP_SEC:
        boundaries.append((last_end, audio_duration_sec))

    gaps = []
    for idx, (g_start, g_end) in enumerate(boundaries):
        gap_type, rms = _classify_gap(wav, sr, g_start, g_end, silence_threshold)
        gaps.append({
            "gap_idx":     idx,
            "start":       round(g_start, 4),
            "end":         round(g_end,   4),
            "duration_ms": round((g_end - g_start) * 1000, 1),
            "type":        gap_type,
            "rms_energy":  round(rms, 6),
            "start_frame": int(round(g_start * fps)),
            "end_frame":   int(round(g_end   * fps)),
        })

    return gaps


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Audio extraction & WhisperX word alignment
# ═══════════════════════════════════════════════════════════════════════════════

def extract_audio(video_path: Path, out_wav: Path) -> bool:
    """Extract 16kHz mono WAV from video using ffmpeg."""
    try:
        cmd = (
            f'ffmpeg -y -i "{video_path}" '
            f'-ac 1 -ar {AUDIO_SR} -vn "{out_wav}" '
            f'-loglevel error'
        )
        ret = os.system(cmd)
        return ret == 0 and out_wav.exists() and out_wav.stat().st_size > 0
    except Exception:
        return False


def load_wav_numpy(wav_path: Path) -> tuple:
    """
    Load WAV into a float32 numpy array.
    Returns (signal, sample_rate).
    """
    sr, data = wavfile.read(str(wav_path))

    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        data = (data.astype(np.float32) - 128) / 128.0
    else:
        data = data.astype(np.float32)

    if data.ndim == 2:      # stereo → mono
        data = data.mean(axis=1)

    return data, sr


def run_whisperx(wav_path: Path, device: str) -> dict:
    """
    Run WhisperX large-v3 to get word-level timestamps + transcript.

    Returns
    -------
    dict: transcript (str), language (str),
          words (list of {"word", "start", "end"})
    """
    try:
        wx = _require("whisperx")

        model  = wx.load_model("large-v3", device=device, compute_type="float16")
        audio  = wx.load_audio(str(wav_path))
        result = model.transcribe(audio, batch_size=8)

        align_model, metadata = wx.load_align_model(
            language_code=result["language"], device=device
        )
        aligned = wx.align(
            result["segments"], align_model, metadata, audio, device=device
        )

        words = []
        for seg in aligned.get("word_segments", []):
            words.append({
                "word":  seg.get("word", "").strip(),
                "start": round(seg.get("start", 0.0), 4),
                "end":   round(seg.get("end",   0.0), 4),
            })

        return {
            "transcript": " ".join(w["word"] for w in words),
            "language":   result.get("language", "en"),
            "words":      words,
        }

    except Exception as e:
        print(f"    [WhisperX] failed: {e}")
        return {"transcript": "", "language": "unknown", "words": []}


def add_frame_indices_to_words(words: list, fps: float) -> list:
    """Attach start_frame / end_frame to each word dict (in-place)."""
    for w in words:
        w["start_frame"] = int(round(w["start"] * fps))
        w["end_frame"]   = int(round(w["end"]   * fps))
    return words


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Per-video processing
# ═══════════════════════════════════════════════════════════════════════════════

def video_is_done(video_out: Path) -> bool:
    return (video_out / "meta.json").exists()


def process_video(video_path: Path, video_out: Path, category: str, device: str):
    """
    Full multi-scale preprocessing for one video.

    Pipeline
    --------
    1.  ffmpeg        → 16kHz mono audio.wav
    2.  scipy wavfile → float32 numpy waveform (for phoneme model + gap RMS)
    3.  WhisperX      → word-level timestamps + transcript
    4.  wav2vec2 CTC  → IPA phoneme sequence at 20ms resolution
    5.  Gap classifier → silence / vocalization labels for inter-word gaps
    6.  Write audio/alignment.json
    7.  MediaPipe + OpenCV → lip crops (scale1), face crops (scale2),
                              full frames (scale3)
    8.  Write meta.json
    """

    lip_dir   = video_out / "scale1_phoneme" / "lip_crops"
    face_dir  = video_out / "scale2_word"    / "face_crops"
    full_dir  = video_out / "scale3_phrase"  / "full_frames"
    audio_dir = video_out / "audio"

    for d in [lip_dir, face_dir, full_dir, audio_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ── 1. Audio extraction ──────────────────────────────────────────────────
    wav_path = audio_dir / "audio.wav"
    if not wav_path.exists():
        ok = extract_audio(video_path, wav_path)
        if not ok:
            print(f"    [SKIP] audio extraction failed: {video_path.name}")
            return False

    # ── 2. Load WAV as numpy ─────────────────────────────────────────────────
    wav_np, wav_sr = load_wav_numpy(wav_path)
    audio_duration = len(wav_np) / wav_sr

    # ── 3. Video metadata ────────────────────────────────────────────────────
    cap          = cv2.VideoCapture(str(video_path))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps

    # ── 4. Build alignment.json (idempotent — skipped if already exists) ─────
    align_path = audio_dir / "alignment.json"
    if align_path.exists():
        with open(align_path) as f:
            alignment = json.load(f)
    else:
        # 4a. WhisperX word alignment
        raw   = run_whisperx(wav_path, device)
        words = add_frame_indices_to_words(raw["words"], fps)

        # 4b. wav2vec2 IPA phoneme extraction
        # Runs on the raw waveform — no transcript needed.
        # Language-agnostic IPA ensures consistent phoneme vocabulary across
        # FakeAVCeleb (English) and PolyGlotFake (6 languages).
        try:
            phonemes = extract_phonemes(wav_np, wav_sr, fps, device)
        except Exception as e:
            print(f"    [Phoneme] extraction failed ({e}), storing empty list")
            phonemes = []

        # 4c. Gap classification with adaptive per-video threshold
        silence_threshold = _compute_adaptive_threshold(wav_np, factor=0.5)
        gaps = extract_gaps(
            word_alignment=words,
            wav=wav_np,
            sr=wav_sr,
            audio_duration_sec=audio_duration,
            fps=fps,
            silence_threshold=silence_threshold,
        )

        alignment = {
            "transcript":        raw["transcript"],
            "language":          raw["language"],
            "silence_threshold": round(silence_threshold, 6),
            "words":             words,
            "phonemes":          phonemes,
            "gaps":              gaps,
        }

        with open(align_path, "w") as f:
            json.dump(alignment, f, indent=2)

    # ── 5. Frame extraction ──────────────────────────────────────────────────
    face_mesh_tuple = load_face_mesh()

    phrase_step   = max(1, round(fps / PHRASE_FPS))
    lip_saved     = 0
    face_saved    = 0
    phrase_saved  = 0
    failed_frames = 0

    frame_idx = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_name = f"frame_{frame_idx:06d}.jpg"
        frame_rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Scale 1 — lip crops at every frame (phoneme-level sync at ~25ms)
        lip_path = lip_dir / frame_name
        if not lip_path.exists():
            lip = crop_lip(frame_rgb, face_mesh_tuple)
            if lip is not None:
                Image.fromarray(lip).save(lip_path, quality=92)
                lip_saved += 1
            else:
                failed_frames += 1

        # Scale 2 — face crops at every frame (word-level sync at ~400ms)
        face_path = face_dir / frame_name
        if not face_path.exists():
            face = crop_face(frame_rgb, face_mesh_tuple)
            if face is not None:
                Image.fromarray(face).save(face_path, quality=92)
                face_saved += 1

        # Scale 3 — full frames at 1fps (phrase/utterance-level context)
        if frame_idx % phrase_step == 0:
            full_path = full_dir / frame_name
            if not full_path.exists():
                full = cv2.resize(frame_rgb, (FULL_W, FULL_H))
                Image.fromarray(full).save(full_path, quality=92)
                phrase_saved += 1

        frame_idx += 1

    cap.release()

    # ── 6. Resolve identity and path components ──────────────────────────────
    parts     = video_path.parts
    cat_idx   = next((i for i, p in enumerate(parts) if p == category), None)
    ethnicity = parts[cat_idx + 1] if cat_idx is not None else "unknown"
    gender    = parts[cat_idx + 2] if cat_idx is not None else "unknown"
    identity  = parts[cat_idx + 3] if cat_idx is not None else "unknown"

    n_vocalizations = sum(
        1 for g in alignment.get("gaps", []) if g["type"] == "vocalization"
    )
    n_silences = sum(
        1 for g in alignment.get("gaps", []) if g["type"] == "silence"
    )

    # ── 7. meta.json ─────────────────────────────────────────────────────────
    meta = {
        "video_id":           video_path.stem,
        "identity":           identity,
        "label":              LABEL_MAP.get(category, -1),
        "label_str":          category,
        "manipulation":       MANIP_MAP.get(category, "unknown"),
        "ethnicity":          ethnicity,
        "gender":             gender,
        "duration_sec":       round(duration_sec, 3),
        "fps":                fps,
        "total_frames":       total_frames,
        "transcript":         alignment.get("transcript", ""),
        "language":           alignment.get("language",   "unknown"),
        "silence_threshold":  alignment.get("silence_threshold", 0.0),
        "n_words":            len(alignment.get("words",    [])),
        "n_phonemes":         len(alignment.get("phonemes", [])),
        "n_gaps":             len(alignment.get("gaps",     [])),
        "n_vocalizations":    n_vocalizations,
        "n_silences":         n_silences,
        "n_lip_frames":       lip_saved,
        "n_face_frames":      face_saved,
        "n_phrase_frames":    phrase_saved,
        "failed_frames":      failed_frames,
    }

    with open(video_out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return True


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Dataset scanner
# ═══════════════════════════════════════════════════════════════════════════════

def scan_dataset(input_dir: Path) -> dict:
    """
    Recursively scan FakeAVCeleb_v1.2 root.

    Returns
    -------
    { category: { ethnicity: { gender: [video_paths] } } }
    """
    tree       = {}
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

    for cat in CATEGORIES:
        cat_path = input_dir / cat
        if not cat_path.exists():
            continue
        tree[cat] = {}

        for eth_path in sorted(cat_path.iterdir()):
            if not eth_path.is_dir():
                continue
            eth = eth_path.name
            tree[cat][eth] = {}

            for gen_path in sorted(eth_path.iterdir()):
                if not gen_path.is_dir():
                    continue
                gen    = gen_path.name
                videos = []
                for root, _, files in os.walk(gen_path):
                    for f in files:
                        if Path(f).suffix.lower() in video_exts:
                            videos.append(Path(root) / f)
                if videos:
                    tree[cat][eth][gen] = sorted(videos)

    return tree


def list_dataset(input_dir: Path):
    tree = scan_dataset(input_dir)
    print("\n" + "=" * 60)
    print("FAKEAVCELEB DATASET CONTENTS")
    print("=" * 60)
    total = 0
    for cat, ethnicities in tree.items():
        cat_total = sum(
            len(vids)
            for genders in ethnicities.values()
            for vids in genders.values()
        )
        print(f"\n  [{LABEL_MAP[cat]}] {cat}  ({cat_total} videos)")
        for eth, genders in ethnicities.items():
            for gen, vids in genders.items():
                print(
                    f"      --ethnicity {eth:20s} "
                    f"--gender {gen:6s}  → {len(vids):4d} videos"
                )
        total += cat_total
    print(f"\n  TOTAL: {total} videos")
    print("=" * 60)
    print("\nExample usage:")
    print("  python preprocess_fakeavceleb_final.py \\")
    print("      --input  <dataset_path> \\")
    print("      --output <output_path>  \\")
    print("      --category FakeVideo-FakeAudio \\")
    print("      --ethnicity African \\")
    print("      --gender women\n")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description=(
            "FakeAVCeleb multi-scale preprocessor — "
            "wav2vec2 IPA phonemes + WhisperX words + gap classification"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input",     required=True,
                        help="Path to FakeAVCeleb_v1.2 root")
    parser.add_argument("--output",    default=None,
                        help="Path to output directory")
    parser.add_argument("--list",      action="store_true",
                        help="List available categories/ethnicities/genders and exit")
    parser.add_argument("--category",  default=None, choices=CATEGORIES,
                        help="Only process this category")
    parser.add_argument("--ethnicity", default=None,
                        help="Only process this ethnicity")
    parser.add_argument("--gender",    default=None, choices=["men", "women"],
                        help="Only process this gender")
    parser.add_argument("--device",    default=None,
                        help="cuda / cpu  (default: auto-detect)")
    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.exists():
        sys.exit(f"[ERROR] Input directory not found: {input_dir}")

    if args.list:
        list_dataset(input_dir)
        return

    if args.output is None:
        sys.exit("[ERROR] --output is required when not using --list")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INFO] Device : {device}")
    print(f"[INFO] Input  : {input_dir}")
    print(f"[INFO] Output : {output_dir}")

    # Pre-load the phoneme model once here so the first video doesn't appear
    # to hang silently while the model downloads / loads
    print("[INFO] Pre-loading wav2vec2 phoneme model ...")
    load_phoneme_model(device)

    tree = scan_dataset(input_dir)

    selected_categories  = [args.category]  if args.category  else list(tree.keys())
    selected_ethnicities = [args.ethnicity] if args.ethnicity else None
    selected_genders     = [args.gender]    if args.gender    else None

    video_queue = []
    for cat in selected_categories:
        if cat not in tree:
            print(f"[WARN] Category '{cat}' not found in dataset, skipping.")
            continue
        for eth, genders in tree[cat].items():
            if selected_ethnicities and eth not in selected_ethnicities:
                continue
            for gen, videos in genders.items():
                if selected_genders and gen not in selected_genders:
                    continue
                for vp in videos:
                    identity  = vp.parent.name
                    video_out = output_dir / cat / eth / gen / identity / vp.stem
                    video_queue.append((vp, video_out, cat))

    if not video_queue:
        sys.exit("[ERROR] No videos matched the filters. Use --list to see options.")

    already_done = sum(1 for _, vo, _ in video_queue if video_is_done(vo))
    to_process   = len(video_queue) - already_done

    print(f"\n[INFO] Videos matched : {len(video_queue)}")
    print(f"[INFO] Already done   : {already_done}  (will be skipped)")
    print(f"[INFO] To process     : {to_process}")
    if args.category:  print(f"[INFO] Category   : {args.category}")
    if args.ethnicity: print(f"[INFO] Ethnicity  : {args.ethnicity}")
    if args.gender:    print(f"[INFO] Gender     : {args.gender}")
    print()

    success = 0
    failed  = 0

    for video_path, video_out, category in tqdm(video_queue, desc="Videos", unit="vid"):

        if video_is_done(video_out):
            continue

        tqdm.write(f"  Processing: {video_path.name}  [{video_path.parent.name}]")

        try:
            ok = process_video(video_path, video_out, category, device)
            if ok:
                success += 1
                tqdm.write(f"  ✓ {video_path.parent.name}/{video_out.name}")
            else:
                failed += 1
                tqdm.write(f"  ✗ Failed (audio/frame issue): {video_path.name}")

        except Exception as e:
            failed += 1
            tqdm.write(f"  ✗ Exception on {video_path.name}: {e}")
            tqdm.write(traceback.format_exc())

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"  Success  : {success}")
    print(f"  Failed   : {failed}")
    print(f"  Skipped  : {already_done}")
    print(f"  Total    : {len(video_queue)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
