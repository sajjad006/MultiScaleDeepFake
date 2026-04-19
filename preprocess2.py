"""
FakeAVCeleb Multi-Scale Preprocessing Pipeline
================================================
Produces the following structure per video:

processed_FakeAVCeleb_multiscale/
└── {Category}/
    └── {Ethnicity}/
        └── {Gender}/
            └── {identity_id}/
                └── {video_id}/
                    ├── scale1_phoneme/
                    │   └── lip_crops/
                    │       ├── frame_000006.jpg   (96×48, 25fps)
                    │       └── ...
                    ├── scale2_word/
                    │   └── face_crops/
                    │       ├── frame_000006.jpg   (96×96, 25fps)
                    │       └── ...
                    ├── scale3_phrase/
                    │   └── full_frames/
                    │       ├── frame_000000.jpg   (224×224, 1fps)
                    │       └── ...
                    ├── audio/
                    │   ├── audio.wav              (16kHz mono)
                    │   └── alignment.json         (WhisperX word+phoneme boundaries)
                    └── meta.json                  (label, transcript, fps, counts)

Usage
-----
# List what's available in the dataset
python preprocess_fakeavceleb.py --input /path/to/FakeAVCeleb_v1.2 --list

# Process a specific subset
python preprocess_fakeavceleb.py \\
    --input  /path/to/FakeAVCeleb_v1.2 \\
    --output /path/to/processed_multiscale \\
    --category FakeVideo-FakeAudio \\
    --ethnicity African \\
    --gender women

# Process an entire category
python preprocess_fakeavceleb.py \\
    --input  /path/to/FakeAVCeleb_v1.2 \\
    --output /path/to/processed_multiscale \\
    --category RealVideo-RealAudio

# Process everything
python preprocess_fakeavceleb.py \\
    --input  /path/to/FakeAVCeleb_v1.2 \\
    --output /path/to/processed_multiscale

# Resume-safe: already-completed videos are skipped automatically.
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
from tqdm import tqdm

# ── optional heavy imports (fail loudly with install hint) ──────────────────

def _require(pkg, pip_name=None):
    import importlib
    try:
        return importlib.import_module(pkg)
    except ImportError:
        name = pip_name or pkg
        sys.exit(f"[ERROR] '{pkg}' not installed. Run:  pip install {name}")

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

LIP_W, LIP_H    = 96, 48
FACE_W, FACE_H  = 96, 96
FULL_W, FULL_H  = 224, 224
PHRASE_FPS      = 1
AUDIO_SR        = 16000

LIP_LANDMARKS = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78,
]

# ── MediaPipe face mesh (supports legacy < 0.10 and tasks >= 0.10) ──────────

def load_face_mesh():
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
            print(f"    [MediaPipe] Downloading face landmarker model to {model_path} ...")
            urllib.request.urlretrieve(url, model_path)

        base_options = mp_tasks.BaseOptions(model_asset_path=model_path)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1,
            min_face_detection_confidence=0.4,
            min_face_presence_confidence=0.4,
        )
        landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        return ("tasks", landmarker)


def get_landmarks(frame_rgb, face_mesh_tuple):
    """Return list of (x_px, y_px) for all landmarks, or None if no face found."""
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
        result = face_mesh.detect(mp_image)
        if not result.face_landmarks:
            return None
        lm = result.face_landmarks[0]
        return [(int(l.x * w), int(l.y * h)) for l in lm]


def crop_lip(frame_rgb, face_mesh_tuple, padding=8):
    """Return 96×48 lip crop or None if face not found."""
    h, w  = frame_rgb.shape[:2]
    pts_all = get_landmarks(frame_rgb, face_mesh_tuple)
    if pts_all is None:
        return None
    pts = [pts_all[i] for i in LIP_LANDMARKS]
    x1 = max(0, min(p[0] for p in pts) - padding)
    y1 = max(0, min(p[1] for p in pts) - padding)
    x2 = min(w, max(p[0] for p in pts) + padding)
    y2 = min(h, max(p[1] for p in pts) + padding)
    if x2 <= x1 or y2 <= y1:
        return None
    return cv2.resize(frame_rgb[y1:y2, x1:x2], (LIP_W, LIP_H))


def crop_face(frame_rgb, face_mesh_tuple, padding=30):
    """Return 96×96 face crop or None if face not found."""
    h, w  = frame_rgb.shape[:2]
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


# ── audio helpers ────────────────────────────────────────────────────────────

def extract_audio(video_path: Path, out_wav: Path) -> bool:
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


def get_phoneme_alignments(words: list, audio_np: np.ndarray, fps: float, device: str) -> list:
    """
    Phoneme forced alignment using torchaudio's wav2vec2 + CTC forced aligner.
    Uses MMS_FA (character-level) as a reliable fallback since true phoneme
    models require extra lexicon files. Characters serve as sub-word units
    and map cleanly to lip motion timing.
    """
    try:
        from torchaudio.pipelines import MMS_FA as bundle
        import torchaudio.functional as F_audio

        model     = bundle.get_model().to(device)
        tokenizer = bundle.get_tokenizer()
        aligner   = bundle.get_aligner()
        labels    = bundle.get_labels()

        phonemes = []

        with torch.inference_mode():
            waveform = torch.from_numpy(audio_np.copy()).unsqueeze(0).to(device)
            if bundle.sample_rate != AUDIO_SR:
                waveform = F_audio.resample(waveform, AUDIO_SR, bundle.sample_rate)
            emission, _ = model(waveform)
            emission     = emission[0]   # (T, vocab)

        duration     = audio_np.shape[0] / AUDIO_SR
        total_frames = emission.shape[0]

        for word_seg in words:
            word_text  = word_seg["word"].strip()
            # lowercase, letters only — MMS_FA is char-level
            word_clean = [c for c in word_text.lower() if c.isalpha()]
            if not word_clean:
                continue

            try:
                tokens = tokenizer(word_clean)
            except Exception as e:
                print(f"    [phoneme] tokenizer failed for '{word_text}': {e}")
                continue

            t_start = max(0, int((word_seg["start"] / duration) * total_frames))
            t_end   = min(total_frames, int((word_seg["end"] / duration) * total_frames))

            if t_end - t_start < len(tokens):
                continue

            # word_emission = emission[t_start:t_end].unsqueeze(0)  # (1, T', vocab)
            word_emission = emission[t_start:t_end]               # (T', vocab)


            try:
                token_spans   = aligner(word_emission, tokens)         # no list wrapper either
                # token_spans   = aligner(word_emission, [tokens])
                word_duration = word_seg["end"] - word_seg["start"]
                span_frames   = t_end - t_start

                for span in token_spans[0]:
                    ph_start = word_seg["start"] + (span.start / span_frames) * word_duration
                    ph_end   = word_seg["start"] + (span.end   / span_frames) * word_duration
                    phonemes.append({
                        "phoneme":     labels[span.token],
                        "word":        word_seg["word"].strip(),
                        "start":       round(ph_start, 4),
                        "end":         round(ph_end,   4),
                        "start_frame": round(ph_start * fps),
                        "end_frame":   round(ph_end   * fps),
                    })
            except Exception as e:
                print(f"    [phoneme] aligner failed for '{word_text}': {e}")
                continue

        print(f"    [phoneme] {len(phonemes)} phonemes across {len(words)} words")
        return phonemes

    except Exception as e:
        print(f"    [phoneme] outer failure: {e}")
        traceback.print_exc()
        return []

def run_whisperx(wav_path: Path, device: str, fps: float) -> dict:
    """
    Run WhisperX for word-level alignment, then wav2vec2 CTC forced
    aligner for phoneme-level alignment.
    """
    try:
        wx    = _require("whisperx")
        model = wx.load_model("large-v3", device=device, compute_type="float16")
        audio = wx.load_audio(str(wav_path))
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

        transcript = " ".join(w["word"] for w in words)

        # ── phoneme alignment via wav2vec2 CTC forced aligner ─────────────
        audio_np = audio if isinstance(audio, np.ndarray) else np.array(audio)
        phonemes = get_phoneme_alignments(words, audio_np, fps, device)

        print(f"    [phoneme] {len(phonemes)} phonemes across {len(words)} words")

        return {
            "transcript": transcript,
            "language":   result.get("language", "en"),
            "words":      words,
            "phonemes":   phonemes,
        }

    except Exception as e:
        print(f"    [WhisperX] failed: {e}")
        return {"transcript": "", "language": "unknown", "words": [], "phonemes": []}

# def add_frame_indices(alignment: dict, fps: float) -> dict:
#     for seg in alignment.get("words", []):
#         seg["start_frame"] = round(seg["start"] * fps)
#         seg["end_frame"]   = round(seg["end"]   * fps)
#     for seg in alignment.get("phonemes", []):
#         seg["start_frame"] = round(seg["start"] * fps)
#         seg["end_frame"]   = round(seg["end"]   * fps)
#     return alignment

def add_frame_indices(alignment: dict, fps: float) -> dict:
    for seg in alignment.get("words", []):
        seg["start_frame"] = round(seg["start"] * fps)
        seg["end_frame"]   = round(seg["end"]   * fps)
    # phonemes already have frame indices added during alignment
    return alignment

def video_is_done(video_out: Path) -> bool:
    return (video_out / "meta.json").exists()


# ── per-video processing ─────────────────────────────────────────────────────

def process_video(video_path: Path, video_out: Path, category: str, device: str):
    """
    Full multi-scale preprocessing for one video.
    video_path: .../Category/Ethnicity/Gender/identity_id/video.mp4
    video_out:  .../Category/Ethnicity/Gender/identity_id/video_id/
    """

    lip_dir   = video_out / "scale1_phoneme" / "lip_crops"
    face_dir  = video_out / "scale2_word"    / "face_crops"
    full_dir  = video_out / "scale3_phrase"  / "full_frames"
    audio_dir = video_out / "audio"

    for d in [lip_dir, face_dir, full_dir, audio_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ── audio extraction ──────────────────────────────────────────────────
    wav_path = audio_dir / "audio.wav"
    if not wav_path.exists():
        ok = extract_audio(video_path, wav_path)
        if not ok:
            print(f"    [SKIP] audio extraction failed for {video_path.name}")
            return False

    # ── video metadata ────────────────────────────────────────────────────
    cap          = cv2.VideoCapture(str(video_path))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps

    # ── WhisperX + pyfoal alignment ───────────────────────────────────────
    align_path = audio_dir / "alignment.json"
    if align_path.exists():
        with open(align_path) as f:
            alignment = json.load(f)
    else:
        # alignment = run_whisperx(wav_path, device)
        alignment = run_whisperx(wav_path, device, fps)
        alignment = add_frame_indices(alignment, fps)
        with open(align_path, "w") as f:
            json.dump(alignment, f, indent=2)

    # ── frame extraction ──────────────────────────────────────────────────
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

        lip_path  = lip_dir  / frame_name
        face_path = face_dir / frame_name

        if not lip_path.exists() or not face_path.exists():
            lip  = crop_lip(frame_rgb, face_mesh_tuple)
            face = crop_face(frame_rgb, face_mesh_tuple)

            if lip is not None:
                Image.fromarray(lip).save(lip_path, quality=92)
                lip_saved += 1
            else:
                failed_frames += 1

            if face is not None:
                Image.fromarray(face).save(face_path, quality=92)
                face_saved += 1

        if frame_idx % phrase_step == 0:
            full_path = full_dir / frame_name
            if not full_path.exists():
                full = cv2.resize(frame_rgb, (FULL_W, FULL_H))
                Image.fromarray(full).save(full_path, quality=92)
                phrase_saved += 1

        frame_idx += 1

    cap.release()

    # ── path component extraction ─────────────────────────────────────────
    parts     = video_path.parts
    cat_idx   = next((i for i, p in enumerate(parts) if p == category), None)
    ethnicity = parts[cat_idx + 1] if cat_idx is not None else "unknown"
    gender    = parts[cat_idx + 2] if cat_idx is not None else "unknown"
    identity  = parts[cat_idx + 3] if cat_idx is not None else "unknown"

    # ── meta.json ─────────────────────────────────────────────────────────
    meta = {
        "video_id":        video_path.stem,
        "identity":        identity,
        "label":           LABEL_MAP.get(category, -1),
        "label_str":       category,
        "manipulation":    MANIP_MAP.get(category, "unknown"),
        "ethnicity":       ethnicity,
        "gender":          gender,
        "duration_sec":    round(duration_sec, 3),
        "fps":             fps,
        "total_frames":    total_frames,
        "transcript":      alignment.get("transcript", ""),
        "language":        alignment.get("language", "unknown"),
        "n_words":         len(alignment.get("words", [])),
        "n_phonemes":      len(alignment.get("phonemes", [])),
        "n_lip_frames":    lip_saved,
        "n_face_frames":   face_saved,
        "n_phrase_frames": phrase_saved,
        "failed_frames":   failed_frames,
    }

    with open(video_out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return True


# ── dataset scanner ──────────────────────────────────────────────────────────

def scan_dataset(input_dir: Path):
    """
    Returns:
    {
      category: {
        ethnicity: {
          gender: [list of video Paths]
        }
      }
    }
    Videos are found by recursing into identity subdirectories.
    """
    tree = {}
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
                gen = gen_path.name
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
                print(f"        --ethnicity {eth:20s} --gender {gen:6s}  → {len(vids):4d} videos")
        total += cat_total
    print(f"\n  TOTAL: {total} videos")
    print("=" * 60)
    print("\nExample usage:")
    print("  python preprocess_fakeavceleb.py \\")
    print("      --input  <dataset_path> \\")
    print("      --output <output_path>  \\")
    print("      --category FakeVideo-FakeAudio \\")
    print("      --ethnicity African \\")
    print("      --gender women\n")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="FakeAVCeleb multi-scale preprocessor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input",     required=True, help="Path to FakeAVCeleb_v1.2 root")
    parser.add_argument("--output",    default=None,  help="Path to output directory")
    parser.add_argument("--list",      action="store_true",
                        help="List available categories/ethnicities/genders and exit")
    parser.add_argument("--category",  default=None, choices=CATEGORIES,
                        help="Only process this category")
    parser.add_argument("--ethnicity", default=None,
                        help="Only process this ethnicity")
    parser.add_argument("--gender",    default=None, choices=["men", "women"],
                        help="Only process this gender")
    parser.add_argument("--device",    default=None,
                        help="cuda / cpu (default: auto-detect)")
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
                    identity  = vp.parent.name          # e.g. id00076
                    video_out = output_dir / cat / eth / gen / identity / vp.stem
                    video_queue.append((vp, video_out, cat))

    if not video_queue:
        sys.exit("[ERROR] No videos matched the given filters. Use --list to see options.")

    already_done = sum(1 for _, vo, _ in video_queue if video_is_done(vo))
    to_process   = len(video_queue) - already_done

    print(f"\n[INFO] Videos matched : {len(video_queue)}")
    print(f"[INFO] Already done   : {already_done}  (will be skipped)")
    print(f"[INFO] To process     : {to_process}")
    if args.category:
        print(f"[INFO] Category   filter : {args.category}")
    if args.ethnicity:
        print(f"[INFO] Ethnicity  filter : {args.ethnicity}")
    if args.gender:
        print(f"[INFO] Gender     filter : {args.gender}")
    print()

    success = 0
    failed  = 0

    for video_path, video_out, category in tqdm(video_queue, desc="Videos", unit="vid"):

        if video_is_done(video_out):
            continue

        tqdm.write(f"  Processing: {video_path.parent.name}/{video_path.name}")

        try:
            ok = process_video(video_path, video_out, category, device)
            if ok:
                success += 1
                tqdm.write(f"  ✓ Done: {video_path.parent.name}/{video_out.name}")
            else:
                failed += 1
                tqdm.write(f"  ✗ Failed (no audio/frames): {video_path.name}")

        except Exception as e:
            failed += 1
            tqdm.write(f"  ✗ Exception on {video_path.name}: {e}")
            tqdm.write(traceback.format_exc())
            continue

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
