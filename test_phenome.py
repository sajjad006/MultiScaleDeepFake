"""
phoneme_extractor.py
====================
Drop-in replacement for the extract_phonemes() function in
preprocess_fakeavceleb_final.py.

Fixes vs the original script
-----------------------------
Bug 1 — Word boundary token '|' was passed through as a phoneme.
         '|' is a separator emitted by wav2vec2-lv-60-espeak-cv-ft between
         words. It has no acoustic meaning — it is purely a positional marker
         that the CTC decoder emits at word breaks. Including it pollutes the
         phoneme list with fake entries and corrupts frame-index lookups.

Bug 2 — 'spn' (spoken noise) token was passed through as a phoneme.
         wav2vec2-lv-60-espeak-cv-ft reserves 'spn' for non-speech sounds
         (breath, lip smack, background noise). These should be logged
         separately (they are recorded as gaps with type='vocalization' by
         the gap classifier) but must never appear in the phoneme list.

Bug 3 — No minimum duration filter.
         CTC sometimes emits 1-frame segments (20ms) for phonemes that are
         acoustically ambiguous or co-articulated. A 20ms segment maps to
         exactly 0-1 video frames at 25fps, making it useless for visual
         crop extraction. Filtered at MIN_PHONEME_MS=30ms.

Bug 4 — No maximum duration cap.
         Forced aligner drift can cause the last phoneme segment to run to
         the end of the logit sequence, inflating its duration by hundreds of
         milliseconds. Capped at MAX_PHONEME_MS.

Not a bug — consecutive identical phonemes (e.g. /l/ in 'hello') produce two
separate entries with a blank between them. CTC guarantees this separation is
acoustically real (the blank represents a closure/release event), so they are
kept as distinct phoneme entries. This is correct behaviour.

Usage
-----
Replace the extract_phonemes() function in preprocess_fakeavceleb_final.py
with the one from this file. The function signature is identical.

You can also test standalone:
    python phoneme_extractor.py --wav /path/to/audio.wav --fps 25
"""

import sys
import json
import argparse
import numpy as np
import torch
import torchaudio
from scipy.io import wavfile

# ── constants ─────────────────────────────────────────────────────────────────

AUDIO_SR        = 16000
WAV2VEC2_STRIDE = 320          # CNN downsampling factor — fixed for all wav2vec2
STRIDE_SEC      = WAV2VEC2_STRIDE / AUDIO_SR   # 0.020s = 20ms per output frame

# Tokens that must never appear in the phoneme output
# '|'   → word boundary marker (positional, not acoustic)
# 'spn' → spoken noise (non-speech, handled by gap classifier)
# '<unk>' '<s>' '</s>' → special tokens from the tokenizer
NON_PHONEME_TOKENS = {"|", "spn", "<unk>", "<s>", "</s>", "<pad>"}

MIN_PHONEME_MS = 30.0    # shorter than this = CTC artifact, discard
MAX_PHONEME_MS = 800.0   # longer than this  = aligner drift, cap

# Module-level model cache — load once per process
_MODEL_CACHE: dict = {}


# ── model loading ─────────────────────────────────────────────────────────────

def load_phoneme_model(device: str):
    """
    Load facebook/wav2vec2-lv-60-espeak-cv-ft once and cache in process memory.

    Why this model:
      - Trained on 60 languages → IPA output is language-agnostic.
        Works identically on FakeAVCeleb (English) and PolyGlotFake
        (Mandarin, French, German, Spanish, Hindi, English).
      - No pronunciation dictionary or transcript required. Runs directly
        on the waveform, so it is robust to WhisperX failures.
      - CNN stride = 320 samples → 20ms resolution at 16kHz, matching the
        ~25ms phoneme scale in the HAVDNet architecture.
      - IPA symbols are consistent across languages, giving the DataLoader
        a unified phoneme vocabulary for all training data.
    """
    if "model" not in _MODEL_CACHE:
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        model_id  = "facebook/wav2vec2-lv-60-espeak-cv-ft"
        print(f"    [Phoneme] Loading {model_id} on {device} ...")
        processor = Wav2Vec2Processor.from_pretrained(model_id)
        model     = Wav2Vec2ForCTC.from_pretrained(model_id).to(device)
        model.eval()
        _MODEL_CACHE["model"]     = model
        _MODEL_CACHE["processor"] = processor
        _MODEL_CACHE["device"]    = device
        # Pre-build the full vocab list once — avoids repeated tokenizer calls
        vocab_size = len(processor.tokenizer)
        _MODEL_CACHE["vocab"] = processor.tokenizer.convert_ids_to_tokens(
            range(vocab_size)
        )
        _MODEL_CACHE["blank_id"] = processor.tokenizer.pad_token_id
        print(f"    [Phoneme] Ready. Vocab size: {vocab_size}, "
              f"blank_id: {_MODEL_CACHE['blank_id']}")

    return (
        _MODEL_CACHE["model"],
        _MODEL_CACHE["processor"],
        _MODEL_CACHE["vocab"],
        _MODEL_CACHE["blank_id"],
    )


# ── CTC decoder ───────────────────────────────────────────────────────────────

def _decode_ctc(ids: list, vocab: list, blank_id: int,
                max_audio_sec: float, fps: float) -> list:
    """
    Greedy CTC blank collapsing with full filtering and time conversion.

    Parameters
    ----------
    ids           : raw argmax token ids from wav2vec2 logits, shape (T,)
    vocab         : list of token strings, indexed by token id
    blank_id      : id of the CTC blank token (<pad>)
    max_audio_sec : total audio duration — used to cap the last segment
    fps           : video frame rate for start_frame / end_frame

    Returns
    -------
    list of phoneme dicts:
        ph_idx, phoneme, start, end, duration_ms, start_frame, end_frame
    """
    raw_segments = []
    prev         = None
    seg_start    = 0

    for t, token_id in enumerate(ids):
        if token_id != prev:
            if prev is not None and prev != blank_id:
                label  = vocab[prev]
                dur_ms = (t - seg_start) * STRIDE_SEC * 1000

                # Bug 1 fix: skip word boundary and noise tokens
                if label in NON_PHONEME_TOKENS:
                    seg_start = t
                    prev      = token_id
                    continue

                # Bug 3 fix: skip sub-phoneme duration artifacts
                if dur_ms < MIN_PHONEME_MS:
                    seg_start = t
                    prev      = token_id
                    continue

                raw_segments.append({
                    "label":      label,
                    "start_step": seg_start,
                    "end_step":   t,
                    "dur_ms":     dur_ms,
                })

            seg_start = t
            prev      = token_id

    # Flush last segment
    if prev is not None and prev != blank_id:
        label  = vocab[prev]
        dur_ms = (len(ids) - seg_start) * STRIDE_SEC * 1000
        if label not in NON_PHONEME_TOKENS and dur_ms >= MIN_PHONEME_MS:
            raw_segments.append({
                "label":      label,
                "start_step": seg_start,
                "end_step":   len(ids),
                "dur_ms":     dur_ms,
            })

    # Convert steps → seconds → frame indices
    phonemes_out = []
    for i, s in enumerate(raw_segments):
        start_sec = s["start_step"] * STRIDE_SEC
        raw_end   = s["end_step"]   * STRIDE_SEC

        # Bug 4 fix: cap aligner drift at max_audio_sec and MAX_PHONEME_MS
        end_sec   = min(raw_end, max_audio_sec)
        end_sec   = min(end_sec, start_sec + MAX_PHONEME_MS / 1000.0)

        actual_dur_ms = (end_sec - start_sec) * 1000.0
        if actual_dur_ms < MIN_PHONEME_MS:
            continue   # post-cap duration too short — discard

        phonemes_out.append({
            "ph_idx":      i,
            "phoneme":     s["label"],
            "start":       round(start_sec,       4),
            "end":         round(end_sec,         4),
            "duration_ms": round(actual_dur_ms,   1),
            "start_frame": int(round(start_sec * fps)),
            "end_frame":   int(round(end_sec   * fps)),
        })

    # Re-index ph_idx after filtering
    for i, p in enumerate(phonemes_out):
        p["ph_idx"] = i

    return phonemes_out


# ── public API ────────────────────────────────────────────────────────────────

def extract_phonemes(wav_np: np.ndarray, sr: int,
                     fps: float, device: str) -> list:
    """
    Run wav2vec2 CTC phoneme decoding on a 16kHz mono waveform.

    Drop-in replacement for the extract_phonemes() in
    preprocess_fakeavceleb_final.py. Signature is identical.

    Parameters
    ----------
    wav_np  : float32 mono waveform (values in [-1, 1])
    sr      : sample rate — resampled to 16kHz automatically if needed
    fps     : video frame rate, used for start_frame / end_frame
    device  : "cuda" or "cpu"

    Returns
    -------
    list of dicts:
        ph_idx (int), phoneme (IPA str), start (sec), end (sec),
        duration_ms (float), start_frame (int), end_frame (int)
    """
    model, processor, vocab, blank_id = load_phoneme_model(device)

    # Resample if needed (pipeline always writes 16kHz, but defensive)
    if sr != AUDIO_SR:
        wav_t  = torch.from_numpy(wav_np).unsqueeze(0)
        wav_t  = torchaudio.functional.resample(wav_t, sr, AUDIO_SR)
        wav_np = wav_t.squeeze(0).numpy()

    max_audio_sec = len(wav_np) / AUDIO_SR

    # Normalise + encode
    inputs = processor(
        wav_np,
        sampling_rate=AUDIO_SR,
        return_tensors="pt",
        padding=True,
    )
    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        logits = model(input_values).logits   # (1, T, vocab_size)

    ids = torch.argmax(logits, dim=-1)[0].cpu().tolist()   # (T,)

    return _decode_ctc(ids, vocab, blank_id, max_audio_sec, fps)


# ── standalone test ───────────────────────────────────────────────────────────

def _load_wav(path: str):
    sr, data = wavfile.read(path)
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


def _print_table(phonemes: list):
    print(f"\n{'='*68}")
    print(f"  Extracted {len(phonemes)} phonemes")
    print(f"{'='*68}")
    print(f"  {'#':<4} {'IPA':<10} {'start_s':>8} {'end_s':>7} "
          f"{'dur_ms':>8} {'f_start':>8} {'f_end':>6}")
    print(f"  {'-'*60}")
    for p in phonemes:
        print(f"  {p['ph_idx']:<4} {p['phoneme']:<10} {p['start']:>8.3f} "
              f"{p['end']:>7.3f} {p['duration_ms']:>8.1f} "
              f"{p['start_frame']:>8} {p['end_frame']:>6}")


def main():
    parser = argparse.ArgumentParser(description="Phoneme extractor test")
    parser.add_argument("--wav",    required=True, help="Path to 16kHz mono WAV")
    parser.add_argument("--fps",    type=float, default=25.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=None, help="Save JSON to this path")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"WAV    : {args.wav}")
    print(f"FPS    : {args.fps}\n")

    wav_np, sr = _load_wav(args.wav)
    print(f"Audio  : {len(wav_np)/sr:.2f}s @ {sr}Hz\n")

    phonemes = extract_phonemes(wav_np, sr, fps=args.fps, device=device)
    _print_table(phonemes)

    if args.output:
        out = {
            "wav":        args.wav,
            "duration_s": round(len(wav_np) / sr, 3),
            "fps":        args.fps,
            "n_phonemes": len(phonemes),
            "phonemes":   phonemes,
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
