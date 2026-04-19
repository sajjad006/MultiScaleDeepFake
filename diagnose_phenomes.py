"""
diagnose_phonemes.py
====================
Run this on the problematic audio file to find exactly why
so few phonemes are being detected.

Usage:
    python diagnose_phonemes.py --wav /path/to/audio/audio.wav
"""

import argparse
import json
import numpy as np
import torch
import torchaudio
from scipy.io import wavfile
from pathlib import Path

AUDIO_SR        = 16000
WAV2VEC2_STRIDE = 320
STRIDE_SEC      = WAV2VEC2_STRIDE / AUDIO_SR   # 20ms
NON_PHONEME_TOKENS = {"|", "spn", "<unk>", "<s>", "</s>", "<pad>"}


def load_wav_raw(path):
    """Load WAV and return BOTH the raw data AND normalized, for comparison."""
    sr, data = wavfile.read(path)
    raw_dtype = data.dtype

    # show raw stats before any normalization
    print(f"\n{'='*60}")
    print("STEP 1 — RAW WAV FILE")
    print(f"{'='*60}")
    print(f"  File        : {path}")
    print(f"  Sample rate : {sr} Hz  (expected: {AUDIO_SR})")
    print(f"  dtype       : {raw_dtype}")
    print(f"  Shape       : {data.shape}")
    print(f"  Duration    : {len(data)/sr:.3f}s")
    if data.ndim == 2:
        print(f"  Channels    : {data.shape[1]}  → will be mixed to mono")

    if data.ndim == 2:
        data = data.mean(axis=1)

    print(f"\n  Raw amplitude stats:")
    print(f"    min  : {data.min()}")
    print(f"    max  : {data.max()}")
    print(f"    mean : {data.mean():.4f}")

    # normalize to float32 [-1, 1]
    if raw_dtype == np.int16:
        wav_f32 = data.astype(np.float32) / 32768.0
    elif raw_dtype == np.int32:
        wav_f32 = data.astype(np.float32) / 2147483648.0
    elif raw_dtype == np.uint8:
        wav_f32 = (data.astype(np.float32) - 128) / 128.0
    else:
        wav_f32 = data.astype(np.float32)

    rms = float(np.sqrt(np.mean(wav_f32 ** 2)))
    peak = float(np.abs(wav_f32).max())

    print(f"\n  Normalized float32 stats:")
    print(f"    peak RMS    : {rms:.4f}  (healthy speech: 0.05 – 0.4)")
    print(f"    peak abs    : {peak:.4f}  (should be ≤ 1.0)")

    if peak < 0.01:
        print(f"\n  *** WARNING: Audio is extremely quiet (peak={peak:.5f})")
        print(f"      Model will see near-zero input → mostly blank outputs")
    elif peak > 1.0:
        print(f"\n  *** WARNING: Audio is clipping (peak={peak:.4f} > 1.0)")

    if sr != AUDIO_SR:
        print(f"\n  *** WARNING: Sample rate mismatch!")
        print(f"      File is {sr}Hz but model expects {AUDIO_SR}Hz")
        print(f"      Will resample automatically")

    return wav_f32, sr


def check_processor_normalization(wav_f32, sr, processor):
    """
    Show exactly what the processor feeds to the model.
    wav2vec2 processor applies zero-mean unit-variance normalization per sample.
    If the input is already near-zero, post-normalization it explodes to NaN/inf.
    """
    print(f"\n{'='*60}")
    print("STEP 2 — PROCESSOR NORMALIZATION CHECK")
    print(f"{'='*60}")

    # resample if needed
    if sr != AUDIO_SR:
        print(f"  Resampling {sr}Hz → {AUDIO_SR}Hz ...")
        t = torch.from_numpy(wav_f32).unsqueeze(0)
        t = torchaudio.functional.resample(t, sr, AUDIO_SR)
        wav_f32 = t.squeeze(0).numpy()

    inputs = processor(
        wav_f32,
        sampling_rate=AUDIO_SR,
        return_tensors="pt",
        padding=True,
    )
    iv = inputs.input_values[0].numpy()

    print(f"  input_values shape : {iv.shape}")
    print(f"  input_values stats :")
    print(f"    min  : {iv.min():.4f}")
    print(f"    max  : {iv.max():.4f}")
    print(f"    mean : {iv.mean():.4f}  (should be ~0.0 after norm)")
    print(f"    std  : {iv.std():.4f}   (should be ~1.0 after norm)")

    has_nan = np.isnan(iv).any()
    has_inf = np.isinf(iv).any()
    if has_nan or has_inf:
        print(f"\n  *** CRITICAL: NaN={has_nan}, Inf={has_inf} in input_values!")
        print(f"      Model will output garbage. Audio is likely silent/zero.")

    return wav_f32, inputs


def run_model_and_diagnose(inputs, model, processor, wav_f32, fps=25.0):
    """
    Run the model and show the full token distribution before any filtering.
    """
    print(f"\n{'='*60}")
    print("STEP 3 — RAW MODEL OUTPUT")
    print(f"{'='*60}")

    device = next(model.parameters()).device
    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        logits = model(input_values).logits

    print(f"  Logits shape : {logits.shape}  (1, T, vocab)")
    print(f"  T frames     : {logits.shape[1]}  × {STRIDE_SEC*1000:.0f}ms = {logits.shape[1]*STRIDE_SEC:.2f}s")

    ids = torch.argmax(logits, dim=-1)[0].cpu().tolist()
    blank_id = processor.tokenizer.pad_token_id
    vocab = processor.tokenizer.convert_ids_to_tokens(range(len(processor.tokenizer)))

    # token distribution
    from collections import Counter
    counts = Counter(ids)
    n_blank = counts[blank_id]
    n_total = len(ids)
    print(f"\n  Token distribution:")
    print(f"    Total frames : {n_total}")
    print(f"    Blank frames : {n_blank}  ({100*n_blank/n_total:.1f}%)")
    print(f"    Non-blank    : {n_total - n_blank}  ({100*(n_total-n_blank)/n_total:.1f}%)")

    if n_blank / n_total > 0.95:
        print(f"\n  *** WARNING: >95% blanks — model sees little speech signal")
        print(f"      Most likely cause: audio too quiet or wrong sample rate")

    # show all non-blank tokens
    non_blank = [(vocab[tid], cnt) for tid, cnt in counts.most_common() if tid != blank_id]
    print(f"\n  Top non-blank tokens:")
    for tok, cnt in non_blank[:20]:
        bar = "#" * min(cnt, 40)
        print(f"    '{tok:8s}' : {cnt:4d} frames  {bar}")

    # raw segments BEFORE any filter
    print(f"\n{'='*60}")
    print("STEP 4 — RAW SEGMENTS (before duration filter)")
    print(f"{'='*60}")

    raw_segs = []
    prev = None
    seg_start = 0
    for t, tid in enumerate(ids):
        if tid != prev:
            if prev is not None and prev != blank_id:
                label  = vocab[prev]
                dur_ms = (t - seg_start) * STRIDE_SEC * 1000
                raw_segs.append({
                    "token":   label,
                    "start_s": round(seg_start * STRIDE_SEC, 3),
                    "end_s":   round(t * STRIDE_SEC, 3),
                    "dur_ms":  round(dur_ms, 1),
                    "frames":  t - seg_start,
                    "is_noise": label in NON_PHONEME_TOKENS,
                })
            seg_start = t
            prev = tid

    if prev is not None and prev != blank_id:
        label  = vocab[prev]
        dur_ms = (len(ids) - seg_start) * STRIDE_SEC * 1000
        raw_segs.append({
            "token":   label,
            "start_s": round(seg_start * STRIDE_SEC, 3),
            "end_s":   round(len(ids) * STRIDE_SEC, 3),
            "dur_ms":  round(dur_ms, 1),
            "frames":  len(ids) - seg_start,
            "is_noise": label in NON_PHONEME_TOKENS,
        })

    n_noise    = sum(1 for s in raw_segs if s["is_noise"])
    n_short    = sum(1 for s in raw_segs if not s["is_noise"] and s["dur_ms"] < 30)
    n_valid    = sum(1 for s in raw_segs if not s["is_noise"] and s["dur_ms"] >= 30)

    print(f"  Total raw segments   : {len(raw_segs)}")
    print(f"  Noise tokens (|,spn) : {n_noise}  → filtered out")
    print(f"  Short < 30ms         : {n_short}  → filtered out by MIN_PHONEME_MS")
    print(f"  Valid phonemes       : {n_valid}  → kept")

    if n_short > n_valid:
        print(f"\n  *** KEY FINDING: more short segments ({n_short}) than valid ones ({n_valid})")
        print(f"      The 30ms filter is removing most detections.")
        print(f"      Recommended fix: lower MIN_PHONEME_MS to 20ms (= 1 CTC frame)")

    print(f"\n  All raw segments:")
    print(f"  {'token':10s} {'start_s':>8} {'dur_ms':>8} {'frames':>7} {'status':>12}")
    print(f"  {'-'*52}")
    for s in raw_segs:
        if s["is_noise"]:
            status = "noise→skip"
        elif s["dur_ms"] < 30:
            status = "short→skip"
        else:
            status = "KEEP"
        print(f"  {s['token']:10s} {s['start_s']:>8.3f} {s['dur_ms']:>8.1f} "
              f"{s['frames']:>7} {status:>12}")

    return ids, vocab, blank_id, raw_segs


def suggest_fixes(raw_segs, wav_f32):
    print(f"\n{'='*60}")
    print("STEP 5 — DIAGNOSIS SUMMARY & FIXES")
    print(f"{'='*60}")

    rms  = float(np.sqrt(np.mean(wav_f32 ** 2)))
    peak = float(np.abs(wav_f32).max())

    n_short = sum(1 for s in raw_segs
                  if not s["is_noise"] and s["dur_ms"] < 30)
    n_valid = sum(1 for s in raw_segs
                  if not s["is_noise"] and s["dur_ms"] >= 30)

    if peak < 0.02:
        print("""
  ROOT CAUSE: Audio too quiet (peak < 0.02)
  FIX: Add amplitude normalization before feeding to wav2vec2:
  
      wav_np = wav_np / (np.abs(wav_np).max() + 1e-8)
  
  Add this line in extract_phonemes() after the resample block.
""")
    elif n_short > n_valid * 2:
        print(f"""
  ROOT CAUSE: MIN_PHONEME_MS=30ms filter is too aggressive.
  Model IS detecting phonemes but they're 1-frame (20ms) each,
  which our filter removes.
  
  FIX: Lower MIN_PHONEME_MS from 30ms to 20ms in phoneme_extractor.py:
  
      MIN_PHONEME_MS = 20.0   # 1 CTC frame minimum
  
  This keeps all genuine 1-frame phoneme detections.
  Short vowels in fast speech are legitimately 20ms.
""")
    else:
        print(f"""
  ROOT CAUSE: Unclear from static analysis.
  Peak amplitude {peak:.3f}, RMS {rms:.4f}.
  {n_valid} valid segments detected after filtering.
  
  Try running with a different clip to rule out clip-specific issues.
""")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav",    required=True, help="Path to audio.wav")
    parser.add_argument("--fps",    type=float, default=25.0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # load audio
    wav_f32, sr = load_wav_raw(args.wav)

    # load model
    print(f"\n{'='*60}")
    print("Loading wav2vec2-lv-60-espeak-cv-ft ...")
    print(f"{'='*60}")
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    processor = Wav2Vec2Processor.from_pretrained(
        "facebook/wav2vec2-lv-60-espeak-cv-ft")
    model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/wav2vec2-lv-60-espeak-cv-ft").to(device)
    model.eval()
    print("Model loaded.")

    # check processor normalization
    wav_f32_resampled, inputs = check_processor_normalization(
        wav_f32, sr, processor)

    # run model and diagnose
    ids, vocab, blank_id, raw_segs = run_model_and_diagnose(
        inputs, model, processor, wav_f32_resampled, fps=args.fps)

    # suggest fixes
    suggest_fixes(raw_segs, wav_f32_resampled)


if __name__ == "__main__":
    main()
