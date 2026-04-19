"""
Syllable Extraction Test Script
================================
Tests syllable extraction on a single audio file using two complementary methods:

  Method A — Energy onset detection (audio-only, no transcript needed)
             Uses RMS energy peaks to find syllable nuclei directly from the
             waveform. Works without any text. Good for validation.

  Method B — Transcript proportional mapping (what the real pipeline uses)
             Takes word-level WhisperX boundaries + transcript, splits each
             word into syllables via pyphen, then proportionally distributes
             the word's time across its syllables by character length.

Usage
-----
  # On a real audio file from your dataset:
  python test_syllable_extraction.py --audio /path/to/audio.wav

  # With a transcript (simulates what WhisperX gives you):
  python test_syllable_extraction.py \\
      --audio /path/to/audio.wav \\
      --transcript "hello world how are you"

  # With WhisperX alignment JSON (full pipeline simulation):
  python test_syllable_extraction.py \\
      --audio /path/to/audio.wav \\
      --alignment /path/to/alignment.json

  # Generate a synthetic test WAV (no real audio needed):
  python test_syllable_extraction.py --synth

Install
-------
  pip install pyphen scipy numpy
  # WhisperX only needed for --alignment mode:
  # pip install whisperx
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, filtfilt, find_peaks


# ─────────────────────────────────────────────────────────────────────────────
# AUDIO UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def load_wav(path: str):
    """Load WAV file, return (signal_float32, sample_rate)."""
    sr, data = wavfile.read(path)

    # convert to float32 in [-1, 1]
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        data = (data.astype(np.float32) - 128) / 128.0
    else:
        data = data.astype(np.float32)

    # stereo → mono
    if data.ndim == 2:
        data = data.mean(axis=1)

    return data, sr


def make_synthetic_wav(output_path: str, transcript_words=None):
    """
    Generate a synthetic WAV with planted syllable onsets.
    Returns (signal, sr, planted_onsets, word_alignment).
    """
    sr = 16000
    duration = 4.0
    t = np.linspace(0, duration, int(sr * duration))

    # default: 'hello world how are you doing'
    if transcript_words is None:
        transcript_words = ["hello", "world", "how", "are", "you", "doing"]

    # plant syllable onsets with ~300ms inter-syllable interval
    import pyphen
    dic = pyphen.Pyphen(lang="en_US")

    word_alignment = []
    syllable_onsets = []
    t_cursor = 0.20  # start at 200ms

    for word in transcript_words:
        syls = dic.inserted(word, hyphen="|").split("|")
        n    = len(syls)
        word_start = t_cursor
        for _ in syls:
            syllable_onsets.append(round(t_cursor, 3))
            t_cursor += 0.28   # ~280ms per syllable
        word_end = t_cursor - 0.05
        word_alignment.append({
            "word":  word,
            "start": round(word_start, 3),
            "end":   round(word_end, 3),
        })
        t_cursor += 0.15   # 150ms pause between words

    # build amplitude envelope from syllable onsets
    envelope = np.zeros_like(t)
    for onset in syllable_onsets:
        idx   = int(onset * sr)
        width = int(0.14 * sr)
        if idx + width < len(envelope):
            pulse = np.exp(-0.5 * ((np.arange(width) - width // 2) / (width / 6)) ** 2)
            envelope[idx:idx + width] += pulse

    carrier = (0.35 * np.sin(2 * np.pi * 150 * t)
               + 0.20 * np.sin(2 * np.pi * 300 * t)
               + 0.10 * np.sin(2 * np.pi * 450 * t))
    signal = (carrier * envelope * 0.8).astype(np.float32)

    wavfile.write(output_path, sr, signal)
    print(f"[synth] WAV written: {output_path}  ({duration}s, {sr}Hz)")
    print(f"[synth] Words      : {transcript_words}")
    print(f"[synth] Planted    : {len(syllable_onsets)} syllable onsets")
    print(f"[synth] Onsets     : {syllable_onsets}\n")

    return signal, sr, syllable_onsets, word_alignment


# ─────────────────────────────────────────────────────────────────────────────
# METHOD A — ENERGY ONSET DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_syllables_energy(signal: np.ndarray, sr: int,
                             frame_ms=10, hop_ms=5,
                             min_gap_ms=80,
                             energy_thresh_pct=0.12,
                             prominence_pct=0.04) -> list:
    """
    Detect syllable nuclei from raw waveform using RMS energy peaks.

    Returns list of dicts:
        {"syl_idx": int, "onset_sec": float, "peak_energy": float,
         "start_frame": int, "end_frame": int}

    Parameters
    ----------
    frame_ms        : RMS window length in ms
    hop_ms          : hop between windows in ms
    min_gap_ms      : minimum gap between syllables (fastest rate ~1000/min_gap_ms syl/s)
    energy_thresh_pct : peak must be >= this fraction of max energy
    prominence_pct  : scipy find_peaks prominence threshold
    """
    # bandpass filter — keep speech (100–4000 Hz), remove rumble and hiss
    nyq  = sr / 2
    b, a = butter(4, [100 / nyq, 4000 / nyq], btype="band")
    filt = filtfilt(b, a, signal)

    # RMS energy per frame
    frame_len = int(sr * frame_ms / 1000)
    hop_len   = int(sr * hop_ms   / 1000)
    frames    = [filt[i:i + frame_len]
                 for i in range(0, len(filt) - frame_len, hop_len)]
    energy    = np.array([np.sqrt(np.mean(f ** 2)) for f in frames])
    times     = np.array([i * hop_ms / 1000 for i in range(len(energy))])

    # smooth to reduce micro-peaks
    energy_smooth = uniform_filter1d(energy, size=10)

    # find peaks
    min_dist = max(1, int(min_gap_ms / hop_ms))
    peaks, _ = find_peaks(
        energy_smooth,
        height=energy_smooth.max() * energy_thresh_pct,
        distance=min_dist,
        prominence=energy_smooth.max() * prominence_pct,
    )

    syllables = []
    for i, pk in enumerate(peaks):
        onset = float(times[pk])
        # estimate end as midpoint to next onset (or +120ms for last)
        if i + 1 < len(peaks):
            end = float(times[peaks[i + 1]]) * 0.5 + onset * 0.5
        else:
            end = min(onset + 0.12, len(signal) / sr)

        syllables.append({
            "syl_idx":     i,
            "method":      "energy",
            "onset_sec":   round(onset, 4),
            "end_sec":     round(end,   4),
            "duration_ms": round((end - onset) * 1000, 1),
            "peak_energy": round(float(energy_smooth[pk]), 5),
            "start_frame": round(onset * 25),
            "end_frame":   round(end   * 25),
        })

    return syllables


# ─────────────────────────────────────────────────────────────────────────────
# METHOD B — TRANSCRIPT PROPORTIONAL MAPPING
# ─────────────────────────────────────────────────────────────────────────────

def _require_pyphen():
    try:
        import pyphen
        return pyphen
    except ImportError:
        sys.exit("[ERROR] pyphen not installed. Run: pip install pyphen")


def split_word_to_syllables(word: str, dic) -> list:
    """Return list of syllable strings for a word using pyphen."""
    clean = word.lower().strip(".,!?;:\"'")
    if not clean:
        return [word]
    return dic.inserted(clean, hyphen="|").split("|")


def map_syllables_from_alignment(word_alignment: list, fps: float = 25.0) -> list:
    """
    Given WhisperX word-level alignment, produce syllable-level timing.

    Each word's duration is distributed across its syllables proportionally
    by character length (equal split is the fallback for 1-syllable words).

    Parameters
    ----------
    word_alignment : list of {"word": str, "start": float, "end": float}
    fps            : frame rate for computing start_frame / end_frame

    Returns list of syllable dicts with full timing and frame indices.
    """
    pyphen = _require_pyphen()
    dic    = pyphen.Pyphen(lang="en_US")

    syllables = []
    syl_idx   = 0

    for word_info in word_alignment:
        word       = word_info["word"]
        w_start    = word_info["start"]
        w_end      = word_info["end"]
        w_dur      = max(w_end - w_start, 0.01)

        syl_parts  = split_word_to_syllables(word, dic)
        n_syl      = len(syl_parts)

        # proportional duration by character length
        char_lens  = [max(len(s), 1) for s in syl_parts]
        total      = sum(char_lens)
        durations  = [w_dur * (c / total) for c in char_lens]

        t = w_start
        for i, (text, dur) in enumerate(zip(syl_parts, durations)):
            s_end = t + dur
            syllables.append({
                "syl_idx":     syl_idx,
                "method":      "transcript",
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


def run_whisperx_alignment(wav_path: str, device: str = "cpu") -> dict:
    """Run WhisperX on a WAV and return word alignment list."""
    try:
        import whisperx
    except ImportError:
        sys.exit("[ERROR] whisperx not installed. Run: pip install whisperx")

    print("[whisperx] Loading model large-v3 ...")
    model  = whisperx.load_model("large-v3", device=device, compute_type="float16")
    audio  = whisperx.load_audio(wav_path)
    result = model.transcribe(audio, batch_size=8)

    print(f"[whisperx] Transcript: {' '.join(s['text'] for s in result['segments'])}")
    print("[whisperx] Running forced alignment ...")

    align_model, metadata = whisperx.load_align_model(
        language_code=result["language"], device=device
    )
    aligned = whisperx.align(
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


# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY
# ─────────────────────────────────────────────────────────────────────────────

def print_syllable_table(syllables: list, title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

    method = syllables[0].get("method", "?") if syllables else "?"

    if method == "energy":
        print(f"  {'#':<4} {'onset_s':>8} {'end_s':>7} {'dur_ms':>8} {'f_start':>8} {'f_end':>7}")
        print(f"  {'-'*52}")
        for s in syllables:
            print(f"  {s['syl_idx']:<4} {s['onset_sec']:>8.3f} {s['end_sec']:>7.3f} "
                  f"{s['duration_ms']:>8.1f} {s['start_frame']:>8} {s['end_frame']:>7}")
    else:
        print(f"  {'#':<4} {'syllable':<10} {'word':<14} {'pos':<6} "
              f"{'start':>7} {'end':>7} {'dur_ms':>8} {'f_start':>8} {'f_end':>7}")
        print(f"  {'-'*75}")
        for s in syllables:
            print(f"  {s['syl_idx']:<4} {s['syllable']:<10} {s['word']:<14} "
                  f"{s['word_pos']:<6} {s['start']:>7.3f} {s['end']:>7.3f} "
                  f"{s['duration_ms']:>8.1f} {s['start_frame']:>8} {s['end_frame']:>7}")

    print(f"\n  Total: {len(syllables)} syllables")


def compare_to_ground_truth(detected: list, ground_truth: list, tolerance_ms=80):
    """Compare energy-detected onsets to planted ground truth."""
    print(f"\n{'='*70}")
    print(f"  Accuracy vs ground truth  (tolerance ±{tolerance_ms}ms)")
    print(f"{'='*70}")

    tol = tolerance_ms / 1000.0
    matched = 0
    for i, gt in enumerate(ground_truth):
        hit = next((d for d in detected if abs(d["onset_sec"] - gt) < tol), None)
        status = f"✓  detected at {hit['onset_sec']:.3f}s" if hit else "✗  MISSED"
        print(f"  GT[{i}] {gt:.3f}s  →  {status}")
        if hit:
            matched += 1

    print(f"\n  Matched: {matched}/{len(ground_truth)}")
    return matched / len(ground_truth) if ground_truth else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Syllable extraction tester")
    parser.add_argument("--audio",      default=None, help="Path to WAV file")
    parser.add_argument("--transcript", default=None, help="Space-separated transcript")
    parser.add_argument("--alignment",  default=None, help="Path to WhisperX alignment.json")
    parser.add_argument("--synth",      action="store_true", help="Generate synthetic test WAV")
    parser.add_argument("--output",     default="syllable_output.json", help="Output JSON path")
    parser.add_argument("--fps",        type=float, default=25.0, help="FPS for frame indices")
    parser.add_argument("--device",     default="cpu", help="cuda or cpu (for WhisperX)")
    args = parser.parse_args()

    ground_truth_onsets = None
    word_alignment      = None
    synth_words         = None

    # ── determine audio source ────────────────────────────────────────────────
    if args.synth:
        synth_words = ["hello", "world", "how", "are", "you", "doing"]
        audio_path  = "/tmp/synth_test.wav"
        signal, sr, ground_truth_onsets, word_alignment = make_synthetic_wav(
            audio_path, synth_words
        )
    elif args.audio:
        audio_path = args.audio
        if not Path(audio_path).exists():
            sys.exit(f"[ERROR] Audio file not found: {audio_path}")
        signal, sr = load_wav(audio_path)
        print(f"Loaded: {audio_path}  ({len(signal)/sr:.2f}s @ {sr}Hz)\n")
    else:
        print("No audio source specified. Using --synth mode.\n")
        synth_words = ["hello", "world", "how", "are", "you", "doing"]
        audio_path  = "/tmp/synth_test.wav"
        signal, sr, ground_truth_onsets, word_alignment = make_synthetic_wav(
            audio_path, synth_words
        )

    # ── METHOD A: energy onset detection ─────────────────────────────────────
    print("[Method A] Running energy onset detection ...")
    energy_syllables = detect_syllables_energy(signal, sr)
    print_syllable_table(energy_syllables, "Method A — Energy onset detection")

    if ground_truth_onsets:
        compare_to_ground_truth(energy_syllables, ground_truth_onsets)

    # ── METHOD B: transcript mapping ──────────────────────────────────────────
    transcript_syllables = []

    # priority: alignment.json > --transcript flag > synth word_alignment
    if args.alignment:
        print(f"\n[Method B] Loading alignment from {args.alignment} ...")
        with open(args.alignment) as f:
            align_data     = json.load(f)
        word_alignment = align_data.get("words", [])
        print(f"  {len(word_alignment)} words loaded from alignment JSON")

    elif args.transcript:
        print(f"\n[Method B] Transcript provided: '{args.transcript}'")
        print("  NOTE: no timing info — using uniform word spacing as placeholder.")
        words    = args.transcript.strip().split()
        duration = len(signal) / sr
        word_dur = duration / len(words) if words else 0
        word_alignment = [
            {"word": w, "start": round(i * word_dur, 3),
             "end": round((i + 1) * word_dur, 3)}
            for i, w in enumerate(words)
        ]

    elif args.audio and not args.synth:
        # real audio with no transcript — try WhisperX
        print(f"\n[Method B] No transcript given. Attempting WhisperX alignment ...")
        try:
            align_data    = run_whisperx_alignment(audio_path, device=args.device)
            word_alignment = align_data.get("words", [])
            print(f"  Transcript: {align_data.get('transcript','')}")
        except SystemExit:
            print("  WhisperX not available — skipping Method B.")
            word_alignment = []

    if word_alignment:
        transcript_syllables = map_syllables_from_alignment(word_alignment, fps=args.fps)
        print_syllable_table(
            transcript_syllables,
            "Method B — Transcript proportional mapping"
        )

    # ── save output ───────────────────────────────────────────────────────────
    output = {
        "audio_file":    audio_path,
        "duration_sec":  round(len(signal) / sr, 3),
        "sample_rate":   sr,
        "fps_reference": args.fps,
        "method_A": {
            "description": "energy onset detection",
            "n_syllables": len(energy_syllables),
            "syllables":   energy_syllables,
        },
        "method_B": {
            "description": "transcript proportional mapping",
            "n_syllables": len(transcript_syllables),
            "word_alignment": word_alignment or [],
            "syllables":   transcript_syllables,
        },
    }

    if ground_truth_onsets:
        output["ground_truth_onsets"] = ground_truth_onsets

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  Output saved to: {args.output}")
    print(f"  Method A: {len(energy_syllables)} syllables detected from audio")
    print(f"  Method B: {len(transcript_syllables)} syllables from transcript")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
