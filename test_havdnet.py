#!/usr/bin/env python3
"""
test_havdnet.py — Evaluation script for HAVDNet-W checkpoints.

Modes:
  1. Standard test     — evaluate a checkpoint on a specified demographic
  2. Cross-demographic — train on one group, test on another
  3. Ablation (Table 2)— disable model components at eval to measure contribution
  4. Threshold sweep (Table 3) — sweep τ1,τ2,τ3 for forgery type identification

Usage examples:
  # Test checkpoint on same demographic it was trained on
  python test_havdnet.py \\
      --checkpoint african_men/best.pt \\
      --data_root /workspace/processed_multiscale \\
      --ethnicity African --gender men

  # Cross-demographic: test African men model on East Asian women
  python test_havdnet.py \\
      --checkpoint african_men/best.pt \\
      --data_root /workspace/processed_multiscale \\
      --ethnicity "Asian (East)" --gender women

  # Full dataset test (no ethnicity/gender filter)
  python test_havdnet.py \\
      --checkpoint checkpoints/best.pt \\
      --data_root /workspace/processed_multiscale

  # Ablation study (Table 2) — runs all 5 ablation variants + full model
  python test_havdnet.py \\
      --checkpoint checkpoints/best.pt \\
      --data_root /workspace/processed_multiscale \\
      --ablation

  # Threshold sweep (Table 3) — forgery type identification
  python test_havdnet.py \\
      --checkpoint checkpoints/best.pt \\
      --data_root /workspace/processed_multiscale \\
      --threshold_sweep

  # Save results to file
  python test_havdnet.py \\
      --checkpoint checkpoints/best.pt \\
      --data_root /workspace/processed_multiscale \\
      --save_dir results/african_men
"""

import os, json, argparse, itertools
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, classification_report,
                             confusion_matrix)
from tqdm import tqdm

# ── reuse dataset + model from training script ────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent))

from train_havdnet_w import (
    FakeAVCelebDataset, collate_fn, grouped_split,
    precache_wav, precache_lip, precache_clip_all,
    set_seed, CLASS_NAMES, MAX_SYLLABLES, MAX_WORDS,
)

# ── Forgery type mapping from paper Table 1 ──────────────────────────────────
# (s1=phrase, s2=word, s3=syllable) → forgery type
# L = si < τi,  H = si >= τi
FORGERY_MAP = {
    (0, 0, 0): "Fully Synthetic",
    (0, 0, 1): "Lip Sync",
    (0, 1, 0): "Face Swap",
    (0, 1, 1): "Face Reenactment",
    (1, 0, 0): "Background Change",
    (1, 0, 1): "Lip Reenactment",
    (1, 1, 0): "Identity Swap",
    (1, 1, 1): "No Forgery",
}


# ═══════════════════════════════════════════════════════════════
# ABLATION WRAPPERS
# Wrap the model to disable specific components at eval time.
# Each wrapper zeroes out or bypasses one component, measuring
# how much that component contributes to final performance.
# ═══════════════════════════════════════════════════════════════

class AblationWrapper(nn.Module):
    """Wraps HAVDNetW and selectively disables components."""
    def __init__(self, model, mode="full"):
        super().__init__()
        self.model = model
        self.mode  = mode
        assert mode in [
            "full",
            "visual_only",   # no audio encoder — audio features zeroed
            "audio_only",    # no visual encoder — visual features zeroed
            "phrase_only",   # only sentence/phrase scale
            "word_only",     # only word scale
            "syllable_only", # only syllable scale
        ], f"Unknown ablation mode: {mode}"

    def forward(self, batch):
        if self.mode == "full":
            return self.model(batch)

        # Modify batch in-place for ablation
        b = {k: v for k, v in batch.items()}

        if self.mode == "visual_only":
            # Zero out audio — waveform becomes silence
            b['waveform'] = torch.zeros_like(batch['waveform'])

        elif self.mode == "audio_only":
            # Zero out all visual inputs
            b['syllable_lip_crops'] = torch.zeros_like(batch['syllable_lip_crops'])
            b['word_face_crops']    = torch.zeros_like(batch['word_face_crops'])
            b['sentence_arcface']   = torch.zeros_like(batch['sentence_arcface'])

        elif self.mode in ("phrase_only", "word_only", "syllable_only"):
            # Keep all inputs but zero out sync scores for non-selected scales
            # by zeroing the features at other scales
            if self.mode == "phrase_only":
                # Zero syllable and word visual/audio
                b['syllable_lip_crops'] = torch.zeros_like(batch['syllable_lip_crops'])
                b['word_face_crops']    = torch.zeros_like(batch['word_face_crops'])
            elif self.mode == "word_only":
                b['syllable_lip_crops'] = torch.zeros_like(batch['syllable_lip_crops'])
                b['sentence_arcface']   = torch.zeros_like(batch['sentence_arcface'])
                b['waveform']           = torch.zeros_like(batch['waveform'])
            elif self.mode == "syllable_only":
                b['word_face_crops']  = torch.zeros_like(batch['word_face_crops'])
                b['sentence_arcface'] = torch.zeros_like(batch['sentence_arcface'])

        return self.model(b)


# ═══════════════════════════════════════════════════════════════
# CORE EVALUATION FUNCTION
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_model(model, loader, device, desc="Test"):
    """
    Run model on loader, collect predictions, sync scores, and features.
    Returns dict with metrics + raw outputs for downstream analysis.
    """
    model.eval()
    preds, labels_all = [], []
    sync_scores_all   = []   # (N, 3) for threshold sweep
    alphas_all        = []   # (N, 3) for interpretability
    gt_weights_all    = []   # (N, 3) game-theoretic fusion weights
    video_ids_all     = []

    for bd, labels, vids in tqdm(loader, desc=f"  {desc}", leave=False):
        for k, v in bd.items():
            if isinstance(v, torch.Tensor): bd[k] = v.to(device)
        labels_dev = labels.to(device)

        out = model(bd)

        preds.extend(out['logits'].argmax(-1).cpu().tolist())
        labels_all.extend(labels.tolist())
        sync_scores_all.append(out['sync_scores'].cpu())
        alphas_all.append(out['alphas'].cpu())
        if 'game_info' in out and 'final_weights' in out['game_info']:
            gt_weights_all.append(out['game_info']['final_weights'].cpu())
        video_ids_all.extend(vids)

    ss_tensor = torch.cat(sync_scores_all)   # (N, 3)
    al_tensor = torch.cat(alphas_all)        # (N, 3)
    gw_tensor = torch.cat(gt_weights_all) if gt_weights_all else None

    acc  = accuracy_score(labels_all, preds)
    f1m  = f1_score(labels_all, preds, average="macro",    zero_division=0)
    f1w  = f1_score(labels_all, preds, average="weighted", zero_division=0)
    prec = precision_score(labels_all, preds, average="macro", zero_division=0)
    rec  = recall_score(labels_all, preds, average="macro",    zero_division=0)
    cm   = confusion_matrix(labels_all, preds)

    return {
        "acc":          acc,
        "f1_macro":     f1m,
        "f1_weighted":  f1w,
        "precision":    prec,
        "recall":       rec,
        "confusion":    cm.tolist(),
        "preds":        preds,
        "labels":       labels_all,
        "sync_scores":  ss_tensor,
        "alphas":       al_tensor,
        "gt_weights":   gw_tensor,
        "video_ids":    video_ids_all,
        "mean_alphas":  al_tensor.mean(0).tolist(),
    }


def print_results(results, name="Test"):
    """Pretty-print evaluation results."""
    print(f"\n  {name}")
    print(f"  {'─'*50}")
    print(f"  Accuracy : {results['acc']:.4f}")
    print(f"  F1 macro : {results['f1_macro']:.4f}")
    print(f"  F1 weight: {results['f1_weighted']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall   : {results['recall']:.4f}")
    a = results['mean_alphas']
    print(f"  α (syl/word/sent): [{a[0]:.3f}, {a[1]:.3f}, {a[2]:.3f}]")

    if results.get('gt_weights') is not None:
        gw = results['gt_weights']
        la = torch.tensor(results['labels'])
        print(f"\n  GT fusion weights per class:")
        print(f"  {'':30s}  Syllable   Word   Sentence")
        for c, nm in enumerate(CLASS_NAMES):
            mask = (la == c)
            if mask.any():
                w = gw[mask].mean(0)
                print(f"  {nm:30s}  {w[0]:.3f}      {w[1]:.3f}    {w[2]:.3f}")

    cm = results['confusion']
    print(f"\n  Confusion matrix:")
    print(f"  {'':28s}  " + "  ".join(f"{cn[:8]:8s}" for cn in CLASS_NAMES))
    for i, nm in enumerate(CLASS_NAMES):
        row = "  ".join(f"{cm[i][j]:8d}" for j in range(4))
        print(f"  {nm:28s}  {row}")


# ═══════════════════════════════════════════════════════════════
# ABLATION STUDY (Table 2)
# ═══════════════════════════════════════════════════════════════

def run_ablation(model, loader, device, save_dir):
    """
    Run 6 ablation variants on the same loader.
    Returns dict: mode → metrics.
    """
    modes = [
        ("full",          "Proposed Model (Full)"),
        ("visual_only",   "Visual Branch only"),
        ("audio_only",    "Audio Branch only"),
        ("phrase_only",   "Phrase Level Sync only"),
        ("word_only",     "Word Level Sync only"),
        ("syllable_only", "Syllable Level Sync only"),
    ]

    results_table = {}
    print(f"\n{'='*60}")
    print("  ABLATION STUDY (Table 2)")
    print(f"{'='*60}")

    for mode, label in modes:
        wrapped = AblationWrapper(model, mode=mode)
        res = evaluate_model(wrapped, loader, device, desc=label)
        results_table[label] = res
        print(f"\n  [{label}]")
        print(f"    Acc={res['acc']:.4f}  F1m={res['f1_macro']:.4f}  "
              f"F1w={res['f1_weighted']:.4f}  "
              f"P={res['precision']:.4f}  R={res['recall']:.4f}")

    # Print as Table 2
    print(f"\n  {'─'*70}")
    print(f"  {'Model Variant':<40}  {'Acc':>6}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}")
    print(f"  {'─'*70}")
    for label, res in results_table.items():
        print(f"  {label:<40}  "
              f"{res['acc']*100:6.2f}  "
              f"{res['precision']*100:6.2f}  "
              f"{res['recall']*100:6.2f}  "
              f"{res['f1_macro']*100:6.2f}")

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        out = {k: {m: v for m, v in r.items()
                   if m not in ('preds', 'labels', 'sync_scores',
                                'alphas', 'gt_weights', 'video_ids',
                                'confusion')}
               for k, r in results_table.items()}
        (Path(save_dir) / "ablation_table2.json").write_text(
            json.dumps(out, indent=2))
        print(f"\n  Saved → {save_dir}/ablation_table2.json")

    return results_table


# ═══════════════════════════════════════════════════════════════
# THRESHOLD SWEEP (Table 3)
# ═══════════════════════════════════════════════════════════════

def forgery_type_from_scores(ss, tau1, tau2, tau3):
    """Map (s1,s2,s3) to forgery type string using Table 1."""
    # ss: (N, 3) — [phrase, word, syllable]
    s1 = (ss[:, 0] >= tau1).int()
    s2 = (ss[:, 1] >= tau2).int()
    s3 = (ss[:, 2] >= tau3).int()
    forgery_types = []
    for i in range(len(ss)):
        key = (s1[i].item(), s2[i].item(), s3[i].item())
        forgery_types.append(FORGERY_MAP[key])
    return forgery_types


def run_threshold_sweep(results_base, save_dir):
    """
    Sweep τ1, τ2, τ3 ∈ {0.25, 0.50, 0.75} and compute forgery type accuracy.
    Since ground-truth forgery types aren't in the dataset, we instead report
    the distribution of predicted forgery types and 4-class acc at each τ combo.
    Table 3 reports 4-class accuracy — NOT forgery type accuracy.
    """
    thresholds = [0.25, 0.50, 0.75]
    ss = results_base['sync_scores']   # (N, 3)
    labels = results_base['labels']
    preds  = results_base['preds']

    print(f"\n{'='*60}")
    print("  THRESHOLD SWEEP (Table 3)")
    print(f"{'='*60}")
    print(f"\n  τ1=phrase, τ2=word, τ3=syllable")
    print(f"  Reporting 4-class accuracy and forgery type distribution\n")

    sweep_results = {}
    best_combo = None
    best_acc   = 0.0

    # Header
    print(f"  {'τ1':>5} {'τ2':>5} {'τ3':>5}  {'Acc%':>6}  Dominant forgery type")
    print(f"  {'─'*60}")

    for tau1, tau2, tau3 in itertools.product(thresholds, repeat=3):
        ftypes = forgery_type_from_scores(ss, tau1, tau2, tau3)
        # Count forgery type distribution
        dist = Counter(ftypes)
        dominant = dist.most_common(1)[0][0]

        # Acc is fixed (predictions don't change with threshold)
        # Table 3 shows how thresholds affect the forgery IDENTIFICATION
        # which is the implicit output — we report type distribution here
        no_forgery_pct = dist.get("No Forgery", 0) / len(ftypes) * 100

        key = (tau1, tau2, tau3)
        sweep_results[str(key)] = {
            "tau1": tau1, "tau2": tau2, "tau3": tau3,
            "forgery_distribution": dict(dist),
            "no_forgery_pct": no_forgery_pct,
            "4class_acc": accuracy_score(labels, preds),
        }

        print(f"  {tau1:5.2f} {tau2:5.2f} {tau3:5.2f}  "
              f"{accuracy_score(labels, preds)*100:6.2f}  {dominant}")

        if no_forgery_pct > best_acc:
            best_acc   = no_forgery_pct
            best_combo = key

    print(f"\n  Best τ for real video identification: "
          f"τ1={best_combo[0]}, τ2={best_combo[1]}, τ3={best_combo[2]}")

    # Print compact 3×3×3 table (like paper Table 3) for τ1=0.50 slice
    print(f"\n  Compact table (τ1=0.50, Acc% for 4-class classification):")
    print(f"  {'τ2\\τ3':>8}  {0.25:>8}  {0.50:>8}  {0.75:>8}")
    acc_val = accuracy_score(labels, preds) * 100
    for tau2 in thresholds:
        row = f"  {tau2:>8}"
        for tau3 in thresholds:
            row += f"  {acc_val:8.2f}"
        print(row)

    if save_dir:
        (Path(save_dir) / "threshold_sweep_table3.json").write_text(
            json.dumps(sweep_results, indent=2))
        print(f"\n  Saved → {save_dir}/threshold_sweep_table3.json")

    return sweep_results


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="HAVDNet-W Test Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)

    p.add_argument("--checkpoint",  required=True,
                   help="Path to .pt checkpoint file (e.g. african_men/best.pt)")
    p.add_argument("--data_root",   required=True)
    p.add_argument("--ethnicity",   default=None,
                   help="Test on this ethnicity. None = full dataset.")
    p.add_argument("--gender",      default=None, choices=["men", "women"],
                   help="Test on this gender. None = both.")
    p.add_argument("--save_dir",    default=None,
                   help="Directory to save results JSON and reports.")
    p.add_argument("--batch_size",  type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device",      default=None)
    p.add_argument("--seed",        type=int, default=42)

    # Modes
    p.add_argument("--ablation",        action="store_true",
                   help="Run Table 2 ablation study (6 variants).")
    p.add_argument("--threshold_sweep", action="store_true",
                   help="Run Table 3 threshold sweep for forgery identification.")
    p.add_argument("--full_report",     action="store_true",
                   help="Print sklearn classification_report.")

    args = p.parse_args()
    set_seed(args.seed)

    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nHAVDNet-W Test Script")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Data root  : {args.data_root}")
    print(f"  Ethnicity  : {args.ethnicity or 'all'}")
    print(f"  Gender     : {args.gender or 'all'}")
    print(f"  Device     : {dev}")

    # ── Load dataset ──────────────────────────────────────────
    print(f"\nLoading dataset...")
    ds = FakeAVCelebDataset(
        args.data_root,
        ethnicity=args.ethnicity,
        gender=args.gender,
        split_name="test")
    print(f"  Total samples: {len(ds)}")

    lc = ds.get_label_counts()
    for i, (nm, c) in enumerate(zip(CLASS_NAMES, lc)):
        print(f"    [{i}] {nm:28s}: {c:5d}")

    if len(ds) == 0:
        print("ERROR: No samples found. Check --ethnicity / --gender values.")
        return

    # ── Precache ──────────────────────────────────────────────
    print(f"\nChecking caches...")
    precache_wav(ds.samples, desc="Wav")
    precache_lip(ds.samples, desc="Lip")

    loader = DataLoader(ds, args.batch_size, shuffle=False,
                        num_workers=args.num_workers,
                        collate_fn=collate_fn,
                        pin_memory=(args.num_workers > 0))

    # ── Load CLIP + Wav2Vec2 ──────────────────────────────────
    print(f"\nLoading backbone models...")
    try:
        from load_models import load_clip, load_wav2vec2
        cm_model, _ = load_clip(dev)
        w2v_model   = load_wav2vec2(dev)
    except ImportError:
        import clip
        from transformers import Wav2Vec2Model
        cm_model, _ = clip.load("ViT-B/32", device=dev)
        w2v_model   = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-large-xlsr-53").to(dev)

    # Set CLIP for dataset sentence extraction
    import train_havdnet_w as train_mod
    train_mod._CLIP_MODEL = cm_model

    # CLIP cache check
    precache_clip_all(ds.samples, cm_model, device=dev, batch_size=256, num_workers=4)

    # ── Load HAVDNetW ─────────────────────────────────────────
    from havdnet_w import HAVDNetW
    model = HAVDNetW(cm_model, w2v_model).to(dev)

    ck = torch.load(args.checkpoint, map_location=dev, weights_only=False)
    model.load_state_dict(ck["state"])
    train_epoch = ck.get("epoch", "?")
    train_f1    = ck.get("f1", "?")
    print(f"  Checkpoint: epoch={train_epoch}, train_best_F1={train_f1}")

    tp = sum(x.numel() for x in model.parameters())
    tr = sum(x.numel() for x in model.parameters() if x.requires_grad)
    print(f"  Params: {tp:,} total  {tr:,} trainable")

    # ── Standard evaluation ───────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  STANDARD EVALUATION")
    print(f"{'='*60}")
    results = evaluate_model(model, loader, dev, desc="Evaluating")
    print_results(results, name="Test Results")

    if args.full_report:
        print(f"\n  Classification Report:")
        rpt = classification_report(
            results['labels'], results['preds'],
            target_names=CLASS_NAMES, digits=4, zero_division=0)
        print(rpt)

    # ── Save standard results ─────────────────────────────────
    if args.save_dir:
        sd = Path(args.save_dir)
        sd.mkdir(parents=True, exist_ok=True)
        out = {
            "checkpoint":  args.checkpoint,
            "ethnicity":   args.ethnicity or "all",
            "gender":      args.gender or "all",
            "n_samples":   len(ds),
            "acc":         results['acc'],
            "f1_macro":    results['f1_macro'],
            "f1_weighted": results['f1_weighted'],
            "precision":   results['precision'],
            "recall":      results['recall'],
            "confusion":   results['confusion'],
            "mean_alphas": results['mean_alphas'],
        }
        (sd / "test_results.json").write_text(json.dumps(out, indent=2))
        (sd / "test_report.txt").write_text(
            classification_report(results['labels'], results['preds'],
                                  target_names=CLASS_NAMES, digits=4,
                                  zero_division=0))
        print(f"\n  Saved → {args.save_dir}/")

    # ── Ablation study ────────────────────────────────────────
    if args.ablation:
        abl_results = run_ablation(model, loader, dev, args.save_dir)

    # ── Threshold sweep ───────────────────────────────────────
    if args.threshold_sweep:
        sweep_results = run_threshold_sweep(results, args.save_dir)

    # ── Summary ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"  Acc    : {results['acc']*100:.2f}%")
    print(f"  F1 (m) : {results['f1_macro']*100:.2f}%")
    print(f"  F1 (w) : {results['f1_weighted']*100:.2f}%")
    if args.save_dir:
        print(f"  Output : {args.save_dir}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
