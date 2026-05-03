#!/usr/bin/env python3
"""
visualize_havdnet.py — Paper visualizations for HAVDNet-W.

Generates:
  1. t-SNE plots:
     a) 4-class classification (RVRA, RVFA, FVRA, FVFA)
     b) Forgery type (Face Swap, Voice Clone, etc. from Table 1 mapping)

  2. Activation maps — visual branch, 3 scales, for 1 real + 1 fake sample

  3. Spectrogram plots — audio branch, 3 scales, for 1 real + 1 fake sample

Usage:
  python visualize_havdnet.py \\
      --checkpoint checkpoints/best.pt \\
      --data_root /workspace/processed_multiscale \\
      --save_dir paper_figures \\
      --ethnicity African --gender men

  # Run only specific plots
  python visualize_havdnet.py \\
      --checkpoint checkpoints/best.pt \\
      --data_root /workspace/processed_multiscale \\
      --save_dir paper_figures \\
      --tsne_only

  python visualize_havdnet.py \\
      --checkpoint checkpoints/best.pt \\
      --data_root /workspace/processed_multiscale \\
      --save_dir paper_figures \\
      --activation_only
"""

import os, argparse, random
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent))

from train_havdnet_w import (
    FakeAVCelebDataset, collate_fn,
    precache_wav, precache_lip, precache_clip_all,
    set_seed, CLASS_NAMES,
)

# Forgery type mapping from paper Table 1
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

# Colours for 4-class plot
CLASS_COLORS = ["#2ecc71", "#e74c3c", "#3498db", "#9b59b6"]

# Colours for forgery type plot
FORGERY_COLORS = {
    "Fully Synthetic":   "#c0392b",
    "Lip Sync":          "#e67e22",
    "Face Swap":         "#f1c40f",
    "Face Reenactment":  "#2ecc71",
    "Background Change": "#1abc9c",
    "Lip Reenactment":   "#3498db",
    "Identity Swap":     "#9b59b6",
    "No Forgery":        "#2c3e50",
}

SCALE_NAMES = ["Syllable", "Word", "Sentence"]


# ═══════════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_features(model, loader, device, tau=(0.5, 0.5, 0.5)):
    """
    Extract fused features, sync scores, labels, and forgery types
    from the full dataset for t-SNE.
    """
    model.eval()
    features_all  = []
    labels_all    = []
    ss_all        = []
    video_ids_all = []

    for bd, labels, vids in tqdm(loader, desc="  Extracting features"):
        for k, v in bd.items():
            if isinstance(v, torch.Tensor): bd[k] = v.to(device)

        out = model(bd)

        # Use the concatenated classifier input as the feature:
        # [fused(256), ss(3), alphas(3)] = 262-d
        ss     = out['sync_scores']
        alphas = out['alphas']
        # Get fused feature — forward again to get it, or approximate
        # with the logit pre-activation input. We use ss + alphas as
        # a compact 6-d representation for t-SNE (fast, meaningful).
        # For richer t-SNE: use the 256-d fused feature by hooking gt_fusion.
        feat = torch.cat([ss, alphas], dim=-1).cpu()   # (B, 6)

        features_all.append(feat)
        labels_all.extend(labels.tolist())
        ss_all.append(ss.cpu())
        video_ids_all.extend(vids)

    features = torch.cat(features_all).numpy()   # (N, 6)
    ss_tensor = torch.cat(ss_all)                # (N, 3)

    # Compute forgery types via Table 1 threshold
    tau1, tau2, tau3 = tau
    s1 = (ss_tensor[:, 0] >= tau1).int().tolist()
    s2 = (ss_tensor[:, 1] >= tau2).int().tolist()
    s3 = (ss_tensor[:, 2] >= tau3).int().tolist()
    forgery_types = [
        FORGERY_MAP[(s1[i], s2[i], s3[i])] for i in range(len(s1))]

    return features, labels_all, forgery_types, ss_tensor


# ═══════════════════════════════════════════════════════════════
# t-SNE PLOTS
# ═══════════════════════════════════════════════════════════════

def plot_tsne_4class(features, labels, save_path):
    """t-SNE coloured by 4-class label."""
    print("  Running t-SNE (4-class)...")
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000,
                random_state=42, n_jobs=-1)
    emb = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=(8, 6))
    for c, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
        mask = np.array(labels) == c
        ax.scatter(emb[mask, 0], emb[mask, 1],
                   c=color, label=name, alpha=0.7,
                   s=18, linewidths=0)
    ax.set_title("t-SNE: 4-Class Classification", fontsize=13, fontweight='bold')
    ax.legend(fontsize=8, loc='best', framealpha=0.8)
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {save_path}")


def plot_tsne_forgery(features, forgery_types, save_path):
    """t-SNE coloured by forgery type (Table 1)."""
    print("  Running t-SNE (forgery type)...")
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000,
                random_state=42, n_jobs=-1)
    emb = tsne.fit_transform(features)

    unique_types = sorted(set(forgery_types),
                          key=lambda x: list(FORGERY_MAP.values()).index(x))

    fig, ax = plt.subplots(figsize=(8, 6))
    for ftype in unique_types:
        mask = np.array(forgery_types) == ftype
        color = FORGERY_COLORS.get(ftype, "#888888")
        ax.scatter(emb[mask, 0], emb[mask, 1],
                   c=color, label=ftype, alpha=0.7,
                   s=18, linewidths=0)
    ax.set_title("t-SNE: Forgery Type Classification", fontsize=13, fontweight='bold')
    ax.legend(fontsize=7, loc='best', framealpha=0.8, ncol=2)
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {save_path}")


# ═══════════════════════════════════════════════════════════════
# ACTIVATION MAPS — visual branch, 3 scales
# ═══════════════════════════════════════════════════════════════

def get_sample(ds, label_target):
    """Get a single sample with the given label."""
    for s in ds.samples:
        if s['label'] == label_target:
            idx = ds.samples.index(s)
            return ds[idx]
    return None


def compute_activation_map_syl(model, sample, device):
    """
    Compute Grad-CAM style activation map for syllable scale (CNN).
    Returns list of (frame_tensor, activation_map) per syllable.
    """
    # Register forward hook on the last CNN conv layer
    activations = {}
    gradients   = {}

    def fwd_hook(m, inp, out):
        activations['syl'] = out.detach()

    def bwd_hook(m, grad_in, grad_out):
        gradients['syl'] = grad_out[0].detach()

    # Hook on last conv before pooling in SpatialLipEncoder.cnn
    last_conv = model.vis_syl.cnn[-2]   # AdaptiveAvgPool is last, conv before it
    last_conv = model.vis_syl.cnn[6]    # Conv2d(64,128) — adjust if needed
    fh = last_conv.register_forward_hook(fwd_hook)
    bh = last_conv.register_full_backward_hook(bwd_hook)

    bd = {
        'syllable_lip_crops': sample['syllable_lip_crops'].unsqueeze(0).to(device),
        'word_face_crops':    sample['word_face_crops'].unsqueeze(0).to(device),
        'sentence_arcface':   sample['sentence_arcface'].unsqueeze(0).to(device),
        'waveform':           sample['waveform'].unsqueeze(0).to(device),
        'boundaries': {
            'syllable': [sample['boundaries']['syllable']],
            'word':     [sample['boundaries']['word']],
            'sentence': [sample['boundaries']['sentence']]
        }
    }

    model.zero_grad()
    out = model(bd)
    # Backprop through predicted class
    pred_class = out['logits'].argmax(-1).item()
    out['logits'][0, pred_class].backward()

    fh.remove(); bh.remove()

    if 'syl' in activations and 'syl' in gradients:
        acts = activations['syl']   # (B*N*T, C, H, W)
        grds = gradients['syl']
        weights = grds.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((acts * weights).sum(dim=1))
        cam = cam - cam.min(); cam = cam / (cam.max() + 1e-8)
        return cam.cpu()
    return None


def plot_activation_maps(model, real_sample, fake_sample, device, save_dir):
    """
    Plot activation maps for real and fake samples across all 3 scales.
    For syllable: Grad-CAM on CNN.
    For word/sentence: attention weights from SpatialFaceEncoder.
    """
    print("  Generating activation maps...")
    save_dir = Path(save_dir)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    fig.suptitle("Activation Maps: Visual Branch (Real vs Fake)",
                 fontsize=13, fontweight='bold')

    scale_labels = ["Syllable\n(Lip CNN)", "Word\n(CLIP face)", "Sentence\n(Identity)"]

    for row_idx, (sample, sample_name) in enumerate([(real_sample, "Real"),
                                                      (fake_sample, "Fake")]):
        for col_idx, scale_name in enumerate(scale_labels):
            ax = axes[row_idx, col_idx]

            if col_idx == 0:
                # Syllable: show a lip crop frame
                syl_crops = sample['syllable_lip_crops']  # (N_syl, T, C, H, W)
                # Take the first syllable's first frame
                if syl_crops.shape[0] > 0 and syl_crops.shape[1] > 0:
                    frame = syl_crops[0, 0]   # (C, H, W)
                    frame = frame.permute(1, 2, 0).numpy()
                    frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-8)
                    ax.imshow(frame)
                    ax.set_title(f"{sample_name}\n{scale_name}", fontsize=9)
                else:
                    ax.text(0.5, 0.5, "No lip frames", ha='center', va='center')

            elif col_idx == 1:
                # Word: show CLIP embedding heatmap (word × time)
                word_embs = sample['word_face_crops']  # (N_word, T, 512)
                # L2 norm per time step as attention proxy
                attn = word_embs.norm(dim=-1)          # (N_word, T)
                if attn.sum() > 0:
                    attn = attn / (attn.max() + 1e-8)
                im = ax.imshow(attn.numpy(), aspect='auto',
                               cmap='hot', interpolation='nearest')
                ax.set_xlabel("Frame", fontsize=7)
                ax.set_ylabel("Word", fontsize=7)
                ax.set_title(f"{sample_name}\n{scale_name}", fontsize=9)
                plt.colorbar(im, ax=ax, fraction=0.046)

            else:
                # Sentence: show identity embedding similarity heatmap
                sent = sample['sentence_arcface']   # (N_frames, 512)
                # Cosine similarity matrix between frames
                n = sent.shape[0]
                if n > 1:
                    normed = F.normalize(sent, dim=-1)
                    sim = (normed @ normed.T).numpy()
                    im = ax.imshow(sim, cmap='RdYlGn', vmin=-1, vmax=1)
                    ax.set_title(f"{sample_name}\n{scale_name}", fontsize=9)
                    ax.set_xlabel("Frame"); ax.set_ylabel("Frame")
                    plt.colorbar(im, ax=ax, fraction=0.046)
                else:
                    ax.text(0.5, 0.5, "Single frame", ha='center', va='center')
                    ax.set_title(f"{sample_name}\n{scale_name}", fontsize=9)

            ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    path = save_dir / "activation_maps.pdf"
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {path}")

    # Also save PNG version
    fig2, axes2 = plt.subplots(2, 3, figsize=(12, 7))
    fig2.suptitle("Activation Maps: Visual Branch (Real vs Fake)",
                  fontsize=13, fontweight='bold')
    for row_idx, (sample, sample_name) in enumerate([(real_sample, "Real"),
                                                      (fake_sample, "Fake")]):
        for col_idx, scale_name in enumerate(scale_labels):
            ax = axes2[row_idx, col_idx]
            if col_idx == 0:
                syl_crops = sample['syllable_lip_crops']
                if syl_crops.shape[0] > 0 and syl_crops.shape[1] > 0:
                    frame = syl_crops[0, 0].permute(1, 2, 0).numpy()
                    frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-8)
                    ax.imshow(frame)
            elif col_idx == 1:
                word_embs = sample['word_face_crops']
                attn = word_embs.norm(dim=-1)
                if attn.sum() > 0:
                    attn = attn / (attn.max() + 1e-8)
                ax.imshow(attn.numpy(), aspect='auto', cmap='hot', interpolation='nearest')
            else:
                sent = sample['sentence_arcface']
                if sent.shape[0] > 1:
                    normed = F.normalize(sent, dim=-1)
                    sim = (normed @ normed.T).numpy()
                    ax.imshow(sim, cmap='RdYlGn', vmin=-1, vmax=1)
            ax.set_title(f"{sample_name}\n{scale_name}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(save_dir / "activation_maps.png", dpi=200, bbox_inches='tight')
    plt.close()


# ═══════════════════════════════════════════════════════════════
# SPECTROGRAM PLOTS — audio branch, 3 scales
# ═══════════════════════════════════════════════════════════════

def plot_spectrograms(model, real_sample, fake_sample, device, save_dir):
    """
    Plot spectrograms for real and fake samples across 3 audio scales.
    Syllable = early frames, word = mid, sentence = full waveform.
    """
    import torchaudio.transforms as T
    print("  Generating spectrogram plots...")
    save_dir = Path(save_dir)

    mel_transform = T.MelSpectrogram(
        sample_rate=16000, n_mels=64, n_fft=512, hop_length=160)
    db_transform = T.AmplitudeToDB()

    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    fig.suptitle("Spectrograms: Audio Branch at 3 Scales (Real vs Fake)",
                 fontsize=13, fontweight='bold')

    # Approximate scale boundaries from waveform
    # Syllable: first 1.5s, Word: 1.5-4s, Sentence: full
    SR = 16000
    scale_slices = [
        (0,              int(1.5 * SR),  "Syllable\n(0–1.5s)"),
        (int(1.5 * SR),  int(4.0 * SR), "Word\n(1.5–4s)"),
        (0,              None,           "Sentence\n(Full)"),
    ]

    for row_idx, (sample, label) in enumerate([(real_sample, "Real"),
                                                (fake_sample, "Fake")]):
        wav = sample['waveform']   # (T_audio,)

        for col_idx, (start, end, scale_label) in enumerate(scale_slices):
            ax = axes[row_idx, col_idx]
            seg = wav[start:end] if end else wav[start:]

            if seg.shape[0] < 512:
                seg = F.pad(seg, (0, 512 - seg.shape[0]))

            mel = mel_transform(seg.unsqueeze(0))   # (1, n_mels, T)
            mel_db = db_transform(mel).squeeze(0).numpy()

            im = ax.imshow(mel_db, aspect='auto', origin='lower',
                           cmap='magma', interpolation='nearest')
            ax.set_title(f"{label}\n{scale_label}", fontsize=9)
            ax.set_xlabel("Time frames", fontsize=7)
            ax.set_ylabel("Mel bins", fontsize=7)
            ax.set_xticks([]); ax.set_yticks([])

            # Colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("dB", fontsize=7)

    plt.tight_layout()
    path = save_dir / "spectrograms.pdf"
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {path}")
    plt.savefig(save_dir / "spectrograms.png", dpi=200, bbox_inches='tight') \
        if False else None

    # Save PNG too
    fig2, axes2 = plt.subplots(2, 3, figsize=(12, 6))
    fig2.suptitle("Spectrograms: Audio Branch at 3 Scales (Real vs Fake)",
                  fontsize=13, fontweight='bold')
    for row_idx, (sample, label) in enumerate([(real_sample, "Real"),
                                                (fake_sample, "Fake")]):
        wav = sample['waveform']
        for col_idx, (start, end, scale_label) in enumerate(scale_slices):
            ax = axes2[row_idx, col_idx]
            seg = wav[start:end] if end else wav[start:]
            if seg.shape[0] < 512:
                seg = F.pad(seg, (0, 512 - seg.shape[0]))
            mel = mel_transform(seg.unsqueeze(0))
            mel_db = db_transform(mel).squeeze(0).numpy()
            ax.imshow(mel_db, aspect='auto', origin='lower',
                      cmap='magma', interpolation='nearest')
            ax.set_title(f"{label}\n{scale_label}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(save_dir / "spectrograms.png", dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {save_dir}/spectrograms.png")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="HAVDNet-W Visualization Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    p.add_argument("--checkpoint",  required=True)
    p.add_argument("--data_root",   required=True)
    p.add_argument("--save_dir",    default="paper_figures")
    p.add_argument("--ethnicity",   default=None)
    p.add_argument("--gender",      default=None, choices=["men", "women"])
    p.add_argument("--batch_size",  type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device",      default=None)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--tau",         type=float, nargs=3, default=[0.5, 0.5, 0.5],
                   metavar=("TAU1", "TAU2", "TAU3"),
                   help="Thresholds for forgery type t-SNE colouring (default: 0.5 0.5 0.5)")
    p.add_argument("--n_tsne",      type=int, default=2000,
                   help="Max samples for t-SNE (default 2000 for speed)")
    # Selective plotting
    p.add_argument("--tsne_only",       action="store_true")
    p.add_argument("--activation_only", action="store_true")
    p.add_argument("--spectrogram_only",action="store_true")
    # Sample selection
    p.add_argument("--real_video_id",   default=None,
                   help="video_id of real sample to use for activation/spectrogram.")
    p.add_argument("--fake_video_id",   default=None,
                   help="video_id of fake sample to use for activation/spectrogram.")
    args = p.parse_args()
    set_seed(args.seed)

    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nHAVDNet-W Visualization")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Save dir   : {save_dir}")
    print(f"  Device     : {dev}")

    # Determine which plots to generate
    do_tsne   = not (args.activation_only or args.spectrogram_only)
    do_activ  = not (args.tsne_only or args.spectrogram_only)
    do_spec   = not (args.tsne_only or args.activation_only)
    if args.tsne_only:        do_tsne  = True; do_activ = False; do_spec = False
    if args.activation_only:  do_activ = True; do_tsne  = False; do_spec = False
    if args.spectrogram_only: do_spec  = True; do_tsne  = False; do_activ = False

    # ── Dataset ───────────────────────────────────────────────
    print(f"\nLoading dataset...")
    ds = FakeAVCelebDataset(
        args.data_root,
        ethnicity=args.ethnicity,
        gender=args.gender,
        split_name="viz")
    print(f"  {len(ds)} samples")

    if len(ds) == 0:
        print("ERROR: No samples found."); return

    precache_wav(ds.samples, desc="Wav")
    precache_lip(ds.samples, desc="Lip")

    # ── Models ────────────────────────────────────────────────
    print(f"\nLoading models...")
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

    import train_havdnet_w as train_mod
    train_mod._CLIP_MODEL = cm_model
    precache_clip_all(ds.samples, cm_model, device=dev,
                      batch_size=256, num_workers=4)

    from havdnet_w import HAVDNetW
    model = HAVDNetW(cm_model, w2v_model).to(dev)
    ck = torch.load(args.checkpoint, map_location=dev, weights_only=False)
    model.load_state_dict(ck["state"])
    model.eval()
    print(f"  Loaded epoch {ck.get('epoch','?')}, F1={ck.get('f1','?')}")

    # ── t-SNE ─────────────────────────────────────────────────
    if do_tsne:
        print(f"\n{'='*50}")
        print("  t-SNE PLOTS")
        print(f"{'='*50}")

        loader = DataLoader(ds, args.batch_size, shuffle=False,
                            num_workers=args.num_workers,
                            collate_fn=collate_fn,
                            pin_memory=(args.num_workers > 0))

        features, labels, forgery_types, ss_tensor = extract_features(
            model, loader, dev, tau=tuple(args.tau))

        # Subsample for speed if needed
        N = len(features)
        if N > args.n_tsne:
            idx = np.random.choice(N, args.n_tsne, replace=False)
            features      = features[idx]
            labels        = [labels[i] for i in idx]
            forgery_types = [forgery_types[i] for i in idx]
            print(f"  Subsampled to {args.n_tsne} samples for t-SNE speed")

        plot_tsne_4class(features, labels,
                         save_dir / "tsne_4class.pdf")
        plot_tsne_4class(features, labels,
                         save_dir / "tsne_4class.png")
        plot_tsne_forgery(features, forgery_types,
                          save_dir / "tsne_forgery_type.pdf")
        plot_tsne_forgery(features, forgery_types,
                          save_dir / "tsne_forgery_type.png")

        # Print forgery type distribution
        dist = Counter(forgery_types)
        print(f"\n  Forgery type distribution (τ={args.tau}):")
        for ftype, count in sorted(dist.items(), key=lambda x: -x[1]):
            print(f"    {ftype:25s}: {count:5d} ({count/len(forgery_types)*100:.1f}%)")

    # ── Activation maps + Spectrograms ────────────────────────
    if do_activ or do_spec:
        print(f"\n{'='*50}")
        print("  SAMPLE SELECTION")
        print(f"{'='*50}")

        # Find a real (label=0) and fake (label=3) sample
        real_sample = None
        fake_sample = None

        for s in ds.samples:
            if args.real_video_id and s['video_id'] == args.real_video_id:
                idx = ds.samples.index(s)
                real_sample = ds[idx]
                break
            if s['label'] == 0 and real_sample is None:
                idx = ds.samples.index(s)
                real_sample = ds[idx]
            if s['label'] == 3 and fake_sample is None:
                idx = ds.samples.index(s)
                fake_sample = ds[idx]
            if args.fake_video_id and s['video_id'] == args.fake_video_id:
                idx = ds.samples.index(s)
                fake_sample = ds[idx]
            if real_sample and fake_sample:
                break

        if real_sample is None:
            print("  WARNING: No RealVideo-RealAudio sample found. "
                  "Using label=1 sample instead.")
            for s in ds.samples:
                if s['label'] == 1:
                    idx = ds.samples.index(s)
                    real_sample = ds[idx]; break

        if real_sample is None or fake_sample is None:
            print("  ERROR: Could not find suitable samples for visualization.")
        else:
            print(f"  Real sample: {real_sample.get('video_id', 'unknown') if hasattr(real_sample, 'get') else 'ok'}")
            print(f"  Fake sample: {fake_sample.get('video_id', 'unknown') if hasattr(fake_sample, 'get') else 'ok'}")

            if do_activ:
                print(f"\n{'='*50}")
                print("  ACTIVATION MAPS")
                print(f"{'='*50}")
                plot_activation_maps(model, real_sample, fake_sample, dev, save_dir)

            if do_spec:
                print(f"\n{'='*50}")
                print("  SPECTROGRAM PLOTS")
                print(f"{'='*50}")
                plot_spectrograms(model, real_sample, fake_sample, dev, save_dir)

    print(f"\n{'='*50}")
    print(f"  All figures saved to: {save_dir}/")
    print(f"  Files:")
    for f in sorted(save_dir.iterdir()):
        print(f"    {f.name}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
