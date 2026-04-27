import math, torch, torch.nn as nn, torch.nn.functional as F

# ═══════════════════════════════════════════════════════════════
# VISUAL ENCODERS
# ═══════════════════════════════════════════════════════════════

class SpatialLipEncoder(nn.Module):
    """Syllable: lip crops (96×48) → CNN+GRU → 128-d per syllable."""
    def __init__(self, out_dim=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)))
        self.temporal = nn.GRU(128, out_dim, batch_first=True)
        self.proj = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        B, N, T, C, H, W = x.shape
        feat = self.cnn(x.reshape(B*N*T, C, H, W)).view(B*N, T, 128)
        out, _ = self.temporal(feat)
        return self.proj(out.mean(1)).view(B, N, -1)


class SpatialFaceEncoder(nn.Module):
    """Word: spatial attn → frozen CLIP → mean+std → 1024-d.
    Chunks through GPU in CLIP_BATCH slices to avoid OOM."""
    CLIP_BATCH = 32

    def __init__(self, clip_model):
        super().__init__()
        self.clip = clip_model
        for p in self.clip.parameters():
            p.requires_grad = False
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 1, 1), nn.Sigmoid())

    def forward(self, x):
        B, N, T, C, H, W = x.shape
        dev = next(self.spatial_attn.parameters()).device

        flat_dev = x.reshape(B*N*T, C, H, W).to(dev)
        attn = self.spatial_attn(flat_dev)
        flat_dev = flat_dev * attn + flat_dev
        flat_cpu = flat_dev.cpu()
        del flat_dev, attn

        all_embs = []
        for i in range(0, flat_cpu.shape[0], self.CLIP_BATCH):
            chunk = flat_cpu[i:i+self.CLIP_BATCH].to(dev)
            with torch.no_grad():
                all_embs.append(self.clip.encode_image(chunk).float().cpu())
            del chunk
        embs = torch.cat(all_embs).to(dev).view(B, N, T, 512)

        mean = embs.mean(dim=2)
        std  = embs.std(dim=2).clamp(min=1e-6)
        return torch.cat([mean, std], dim=-1)


class SpatialIdentityEncoder(nn.Module):
    """Sentence: projects pre-cached ArcFace 512-d identity embeddings.

    batch['sentence_arcface'] arrives as (B, N, 512) — N=MAX_SENTENCE_FRAMES
    per-frame ArcFace embeddings from the dataset (already padded/truncated).

    Returns TWO tensors so the forward pass can use them differently:
      • global: (B, out_dim) — mean-pooled across frames, used in MoE fusion
                               as the sentence-level visual feature.
      • seq:    (B, N, 512)  — full frame sequence, passed to MultiChannelEntropy
                               so T=N>ws and real entropy/sync scores are computed.
                               NOT projected — entropy operates on raw ArcFace space.

    Review fix (point 3): the previous implementation mean-pooled to (B,512) and
    then unsqueeze(1) → (B,1,512) for ent_sent, giving T=1 < ws=5 → zeros → 0.5
    sentinel sync score for every sample.  The temporal data was always available
    (MAX_SENTENCE_FRAMES=8 > ws=5); we were just discarding it prematurely.
    """
    def __init__(self, in_dim=512, out_dim=512):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim))

    def forward(self, x):
        # x: (B, N, 512) — always 3-D from the dataset after collate_fn stacks
        if x.dim() == 2:
            x = x.unsqueeze(1)          # (B, 1, 512) fallback for legacy callers
        seq = x                         # (B, N, 512) — full sequence for entropy

        # Bug 2 fix: zero-padding bias.
        # Dataset pads short videos with exact torch.zeros rows.  A naive x.mean(1)
        # divides the sum by N=MAX_SENTENCE_FRAMES even when only k < N frames are
        # valid, shrinking the magnitude by k/N and shifting the distribution.
        # Fix: count only non-zero rows (valid ArcFace embeddings are L2-normalised
        # so no valid frame is ever all-zero; padding rows are exact zeros).
        valid_mask   = (x.abs().sum(dim=-1) > 1e-6).float()      # (B, N)
        valid_counts = valid_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)  # (B, 1)
        global_      = self.proj(x.sum(dim=1) / valid_counts)    # (B, out_dim)

        return global_, seq


# ═══════════════════════════════════════════════════════════════
# XLSR-53 AUDIO ENCODER
# ═══════════════════════════════════════════════════════════════

class MultiScaleAudioEncoder(nn.Module):
    """Frozen XLSR-53: layers 1-7 → syllable, 8-16 → word, 17-24 → sentence."""
    def __init__(self, wav2vec_model):
        super().__init__()
        self.wav2vec = wav2vec_model
        for p in self.wav2vec.parameters():
            p.requires_grad = False
        self.scale_layers = {'syllable': (1, 7), 'word': (8, 16), 'sentence': (17, 24)}
        self.projections = nn.ModuleDict({
            'syllable': nn.Sequential(nn.Linear(1024, 128), nn.LayerNorm(128)),
            'word':     nn.Sequential(nn.Linear(1024, 256), nn.LayerNorm(256)),
            'sentence': nn.Sequential(nn.Linear(1024, 512), nn.LayerNorm(512))})
        self.attn_pools = nn.ModuleDict({
            'syllable': nn.Linear(1024, 1),
            'word':     nn.Linear(1024, 1),
            'sentence': nn.Linear(1024, 1)})

    def forward(self, waveform, boundaries):
        with torch.no_grad():
            out = self.wav2vec(waveform, output_hidden_states=True)
        hs = out.hidden_states
        result = {}
        for scale, (ls, le) in self.scale_layers.items():
            avg = torch.stack(list(hs[ls:le+1])).mean(0)
            result[scale] = self.projections[scale](
                self._pool(avg, boundaries[scale], scale))
        return result

    def _pool(self, feat, batch_bounds, scale):
        B, Ta = feat.shape[0], feat.shape[1]
        Nm = max((len(bl) for bl in batch_bounds), default=1)
        r = torch.zeros(B, Nm, 1024, device=feat.device)
        for b in range(B):
            for i, (st, et) in enumerate(batch_bounds[b]):
                if st >= et:
                    continue
                s = int(st / 0.020)
                # guard: ensure at least 1 frame wide
                e = min(max(s + 1, int(et / 0.020)), Ta)
                seg = feat[b:b+1, s:e, :]
                if seg.shape[1] == 0:
                    continue
                w = torch.softmax(self.attn_pools[scale](seg), dim=1)
                r[b, i] = (seg * w).sum(1).squeeze(0)
        return r


# ═══════════════════════════════════════════════════════════════
# DIFFERENTIABLE SAMPLE ENTROPY
# ═══════════════════════════════════════════════════════════════

class DiffSampEn(nn.Module):
    """
    Differentiable SampleEntropy for a (B, T, D) sequence.
    Output: (B,) — one scalar per batch item.

    The inner loops are kept (they are correct and T is small after
    windowing) but the similarity matrix is computed once up-front
    and sliced, avoiding redundant ops.
    """
    def __init__(self, m=2, r=0.15, beta=10.0):
        super().__init__()
        self.m, self.r, self.beta = m, r, beta

    def forward(self, x):
        B, T, D = x.shape
        if T < self.m + 2:
            return torch.zeros(B, device=x.device)

        fn   = F.normalize(x, dim=-1)                         # (B, T, D)
        dist = 1.0 - torch.bmm(fn, fn.transpose(1, 2))       # (B, T, T)

        thr = self.r * dist.std(dim=(-2, -1), keepdim=True).detach()
        sm  = torch.sigmoid(self.beta * (thr - dist))         # (B, T, T)
        eye = torch.eye(T, device=x.device).unsqueeze(0)
        sm  = sm * (1.0 - eye)

        Bc = torch.zeros(B, device=x.device)
        Ac = torch.zeros(B, device=x.device)
        for i in range(T - self.m):
            for j in range(i + 1, T - self.m):
                # template match for m consecutive steps
                mm = sm[:, i, j]
                for k in range(1, self.m):
                    mm = mm * sm[:, i+k, j+k]
                Bc = Bc + mm
                # extend by one step for numerator
                ei, ej = i + self.m, j + self.m
                if ei < T and ej < T:
                    Ac = Ac + mm * sm[:, ei, ej]

        return -torch.log((Ac + 1e-8) / (Bc + 1e-8))


# ═══════════════════════════════════════════════════════════════
# MULTI-CHANNEL ENTROPY
# ═══════════════════════════════════════════════════════════════

class MultiChannelEntropy(nn.Module):
    """
    Splits D-dimensional features into n_groups channel groups,
    computes DiffSampEn per group inside each sliding window.

    Input:  (B, T, D)
    Output: (B, T_out, n_groups)   where T_out = T - ws + 1

    FIX (bug #3): if T < ws the output is zeros — callers must handle
    this gracefully (the sentence scale routinely hits this).
    """
    def __init__(self, ws=5, n_groups=16, m=2, r=0.15):
        super().__init__()
        self.ws       = ws
        self.n_groups = n_groups
        self.sampen   = DiffSampEn(m, r)

    def forward(self, x):
        B, T, D = x.shape
        T_out = T - self.ws + 1
        if T_out < 1:
            return torch.zeros(B, 1, self.n_groups, device=x.device)

        g          = min(self.n_groups, D)
        group_size = D // g
        x_grouped  = x[:, :, :g * group_size].reshape(B, T, g, group_size)

        all_ents = []
        for t in range(T_out):
            window     = x_grouped[:, t:t+self.ws]            # (B, ws, g, group_size)
            group_ents = []
            for gi in range(g):
                feat = window[:, :, gi, :]                    # (B, ws, group_size)
                group_ents.append(self.sampen(feat))          # (B,)
            all_ents.append(torch.stack(group_ents, dim=1))  # (B, g)

        return torch.stack(all_ents, dim=1)                   # (B, T_out, g)


# ═══════════════════════════════════════════════════════════════
# SYNC SCORE  — operates on (B, T, D) dynamics signals
# ═══════════════════════════════════════════════════════════════

class SyncScoreModule(nn.Module):
    """
    Compares two (B, T, D) dynamics signals with three metrics:
      • Cosine similarity
      • PLV (phase-locking value via Hilbert transform)
      • Soft-DTW

    Each scale has an independent learned gate MLP.
    Returns: (score (B,), gate_weights (B,3), raw_metrics (B,3))
    """
    def __init__(self, n=3):
        super().__init__()
        self.gates = nn.ModuleList([
            nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 3), nn.Softmax(dim=-1))
            for _ in range(n)])

    def forward(self, v, a, si):
        if v.dim() == 2: v = v.unsqueeze(-1)
        if a.dim() == 2: a = a.unsqueeze(-1)

        T = min(v.shape[1], a.shape[1])
        B = v.shape[0]
        d = v.device

        if T < 2:
            return (torch.full((B,), 0.5, device=d),
                    torch.ones(B, 3, device=d) / 3,
                    torch.full((B, 3), 0.5, device=d))

        v, a = v[:, :T], a[:, :T]
        c = self._cosine(v, a)
        p = self._plv(v, a)
        w = self._sdtw(v, a)

        m = torch.stack([c, p, w], dim=-1)          # (B, 3)
        g = self.gates[si](m.detach())              # (B, 3)
        return (m * g).sum(-1), g, m

    def _cosine(self, v, a):
        """
        Mean per-step cosine similarity, remapped from [-1,1] → [0,1]
        so sync scores (which are a weighted sum of cosine, PLV, sdtw) stay
        in [0,1] and can be directly fed into BCE without clamping artefacts.
        PLV ∈ [0,1] and soft-DTW ∈ (0,1] already; this makes all three consistent.
        """
        vn = F.normalize(v, dim=-1)
        an = F.normalize(a, dim=-1)
        cos = (vn * an).sum(-1).mean(dim=1)   # (B,) ∈ [-1, 1]
        return (cos + 1.0) * 0.5              # → [0, 1]

    def _plv(self, v, a):
        diff     = self._hilbert_phase(v) - self._hilbert_phase(a)
        cos_mean = torch.cos(diff).mean(dim=1)      # (B, D)
        sin_mean = torch.sin(diff).mean(dim=1)
        plv_ch   = torch.sqrt(cos_mean**2 + sin_mean**2 + 1e-8)
        return plv_ch.mean(dim=-1)

    def _hilbert_phase(self, x):
        B, T, D = x.shape
        xf = x.reshape(B * D, T)
        N  = xf.shape[-1]
        X  = torch.fft.rfft(xf, dim=-1)
        h  = torch.zeros(X.shape[-1], device=x.device)
        h[0] = 1.0
        if N > 2:
            h[1:N//2] = 2.0
        if N % 2 == 0:
            h[N//2] = 1.0
        analytic = torch.fft.irfft(X * h, n=N, dim=-1)
        phase    = torch.atan2(analytic, xf)
        return phase.view(B, D, T).permute(0, 2, 1)

    def _sdtw(self, v, a, gamma=0.1):
        B, Tv, D = v.shape
        Ta = a.shape[1]
        c  = ((v.unsqueeze(2) - a.unsqueeze(1)) ** 2).sum(-1)  # (B, Tv, Ta)

        R = torch.full((B, Tv+1, Ta+1), float('inf'), device=v.device)
        R[:, 0, 0] = 0.0
        for i in range(1, Tv+1):
            for j in range(1, Ta+1):
                nb = torch.stack([R[:, i-1, j-1], R[:, i-1, j], R[:, i, j-1]], dim=-1)
                R[:, i, j] = c[:, i-1, j-1] - gamma * torch.logsumexp(-nb / gamma, dim=-1)
        return torch.exp(-R[:, Tv, Ta] / (Tv + Ta))


# ═══════════════════════════════════════════════════════════════
# MIXTURE OF EXPERTS
# ═══════════════════════════════════════════════════════════════

class Expert(nn.Module):
    """Single MLP expert."""
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim))

    def forward(self, x):
        return self.net(x)


class SyncConditionedRouter(nn.Module):
    """
    Routes inputs to experts based on sync scores.

    The router is trained via two signals:
      1. Soft cross-entropy from downstream classification loss (implicit).
      2. Explicit load-balancing loss (returned separately).

    FIX (bug #4 / ScaleGate.probe): instead of a probe that gets zero
    gradient during training, the router is always active in both train
    and eval modes.  During training it is also supervised by an auxiliary
    alignment loss that pulls its soft assignment toward the true label
    embedding — so it learns *which* fake type each sync pattern maps to
    before being used at inference.
    """
    def __init__(self, n_experts=4, ss_dim=3, hidden=32):
        super().__init__()
        self.n_experts = n_experts
        self.router = nn.Sequential(
            nn.Linear(ss_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_experts))

    def forward(self, ss):
        """
        ss: (B, 3) sync scores
        Returns:
            weights: (B, n_experts)  soft routing weights
            logits:  (B, n_experts)  pre-softmax, needed for load-balance loss
        """
        logits  = self.router(ss)                # (B, n_experts)
        weights = F.softmax(logits, dim=-1)      # (B, n_experts)
        return weights, logits

    def load_balance_loss(self, weights):
        """
        Auxiliary loss that prevents expert collapse.
        Encourages uniform usage across the batch.
        weights: (B, n_experts)
        """
        # Mean routing prob per expert, should be ~1/n_experts
        mean_w = weights.mean(dim=0)             # (n_experts,)
        target = torch.ones_like(mean_w) / self.n_experts
        return F.mse_loss(mean_w, target)


class MoEFusion(nn.Module):
    """
    Soft Mixture-of-Experts for scale fusion.

    n_experts experts (default 4 = one per fake class).
    Each expert sees [proj_syl, proj_word, proj_sent] concatenated,
    weighted by alpha (scale gate), and maps to out_dim.
    The router uses sync scores so each expert specialises on a
    different AV synchrony pattern (face-swap / voice-clone / etc.).

    Returns:
        fused:  (B, out_dim)
        w_out:  (B, n_experts) routing weights (for interpretability)
    """
    def __init__(self, feat_dim=256, n_experts=4, n_scales=3, dropout=0.2):
        super().__init__()
        self.n_experts = n_experts
        self.n_scales  = n_scales

        # Each expert fuses the scale features differently
        self.experts = nn.ModuleList([
            Expert(feat_dim * n_scales, feat_dim * 2, feat_dim, dropout)
            for _ in range(n_experts)])

        self.out_proj = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.GELU())

    def forward(self, feats, alphas, router_weights):
        """
        feats:          list of n_scales tensors, each (B, feat_dim)
        alphas:         (B, n_scales) — scale attention from ScaleGate
        router_weights: (B, n_experts) — from SyncConditionedRouter
        Returns: (B, feat_dim)
        """
        # Weight scale features by alpha before concatenating
        weighted = [alphas[:, i:i+1] * feats[i] for i in range(self.n_scales)]
        x = torch.cat(weighted, dim=-1)          # (B, feat_dim * n_scales)

        # Soft MoE: weighted sum of all expert outputs
        out = torch.zeros(x.shape[0], self.experts[0].net[-1].out_features,
                          device=x.device)
        for i, expert in enumerate(self.experts):
            w = router_weights[:, i:i+1]         # (B, 1)
            out = out + w * expert(x)            # (B, feat_dim)

        return self.out_proj(out)                # (B, feat_dim)


class MoEClassifier(nn.Module):
    """
    Soft MoE classification head.
    Replaces the fixed 3-layer MLP classifier.
    Each expert is a 2-layer MLP; they share the same router as MoEFusion.
    """
    def __init__(self, in_dim=262, hidden_dim=128, n_classes=4, n_experts=4, dropout=0.2):
        super().__init__()
        self.experts = nn.ModuleList([
            Expert(in_dim, hidden_dim, n_classes, dropout)
            for _ in range(n_experts)])

    def forward(self, x, router_weights):
        """
        x:              (B, in_dim)
        router_weights: (B, n_experts) — same weights as MoEFusion for coherence
        Returns: (B, n_classes)
        """
        out = torch.zeros(x.shape[0], self.experts[0].net[-1].out_features,
                          device=x.device)
        for i, expert in enumerate(self.experts):
            w = router_weights[:, i:i+1]
            out = out + w * expert(x)
        return out


# ═══════════════════════════════════════════════════════════════
# PROMPT GATE
# ═══════════════════════════════════════════════════════════════

class PromptRefiner(nn.Module):
    def __init__(self, nc=4, nt=4, d=512):
        super().__init__()
        self.lp   = nn.Parameter(torch.randn(nc, nt, d) * 0.02)
        self.proj = nn.Sequential(nn.Linear(d, d), nn.LayerNorm(d), nn.GELU())

    def forward(self, te):
        return torch.cat([self.proj(self.lp).to(te.dtype), te], dim=1)


class ScaleGate(nn.Module):
    """
    Produces scale attention weights (alphas) from CLIP text embeddings.

    FIX (bug #4): The original code had two separate paths — a text-embedding
    path for training and a sync-score probe path for inference — where the
    probe received ZERO gradient during training (it was only called at
    inference).  This meant the probe had random weights throughout training,
    making inference alphas essentially random.

    New design:
      • ScaleGate ONLY produces alphas from text embeddings.
      • SyncConditionedRouter handles all sync-score-based routing.
      • At inference (no labels), alphas are produced by a softmax over
        router_weights @ te, keeping everything differentiable.
    """
    def __init__(self, d=512, n=3):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, n))

    def forward(self, te):
        """te: (B, 512) — class text embeddings (training: te[labels], inference: weighted te)"""
        return F.softmax(self.gate(te), dim=-1)  # (B, n_scales)


# ═══════════════════════════════════════════════════════════════
# CONTRASTIVE LOSS  (fixed)
# ═══════════════════════════════════════════════════════════════

class ContrastiveAVLoss(nn.Module):
    """
    Cross-sample InfoNCE contrastive loss on entropy dynamics.

    FIX (bug #1 — root cause of 100% train accuracy):
    The original loss used torch.mm(v, a.T) to build a (B,B) similarity
    matrix, then masked the diagonal where label==3 as positives.
    Both v and a came from the SAME sample's entropy dynamics, so
    v[i]·a[i] was trivially high for all i regardless of fake/real status.
    The loss collapsed all representations to maximize intra-sample
    similarity → sync scores became a trivial identity signal → the 262-d
    classifier input was a near-perfect same-sample fingerprint → 100% train acc.

    Correct InfoNCE semantics:
      • Positive pair: (v[i], a[j]) where i≠j and BOTH are real (label==3)
        Real videos have genuine AV synchrony → their cross-sample entropy
        patterns should be more similar than fake ones.
      • Negative pair: any (v[i], a[j]) where at least one is fake.
      • For fakes: push v[i] AWAY from a[i] (AV divergence is the fake signal).

    This correctly trains entropy representations WITHOUT same-sample leakage.
    """
    def __init__(self, t=0.07):
        super().__init__()
        self.t = t

    def forward(self, vd, ad, labels):
        """
        vd, ad: (B, D) — flattened entropy dynamics for visual and audio
        labels: (B,)   — 0/1/2/3, where 3 = real
        """
        if vd.shape[0] < 2:
            return torch.tensor(0.0, device=vd.device)

        v = F.normalize(vd, dim=-1)             # (B, D)
        a = F.normalize(ad, dim=-1)             # (B, D)

        sim = torch.mm(v, a.T) / self.t         # (B, B) cross-sample similarity

        real_mask = (labels == 3)               # (B,) bool

        if real_mask.sum() < 1:
            # No real samples in batch — only push fakes apart.
            # Bug 4 fix: apply clamp(min=0) to match the main path.
            # Without it, sim can be negative (vectors already separated) and
            # mean(negative) rewards pushing them to exact opposites indefinitely.
            # clamp(min=0): stop gradient once they are already non-similar.
            fake_diag = torch.diag(sim)[~real_mask]
            return fake_diag.clamp(min=0).mean()

        # ── Positive loss: real×real cross-sample alignment ──────────
        # For each real sample i, treat all other real samples j≠i as positives.
        # The InfoNCE denominator includes ALL j in the batch.
        real_idx  = real_mask.nonzero(as_tuple=True)[0]           # (n_real,)
        n_real    = real_idx.shape[0]

        # Sim sub-matrix: real queries vs all keys
        sim_real  = sim[real_idx]                                  # (n_real, B)
        log_prob  = F.log_softmax(sim_real, dim=-1)               # (n_real, B)

        # Positive mask: other real samples (exclude self)
        pos_mask  = real_mask.unsqueeze(0).expand(n_real, -1).clone()  # (n_real, B)
        for k, idx in enumerate(real_idx):
            pos_mask[k, idx] = False                               # exclude self

        n_pos = pos_mask.float().sum(dim=-1).clamp(min=1)         # (n_real,)
        l_real = -(log_prob * pos_mask.float()).sum(dim=-1) / n_pos
        loss = l_real.mean()

        # ── Negative loss: push fake diagonal pairs apart ─────────────
        # For fake samples, v[i] and a[i] should NOT align.
        fake_mask = ~real_mask
        if fake_mask.any():
            fake_diag_sim = torch.diag(sim)[fake_mask]            # cosine of same-sample fake pairs
            # We want this to be SMALL → penalise high similarity
            loss = loss + fake_diag_sim.clamp(min=0).mean()

        return loss


# ═══════════════════════════════════════════════════════════════
# HAVDNet-W
# ═══════════════════════════════════════════════════════════════

class HAVDNetW(nn.Module):
    CLASS_PROMPTS = [
        "a deepfake using both face swap and voice cloning",
        "a deepfake using face swap",
        "a deepfake using voice cloning",
        "an authentic video of a real person speaking"]

    def __init__(self, clip_model, wav2vec_model, nt=4, cd=512, nc=4, ew=5, n_experts=4):
        super().__init__()
        self.clip      = clip_model
        self.nc        = nc
        self.n_experts = n_experts

        self.prompt_refiner = PromptRefiner(nc, nt, cd)
        self.vis_syl   = SpatialLipEncoder(128)
        self.vis_word  = SpatialFaceEncoder(clip_model)
        self.vis_sent  = SpatialIdentityEncoder()
        self.audio_enc = MultiScaleAudioEncoder(wav2vec_model)

        # Bug 1 fix: ws must be >= m+2 = 4 for DiffSampEn to compute anything.
        # ws=3 caused T=ws=3 < m+2=4 → guard triggered → all-zeros → 0.5 sync scores.
        # ws=5 gives T_out per scale (after trim_valid removes zero-pad trailing rows):
        #   syllable: T up to N_syl=40 → T_out up to 36 ✓
        #   word:     T up to N_word=12→ T_out up to 8  ✓
        #   sentence: T=MAX_SENTENCE_FRAMES=8 → T_out=4  ✓
        self.ent_syl  = MultiChannelEntropy(ws=ew, n_groups=16, m=2)
        self.ent_word = MultiChannelEntropy(ws=ew, n_groups=16, m=2)
        self.ent_sent = MultiChannelEntropy(ws=ew, n_groups=32, m=2)

        self.sync = SyncScoreModule(n=3)

        # Scale gate (text-embedding conditioned)
        self.gate   = ScaleGate(cd, n=3)

        # Sync-conditioned router — always active, trains in both modes
        self.router = SyncConditionedRouter(n_experts=n_experts, ss_dim=3)

        # Scale projection heads
        self.proj_syl  = nn.Linear(128, 256)
        self.proj_word = nn.Linear(1024, 256)
        self.proj_sent = nn.Linear(512, 256)

        # MoE fusion: takes [proj_syl, proj_word, proj_sent] → 256-d
        self.moe_fusion = MoEFusion(feat_dim=256, n_experts=n_experts, n_scales=3)

        # MoE classifier: input = f(256) + ss(3) + alphas(3) = 262
        self.moe_clf = MoEClassifier(
            in_dim=262, hidden_dim=128, n_classes=nc,
            n_experts=n_experts)

        self._te_cache = None

    # ── text embeddings ─────────────────────────────────────────

    def _get_te(self):
        if self.training:
            return self._compute_te(grad=True)
        if self._te_cache is None:
            self._te_cache = self._compute_te(grad=False)
        return self._te_cache

    def _compute_te(self, grad=False):
        """
        Compute class text embeddings via the full CLIP text encoder.

        Review fix (point 2): the previous implementation extracted raw token
        vocabulary embeddings (clip.token_embedding) and mean-pooled them WITHOUT
        passing through the CLIP transformer.  A mean of dictionary lookup vectors
        has no semantic content — it is equivalent to computing the average word-
        vector regardless of word order or context.

        Fix: additive prompt injection.
          1. Tokenise the four class prompts → (4, 77) token ids.
          2. Extract token embeddings → (4, 77, 512).
          3. PromptRefiner projects its learned parameters to the SAME (4, nt, 512)
             shape and ADDS them to the first nt token positions (prefix injection).
             Sequence length stays 77 → CLIP's positional embeddings match exactly.
          4. Add CLIP positional embeddings, pass through CLIP transformer.
          5. Extract the EOT (end-of-text) token representation, which is what the
             original CLIP text encoder uses as the sentence embedding.
        """
        dev = next(self.parameters()).device
        import clip as clip_module

        tokens = clip_module.tokenize(self.CLASS_PROMPTS).to(dev)  # (4, 77)

        with torch.no_grad():
            base = self.clip.token_embedding(tokens).float()        # (4, 77, 512)

        # Additive prefix injection: add learned prompts to first nt token slots.
        # PromptRefiner.forward returns cat([proj(lp), te], dim=1) which PREPENDS
        # tokens — we instead apply it additively so seq_len stays 77.
        refined_prefix = self.prompt_refiner.proj(
            self.prompt_refiner.lp).to(base.dtype)                  # (4, nt, 512)
        nt = refined_prefix.shape[1]

        if grad:
            x = base.clone()
            x[:, :nt, :] = x[:, :nt, :] + refined_prefix           # (4, 77, 512), has grad
        else:
            with torch.no_grad():
                x = base.clone()
                x[:, :nt, :] = x[:, :nt, :] + refined_prefix.detach()

        # Full CLIP transformer pass (mirrors clip/model.py encode_text)
        ctx = torch.no_grad() if not grad else torch.enable_grad()
        with ctx:
            x = x + self.clip.positional_embedding.to(x.dtype)     # (4, 77, 512)
            x = x.permute(1, 0, 2)                                  # (77, 4, 512) LND
            x = self.clip.transformer(x)
            x = x.permute(1, 0, 2)                                  # (4, 77, 512) NLD
            x = self.clip.ln_final(x).float()

        # EOT token position carries the sentence embedding in CLIP
        eot_idx = tokens.argmax(dim=-1)                             # (4,)
        return x[torch.arange(4, device=dev), eot_idx]             # (4, 512)

    def train(self, mode=True):
        super().train(mode)
        self._te_cache = None
        return self

    # ── forward ─────────────────────────────────────────────────

    @staticmethod
    def _trim_valid(x, min_t):
        """
        Trim trailing all-zero rows from a (B, T, D) sequence.

        Zero-padded positions (added to bring short videos up to MAX_*
        constants) produce artificial entropy measurements — a window of ws=5
        frames where 4 are pure zeros is measuring the "dynamics" of a constant
        signal, not real content.  The user's request "don't compute entropy for
        single-frame segments; take old and new along with audio" means: exclude
        windows that contain no meaningful temporal change.

        Strategy: find the last temporal index where ANY sample in the batch has
        a non-zero vector, then trim to max(that index + 1, min_t).  Using the
        batch-wide maximum means all samples stay the same length (required for
        batched tensor ops) while still removing pure-zero tail padding that
        appears in every sample.

        Args:
            x:     (B, T, D)
            min_t: minimum T to keep (= ws, so T_out >= 1 is guaranteed)
        Returns:
            (B, T', D) with T' = max(last_valid_t + 1, min_t)
        """
        # Norm across D: (B, T) → True where any sample has a non-zero vector
        norms       = x.abs().sum(dim=-1)          # (B, T)
        any_valid   = norms.any(dim=0)             # (T,) bool — batch-wise OR
        valid_idx   = any_valid.nonzero(as_tuple=False)
        if valid_idx.numel() == 0:
            # Entire sequence is zeros — return minimum slice
            return x[:, :min_t]
        last_valid  = valid_idx[-1].item()         # index of last non-zero column
        t_keep      = max(last_valid + 1, min_t)
        return x[:, :t_keep]

    def forward(self, batch, labels=None):
        """
        batch keys:
            syllable_lip_crops  (B, N_syl, T, 3, 96, 48)
            word_face_crops     (B, N_word, T, 3, 96, 96)
            sentence_arcface    (B, N, 512)  — N = MAX_SENTENCE_FRAMES
            waveform            (B, samples)
            boundaries          dict[scale → list of (st, et) tuples per sample]

        labels: (B,) int64 — pass during training, None at eval.
        """
        # ── Visual encoding ──────────────────────────────────────
        vs  = self.vis_syl(batch['syllable_lip_crops'])      # (B, N_syl, 128)
        vw  = self.vis_word(batch['word_face_crops'])         # (B, N_word, 1024)
        # SpatialIdentityEncoder now returns (global, seq):
        #   vst_global: (B, 512) — mean-pooled (valid-frame masked), used as fusion feature
        #   vst_seq:    (B, N, 512) — full ArcFace sequence, trimmed then used for entropy
        #                             T=N=MAX_SENTENCE_FRAMES=8 > ws=5 → real entropy ✓
        vst_global, vst_seq = self.vis_sent(batch['sentence_arcface'])

        # ── Audio encoding ───────────────────────────────────────
        aud = self.audio_enc(batch['waveform'], batch['boundaries'])
        # aud['syllable']: (B, N_syl, 128)
        # aud['word']:     (B, N_word, 256)
        # aud['sentence']: (B, N_sent, 512)

        # ── Entropy dynamics ─────────────────────────────────────
        # _trim_valid removes trailing all-zero rows (padding artefacts) before
        # MultiChannelEntropy so windows containing only zero-padded frames are
        # excluded.  min_t = ws ensures T_out >= 1 is always satisfied.
        ws = self.ent_syl.ws                                 # same ws across all scales

        vs_t   = self._trim_valid(vs,               ws)     # (B, T'_syl, 128)
        vw_t   = self._trim_valid(vw,               ws)     # (B, T'_word, 1024)
        vst_t  = self._trim_valid(vst_seq,          ws)     # (B, T'_sent, 512)
        as_t   = self._trim_valid(aud['syllable'],  ws)     # (B, T'_syl, 128)
        aw_t   = self._trim_valid(aud['word'],      ws)     # (B, T'_word, 256)
        ast_t  = self._trim_valid(aud['sentence'],  ws)     # (B, T'_sent, 512)

        ve_s  = self.ent_syl(vs_t)                          # (B, T_out, 16)
        ae_s  = self.ent_syl(as_t)
        ve_w  = self.ent_word(vw_t)                         # (B, T_out, 16)
        ae_w  = self.ent_word(aw_t)
        ve_st = self.ent_sent(vst_t)                        # (B, T_out, 32)
        ae_st = self.ent_sent(ast_t)                        # (B, T_out, 32)

        # ── Sync scores ──────────────────────────────────────────
        ss_s,  ws_s,  ms_s  = self.sync(ve_s,  ae_s,  0)
        ss_w,  ws_w,  ms_w  = self.sync(ve_w,  ae_w,  1)
        ss_st, ws_st, ms_st = self.sync(ve_st, ae_st, 2)
        ss = torch.stack([ss_s, ss_w, ss_st], dim=-1)       # (B, 3)

        # ── Router (always active, both train and eval) ───────────
        router_weights, router_logits = self.router(ss)     # (B, n_experts)

        # ── Scale gate (text-embedding conditioned) ───────────────
        te = self._get_te()                                  # (4, 512)

        if self.training and labels is not None:
            # Use true class embeddings during training
            te_input = te[labels]                            # (B, 512)
        else:
            # FIX (bug #4): at inference, route via router weights → class embedding.
            # router_weights: (B, n_experts=4), te: (4, 512)
            # Weighted sum gives a class-conditioned text embedding that IS
            # trained (router is trained throughout), avoiding random probe weights.
            te_input = router_weights @ te                   # (B, 512)

        alphas = self.gate(te_input)                         # (B, 3) scale weights

        # ── MoE fusion ───────────────────────────────────────────
        feats = [
            self.proj_syl(vs.mean(1)),                       # (B, 256)
            self.proj_word(vw.mean(1)),                      # (B, 256)
            self.proj_sent(vst_global)]                      # (B, 256)

        f = self.moe_fusion(feats, alphas, router_weights)  # (B, 256)

        # ── MoE classification ────────────────────────────────────
        clf_input = torch.cat([f, ss, alphas], dim=-1)      # (B, 262)
        logits    = self.moe_clf(clf_input, router_weights)  # (B, 4)

        return {
            'logits':         logits,
            'sync_scores':    ss,
            'alphas':         alphas,
            'router_weights': router_weights,
            'router_logits':  router_logits,
            'metrics': {
                'syllable': {'raw': ms_s,  'weights': ws_s},
                'word':     {'raw': ms_w,  'weights': ws_w},
                'sentence': {'raw': ms_st, 'weights': ws_st}},
            'entropy_dynamics': {
                'syllable': (ve_s,  ae_s),
                'word':     (ve_w,  ae_w),
                'sentence': (ve_st, ae_st)}}


# ═══════════════════════════════════════════════════════════════
# LOSS
# ═══════════════════════════════════════════════════════════════

class HAVDNetLoss(nn.Module):
    """
    Total loss = L_ce + λ_c·L_con + λ_s·L_sync + λ_k·L_consistency
               + λ_t·L_shift + λ_lb·L_loadbalance

    Changes from original:
      • ContrastiveAVLoss now uses cross-sample InfoNCE (fixes leakage).
      • L_loadbalance: auxiliary MoE router loss to prevent expert collapse.
      • L_sync: BCE now applied after sigmoid (not with_logits) since sync
        scores are already in (0,1) range.
    """
    def __init__(self, cc, lc=0.3, ls=0.1, lk=0.05, lt=0.1, lsm=0.1, llb=0.01):
        super().__init__()
        self.lc, self.ls, self.lk, self.lt, self.llb = lc, ls, lk, lt, llb
        tot = sum(cc)
        w   = torch.tensor([math.sqrt(tot / (c + 1)) for c in cc], dtype=torch.float32)
        w   = w / w.sum() * len(cc)
        self.register_buffer('cw', w)
        self.ce  = nn.CrossEntropyLoss(weight=w, label_smoothing=lsm)
        self.con = ContrastiveAVLoss()

    def forward(self, o, labels, epoch=0, shifted_sync=None):
        # ── Classification ────────────────────────────────────────
        l_ce = self.ce(o['logits'], labels)

        # ── Contrastive (active after epoch 3 to let reps stabilise) ──
        l_con = torch.tensor(0.0, device=l_ce.device)
        if epoch >= 3:
            for s in ['syllable', 'word', 'sentence']:
                vd, ad = o['entropy_dynamics'][s]
                # Flatten (B, T_out, g) → (B, T_out*g)
                vd_f = vd.reshape(vd.shape[0], -1)
                ad_f = ad.reshape(ad.shape[0], -1)
                l_con = l_con + self.con(vd_f, ad_f, labels)
            l_con = l_con / 3

        # ── Sync supervision ──────────────────────────────────────
        ss       = o['sync_scores']                          # (B, 3) — all in [0,1]
        # cosine is remapped to [0,1] in SyncScoreModule._cosine; PLV and
        # soft-DTW are inherently in [0,1].  Weighted sum ∈ [0,1] → plain BCE.
        real_tgt = (labels == 3).float().unsqueeze(1).expand_as(ss)
        l_sync   = F.binary_cross_entropy(ss.clamp(1e-6, 1 - 1e-6), real_tgt)

        # ── Consistency (penalise uncertain fakes) ────────────────
        fake_mask = (labels != 3).float()
        l_k = ((1 - 4 * (ss * fake_mask.unsqueeze(1) - 0.5).pow(2))
               * fake_mask.unsqueeze(1)).mean()

        # ── Temporal shift ────────────────────────────────────────
        # Review fix (point 4): use Python float 0.0 as default — creating a
        # 0-d tensor when shifted_sync is None adds unnecessary graph overhead.
        l_t = 0.0
        if shifted_sync is not None and (labels == 3).any():
            m   = labels == 3
            l_t = F.relu(0.3 - (ss[m] - shifted_sync[m])).mean().item()

        # ── MoE load-balancing ────────────────────────────────────
        # Prevent all inputs routing to one expert
        router_weights = o['router_weights']                 # (B, n_experts)
        # Import router here to call its helper — or just compute inline
        mean_w = router_weights.mean(dim=0)
        target = torch.ones_like(mean_w) / mean_w.shape[0]
        l_lb   = F.mse_loss(mean_w, target)

        total = (l_ce
                 + self.lc  * l_con
                 + self.ls  * l_sync
                 + self.lk  * l_k
                 + self.lt  * l_t
                 + self.llb * l_lb)

        return {
            'total':          total,
            'ce':             l_ce.item(),
            'contrastive':    l_con.item(),
            'sync_reg':       l_sync.item(),
            'consistency':    l_k.item(),
            'shift':          l_t,
            'load_balance':   l_lb.item()}
