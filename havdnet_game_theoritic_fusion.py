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
    Keeps flat tensor on CPU; moves CLIP_BATCH-sized chunks to device."""
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

        # Attention on device, then back to CPU to avoid holding full flat tensor on GPU
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
    """Sentence: projects pre-cached ArcFace 512-d identity embeddings."""
    def __init__(self, in_dim=512, out_dim=512):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim))

    def forward(self, x):
        return self.proj(x.mean(dim=1))


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
                if st >= et: continue
                s = int(st / 0.020)
                e = min(max(int(st / 0.020) + 1, int(et / 0.020)), Ta)
                seg = feat[b:b+1, s:e, :]
                if seg.shape[1] == 0: continue
                w = torch.softmax(self.attn_pools[scale](seg), dim=1)
                r[b, i] = (seg * w).sum(1).squeeze(0)
        return r



# ═══════════════════════════════════════════════════════════════
# DIFFERENTIABLE SAMPLE ENTROPY
# ═══════════════════════════════════════════════════════════════

class DiffSampEn(nn.Module):
    """
    Differentiable SampleEntropy for a sequence of D-dimensional vectors.
    Input:  (B, T, D)
    Output: (B,)  — one scalar per batch element.
    """
    def __init__(self, m=2, r=0.15, beta=10.0):
        super().__init__()
        self.m, self.r, self.beta = m, r, beta

    def forward(self, x):
        B, T, D = x.shape
        if T < self.m + 2:
            return torch.zeros(B, device=x.device)
        fn = F.normalize(x, dim=-1)
        dist = 1.0 - torch.bmm(fn, fn.transpose(1, 2))
        thr = self.r * dist.std(dim=(-2, -1), keepdim=True).detach()
        sm  = torch.sigmoid(self.beta * (thr - dist))
        sm  = sm * (1.0 - torch.eye(T, device=x.device).unsqueeze(0))
        Bc = Ac = torch.zeros(B, device=x.device)
        for i in range(T - self.m):
            for j in range(i + 1, T - self.m):
                mm = torch.ones(B, device=x.device)
                for k in range(self.m):
                    mm = mm * sm[:, i+k, j+k]
                Bc = Bc + mm
                if i + self.m < T and j + self.m < T:
                    Ac = Ac + mm * sm[:, i+self.m, j+self.m]
        return -torch.log((Ac + 1e-8) / (Bc + 1e-8))


# ═══════════════════════════════════════════════════════════════
# MULTI-CHANNEL ENTROPY  (replaces SlidingEntropy)
# ═══════════════════════════════════════════════════════════════

class MultiChannelEntropy(nn.Module):
    """
    Splits D-dimensional features into n_groups channel groups, computes
    DiffSampEn per group inside each sliding window.

    Input:  (B, T, D)
    Output: (B, T_out, n_groups)   where T_out = T - ws + 1

    This preserves multi-dimensional structure so that downstream sync
    metrics (cosine, PLV, soft-DTW) operate on n_groups-d vectors rather
    than collapsing everything to a scalar.
    """
    def __init__(self, ws=3, n_groups=16, m=2, r=0.15):
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
        # Truncate to an exact multiple and reshape → (B, T, g, group_size)
        x_grouped = x[:, :, :g * group_size].reshape(B, T, g, group_size)

        all_ents = []
        for t in range(T_out):
            window     = x_grouped[:, t:t+self.ws]     # (B, ws, g, group_size)
            group_ents = []
            for gi in range(g):
                feat = window[:, :, gi, :]              # (B, ws, group_size)
                group_ents.append(self.sampen(feat))    # (B,)
            all_ents.append(torch.stack(group_ents, dim=1))  # (B, g)

        return torch.stack(all_ents, dim=1)             # (B, T_out, g)


# ═══════════════════════════════════════════════════════════════
# SYNC SCORE  — operates on (B, T, D) dynamics signals
# ═══════════════════════════════════════════════════════════════

class SyncScoreModule(nn.Module):
    """
    Compares two (B, T, D) dynamics signals with three metrics:
      • Cosine similarity  — mean over time of per-step cosine
      • PLV               — per-channel phase-locking value, mean over channels
      • Soft-DTW          — D-dimensional cost matrix, exponential warp score

    The three scalars are gated per scale with a learned MLP.
    Returns: (score (B,), gate_weights (B,3), raw_metrics (B,3))
    """
    def __init__(self, n=3):
        super().__init__()
        self.gates = nn.ModuleList([
            nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 3), nn.Softmax(dim=-1))
            for _ in range(n)])

    def forward(self, v, a, si):
        # Handle 2-D input (B, T) — e.g. if something upstream collapses D
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

        c = self._cosine(v, a)   # (B,)
        p = self._plv(v, a)      # (B,)
        w = self._sdtw(v, a)     # (B,)

        m = torch.stack([c, p, w], dim=-1)          # (B, 3)
        g = self.gates[si](m.detach())              # (B, 3)
        return (m * g).sum(-1), g, m

    # ── metric helpers ──────────────────────────────────────────

    def _cosine(self, v, a):
        """Mean over time of per-step cosine similarity."""
        vn = F.normalize(v, dim=-1)
        an = F.normalize(a, dim=-1)
        return (vn * an).sum(-1).mean(dim=1)        # (B,)

    def _plv(self, v, a):
        """
        Per-channel PLV (analytic signal via real FFT Hilbert),
        averaged across channels.  Works for D >= 1.
        """
        B, T, D = v.shape
        vp = self._hilbert_phase(v)  # (B, T, D)
        ap = self._hilbert_phase(a)
        diff = vp - ap               # (B, T, D)
        # PLV per channel: |mean_t exp(j*diff)|
        plv_per_ch = torch.abs(
            torch.mean(
                torch.stack([torch.cos(diff), torch.sin(diff)], dim=-1)
                    .view(B, T, D, 2)
                    .permute(0, 2, 1, 3)               # (B, D, T, 2)
                    .contiguous()
                    .view(B * D, T, 2)
                    .mean(dim=1),                       # (B*D, 2)
                dim=-1))                               # not quite — fix below
        # cleaner version:
        cos_mean = torch.cos(diff).mean(dim=1)         # (B, D)
        sin_mean = torch.sin(diff).mean(dim=1)         # (B, D)
        plv_ch   = torch.sqrt(cos_mean**2 + sin_mean**2 + 1e-8)  # (B, D)
        return plv_ch.mean(dim=-1)                     # (B,)

    def _hilbert_phase(self, x):
        """Approximate analytic phase via real FFT Hilbert transform, per channel."""
        B, T, D = x.shape
        xf = x.reshape(B * D, T)
        N  = xf.shape[-1]
        X  = torch.fft.rfft(xf, dim=-1)               # (B*D, N//2+1)
        h  = torch.zeros(X.shape[-1], device=x.device)
        h[0] = 1.0
        if N > 2: h[1:N//2] = 2.0
        if N % 2 == 0: h[N//2] = 1.0
        analytic = torch.fft.irfft(X * h, n=N, dim=-1)  # (B*D, T)
        phase = torch.atan2(analytic, xf)               # (B*D, T)
        return phase.view(B, D, T).permute(0, 2, 1)    # (B, T, D)

    def _sdtw(self, v, a, gamma=0.1):
        """
        Soft-DTW on multi-dimensional sequences.
        Cost matrix uses squared L2 distance between D-dim vectors.
        Returns an exponential warp score in (0, 1].
        """
        B, Tv, D = v.shape
        Ta       = a.shape[1]
        # Cost matrix: (B, Tv, Ta) — squared L2
        c = ((v.unsqueeze(2) - a.unsqueeze(1)) ** 2).sum(-1)   # (B, Tv, Ta)

        R = torch.full((B, Tv+1, Ta+1), float('inf'), device=v.device)
        R[:, 0, 0] = 0.0
        for i in range(1, Tv+1):
            for j in range(1, Ta+1):
                nb = torch.stack([R[:, i-1, j-1], R[:, i-1, j], R[:, i, j-1]], dim=-1)
                R[:, i, j] = c[:, i-1, j-1] - gamma * torch.logsumexp(-nb / gamma, dim=-1)
        return torch.exp(-R[:, Tv, Ta] / (Tv + Ta))
    

class GameTheoreticFusion(nn.Module):
    """
    Game-Theoretic Fusion treating each scale as a rational player.

    Framework:
      - Each player i has a utility function Uᵢ(fᵢ, ss_i) that reflects
        how informative its features are given audio-visual sync.
      - Nash Bargaining Solution finds weights that maximise the product
        of individual surpluses over a disagreement point (sync floor).
      - Shapley values (approximated via sampling) determine each player's
        marginal contribution to the coalition utility.
      - Final fusion = Shapley-weighted sum of player features.

    Input:
        feats  : [(B,256), (B,256), (B,256)]   projected scale features
        ss     : (B, 3)                         sync scores per scale
        alphas : (B, 3)                         ScaleGate weights (prior)

    Output: (B, 256) fused representation
    """
    def __init__(self, feat_dim=256, n_players=3, n_shapley_samples=6):
        super().__init__()
        self.n   = n_players
        self.ns  = n_shapley_samples
        self.D   = feat_dim

        # Per-player utility network: maps (feat_dim + 1) → scalar utility
        # The +1 is the sync score for that scale
        self.utility_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feat_dim + 1, 64),
                nn.LayerNorm(64),
                nn.GELU(),
                nn.Linear(64, 1)          # raw utility score (unbounded)
            ) for _ in range(n_players)
        ])

        # Disagreement point network: maps sync scores → per-player floor
        # d_i is the utility player i gets when NOT in the coalition
        self.disagreement_net = nn.Sequential(
            nn.Linear(n_players, 32),
            nn.ReLU(),
            nn.Linear(32, n_players),
            nn.Sigmoid()                  # d ∈ (0,1) floor
        )

        # Coalition value network: given a subset of player features,
        # estimate the coalition's joint utility
        # Input: feat_dim (mean-pooled coalition features)
        self.coalition_net = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()                  # v(S) ∈ (0,1)
        )

        # Output projection after weighted sum
        self.out_proj = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.GELU()
        )

    # ── Utility ───────────────────────────────────────────────────────────

    def _compute_utilities(self, feats, ss):
        """
        Compute raw utility for each player.
        U_i = utility_net_i( [feat_i | ss_i] )

        Returns: (B, n_players) utilities, normalised to (0,1) via sigmoid
        """
        utils = []
        for i, net in enumerate(self.utility_nets):
            inp = torch.cat([feats[i], ss[:, i:i+1]], dim=-1)  # (B, D+1)
            utils.append(torch.sigmoid(net(inp)))               # (B, 1)
        return torch.cat(utils, dim=-1)                         # (B, n)

    # ── Disagreement point ────────────────────────────────────────────────

    def _compute_disagreement(self, ss):
        """
        d_i = what player i can guarantee alone (without coalition).
        Grounded in sync scores: low sync → low disagreement point
        (the player has little to offer independently).

        Returns: (B, n_players)
        """
        return self.disagreement_net(ss)                        # (B, n)

    # ── Nash Bargaining Solution ──────────────────────────────────────────

    def _nash_weights(self, utilities, disagreement):
        """
        Nash Bargaining Solution:
            max_{w} Π_i (U_i - d_i)   s.t. w ≥ 0, Σw = 1

        The closed-form NBS for symmetric bargaining:
            w_i ∝ (U_i - d_i) / Σ_j (U_j - d_j)

        Surplus clipped to ≥ ε to avoid degenerate collapse
        where one player gets all weight.

        Returns: (B, n_players) Nash weights
        """
        surplus = (utilities - disagreement).clamp(min=1e-3)   # (B, n)
        return surplus / surplus.sum(dim=-1, keepdim=True)      # (B, n)

    # ── Shapley Value Approximation ───────────────────────────────────────

    def _shapley_weights(self, feats, ss):
        """
        Approximate Shapley values via random coalition sampling.

        φ_i = E_{S ⊆ N\{i}} [ v(S ∪ {i}) - v(S) ]

        For each sample:
          1. Pick a random subset S of players (not including i)
          2. Compute coalition value v(S) = coalition_net(mean(feats in S))
          3. Compute v(S ∪ {i}) same way
          4. Marginal contribution = v(S ∪ {i}) - v(S)

        Average marginal contributions → Shapley value φ_i.

        Returns: (B, n_players) Shapley weights (normalised)
        """
        B     = feats[0].shape[0]
        dev   = feats[0].device
        phi   = torch.zeros(B, self.n, device=dev)

        for _ in range(self.ns):
            # Random permutation defines coalition order
            perm = torch.randperm(self.n, device=dev)
            for pos, i in enumerate(perm):
                # S = players before i in the permutation
                S = perm[:pos].tolist()

                # v(S): mean-pool S's features → coalition value
                if len(S) == 0:
                    v_S = torch.zeros(B, 1, device=dev)
                else:
                    pool_S = torch.stack([feats[j] for j in S], dim=1).mean(1)
                    v_S    = self.coalition_net(pool_S)             # (B,1)

                # v(S ∪ {i}): add player i's feature
                S_i    = S + [i.item()]
                pool_Si = torch.stack([feats[j] for j in S_i], dim=1).mean(1)
                v_Si   = self.coalition_net(pool_Si)                # (B,1)

                phi[:, i] = phi[:, i] + (v_Si - v_S).squeeze(-1)

        phi = phi / self.ns                                         # average

        # Normalise to weights — shift to positive first
        phi = phi - phi.min(dim=-1, keepdim=True).values + 1e-4
        return phi / phi.sum(dim=-1, keepdim=True)                  # (B, n)

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(self, feats, ss, alphas):
        """
        feats  : [(B,256), (B,256), (B,256)]
        ss     : (B, 3)
        alphas : (B, 3)   ScaleGate prior (used as regulariser)
        """
        # Step 1: per-player utilities
        utilities    = self._compute_utilities(feats, ss)          # (B, n)

        # Step 2: disagreement points
        disagreement = self._compute_disagreement(ss)              # (B, n)

        # Step 3: Nash bargaining weights
        nash_w       = self._nash_weights(utilities, disagreement) # (B, n)

        # Step 4: Shapley marginal contribution weights
        shapley_w    = self._shapley_weights(feats, ss)            # (B, n)

        # Step 5: Final weight = geometric mean of Nash + Shapley + ScaleGate prior
        # This blends game-theoretic solutions with learned scale attention
        combined_w   = (nash_w * shapley_w * alphas).pow(1/3)     # (B, n)
        combined_w   = combined_w / combined_w.sum(dim=-1, keepdim=True)

        # Step 6: Weighted fusion
        fused = sum(
            combined_w[:, i:i+1] * feats[i]
            for i in range(self.n)
        )                                                           # (B, 256)

        return self.out_proj(fused), {
            'utilities':    utilities,
            'disagreement': disagreement,
            'nash_weights': nash_w,
            'shapley_weights': shapley_w,
            'final_weights':   combined_w
        }




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
    Training: gate conditioned on the true class text embedding te[labels].
    Inference: gate conditioned on sync scores (label-free).
    The switch is driven by self.training of the parent module — NOT by
    whether labels are passed.
    """
    def __init__(self, d=512, n=3):
        super().__init__()
        self.gate  = nn.Sequential(nn.Linear(d, 128), nn.ReLU(),
                                   nn.Linear(128, 64), nn.ReLU(),
                                   nn.Linear(64, n))
        self.probe = nn.Sequential(nn.Linear(n, 32), nn.ReLU(),
                                   nn.Linear(32, 4), nn.Softmax(dim=-1))

    def forward(self, te=None, ss=None, ace=None, use_labels=True):
        if use_labels:
            return F.softmax(self.gate(te), dim=-1)
        return F.softmax(self.gate(self.probe(ss) @ ace), dim=-1)


# ═══════════════════════════════════════════════════════════════
# CONTRASTIVE LOSS
# ═══════════════════════════════════════════════════════════════

class ContrastiveAVLoss(nn.Module):
    def __init__(self, t=0.07):
        super().__init__()
        self.t = t

    def forward(self, vd, ad, labels):
        T = min(vd.shape[-1], ad.shape[-1])
        if T < 1:
            return torch.tensor(0.0, device=vd.device)
        v   = F.normalize(vd[:, :T], dim=-1)
        a   = F.normalize(ad[:, :T], dim=-1)
        sim = torch.mm(v, a.T) / self.t
        pos = torch.diag((labels == 3).float())
        return -(F.log_softmax(sim, dim=-1) * pos).sum() / (pos.sum() + 1e-8)


# ═══════════════════════════════════════════════════════════════
# HAVDNet-W
# ═══════════════════════════════════════════════════════════════

class HAVDNetW(nn.Module):
    CLASS_PROMPTS = [
        "a deepfake using both face swap and voice cloning",
        "a deepfake using face swap",
        "a deepfake using voice cloning",
        "an authentic video of a real person speaking"]

    def __init__(self, clip_model, wav2vec_model, nt=4, cd=512, nc=4, ew=3):
        super().__init__()
        self.clip = clip_model
        self.nc   = nc

        self.prompt_refiner = PromptRefiner(nc, nt, cd)
        self.vis_syl  = SpatialLipEncoder(128)
        self.vis_word = SpatialFaceEncoder(clip_model)
        self.vis_sent = SpatialIdentityEncoder()
        self.audio_enc = MultiScaleAudioEncoder(wav2vec_model)

        # Multi-channel entropy per scale.
        # n_groups chosen so group_size = D / n_groups is reasonable.
        # syllable: D=128 → 16 groups of 8
        # word:     D=256 → 16 groups of 16
        # sentence: D=512 → 32 groups of 16
        self.ent_syl  = MultiChannelEntropy(ws=ew, n_groups=16)
        self.ent_word = MultiChannelEntropy(ws=ew, n_groups=16)
        self.ent_sent = MultiChannelEntropy(ws=ew, n_groups=32)

        self.sync = SyncScoreModule(n=3)
        # self.gate = ScaleGate(cd, n=3)
        self.gate           = ScaleGate(cd, n=3)          # kept — feeds as prior
        self.gt_fusion      = GameTheoreticFusion(feat_dim=256, n_players=3, n_shapley_samples=6)

        self.proj_syl  = nn.Linear(128, 256)
        self.proj_word = nn.Linear(1024, 256)
        self.proj_sent = nn.Linear(512, 256)

        # Classifier input: f(256) + ss(3) + alphas(3) = 262
        self.clf = nn.Sequential(
            nn.Linear(262, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, nc))

        self._te_cache = None

    # ── text embeddings ─────────────────────────────────────────

    def _get_te(self):
        if self.training:
            # Recompute every forward — prompt_refiner params change each step
            return self._compute_te(grad=True)
        if self._te_cache is None:
            self._te_cache = self._compute_te(grad=False)
        return self._te_cache

    def _compute_te(self, grad=False):
        dev = next(self.parameters()).device
        import clip as clip_module
        tokens = clip_module.tokenize(self.CLASS_PROMPTS).to(dev)
        with torch.no_grad():
            base = self.clip.token_embedding(tokens).float()  # (4, L, 512)
        if grad:
            return self.prompt_refiner(base).mean(dim=1)      # has grad via prompt_refiner
        with torch.no_grad():
            return self.prompt_refiner(base).mean(dim=1)

    def train(self, mode=True):
        super().train(mode)
        self._te_cache = None   # invalidate on any mode switch
        return self

    # ── forward ─────────────────────────────────────────────────

    def forward(self, batch, labels=None):
        """
        labels: pass during training so the ScaleGate can use the class embedding.
                Do NOT pass during evaluation — gate uses sync scores instead.
                The self.training flag (set by model.train() / model.eval()) controls
                all other train/eval branching internally.
        """
        vs  = self.vis_syl(batch['syllable_lip_crops'])     # (B, N_syl, 128)
        vw  = self.vis_word(batch['word_face_crops'])        # (B, N_word, 1024)
        vst = self.vis_sent(batch['sentence_arcface'])       # (B, 512)

        aud = self.audio_enc(batch['waveform'], batch['boundaries'])
        # aud['syllable']: (B, N_syl, 128)
        # aud['word']:     (B, N_word, 256)
        # aud['sentence']: (B, N_sent, 512)

        # Multi-channel entropy dynamics: (B, T_out, n_groups)
        ve_s  = self.ent_syl(vs)
        ae_s  = self.ent_syl(aud['syllable'])
        ve_w  = self.ent_word(vw)
        ae_w  = self.ent_word(aud['word'])

        # For sentence-level: unsqueeze N dim so shape is (B, 1, D) → ent gives (B, 1-ws+1, g)
        # If N_sent=1 and ws=3, T_out<1; clamp to at least sentence_arcface frames
        ve_st = self.ent_sent(batch['sentence_arcface'])     # (B, T_out, 32) or zeros
        ae_st = self.ent_sent(aud['sentence'])

        # Sync on multi-channel dynamics (B, T_out, n_groups)
        ss_s,  ws_s,  ms_s  = self.sync(ve_s,  ae_s,  0)
        ss_w,  ws_w,  ms_w  = self.sync(ve_w,  ae_w,  1)
        ss_st, ws_st, ms_st = self.sync(ve_st, ae_st, 2)
        ss = torch.stack([ss_s, ss_w, ss_st], dim=-1)       # (B, 3)

        te = self._get_te()                                  # (4, 512)

        # Gate: use label embeddings during training, sync scores during inference
        if self.training and labels is not None:
            alphas = self.gate(te=te[labels], use_labels=True)
        else:
            alphas = self.gate(ss=ss, ace=te, use_labels=False)

        feats = [
            self.proj_syl(vs.mean(1)),     # (B, 256)
            self.proj_word(vw.mean(1)),    # (B, 256)
            self.proj_sent(vst)            # (B, 256)
        ]
        f, game_info = self.gt_fusion(feats, ss, alphas)   # (B, 256)
        
        logits = self.clf(torch.cat([f, ss, alphas], dim=-1))  # (B, 4)

        return {
            'logits':      logits,
            'sync_scores': ss,
            'alphas':      alphas,
            'game_info':   game_info,      # ← ADD THIS
            
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
    def __init__(self, cc, lc=0.3, ls=0.1, lk=0.05, lt=0.1, lsm=0.1):
        super().__init__()
        self.lc, self.ls, self.lk, self.lt = lc, ls, lk, lt
        tot = sum(cc)
        w   = torch.tensor([math.sqrt(tot / (c + 1)) for c in cc], dtype=torch.float32)
        w   = w / w.sum() * len(cc)
        self.register_buffer('cw', w)
        self.ce  = nn.CrossEntropyLoss(weight=w, label_smoothing=lsm)
        self.con = ContrastiveAVLoss()

    def forward(self, o, labels, epoch=0, shifted_sync=None):
        l_ce  = self.ce(o['logits'], labels)

        l_con = torch.tensor(0.0, device=l_ce.device)
        if epoch >= 3:
            for s in ['syllable', 'word', 'sentence']:
                vd, ad = o['entropy_dynamics'][s]
                # Flatten (B, T_out, g) → (B, T_out*g) for contrastive
                vd_flat = vd.reshape(vd.shape[0], -1)
                ad_flat = ad.reshape(ad.shape[0], -1)
                l_con   = l_con + self.con(vd_flat, ad_flat, labels)
            l_con = l_con / 3

        ss = o['sync_scores']
        l_sync = F.binary_cross_entropy_with_logits(
            ss, (labels == 3).float().unsqueeze(1).expand_as(ss))

        fake   = (labels != 3).float()
        l_k    = ((1 - 4 * (ss * fake.unsqueeze(1) - 0.5).pow(2)) * fake.unsqueeze(1)).mean()

        l_t = torch.tensor(0.0, device=l_ce.device)
        if shifted_sync is not None and (labels == 3).any():
            m   = labels == 3
            l_t = F.relu(0.3 - (ss[m] - shifted_sync[m])).mean()

        total = l_ce + self.lc*l_con + self.ls*l_sync + self.lk*l_k + self.lt*l_t
        return {
            'total':       total,
            'ce':          l_ce.item(),
            'contrastive': l_con.item(),
            'sync_reg':    l_sync.item(),
            'consistency': l_k.item(),
            'shift':       l_t.item() if isinstance(l_t, torch.Tensor) else l_t}
