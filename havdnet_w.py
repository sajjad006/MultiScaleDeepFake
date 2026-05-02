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
    """Word: pre-cached CLIP embeddings → lightweight refinement → mean+std → 1024-d.

    CLIP is frozen and called only during precaching (once per video).
    At training time reads cached (N, T, 512) tensors from clip_word.pt sidecars.
    Removes 768 CLIP image inferences per batch from the forward pass.
    """
    def __init__(self, clip_model):
        super().__init__()
        self.clip = clip_model
        for p in self.clip.parameters():
            p.requires_grad = False
        # Lightweight refinement on cached embeddings
        self.refine = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 512))

    def forward(self, x):
        # x: (B, N, T, 512) pre-cached CLIP embeddings
        B, N, T, D = x.shape
        refined = self.refine(x)                             # (B, N, T, 512)
        mean = refined.mean(dim=2)                           # (B, N, 512)
        std  = refined.std(dim=2).clamp(min=1e-6)
        return torch.cat([mean, std], dim=-1)                # (B, N, 1024)

    @torch.no_grad()
    def extract_and_cache(self, face_crops_dir, word_segments, fps, total_frames,
                          cache_path, device="cuda",
                          max_words=12, max_frames_per_word=4,
                          min_context_frames=3):
        """Called once during precaching — saves (N_word, T_word, 512) tensor."""
        from PIL import Image as _Image
        from torchvision import transforms as _T
        from pathlib import Path as _Path

        def _pad(sf, ef, fps, total, mn=min_context_frames):
            n = ef - sf
            if n >= mn: return sf, ef
            deficit = mn - n
            ns = max(0, sf - deficit // 2)
            ne = min(total, ef + deficit - deficit // 2)
            if ne - ns < mn:
                if ns == 0: ne = min(total, mn)
                else:       ns = max(0, ne - mn)
            return ns, ne

        tfm = _T.Compose([
            _T.Resize((224, 224)), _T.ToTensor(),
            _T.Normalize([.485, .456, .406], [.229, .224, .225])])

        CLIP_BATCH = 32
        all_segs = []
        for seg in word_segments[:max_words]:
            sf, ef = seg.get("start_frame", 0), seg.get("end_frame", 0)
            sf, ef = _pad(sf, ef, fps, total_frames)
            frames = []
            for i in range(sf, min(ef, sf + max_frames_per_word)):
                fp = _Path(face_crops_dir) / f"frame_{i:06d}.jpg"
                if fp.exists():
                    frames.append(tfm(_Image.open(fp).convert("RGB")))
            if not frames: frames = [torch.zeros(3, 224, 224)]
            while len(frames) < max_frames_per_word: frames.append(frames[-1])
            all_segs.append(torch.stack(frames[:max_frames_per_word]))

        while len(all_segs) < max_words:
            all_segs.append(torch.zeros(max_frames_per_word, 3, 224, 224))

        imgs = torch.stack(all_segs)
        N, T, C, H, W = imgs.shape
        flat = imgs.reshape(N * T, C, H, W)
        embs = []
        for i in range(0, flat.shape[0], CLIP_BATCH):
            chunk = flat[i:i + CLIP_BATCH].to(device)
            embs.append(self.clip.encode_image(chunk).float().cpu())
        embs = torch.cat(embs).view(N, T, 512)
        torch.save(F.normalize(embs, dim=-1), cache_path)


class SpatialIdentityEncoder(nn.Module):
    """Sentence: CLIP face embeddings → global pooled + full sequence.

    Returns TWO tensors:
      global: (B, out_dim) — valid-frame masked mean for MoE fusion
      seq:    (B, N, 512)  — full sequence for entropy dynamics
    """
    def __init__(self, in_dim=512, out_dim=512):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim))

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        seq = x                                               # (B, N, 512)
        # Masked mean — ignores zero-padded rows
        valid_mask   = (x.abs().sum(dim=-1) > 1e-6).float()  # (B, N)
        valid_counts = valid_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
        global_      = self.proj(x.sum(dim=1) / valid_counts) # (B, out_dim)
        return global_, seq


# ═══════════════════════════════════════════════════════════════
# AUDIO ENCODER
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
                e = min(max(s + 1, int(et / 0.020)), Ta)
                seg = feat[b:b+1, s:e, :]
                if seg.shape[1] == 0: continue
                w = torch.softmax(self.attn_pools[scale](seg), dim=1)
                r[b, i] = (seg * w).sum(1).squeeze(0)
        return r


# ═══════════════════════════════════════════════════════════════
# DIFFERENTIABLE SAMPLE ENTROPY
# ═══════════════════════════════════════════════════════════════

class DiffSampEn(nn.Module):
    """Differentiable SampleEntropy. Input: (B,T,D) → Output: (B,)"""
    def __init__(self, m=2, r=0.15, beta=10.0):
        super().__init__()
        self.m, self.r, self.beta = m, r, beta

    def forward(self, x):
        B, T, D = x.shape
        if T < self.m + 2:
            return torch.zeros(B, device=x.device)
        fn   = F.normalize(x, dim=-1)
        dist = 1.0 - torch.bmm(fn, fn.transpose(1, 2))
        thr  = self.r * dist.std(dim=(-2, -1), keepdim=True).detach()
        sm   = torch.sigmoid(self.beta * (thr - dist))
        sm   = sm * (1.0 - torch.eye(T, device=x.device).unsqueeze(0))
        Bc = torch.zeros(B, device=x.device)
        Ac = torch.zeros(B, device=x.device)
        for i in range(T - self.m):
            for j in range(i + 1, T - self.m):
                mm = sm[:, i, j]
                for k in range(1, self.m):
                    mm = mm * sm[:, i+k, j+k]
                Bc = Bc + mm
                ei, ej = i + self.m, j + self.m
                if ei < T and ej < T:
                    Ac = Ac + mm * sm[:, ei, ej]
        return -torch.log((Ac + 1e-8) / (Bc + 1e-8))


class MultiChannelEntropy(nn.Module):
    """Sliding window DiffSampEn per channel group.
    Input: (B,T,D) → Output: (B, T_out, n_groups)
    ws must be >= m+2 = 4. Default ws=5 satisfies this.
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
            window     = x_grouped[:, t:t+self.ws]
            group_ents = []
            for gi in range(g):
                feat = window[:, :, gi, :]
                group_ents.append(self.sampen(feat))
            all_ents.append(torch.stack(group_ents, dim=1))
        return torch.stack(all_ents, dim=1)


# ═══════════════════════════════════════════════════════════════
# SYNC SCORE MODULE
# ═══════════════════════════════════════════════════════════════

class SyncScoreModule(nn.Module):
    """Cosine · PLV · soft-DTW per scale → scalar sync score ∈ [0,1]."""
    def __init__(self, n=3):
        super().__init__()
        self.gates = nn.ModuleList([
            nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 3), nn.Softmax(dim=-1))
            for _ in range(n)])

    def forward(self, v, a, si):
        if v.dim() == 2: v = v.unsqueeze(-1)
        if a.dim() == 2: a = a.unsqueeze(-1)
        T = min(v.shape[1], a.shape[1])
        B = v.shape[0]; d = v.device
        if T < 2:
            return (torch.full((B,), 0.5, device=d),
                    torch.ones(B, 3, device=d) / 3,
                    torch.full((B, 3), 0.5, device=d))
        v, a = v[:, :T], a[:, :T]
        c = self._cosine(v, a)
        p = self._plv(v, a)
        w = self._sdtw(v, a)
        m = torch.stack([c, p, w], dim=-1)
        g = self.gates[si](m.detach())
        return (m * g).sum(-1), g, m

    def _cosine(self, v, a):
        vn = F.normalize(v, dim=-1); an = F.normalize(a, dim=-1)
        cos = (vn * an).sum(-1).mean(dim=1)
        return (cos + 1.0) * 0.5   # remap [-1,1] → [0,1]

    def _plv(self, v, a):
        diff     = self._hilbert_phase(v) - self._hilbert_phase(a)
        cos_mean = torch.cos(diff).mean(dim=1)
        sin_mean = torch.sin(diff).mean(dim=1)
        return torch.sqrt(cos_mean**2 + sin_mean**2 + 1e-8).mean(dim=-1)

    def _hilbert_phase(self, x):
        B, T, D = x.shape
        xf = x.reshape(B * D, T)
        N  = xf.shape[-1]
        X  = torch.fft.rfft(xf, dim=-1)
        h  = torch.zeros(X.shape[-1], device=x.device)
        h[0] = 1.0
        if N > 2: h[1:N//2] = 2.0
        if N % 2 == 0: h[N//2] = 1.0
        analytic = torch.fft.irfft(X * h, n=N, dim=-1)
        return torch.atan2(analytic, xf).view(B, D, T).permute(0, 2, 1)

    def _sdtw(self, v, a, gamma=0.1):
        B, Tv, D = v.shape; Ta = a.shape[1]
        c = ((v.unsqueeze(2) - a.unsqueeze(1)) ** 2).sum(-1)
        R = torch.full((B, Tv+1, Ta+1), float('inf'), device=v.device)
        R[:, 0, 0] = 0.0
        for i in range(1, Tv+1):
            for j in range(1, Ta+1):
                nb = torch.stack([R[:, i-1, j-1], R[:, i-1, j], R[:, i, j-1]], dim=-1)
                R[:, i, j] = c[:, i-1, j-1] - gamma * torch.logsumexp(-nb / gamma, dim=-1)
        return torch.exp(-R[:, Tv, Ta] / (Tv + Ta))


# ═══════════════════════════════════════════════════════════════
# GAME-THEORETIC FUSION
# ═══════════════════════════════════════════════════════════════

class GameTheoreticFusion(nn.Module):
    """
    Game-Theoretic Fusion with clean separation of signals:

      Nash weights   ← sync scores (ss)   — measured AV agreement per scale.
                                             ss_i IS player i's utility.
                                             No learned net needed — sync directly
                                             encodes how much the scale contributes.

      Shapley weights ← AV-fused features — coalition value estimated from what
                                             the feature vectors actually look like,
                                             independent of sync agreement.

      ScaleGate prior ← text embeddings   — semantic prior: which scale matters
                                             for this fake type (alphas from gate).

    Final weights = geometric mean(nash_w, shapley_w, alphas), renormalised.
    Output: (B, feat_dim) fused representation + interpretability dict.
    """
    def __init__(self, feat_dim=256, n_players=3, n_shapley_samples=6):
        super().__init__()
        self.n  = n_players
        self.ns = n_shapley_samples

        # Coalition value network: mean-pooled AV features → v(S) ∈ (0,1)
        # Used only for Shapley estimation — operates on feature content,
        # not sync scores (those drive Nash separately).
        self.coalition_net = nn.Sequential(
            nn.Linear(feat_dim, 64), nn.LayerNorm(64), nn.GELU(),
            nn.Linear(64, 1), nn.Sigmoid())

        self.out_proj = nn.Sequential(
            nn.Linear(feat_dim, feat_dim), nn.LayerNorm(feat_dim), nn.GELU())

    # ── Nash: sync scores ARE the utilities ──────────────────────

    def _nash_weights(self, ss):
        """
        Nash Bargaining Solution.
        Utility for player i = ss_i (sync score for that scale).
        Disagreement point = uniform floor (each player's fallback = 1/n).
        Surplus = utility - floor, clipped positive.
        NBS weights ∝ surplus.
        """
        floor   = torch.full_like(ss, 1.0 / self.n)           # (B, n) uniform floor
        surplus = (ss - floor).clamp(min=1e-3)                 # (B, n)
        return surplus / surplus.sum(dim=-1, keepdim=True)     # (B, n)

    # ── Shapley: marginal contributions from feature content ─────

    def _shapley_weights(self, feats):
        """
        Approximate Shapley values via random coalition sampling.
        coalition_net estimates v(S) from mean-pooled features of coalition S.
        Marginal contribution of player i = v(S∪{i}) - v(S).
        """
        B   = feats[0].shape[0]
        dev = feats[0].device
        phi = torch.zeros(B, self.n, device=dev)

        for _ in range(self.ns):
            perm = torch.randperm(self.n, device=dev)
            for pos, i in enumerate(perm):
                S = perm[:pos].tolist()
                v_S = (torch.zeros(B, 1, device=dev) if not S else
                       self.coalition_net(
                           torch.stack([feats[j] for j in S], dim=1).mean(1)))
                S_i  = S + [i.item()]
                v_Si = self.coalition_net(
                    torch.stack([feats[j] for j in S_i], dim=1).mean(1))
                phi[:, i] = phi[:, i] + (v_Si - v_S).squeeze(-1)

        phi = phi / self.ns
        # Shift to positive and normalise
        phi = phi - phi.min(dim=-1, keepdim=True).values + 1e-4
        return phi / phi.sum(dim=-1, keepdim=True)             # (B, n)

    # ── Forward ───────────────────────────────────────────────────

    def forward(self, feats, ss, alphas):
        """
        feats  : [(B,256), (B,256), (B,256)]  AV-fused features per scale
        ss     : (B, 3)                        sync scores — player utilities
        alphas : (B, 3)                        ScaleGate text-conditioned prior
        """
        nash_w    = self._nash_weights(ss)                     # (B, n) from ss
        shapley_w = self._shapley_weights(feats)               # (B, n) from feats

        # Geometric mean: all three must agree for a scale to dominate
        combined_w = (nash_w * shapley_w * alphas).pow(1/3)
        combined_w = combined_w / combined_w.sum(dim=-1, keepdim=True)

        fused = sum(combined_w[:, i:i+1] * feats[i] for i in range(self.n))

        return self.out_proj(fused), {
            'nash_weights':    nash_w,
            'shapley_weights': shapley_w,
            'final_weights':   combined_w}


# ═══════════════════════════════════════════════════════════════
# PROMPT GATE + SCALE GATE
# ═══════════════════════════════════════════════════════════════

class PromptRefiner(nn.Module):
    def __init__(self, nc=4, nt=4, d=512):
        super().__init__()
        self.lp   = nn.Parameter(torch.randn(nc, nt, d) * 0.02)
        self.proj = nn.Sequential(nn.Linear(d, d), nn.LayerNorm(d), nn.GELU())

    def forward(self, te):
        return torch.cat([self.proj(self.lp).to(te.dtype), te], dim=1)


class SyncRouter(nn.Module):
    """
    Maps sync scores (B, 3) → soft class distribution (B, 4).
    Used at inference to select a weighted text embedding without the true label.
    Trained end-to-end — receives gradient from classification loss.

    Nonlinear (2-layer MLP) to handle non-linear class boundaries in sync space:
      - Real:       ss_syl high, ss_word high, ss_sent high
      - Face-swap:  ss_word low, others ok
      - Voice-clone: ss_sent low, others ok
      - Both-fake:  ss_word low AND ss_sent low

    These regions are not linearly separable from the origin.
    """
    def __init__(self, n_sync=3, n_classes=4, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_sync, hidden), nn.ReLU(),
            nn.Linear(hidden, n_classes), nn.Softmax(dim=-1))

    def forward(self, ss):
        return self.net(ss)   # (B, 4) soft class distribution


class ScaleGate(nn.Module):
    """Text-embedding conditioned scale attention weights."""
    def __init__(self, d=512, n=3):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, n))

    def forward(self, te):
        return F.softmax(self.gate(te), dim=-1)   # (B, n)


# ═══════════════════════════════════════════════════════════════
# HAVDNET-W
# ═══════════════════════════════════════════════════════════════

class HAVDNetW(nn.Module):
    CLASS_PROMPTS = [
        "a deepfake using both face swap and voice cloning",
        "a deepfake using face swap",
        "a deepfake using voice cloning",
        "an authentic video of a real person speaking"]

    def __init__(self, clip_model, wav2vec_model, nt=4, cd=512, nc=4, ew=5):
        super().__init__()
        self.clip = clip_model
        self.nc   = nc

        self.prompt_refiner = PromptRefiner(nc, nt, cd)
        self.vis_syl   = SpatialLipEncoder(128)
        self.vis_word  = SpatialFaceEncoder(clip_model)
        self.vis_sent  = SpatialIdentityEncoder()
        self.audio_enc = MultiScaleAudioEncoder(wav2vec_model)

        # ws=5 ensures T=ws=5 >= m+2=4 → DiffSampEn actually computes
        self.ent_syl  = MultiChannelEntropy(ws=ew, n_groups=16, m=2)
        self.ent_word = MultiChannelEntropy(ws=ew, n_groups=16, m=2)
        self.ent_sent = MultiChannelEntropy(ws=ew, n_groups=32, m=2)

        self.sync       = SyncScoreModule(n=3)
        self.sync_router = SyncRouter(n_sync=3, n_classes=nc, hidden=32)
        self.gate       = ScaleGate(cd, n=3)

        # AV-fused projection per scale:
        # syl:  visual(128) + audio(128) = 256 → 256
        # word: visual(1024) + audio(256) = 1280 → 256
        # sent: visual(512) + audio(512) = 1024 → 256
        self.proj_syl  = nn.Linear(128 + 128,   256)
        self.proj_word = nn.Linear(1024 + 256,  256)
        self.proj_sent = nn.Linear(512 + 512,   256)

        # Game-theoretic fusion
        self.gt_fusion = GameTheoreticFusion(feat_dim=256, n_players=3,
                                             n_shapley_samples=6)

        # Classifier: f(256) + ss(3) + alphas(3) = 262
        self.clf = nn.Sequential(
            nn.Linear(262, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, nc))

        self._te_cache = None

    # ── text embeddings ─────────────────────────────────────────

    def _get_te(self):
        if self.training:
            return self._compute_te(grad=True)
        if self._te_cache is None:
            self._te_cache = self._compute_te(grad=False)
        return self._te_cache

    def _compute_te(self, grad=False):
        """Full CLIP transformer pass in correct dtype. Always returns fp32."""
        dev = next(self.parameters()).device
        import clip as clip_module
        tokens = clip_module.tokenize(self.CLASS_PROMPTS).to(dev)

        with torch.no_grad():
            base = self.clip.token_embedding(tokens).float()        # (4, 77, 512) fp32

        # Additive prefix injection — keeps seq_len=77 for positional embeddings
        refined_prefix = self.prompt_refiner.proj(
            self.prompt_refiner.lp).float()
        nt = refined_prefix.shape[1]

        if grad:
            x = base.clone()
            x[:, :nt, :] = x[:, :nt, :] + refined_prefix
        else:
            with torch.no_grad():
                x = base.clone()
                x[:, :nt, :] = x[:, :nt, :] + refined_prefix.detach()

        ctx = torch.no_grad() if not grad else torch.enable_grad()
        with ctx:
            # Match transformer weight dtype to avoid fp16/fp32 mismatch
            dtype = next(self.clip.transformer.parameters()).dtype
            x = x + self.clip.positional_embedding.float()
            x = x.to(dtype).permute(1, 0, 2)                        # LND
            x = self.clip.transformer(x)
            x = x.permute(1, 0, 2)                                  # NLD
            x = self.clip.ln_final(x).float()

        eot_idx = tokens.argmax(dim=-1)
        return x[torch.arange(4, device=dev), eot_idx]             # (4, 512) fp32

    def train(self, mode=True):
        super().train(mode)
        self._te_cache = None
        return self

    # ── trim zero-padded tails before entropy ──────────────────

    @staticmethod
    def _trim_valid(x, min_t):
        """Remove trailing all-zero rows. Keeps at least min_t frames."""
        norms     = x.abs().sum(dim=-1)
        any_valid = norms.any(dim=0)
        valid_idx = any_valid.nonzero(as_tuple=False)
        if valid_idx.numel() == 0:
            return x[:, :min_t]
        return x[:, :max(valid_idx[-1].item() + 1, min_t)]

    # ── forward ─────────────────────────────────────────────────

    def forward(self, batch, labels=None):
        # ── Visual encoding ──────────────────────────────────────
        vs  = self.vis_syl(batch['syllable_lip_crops'])       # (B, N_syl, 128)
        vw  = self.vis_word(batch['word_face_crops'])          # (B, N_word, 1024)
        vst_global, vst_seq = self.vis_sent(batch['sentence_arcface'])
        # vst_global: (B, 512)   for AV fusion
        # vst_seq:    (B, N, 512) for entropy

        # ── Audio encoding ───────────────────────────────────────
        aud = self.audio_enc(batch['waveform'], batch['boundaries'])
        # aud['syllable']: (B, N, 128)
        # aud['word']:     (B, N, 256)
        # aud['sentence']: (B, N, 512)

        # ── Entropy dynamics (trim zero padding first) ────────────
        ws = self.ent_syl.ws
        vs_t  = self._trim_valid(vs,              ws)
        vw_t  = self._trim_valid(vw,              ws)
        vst_t = self._trim_valid(vst_seq,         ws)
        as_t  = self._trim_valid(aud['syllable'], ws)
        aw_t  = self._trim_valid(aud['word'],     ws)
        ast_t = self._trim_valid(aud['sentence'], ws)

        ve_s  = self.ent_syl(vs_t)
        ae_s  = self.ent_syl(as_t)
        ve_w  = self.ent_word(vw_t)
        ae_w  = self.ent_word(aw_t)
        ve_st = self.ent_sent(vst_t)
        ae_st = self.ent_sent(ast_t)

        # ── Sync scores ──────────────────────────────────────────
        ss_s,  ws_s,  ms_s  = self.sync(ve_s,  ae_s,  0)
        ss_w,  ws_w,  ms_w  = self.sync(ve_w,  ae_w,  1)
        ss_st, ws_st, ms_st = self.sync(ve_st, ae_st, 2)
        ss = torch.stack([ss_s, ss_w, ss_st], dim=-1)         # (B, 3)

        # ── Text embeddings ──────────────────────────────────────
        te = self._get_te()                                    # (4, 512) fp32

        # ── Scale gate — NO label leakage at train or eval ───────
        # At training:  use SyncRouter to get soft class probs from ss,
        #               then weight te — same path as inference.
        #               This trains the router end-to-end.
        # At inference: identical path — no label needed, no leakage.
        #
        # Note: we deliberately do NOT use te[labels] even during training.
        # Using true labels would cause the gate to depend on information
        # unavailable at inference, creating a train/eval mismatch.
        # The router learns to approximate te[true_label] from ss alone.
        class_probs = self.sync_router(ss)                     # (B, 4) — grad flows
        te_input    = class_probs @ te                         # (B, 512)
        alphas      = self.gate(te_input)                      # (B, 3)

        # ── AV-fused features per scale ───────────────────────────
        # Concatenate visual mean-pool + audio mean-pool per scale,
        # then project to 256-d. Each player in the game represents
        # the joint AV signal at that scale, not just visual.
        f_syl  = self.proj_syl(
            torch.cat([vs.mean(1), aud['syllable'].mean(1)], dim=-1))   # (B,256)
        f_word = self.proj_word(
            torch.cat([vw.mean(1), aud['word'].mean(1)],     dim=-1))   # (B,256)
        f_sent = self.proj_sent(
            torch.cat([vst_global, aud['sentence'].mean(1)], dim=-1))   # (B,256)
        feats  = [f_syl, f_word, f_sent]

        # ── Game-theoretic fusion ─────────────────────────────────
        # Nash:    from ss (sync scores = player utilities)
        # Shapley: from feats (AV feature content)
        # Prior:   from alphas (text-conditioned scale attention)
        f, game_info = self.gt_fusion(feats, ss, alphas)       # (B, 256)

        # ── Classification ────────────────────────────────────────
        logits = self.clf(torch.cat([f, ss, alphas], dim=-1))  # (B, 262→4)

        return {
            'logits':      logits,
            'sync_scores': ss,
            'alphas':      alphas,
            'game_info':   game_info,
            'metrics': {
                'syllable': {'raw': ms_s,  'weights': ws_s},
                'word':     {'raw': ms_w,  'weights': ws_w},
                'sentence': {'raw': ms_st, 'weights': ws_st}},
            'entropy_dynamics': {
                'syllable': (ve_s,  ae_s),
                'word':     (ve_w,  ae_w),
                'sentence': (ve_st, ae_st)}}


# ═══════════════════════════════════════════════════════════════
# LOSS — Simple CrossEntropy + sync supervision only
# ═══════════════════════════════════════════════════════════════

class HAVDNetLoss(nn.Module):
    """
    Loss: weighted CrossEntropy + per-scale sync supervision.

    Sync supervision targets per scale:
      ss_syl  should be HIGH for classes 0 (real+real) and 2 (fake_video+real_audio)
              — real audio means phoneme-level sync should still hold
              should be LOW for classes 1 and 3 (fake audio)

      ss_word should be HIGH for classes 0 and 1 (real video)
              — real face means word-level face dynamics are natural
              should be LOW for classes 2 and 3 (fake video)

      ss_sent should be HIGH for class 0 only (both real)
              — only fully real videos have consistent sentence-level AV sync
              should be LOW for classes 1, 2, 3 (any fake)

    This is the correct semantic supervision — not a single real/fake label
    applied uniformly to all scales.
    """
    def __init__(self, cc, ls=0.1, lsm=0.15):
        super().__init__()
        self.ls = ls
        tot = sum(cc)
        w   = torch.tensor([math.sqrt(tot / (c + 1)) for c in cc], dtype=torch.float32)
        w   = w / w.sum() * len(cc)
        self.register_buffer('cw', w)
        self.ce = nn.CrossEntropyLoss(weight=w, label_smoothing=lsm)

    def forward(self, o, labels, epoch=0, shifted_sync=None):
        l_ce = self.ce(o['logits'], labels)

        ss = o['sync_scores']   # (B, 3): [ss_syl, ss_word, ss_sent]
        B  = labels.shape[0]
        dev = ss.device

        # Per-scale sync targets — each scale has different semantics
        # labels: 0=RealVideo-RealAudio, 1=RealVideo-FakeAudio,
        #         2=FakeVideo-RealAudio, 3=FakeVideo-FakeAudio
        real_audio = ((labels == 0) | (labels == 2)).float()   # (B,)
        real_video = ((labels == 0) | (labels == 1)).float()   # (B,)
        fully_real = (labels == 0).float()                     # (B,)

        # Stack per-scale targets: (B, 3)
        sync_tgt = torch.stack([real_audio, real_video, fully_real], dim=-1)

        with torch.amp.autocast('cuda', enabled=False):
            l_sync = F.binary_cross_entropy(
                ss.float().clamp(1e-6, 1 - 1e-6), sync_tgt.float())

        total = l_ce + self.ls * l_sync
        return {
            'total':       total,
            'ce':          l_ce.item(),
            'sync_reg':    l_sync.item(),
            'contrastive': 0.0,
            'consistency': 0.0,
            'shift':       0.0,
            'load_balance':0.0}
