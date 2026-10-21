"""Building blocks: spectral stem, Mamba branch, windowed-attention branch, gate fusion.

Design notes (see work plan D2/D3):
- Mamba branch runs at every stage (linear in pixel count).
- Attention branch only at coarser stages by default (12 GB VRAM constraint); windowed
  (Swin-style, no shift) when window_size > 0, global when window_size == 0.
- Fusion = the WHISPERS channel-attention gate, generalized to dense feature maps.
  ConcatFusion kept as the ablation.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except ImportError:  # CPU-only machines / CI: tests still run with the fallback
    HAS_MAMBA = False


class SpectralStem(nn.Module):
    """1x1 conv projection B bands -> D channels (as in the WHISPERS model)."""
    def __init__(self, in_bands: int, dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_bands, dim, 1), nn.BatchNorm2d(dim), nn.ReLU(inplace=True))

    def forward(self, x):  # [N,B,H,W] -> [N,D,H,W]
        return self.proj(x)


class MambaBranch(nn.Module):
    """Flatten spatial dims -> Mamba over the token sequence -> reshape. Optionally
    bidirectional (forward + flipped scan, averaged)."""
    def __init__(self, dim: int, bidirectional: bool = True):
        super().__init__()
        self.bidirectional = bidirectional
        if HAS_MAMBA:
            self.fwd = Mamba(d_model=dim)
            self.bwd = Mamba(d_model=dim) if bidirectional else None
        else:  # lightweight fallback so shape tests run without CUDA
            self.fwd = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
            self.bwd = None
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):  # [N,D,H,W]
        N, D, H, W = x.shape
        seq = x.flatten(2).transpose(1, 2)          # [N, HW, D]
        seq = self.norm(seq)
        out = self.fwd(seq)
        if self.bwd is not None:
            out = 0.5 * (out + torch.flip(self.bwd(torch.flip(seq, dims=[1])), dims=[1]))
        return out.transpose(1, 2).reshape(N, D, H, W)


class AttnBranch(nn.Module):
    """Windowed multi-head self-attention (non-overlapping windows) + FFN.
    window == 0 -> global attention over all tokens (use only at coarse stages)."""
    def __init__(self, dim: int, num_heads: int = 4, window: int = 8, mlp_ratio: float = 4.0,
                 drop: float = 0.0):
        super().__init__()
        self.window = window
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x):  # [N,D,H,W]
        N, D, H, W = x.shape
        w = self.window
        if w and (H > w or W > w):
            ph, pw = (w - H % w) % w, (w - W % w) % w
            xp = F.pad(x, (0, pw, 0, ph))
            Hp, Wp = H + ph, W + pw
            t = xp.reshape(N, D, Hp // w, w, Wp // w, w).permute(0, 2, 4, 3, 5, 1)
            t = t.reshape(N * (Hp // w) * (Wp // w), w * w, D)       # windows as batch
        else:
            t = x.flatten(2).transpose(1, 2)                          # global
            Hp = Wp = None
        h = self.norm1(t)
        t = t + self.attn(h, h, h, need_weights=False)[0]
        t = t + self.mlp(self.norm2(t))
        if Hp is not None:
            t = t.reshape(N, Hp // w, Wp // w, w, w, D).permute(0, 5, 1, 3, 2, 4)
            t = t.reshape(N, D, Hp, Wp)[:, :, :H, :W]
        else:
            t = t.transpose(1, 2).reshape(N, D, H, W)
        return t


class GateFusion(nn.Module):
    """WHISPERS channel-attention gate: alpha = sigmoid(MLP(GAP([M;T]))),
    fused = Conv1x1(alpha * M + (1 - alpha) * T)."""
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(2 * dim, dim), nn.ReLU(inplace=True), nn.Linear(dim, dim), nn.Sigmoid())
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, m, t):  # both [N,D,H,W]
        g = torch.cat([m, t], dim=1).mean(dim=(2, 3))     # GAP -> [N, 2D]
        alpha = self.gate(g)[:, :, None, None]            # [N,D,1,1]
        return self.proj(alpha * m + (1.0 - alpha) * t), alpha.squeeze(-1).squeeze(-1)


class ConcatFusion(nn.Module):
    """Ablation: plain concatenation + 1x1 conv."""
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Conv2d(2 * dim, dim, 1)

    def forward(self, m, t):
        return self.proj(torch.cat([m, t], dim=1)), None


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.block(x)


class _MambaSeq(nn.Module):
    """Mamba over an arbitrary token sequence [P, L, C] (used for the spectral axis)."""
    def __init__(self, dim: int, bidirectional: bool = True):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        if HAS_MAMBA:
            self.fwd = Mamba(d_model=dim)
            self.bwd = Mamba(d_model=dim) if bidirectional else None
        else:
            self.fwd = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
            self.bwd = None

    def forward(self, s):
        s = self.norm(s)
        o = self.fwd(s)
        if self.bwd is not None:
            o = 0.5 * (o + torch.flip(self.bwd(torch.flip(s, dims=[1])), dims=[1]))
        return o


class SpectralStemDeep(nn.Module):
    """Deeper spectral front-end.

    mode="mlp"   : per-pixel spectral MLP (three 1x1 convs) - cheap spectral depth.
    mode="mamba" : bands split into groups, treated as a sequence, scanned by a
                   state-space model - spectral sequence modelling (S2Mamba-style).
    """
    def __init__(self, in_bands: int, dim: int, mode: str = "mlp", n_groups: int = 20,
                 group_dim: int = 32, bidirectional: bool = True, chunk: int = 65536):
        super().__init__()
        self.mode, self.dim, self.chunk = mode, dim, chunk
        if mode == "mlp":
            h1, h2 = 256, 128
            self.proj = nn.Sequential(
                nn.Conv2d(in_bands, h1, 1), nn.BatchNorm2d(h1), nn.ReLU(inplace=True),
                nn.Conv2d(h1, h2, 1), nn.BatchNorm2d(h2), nn.ReLU(inplace=True),
                nn.Conv2d(h2, dim, 1), nn.BatchNorm2d(dim), nn.ReLU(inplace=True))
        elif mode == "mamba":
            self.n_groups = n_groups
            self.gsize = math.ceil(in_bands / n_groups)
            self.pad = n_groups * self.gsize - in_bands
            self.embed = nn.Linear(self.gsize, group_dim)
            self.seq = _MambaSeq(group_dim, bidirectional)
            self.out = nn.Sequential(nn.Linear(group_dim, dim), nn.ReLU(inplace=True))
            self.bn = nn.BatchNorm2d(dim)
        else:
            raise ValueError(f"unknown spectral stem mode: {mode}")

    def forward(self, x):  # [N,B,H,W] -> [N,D,H,W]
        if self.mode == "mlp":
            return self.proj(x)
        N, B, H, W = x.shape
        if self.pad:
            x = F.pad(x, (0, 0, 0, 0, 0, self.pad))
        t = x.permute(0, 2, 3, 1).reshape(N * H * W, self.n_groups, self.gsize)
        t = self.embed(t)
        outs = [self.seq(t[i:i + self.chunk]) for i in range(0, t.shape[0], self.chunk)]
        t = torch.cat(outs, 0).mean(1)
        t = self.out(t).reshape(N, H, W, self.dim).permute(0, 3, 1, 2).contiguous()
        return self.bn(t)


class ZeroSumSpectralFilter(nn.Module):
    """Bank of F local filters along the spectral axis, constrained to sum to zero.

    Re-parameterised as w = v - mean(v), so the constraint holds exactly at every step
    with no penalty term and no projection. For a locally affine per-scene corruption
    x -> a*x + b (a, b roughly constant over the K-band support):

        w * (a x + b) = a (w * x) + b * sum(w) = a (w * x)

    the additive offset dies exactly. L2 normalisation along the band axis then removes
    the multiplicative gain, giving a representation invariant to both.
    """

    def __init__(self, n_filters: int = 4, kernel: int = 5):
        super().__init__()
        self.k = kernel
        self.v = nn.Parameter(torch.randn(n_filters, 1, kernel, 1, 1) * 0.1)

    def weight(self):
        return self.v - self.v.mean(dim=2, keepdim=True)      # zero-sum by construction

    def forward(self, x):                                      # [N,B,H,W] -> [N,F,B,H,W]
        y = F.conv3d(x.unsqueeze(1), self.weight(), padding=(self.k // 2, 0, 0))
        return y / (y.norm(dim=2, keepdim=True) + 1e-6)        # normalise along bands


class SpectralStemInvariant(nn.Module):
    """Two-branch spectral stem: raw projection + provably scene-invariant branch.

    Branch R keeps all information but is scene-dependent (retains albedo level).
    Branch I is invariant to locally-affine per-scene corruption but discards that level.
    The gate learns how much invariance to buy; alpha is stored for analysis, and is
    expected to shift toward branch I under distribution shift.
    """

    def __init__(self, in_bands: int, dim: int, n_filters: int = 4, kernel: int = 5,
                 fusion: str = "gate", zero_sum: bool = True, normalize: bool = True):
        super().__init__()
        self.raw = SpectralStem(in_bands, dim)
        self.filt = ZeroSumSpectralFilter(n_filters, kernel)
        self.zero_sum = zero_sum                                # False = ablation
        self.normalize = normalize                              # False = norm ablation
        self.inv_proj = nn.Sequential(
            nn.Conv2d(n_filters * in_bands, dim, 1), nn.BatchNorm2d(dim), nn.ReLU(inplace=True))
        self.fuse = GateFusion(dim) if fusion == "gate" else ConcatFusion(dim)
        self.last_alpha = None

    def forward(self, x):                                      # [N,B,H,W] -> [N,D,H,W]
        N, B, H, W = x.shape
        if self.zero_sum:
            y = self.filt(x)
        else:                                                  # ablation: same bank, no constraint
            y = F.conv3d(x.unsqueeze(1), self.filt.v, padding=(self.filt.k // 2, 0, 0))
            if self.normalize:
                y = y / (y.norm(dim=2, keepdim=True) + 1e-6)
        inv = self.inv_proj(y.reshape(N, -1, H, W))
        r = self.raw(x)
        f, alpha = self.fuse(inv, r)
        self.last_alpha = alpha.detach() if alpha is not None else None
        return f
