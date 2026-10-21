"""DenseHMT: dense hybrid Mamba-Transformer encoder-decoder for HSI segmentation.

Encoder: 3 stages. Each stage runs the branches enabled in the config
(Mamba everywhere; attention at coarse stages by default), fuses with the
channel-attention gate, then downsamples 2x. Decoder: UNet-style with skips.
Output: per-pixel logits [N, C, H, W]. Train with MaskedCrossEntropy (-1 = ignore).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .components import (SpectralStem, SpectralStemDeep, SpectralStemInvariant, MambaBranch, AttnBranch, GateFusion,
                         ConcatFusion, ConvBlock)


class EncoderStage(nn.Module):
    def __init__(self, dim, use_mamba, use_attn, window, heads, mlp_ratio, drop,
                 bidirectional, fusion):
        super().__init__()
        self.mamba = MambaBranch(dim, bidirectional) if use_mamba else None
        self.attn = AttnBranch(dim, heads, window, mlp_ratio, drop) if use_attn else None
        if self.mamba is not None and self.attn is not None:
            self.fuse = GateFusion(dim) if fusion == "gate" else ConcatFusion(dim)
        else:
            self.fuse = None
        self.refine = ConvBlock(dim, dim)
        self.last_alpha = None  # stored for gate-visualisation figures

    def forward(self, x):
        m = self.mamba(x) if self.mamba is not None else None
        t = self.attn(x) if self.attn is not None else None
        if m is not None and t is not None:
            f, alpha = self.fuse(m, t)
            self.last_alpha = alpha.detach() if alpha is not None else None
        else:
            f = m if m is not None else t
        # a stage with neither branch (e.g. mamba-off ablation at stage 0)
        # degenerates to a plain conv block
        return self.refine(x if f is None else x + f)


class DenseHMT(nn.Module):
    def __init__(self, in_bands: int, num_classes: int, cfg):
        super().__init__()
        d = cfg.embed_dim
        _sm = getattr(cfg, "spectral_stem", "linear")
        if _sm == "linear":
            self.stem = SpectralStem(in_bands, d)
        elif _sm in ("invariant", "invariant_nozs", "invariant_nonorm"):
            self.stem = SpectralStemInvariant(
                in_bands, d, fusion=cfg.fusion, zero_sum=(_sm == "invariant"), normalize=(_sm != "invariant_nonorm"))
        else:
            self.stem = SpectralStemDeep(in_bands, d, mode=_sm)
        self.stages = nn.ModuleList()
        n_stages = len(cfg.mamba_stages)
        for s in range(n_stages):
            self.stages.append(EncoderStage(
                d, cfg.mamba_stages[s], cfg.attn_stages[s], cfg.window_size,
                cfg.num_heads, cfg.mlp_ratio, cfg.drop, cfg.bidirectional_mamba, cfg.fusion))
        self.pool = nn.MaxPool2d(2, ceil_mode=True)
        # UNet decoder: n_stages-1 up steps, skip from each earlier stage
        self.up_blocks = nn.ModuleList(ConvBlock(2 * d, d) for _ in range(n_stages - 1))
        self.head = nn.Conv2d(d, num_classes, 1)

    def forward(self, x):  # [N,B,H,W] -> [N,C,H,W]
        x = self.stem(x)
        skips, sizes = [], []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i < len(self.stages) - 1:
                skips.append(x)
                sizes.append(x.shape[-2:])
                x = self.pool(x)
        for up, skip, size in zip(self.up_blocks, reversed(skips), reversed(sizes)):
            x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
            x = up(torch.cat([x, skip], dim=1))
        return self.head(x)


def build_model(in_bands: int, num_classes: int, cfg) -> nn.Module:
    if cfg.name == "dense_hmt":
        return DenseHMT(in_bands, num_classes, cfg)
    raise ValueError(f"unknown model {cfg.name}")
