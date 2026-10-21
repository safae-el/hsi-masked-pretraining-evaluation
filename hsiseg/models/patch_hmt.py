"""WHISPERS model, ported verbatim from hybrid_weighted.py (M2 reproduction gate).
Constants = paper values: D=64, mamba(16,4,2), 4 heads, FF 256, dropout 0.1."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba


class MambaBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.mamba = Mamba(d_model=d, d_state=16, d_conv=4, expand=2)

    def forward(self, x):
        return x + self.mamba(self.norm(x))


class TransformerBlock(nn.Module):
    def __init__(self, d, heads=4, ff=256, drop=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Dropout(drop),
                                nn.Linear(ff, d), nn.Dropout(drop))

    def forward(self, x):
        n = self.norm1(x)
        a, _ = self.attn(n, n, n)
        x = x + a
        return x + self.ff(self.norm2(x))


class PositionalEncoding(nn.Module):
    def __init__(self, d, max_len=512):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_len, d))
        nn.init.trunc_normal_(self.pe, std=0.02)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class AttentionFusionBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d
        self.gate = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Linear(d * 2, d), nn.ReLU(inplace=True),
                                  nn.Linear(d, d), nn.Sigmoid())
        self.refine = nn.Sequential(nn.Conv2d(d, d, 1, bias=False),
                                    nn.BatchNorm2d(d), nn.ReLU(inplace=True))

    def forward(self, m, t):
        alpha = self.gate(torch.cat([m, t], dim=1)).view(-1, self.d, 1, 1)
        return self.refine(alpha * m + (1 - alpha) * t)


class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.block(x)


class PatchHMT(nn.Module):
    def __init__(self, in_bands, num_classes, d=64):
        super().__init__()
        self.spec_proj = nn.Sequential(nn.Conv2d(in_bands, d, 1, bias=False),
                                       nn.BatchNorm2d(d), nn.ReLU(inplace=True))
        self.mamba1, self.mamba2, self.mamba3 = (MambaBlock(d) for _ in range(3))
        self.pos_enc = PositionalEncoding(d)
        self.trans1, self.trans2, self.trans3 = (TransformerBlock(d) for _ in range(3))
        self.pool = nn.MaxPool2d(2)
        self.fuse1, self.fuse2, self.fuse3 = (AttentionFusionBlock(d) for _ in range(3))
        self.dec1 = ConvBlock(d + d, d)
        self.dec2 = ConvBlock(d + d, 32)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Linear(32, num_classes))

    @staticmethod
    def _to_seq(x):
        B, C, H, W = x.shape
        return x.flatten(2).transpose(1, 2), B, C, H, W

    @staticmethod
    def _to_spatial(seq, B, C, H, W):
        return seq.transpose(1, 2).view(B, C, H, W)

    def forward(self, x):
        x = self.spec_proj(x)
        seq, B, C, H, W = self._to_seq(x)
        m1 = self._to_spatial(self.mamba1(seq), B, C, H, W)
        t1 = self._to_spatial(self.trans1(self.pos_enc(seq)), B, C, H, W)
        f1 = self.fuse1(m1, t1)
        x2 = self.pool(f1)
        seq2, B, C, H2, W2 = self._to_seq(x2)
        m2 = self._to_spatial(self.mamba2(seq2), B, C, H2, W2)
        t2 = self._to_spatial(self.trans2(self.pos_enc(seq2)), B, C, H2, W2)
        f2 = self.fuse2(m2, t2)
        x3 = self.pool(f2)
        seq3, B, C, H3, W3 = self._to_seq(x3)
        m3 = self._to_spatial(self.mamba3(seq3), B, C, H3, W3)
        t3 = self._to_spatial(self.trans3(self.pos_enc(seq3)), B, C, H3, W3)
        f3 = self.fuse3(m3, t3)
        d1 = self.up(f3)
        d1 = self.dec1(torch.cat([d1, f2], dim=1))
        d2 = self.up(d1)
        f1r = F.interpolate(f1, size=d2.shape[2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, f1r], dim=1))
        return self.head(d2)
