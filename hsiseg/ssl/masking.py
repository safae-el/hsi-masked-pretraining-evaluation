"""Masked spectral-spatial corruption for MAE-style pretraining."""
import torch


def mask_spatial_blocks(x, ratio=0.6, block=8, generator=None):
    """Zero out random block x block spatial regions. x: [N,B,H,W].
    Returns (corrupted, mask) where mask is [N,1,H,W], 1 = masked."""
    N, _, H, W = x.shape
    gh, gw = (H + block - 1) // block, (W + block - 1) // block
    keep = torch.rand(N, 1, gh, gw, device=x.device, generator=generator) < ratio
    m = keep.float().repeat_interleave(block, 2).repeat_interleave(block, 3)[:, :, :H, :W]
    return x * (1 - m), m


def mask_spectral_groups(x, ratio=0.3, n_groups=10, generator=None):
    """Zero out random contiguous band groups. x: [N,B,H,W]."""
    N, B, _, _ = x.shape
    g = torch.rand(N, n_groups, device=x.device, generator=generator) < ratio
    m = g.float().repeat_interleave((B + n_groups - 1) // n_groups, 1)[:, :B]
    m = m[:, :, None, None]
    return x * (1 - m), m


def corrupt(x, spatial_ratio=0.6, spectral_ratio=0.3, block=8, generator=None):
    """Both corruptions. Reconstruction loss is computed on spatially masked pixels."""
    xc, sm = mask_spatial_blocks(x, spatial_ratio, block, generator)
    xc, _ = mask_spectral_groups(xc, spectral_ratio, generator=generator)
    return xc, sm
