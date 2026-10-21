"""Datasets for the three training modes.

WholeSceneDataset : one item = the full scene (dense training on classic benchmarks)
TileDataset       : random crops with label masks (dense training at scale / WHU-OHS)
PatchDataset      : 9x9 center-pixel patches (WHISPERS baseline reproduction)

All return targets where -1 = ignore (masked out of the loss).
"""
import numpy as np
import torch
from torch.utils.data import Dataset


def _masked_targets(labels: np.ndarray, keep_idx: np.ndarray) -> np.ndarray:
    """Full-size target map with ONLY keep_idx labeled; everything else -1."""
    t = np.full(labels.size, -1, dtype=np.int64)
    t[keep_idx] = labels.ravel()[keep_idx]
    return t.reshape(labels.shape)


class WholeSceneDataset(Dataset):
    def __init__(self, scene, train_idx: np.ndarray):
        self.cube = torch.from_numpy(scene.cube).permute(2, 0, 1).contiguous()  # [B,H,W]
        self.target = torch.from_numpy(_masked_targets(scene.labels, train_idx))

    def __len__(self):
        return 1

    def __getitem__(self, _):
        return self.cube, self.target


class TileDataset(Dataset):
    def __init__(self, scene, train_idx: np.ndarray, tile: int = 128, tiles_per_epoch=None,
                 seed: int = 0, augment: bool = True, coverage: float = 4.0):
        # tiles_per_epoch=None -> scale with scene area so every dataset gets the
        # same per-epoch coverage (`coverage` passes over the scene).
        if tiles_per_epoch is None:
            import math
            _H, _W = scene.labels.shape
            tiles_per_epoch = max(64, math.ceil(coverage * _H * _W / (tile * tile)))
        self.cube = torch.from_numpy(scene.cube).permute(2, 0, 1).contiguous()
        self.target = torch.from_numpy(_masked_targets(scene.labels, train_idx))
        self.tile = tile
        self.n = tiles_per_epoch
        self.rng = np.random.default_rng(seed)
        self.augment = True

    def __len__(self):
        return self.n

    def __getitem__(self, _):
        _, H, W = self.cube.shape
        t = self.tile
        if H <= t or W <= t:  # scene smaller than tile: return whole scene
            return self.cube, self.target
        # resample until the tile contains at least one labelled pixel; empty
        # tiles make the masked cross-entropy average over nothing -> NaN.
        for _try in range(50):
            y = int(self.rng.integers(0, H - t))
            x = int(self.rng.integers(0, W - t))
            cy = self.target[y:y + t, x:x + t]
            if bool((cy >= 0).any()):
                break
        cx = self.cube[:, y:y + t, x:x + t]
        if self.augment:
            if self.rng.random() < 0.5:
                cx, cy = torch.flip(cx, [2]), torch.flip(cy, [1])
            if self.rng.random() < 0.5:
                cx, cy = torch.flip(cx, [1]), torch.flip(cy, [0])
            k = int(self.rng.integers(0, 4))
            if k:
                cx, cy = torch.rot90(cx, k, (1, 2)), torch.rot90(cy, k, (0, 1))
        return cx, cy


class PatchDataset(Dataset):
    """Center-pixel classification patches (mirror-padded), for patch_hmt / baselines."""
    def __init__(self, scene, pixel_idx: np.ndarray, patch: int = 9):
        assert patch % 2 == 1
        self.p = patch
        r = patch // 2
        cube = np.pad(scene.cube, ((r, r), (r, r), (0, 0)), mode="reflect")
        self.cube = torch.from_numpy(cube).permute(2, 0, 1).contiguous()  # [B,H+2r,W+2r]
        H, W = scene.labels.shape
        self.rows, self.cols = np.unravel_index(pixel_idx, (H, W))
        self.y = torch.from_numpy(scene.labels.ravel()[pixel_idx])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        r, c, p = int(self.rows[i]), int(self.cols[i]), self.p
        return self.cube[:, r:r + p, c:c + p], self.y[i]
