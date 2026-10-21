"""Scene loading and normalization. A Scene = cube [H,W,B] float32 + labels [H,W] int64.

Labels follow the benchmark convention: 0 = unlabeled, 1..C = classes.
Internally we convert to: -1 = unlabeled/ignore, 0..C-1 = classes.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io as sio


@dataclass
class Scene:
    name: str
    cube: np.ndarray      # [H, W, B] float32, per-band standardized
    labels: np.ndarray    # [H, W] int64, -1 = ignore, 0..C-1 = classes
    num_classes: int

    @property
    def shape(self):
        return self.cube.shape

    def labeled_indices(self) -> np.ndarray:
        """Flat indices (row-major) of all labeled pixels."""
        return np.flatnonzero(self.labels.ravel() >= 0)


def _load_mat(path: Path, key: str) -> np.ndarray:
    mat = sio.loadmat(str(path))
    if key not in mat:
        # be forgiving about capitalization differences between mirrors
        candidates = [k for k in mat.keys() if not k.startswith("__")]
        if len(candidates) == 1:
            key = candidates[0]
        else:
            raise KeyError(f"{key} not in {path.name}; found {candidates}")
    return mat[key]


def load_scene(root: str | Path, cfg) -> Scene:
    """cfg: hydra dataset config (cube_file, gt_file, cube_key, gt_key, num_classes, name)."""
    root = Path(root).expanduser()
    cube = _load_mat(root / cfg.cube_file, cfg.cube_key).astype(np.float32)
    gt = _load_mat(root / cfg.gt_file, cfg.gt_key).astype(np.int64)

    # per-band z-score over all pixels (scene-wise; simple and standard)
    mean = cube.reshape(-1, cube.shape[-1]).mean(0)
    std = cube.reshape(-1, cube.shape[-1]).std(0) + 1e-8
    cube = (cube - mean) / std

    labels = gt - 1  # 0 (unlabeled) -> -1 ; 1..C -> 0..C-1
    return Scene(name=cfg.name, cube=cube, labels=labels, num_classes=int(cfg.num_classes))
