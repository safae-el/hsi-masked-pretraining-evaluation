"""Split generation. Splits are FROZEN .npz artifacts — generated once by scripts/make_splits.py,
then loaded by every model. Never regenerate silently.

Artifact format (.npz):
    train_idx : flat row-major indices of training pixels
    test_idx  : flat indices of test pixels
    kind      : 'standard10' | 'disjoint' | 'budgetN'
    seed      : int
"""
from pathlib import Path

import numpy as np


def stratified_split(labels: np.ndarray, fraction: float, seed: int):
    """Per-class random sampling of `fraction` of labeled pixels (WHISPERS/benchmark protocol).
    Guarantees >=1 train and >=1 test pixel per class."""
    rng = np.random.default_rng(seed)
    flat = labels.ravel()
    train, test = [], []
    for c in np.unique(flat[flat >= 0]):
        idx = np.flatnonzero(flat == c)
        rng.shuffle(idx)
        n_train = int(round(len(idx) * fraction))
        n_train = max(1, min(n_train, len(idx) - 1))
        train.append(idx[:n_train])
        test.append(idx[n_train:])
    return np.sort(np.concatenate(train)), np.sort(np.concatenate(test))


def budget_split(labels: np.ndarray, n_per_class: int, seed: int):
    """Exactly n labeled pixels per class for training (fewer if the class is smaller);
    everything else labeled goes to test."""
    rng = np.random.default_rng(seed)
    flat = labels.ravel()
    train, test = [], []
    for c in np.unique(flat[flat >= 0]):
        idx = np.flatnonzero(flat == c)
        rng.shuffle(idx)
        n = max(1, min(n_per_class, len(idx) - 1))
        train.append(idx[:n])
        test.append(idx[n:])
    return np.sort(np.concatenate(train)), np.sort(np.concatenate(test))


def disjoint_block_split(labels: np.ndarray, block: int = 16, train_frac: float = 0.3,
                         buffer: int = 4, seed: int = 0):
    """Spatially disjoint split: the scene is partitioned into block x block cells,
    cells are randomly assigned to train/test, and a `buffer`-pixel guard band around
    train cells is REMOVED from the test set (kills receptive-field leakage).

    Returns (train_idx, test_idx). Some classes may be absent from train at small
    block sizes — make_splits.py retries with different seeds and reports coverage.
    """
    rng = np.random.default_rng(seed)
    H, W = labels.shape
    bh, bw = (H + block - 1) // block, (W + block - 1) // block
    cell_train = rng.random((bh, bw)) < train_frac

    cell_map = np.zeros((H, W), dtype=bool)
    for i in range(bh):
        for j in range(bw):
            if cell_train[i, j]:
                cell_map[i * block:(i + 1) * block, j * block:(j + 1) * block] = True

    labeled = labels >= 0
    train_mask = cell_map & labeled

    # guard band: dilate train region by `buffer` pixels (square kernel = Chebyshev
    # distance, so diagonal neighbours are excluded too); exclude from test
    from scipy.ndimage import binary_dilation
    guard = binary_dilation(cell_map, structure=np.ones((2 * buffer + 1, 2 * buffer + 1)))
    test_mask = (~guard) & labeled

    return np.flatnonzero(train_mask.ravel()), np.flatnonzero(test_mask.ravel())


def class_coverage(labels: np.ndarray, train_idx: np.ndarray) -> dict:
    flat = labels.ravel()
    classes = np.unique(flat[flat >= 0])
    counts = {int(c): int((flat[train_idx] == c).sum()) for c in classes}
    return counts


def save_split(path: Path, train_idx, test_idx, kind: str, seed: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, train_idx=train_idx, test_idx=test_idx, kind=kind, seed=seed)


def load_split(path: Path):
    z = np.load(path, allow_pickle=False)
    return z["train_idx"], z["test_idx"]
