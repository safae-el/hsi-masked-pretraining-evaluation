"""Class-aware weighted cross-entropy on labeled pixels only (ignore_index=-1).

Weights: w_c = (N / n_c)^smooth, normalized to mean 1. smooth in [0,1]:
0 = unweighted, 1 = full inverse frequency. WHISPERS-style behaviour around 0.5-1.0.
"""
import numpy as np
import torch
import torch.nn as nn


def class_weights(labels: np.ndarray, train_idx: np.ndarray, num_classes: int,
                  smooth: float = 0.5) -> torch.Tensor:
    flat = labels.ravel()[train_idx]
    counts = np.bincount(flat[flat >= 0], minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    w = (counts.sum() / counts) ** smooth
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


class MaskedCrossEntropy(nn.Module):
    """logits [N,C,H,W] (dense) or [N,C] (patch); target [N,H,W] or [N] with -1 = ignore."""
    def __init__(self, weights: torch.Tensor | None = None,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weights, ignore_index=-1,
                                      label_smoothing=label_smoothing)

    def forward(self, logits, target):
        if not bool((target >= 0).any()):          # no supervised pixel in this batch
            return logits.sum() * 0.0              # zero loss, zero grad, never NaN
        return self.ce(logits, target)
