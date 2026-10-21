"""
hybrid_weighted.py
==================
Hybrid Mamba-Transformer with attention fusion + weighted cross-entropy loss.
Upweights tiny classes (Alfalfa, Grass-pasture-mowed, Oats) so the model
is penalised more for missing them.

Same config as all baselines (EPOCHS=30, same split, same optimizer).

Weight strategy
---------------
- Tiny classes (< 30 train pixels) : weight = 10.0
- Small classes (< 100 train pixels): weight = 3.0
- Normal classes                    : weight = 1.0

Saves to results/
    hybrid_weighted_metrics.npy
    hybrid_weighted_preds_map.npy
    hybrid_weighted_cm.npy
    hybrid_weighted_history.npy
    hybrid_weighted_config.json
    hybrid_weighted_best.pth

Run
---
    cd /mnt/c/Users/Safae/mamba-transformer-fusion
    python models/hybrid_weighted.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from mamba_ssm import Mamba
import json
import random

# ── shared config ─────────────────────────────────────────────────────────────
from config import (
    BASE_DIR, HSI_PATH, GT_PATH, SPLIT_PATH, RESULTS_DIR,
    NUM_CLASSES, IN_BANDS, IMG_SIZE, PATCH_SIZE,
    CLASS_NAMES, SEED, BATCH_SIZE, EPOCHS, LR, WEIGHT_DECAY
)

# ── reproducibility ───────────────────────────────────────────────────────────
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── class weights ─────────────────────────────────────────────────────────────
# Based on training pixel counts from split_seed42.npy:
# Alfalfa=4, Corn-notill=142, Corn-mintill=83, Corn=23,
# Grass-pasture=48, Grass-trees=73, Grass-pasture-mowed=2,
# Hay-windrowed=47, Oats=2, Soybean-notill=97, Soybean-mintill=245,
# Soybean-clean=59, Wheat=20, Woods=126, Buildings=38, Stone=9
#
# Strategy:
#   < 5  train pixels  → weight 15.0  (Alfalfa=4, Grass-pasture-mowed=2, Oats=2)
#   < 25 train pixels  → weight 5.0   (Corn=23, Wheat=20, Stone=9)
#   < 50 train pixels  → weight 2.0   (Grass-pasture=48, Hay-windrowed=47)
#   >= 50 train pixels → weight 1.0

CLASS_WEIGHTS = torch.tensor([
    15.0,  # Alfalfa                      — 4 train px
     1.0,  # Corn-notill                  — 142
     1.0,  # Corn-mintill                 — 83
     5.0,  # Corn                         — 23
     2.0,  # Grass-pasture                — 48
     1.0,  # Grass-trees                  — 73
    15.0,  # Grass-pasture-mowed          — 2 train px
     2.0,  # Hay-windrowed                — 47
    15.0,  # Oats                         — 2 train px
     1.0,  # Soybean-notill               — 97
     1.0,  # Soybean-mintill              — 245
     1.0,  # Soybean-clean                — 59
     5.0,  # Wheat                        — 20
     1.0,  # Woods                        — 126
     2.0,  # Buildings-Grass-Trees-Drives — 38
     5.0,  # Stone-Steel-Towers           — 9
], dtype=torch.float32)

# ── model config ──────────────────────────────────────────────────────────────
D_MODEL       = 64
MAMBA_D_STATE = 16
MAMBA_D_CONV  = 4
MAMBA_EXPAND  = 2
TRANS_N_HEADS = 4
TRANS_DIM_FF  = 256
TRANS_DROPOUT = 0.1


# ── dataset ───────────────────────────────────────────────────────────────────
class HSIPatchDataset(Dataset):
    def __init__(self, hsi, gt, indices, labels, patch_size):
        self.idx    = indices
        self.labels = labels
        pad         = patch_size // 2
        self.padded = np.pad(hsi,
                             ((pad, pad), (pad, pad), (0, 0)),
                             mode='reflect')
        self.pad = pad

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        r, c   = self.idx[i]
        rp, cp = r + self.pad, c + self.pad
        patch  = self.padded[rp-self.pad : rp+self.pad+1,
                              cp-self.pad : cp+self.pad+1, :]
        patch  = torch.tensor(patch.transpose(2, 0, 1), dtype=torch.float32)
        label  = int(self.labels[i]) - 1
        return patch, label


# ── building blocks ───────────────────────────────────────────────────────────
class MambaBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm  = nn.LayerNorm(D_MODEL)
        self.mamba = Mamba(d_model=D_MODEL, d_state=MAMBA_D_STATE,
                           d_conv=MAMBA_D_CONV, expand=MAMBA_EXPAND)

    def forward(self, x):
        return x + self.mamba(self.norm(x))


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(D_MODEL)
        self.attn  = nn.MultiheadAttention(
            embed_dim=D_MODEL, num_heads=TRANS_N_HEADS,
            dropout=TRANS_DROPOUT, batch_first=True)
        self.norm2 = nn.LayerNorm(D_MODEL)
        self.ff    = nn.Sequential(
            nn.Linear(D_MODEL, TRANS_DIM_FF),
            nn.GELU(),
            nn.Dropout(TRANS_DROPOUT),
            nn.Linear(TRANS_DIM_FF, D_MODEL),
            nn.Dropout(TRANS_DROPOUT),
        )

    def forward(self, x):
        n = self.norm1(x)
        a, _ = self.attn(n, n, n)
        x = x + a
        x = x + self.ff(self.norm2(x))
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, max_len=512):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_len, D_MODEL))
        nn.init.trunc_normal_(self.pe, std=0.02)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class AttentionFusionBlock(nn.Module):
    """
    Channel attention gate:
        alpha = sigmoid(gate(concat(mamba, trans)))
        output = alpha * mamba + (1-alpha) * trans
    """
    def __init__(self):
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(D_MODEL * 2, D_MODEL),
            nn.ReLU(inplace=True),
            nn.Linear(D_MODEL, D_MODEL),
            nn.Sigmoid(),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(D_MODEL, D_MODEL, 1, bias=False),
            nn.BatchNorm2d(D_MODEL),
            nn.ReLU(inplace=True),
        )

    def forward(self, mamba_feat, trans_feat):
        combined = torch.cat([mamba_feat, trans_feat], dim=1)
        alpha    = self.gate(combined).view(-1, D_MODEL, 1, 1)
        fused    = alpha * mamba_feat + (1 - alpha) * trans_feat
        return self.refine(fused)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class HSIHybridWeightedNet(nn.Module):
    """
    Same architecture as hybrid_attention.py.
    Difference is in the loss function — weighted cross-entropy.
    """
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.spec_proj = nn.Sequential(
            nn.Conv2d(IN_BANDS, D_MODEL, 1, bias=False),
            nn.BatchNorm2d(D_MODEL),
            nn.ReLU(inplace=True),
        )
        self.mamba1  = MambaBlock()
        self.mamba2  = MambaBlock()
        self.mamba3  = MambaBlock()
        self.pos_enc = PositionalEncoding()
        self.trans1  = TransformerBlock()
        self.trans2  = TransformerBlock()
        self.trans3  = TransformerBlock()
        self.pool    = nn.MaxPool2d(2)
        self.fuse1   = AttentionFusionBlock()
        self.fuse2   = AttentionFusionBlock()
        self.fuse3   = AttentionFusionBlock()
        self.dec1    = ConvBlock(D_MODEL + D_MODEL, D_MODEL)
        self.dec2    = ConvBlock(D_MODEL + D_MODEL, 32)
        self.up      = nn.Upsample(scale_factor=2, mode='bilinear',
                                    align_corners=False)
        self.head    = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, num_classes),
        )

    def _to_seq(self, x):
        B, C, H, W = x.shape
        return x.flatten(2).transpose(1, 2), B, C, H, W

    def _to_spatial(self, seq, B, C, H, W):
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
        f1_r = F.interpolate(f1, size=d2.shape[2:],
                              mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([d2, f1_r], dim=1))
        return self.head(d2)


# ── training helpers ──────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for patches, labels in loader:
        patches, labels = patches.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(patches)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += len(labels)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    for patches, labels in loader:
        patches = patches.to(DEVICE)
        preds   = model(patches).argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
    return np.array(all_preds), np.array(all_labels)


def compute_metrics(preds, labels):
    cm    = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))
    oa    = float(cm.diagonal().sum() / cm.sum())
    per_class_acc = []
    for c in range(NUM_CLASSES):
        if cm[c].sum() > 0:
            per_class_acc.append(float(cm[c, c] / cm[c].sum()))
        else:
            per_class_acc.append(0.0)
    aa    = float(np.mean(per_class_acc))
    kappa = float(cohen_kappa_score(labels, preds))
    ious  = []
    for c in range(NUM_CLASSES):
        tp    = cm[c, c]
        fp    = cm[:, c].sum() - tp
        fn    = cm[c, :].sum() - tp
        denom = tp + fp + fn
        ious.append(float(tp / denom) if denom > 0 else 0.0)
    miou = float(np.mean(ious))
    return dict(OA=oa, AA=aa, Kappa=kappa, mIoU=miou,
                per_class_acc=per_class_acc, class_ious=ious, cm=cm)


def reconstruct_map(preds, test_idx, gt_shape):
    pred_map = np.zeros(gt_shape, dtype=np.int32)
    for (r, c), p in zip(test_idx, preds):
        pred_map[r, c] = p + 1
    return pred_map


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Model  : Hybrid Mamba-Transformer (Weighted Loss)")
    print("Fusion : Channel attention gate")
    print("Loss   : Weighted cross-entropy — upweights tiny classes")
    print(f"Epochs : {EPOCHS}  (same as all baselines)")
    print("Dataset: Indian Pines | Split: split_seed42.npy")
    print("=" * 60)
    print(f"Device : {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU    : {torch.cuda.get_device_name(0)}")

    print("\nClass weights:")
    for name, w in zip(CLASS_NAMES, CLASS_WEIGHTS.tolist()):
        marker = " ← upweighted" if w > 1.0 else ""
        print(f"  {name:<35} {w:.1f}{marker}")

    print("\n[1/4] Loading data and split ...")
    hsi_data = scipy.io.loadmat(HSI_PATH)
    gt_data  = scipy.io.loadmat(GT_PATH)
    hsi_key  = [k for k in hsi_data if not k.startswith('_')][0]
    gt_key   = [k for k in gt_data  if not k.startswith('_')][0]

    hsi = hsi_data[hsi_key].astype(np.float32)
    gt  = gt_data[gt_key].astype(np.int32)

    hsi_min = hsi.min(axis=(0, 1), keepdims=True)
    hsi_max = hsi.max(axis=(0, 1), keepdims=True)
    hsi     = (hsi - hsi_min) / (hsi_max - hsi_min + 1e-8)

    split        = np.load(SPLIT_PATH, allow_pickle=True).item()
    train_idx    = split['train_idx']
    test_idx     = split['test_idx']
    train_labels = split['train_labels']
    test_labels  = split['test_labels']

    print(f"  Train pixels : {len(train_idx)}")
    print(f"  Test pixels  : {len(test_idx)}")

    train_ds = HSIPatchDataset(hsi, gt, train_idx, train_labels, PATCH_SIZE)
    test_ds  = HSIPatchDataset(hsi, gt, test_idx,  test_labels,  PATCH_SIZE)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=4, pin_memory=True)

    print("\n[2/4] Building model ...")
    model   = HSIHybridWeightedNet().to(DEVICE)
    n_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters : {n_param:,}")

    # weighted cross-entropy — key difference from other models
    criterion = nn.CrossEntropyLoss(
        weight=CLASS_WEIGHTS.to(DEVICE)
    )
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)

    print(f"\n[3/4] Training for {EPOCHS} epochs ...")
    history   = []
    best_oa   = 0.0
    best_path = os.path.join(RESULTS_DIR, "hybrid_weighted_best.pth")

    for epoch in range(1, EPOCHS + 1):
        loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            preds, labels = evaluate(model, test_loader)
            test_oa = float((preds == labels).mean())
            print(f"  Epoch {epoch:3d}/{EPOCHS} | "
                  f"loss {loss:.4f} | "
                  f"train acc {train_acc:.3f} | "
                  f"test OA {test_oa:.3f}")
            history.append({"epoch": epoch, "loss": round(loss, 6),
                            "train_acc": round(train_acc, 6),
                            "test_OA": round(test_oa, 6)})
            if test_oa > best_oa:
                best_oa = test_oa
                torch.save(model.state_dict(), best_path)
        else:
            history.append({"epoch": epoch, "loss": round(loss, 6),
                            "train_acc": round(train_acc, 6),
                            "test_OA": None})

    print("\n[4/4] Final evaluation ...")
    model.load_state_dict(torch.load(best_path))
    preds, labels = evaluate(model, test_loader)
    m = compute_metrics(preds, labels)

    print("\n" + "=" * 60)
    print("RESULTS — Hybrid Weighted Loss on Indian Pines")
    print("=" * 60)
    print(f"  OA    : {m['OA']   * 100:.2f}%")
    print(f"  AA    : {m['AA']   * 100:.2f}%")
    print(f"  Kappa : {m['Kappa']:.4f}")
    print(f"  mIoU  : {m['mIoU'] * 100:.2f}%")
    print("=" * 60)
    print(f"\nPer-class IoU:")
    for name, iou, w in zip(CLASS_NAMES, m['class_ious'],
                             CLASS_WEIGHTS.tolist()):
        bar    = "█" * int(iou * 40)
        marker = " ←" if w > 1.0 else ""
        print(f"  {name:<35} {iou*100:5.1f}%  {bar}{marker}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.save(os.path.join(RESULTS_DIR, "hybrid_weighted_metrics.npy"),
            {k: v for k, v in m.items() if k != 'cm'}, allow_pickle=True)
    np.save(os.path.join(RESULTS_DIR, "hybrid_weighted_cm.npy"), m['cm'])
    pred_map = reconstruct_map(preds, test_idx, gt.shape)
    np.save(os.path.join(RESULTS_DIR, "hybrid_weighted_preds_map.npy"),
            pred_map)
    np.save(os.path.join(RESULTS_DIR, "hybrid_weighted_preds_flat.npy"),
            {"preds": preds, "labels": labels}, allow_pickle=True)
    np.save(os.path.join(RESULTS_DIR, "hybrid_weighted_history.npy"),
            history, allow_pickle=True)

    config_out = {
        "model":         "Hybrid Mamba-Transformer, weighted loss",
        "fusion":        "channel attention gate",
        "loss":          "weighted cross-entropy",
        "class_weights": CLASS_WEIGHTS.tolist(),
        "epochs":        EPOCHS,
        "dataset":       "Indian Pines (corrected)",
        "split_file":    "split_seed42.npy",
        "seed":          SEED,
        "patch_size":    PATCH_SIZE,
        "in_bands":      IN_BANDS,
        "num_classes":   NUM_CLASSES,
        "batch_size":    BATCH_SIZE,
        "lr":            LR,
        "weight_decay":  WEIGHT_DECAY,
        "optimizer":     "Adam",
        "scheduler":     "CosineAnnealingLR",
        "d_model":       D_MODEL,
        "parameters":    n_param,
        "best_OA":       round(m['OA'],    6),
        "best_AA":       round(m['AA'],    6),
        "best_Kappa":    round(m['Kappa'], 6),
        "best_mIoU":     round(m['mIoU'],  6),
    }
    with open(os.path.join(RESULTS_DIR, "hybrid_weighted_config.json"),
              "w") as f:
        json.dump(config_out, f, indent=2)

    print(f"\nArtifacts saved to {RESULTS_DIR}/")
    print("\n" + "=" * 60)
    print("FULL COMPARISON — all models, 30 epochs, same split")
    print("=" * 60)
    print(f"  U-Net              → OA 96.30% | AA 78.76% | mIoU 75.40%")
    print(f"  Transformer        → OA 96.98% | AA 78.72% | mIoU 76.44%")
    print(f"  Mamba              → OA 97.90% | AA 80.13% | mIoU 78.47%")
    print(f"  Hybrid concat      → OA 97.44% | AA 79.68% | mIoU 77.40%")
    print(f"  Hybrid attention   → OA 97.67% | AA 80.07% | mIoU 78.10%")
    print(f"  Hybrid weighted    → OA {m['OA']*100:.2f}% | "
          f"AA {m['AA']*100:.2f}% | "
          f"mIoU {m['mIoU']*100:.2f}%")
    delta_oa   = (m['OA']   - 0.9790) * 100
    delta_aa   = (m['AA']   - 0.8013) * 100
    delta_miou = (m['mIoU'] - 0.7847) * 100
    print(f"\n  vs Mamba (strongest baseline):")
    print(f"  OA {delta_oa:+.2f}% | AA {delta_aa:+.2f}% | "
          f"mIoU {delta_miou:+.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()