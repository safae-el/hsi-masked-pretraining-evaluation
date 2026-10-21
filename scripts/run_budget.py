"""Label-budget fine-tuning: DenseHMT-tiled with/without SSL pretraining.

  python scripts/run_budget.py --dataset ip --budget 5
  python scripts/run_budget.py --dataset ip --budget 5 --ssl ~/hsi_runs/ssl/ip_mae_seed0.pt
"""
import argparse, sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from hsiseg.data.scenes import load_scene
from hsiseg.data.splits import load_split
from hsiseg.data.samplers import TileDataset
from hsiseg.models.dense_hmt import DenseHMT
from hsiseg.losses import MaskedCrossEntropy, class_weights
from hsiseg.ssl.transfer import load_encoder
from hsiseg.engine import train_dense, write_csv_row

DS = {"ip": ("Indian_pines_corrected.mat", "Indian_pines_gt.mat",
             "indian_pines_corrected", "indian_pines_gt", 16),
      "paviau": ("PaviaU.mat", "PaviaU_gt.mat", "paviaU", "paviaU_gt", 9),
      "salinas": ("Salinas_corrected.mat", "Salinas_gt.mat",
                  "salinas_corrected", "salinas_gt", 16)}

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True, choices=DS)
ap.add_argument("--budget", type=int, required=True, choices=[2, 3, 5, 10, 20, 30])
ap.add_argument("--ssl", default=None, help="path to pretrained encoder .pt")
ap.add_argument("--freeze", action="store_true")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--epochs", type=int, default=100)
ap.add_argument("--tile", type=int, default=64)
ap.add_argument("--enc_lr_mult", type=float, default=1.0)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--split", default="budget", help="split-file prefix: budget | dbudget")
ap.add_argument("--out", default="results/budgets.csv")
ap.add_argument("--test_budget", type=int, default=0,
                help="evaluate on the test split of this budget (common test set)")
ap.add_argument("--label_smoothing", type=float, default=0.1,
                help="0.1 = value used by the August 2026 budget runs")
a = ap.parse_args()
import subprocess
_commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()

torch.manual_seed(a.seed); np.random.seed(a.seed)
cf = DS[a.dataset]
scene = load_scene(Path.home() / "hsi_data",
                   SimpleNamespace(name=a.dataset, cube_file=cf[0], gt_file=cf[1],
                                   cube_key=cf[2], gt_key=cf[3], num_classes=cf[4]))
tr, te = load_split(Path.home() / "hsi_data" / "splits" /
                    f"{a.dataset}_{a.split}{a.budget}_seed{a.seed}.npz")
if a.test_budget:
    _, te = load_split(Path.home() / "hsi_data" / "splits" /
                       f"{a.dataset}_{a.split}{a.test_budget}_seed{a.seed}.npz")
    assert len(np.intersect1d(tr, te)) == 0, "train/test overlap"
print(f"{a.dataset} budget={a.budget}/class: {len(tr)} train px, {len(te)} test px, "
      f"ssl={'yes' if a.ssl else 'no'}")

mc = OmegaConf.load("configs/model/dense_hmt.yaml")
model = DenseHMT(scene.cube.shape[-1], scene.num_classes, mc)
if a.ssl:
    load_encoder(model, Path(a.ssl).expanduser(), freeze=a.freeze)
elif a.freeze:  # random-encoder probe: freeze exactly what load_encoder freezes
    for n, p in model.named_parameters():
        if not n.startswith("head."):
            p.requires_grad = False
    print("random encoder frozen; training head only")

print(f"trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
tcfg = OmegaConf.load("configs/train/supervised.yaml")
tcfg.mode = "tiles"; tcfg.tile_size = a.tile; tcfg.batch_size = a.batch
tcfg.epochs = a.epochs; tcfg.resume = False; tcfg.enc_lr_mult = a.enc_lr_mult

ds = TileDataset(scene, tr, a.tile, seed=a.seed)
loader = DataLoader(ds, batch_size=a.batch, shuffle=False)
w = class_weights(scene.labels, tr, scene.num_classes, tcfg.weight_smooth)
crit = MaskedCrossEntropy(w, a.label_smoothing)

run_dir = Path.home() / "hsi_runs" / f"budget_{a.dataset}_{a.budget}_ssl{int(bool(a.ssl))}_s{a.seed}"
out = train_dense(model, loader, scene, te, crit, tcfg, run_dir, "cuda")
fin = out["final"]
print(f"FINAL budget={a.budget} ssl={bool(a.ssl)}: mIoU {fin['miou']*100:.2f}  "
      f"OA {fin['oa']*100:.2f}  AA {fin['aa']*100:.2f}  "
      f"worst-class IoU {min(fin['per_class_iou'])*100:.1f}")
write_csv_row(Path(a.out),
              {"dataset": a.dataset, "budget": a.budget, "ssl": int(bool(a.ssl)),
               "freeze": int(a.freeze), "enc_lr_mult": a.enc_lr_mult,
               "ssl_tag": (Path(a.ssl).stem if a.ssl else "none"), "seed": a.seed, "mode": "tiles", "split": a.split, "commit": _commit, "label_smoothing": a.label_smoothing, "test_budget": a.test_budget,
               **{k: v for k, v in fin.items() if k != "per_class_iou"},
               "min_class_iou": min(fin["per_class_iou"]),
               "zero_iou_classes": sum(1 for x in fin["per_class_iou"] if x == 0.0),
               "per_class_iou": ";".join(f"{x:.4f}" for x in fin["per_class_iou"])})
