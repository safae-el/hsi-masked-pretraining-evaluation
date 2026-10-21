"""Multi-scene self-supervised pretraining (no labels used).
  python scripts/pretrain_ssl_multi.py --epochs 300
Writes ~/hsi_runs/ssl/multi_<scene>_mae_seed<N>.pt per scene: that scene's stem
plus the shared backbone, in DenseHMT key format so load_encoder works unchanged.
"""
import argparse, sys, time
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from omegaconf import OmegaConf
from hsiseg.data.scenes import load_scene
from hsiseg.ssl.masking import corrupt
from hsiseg.ssl.multi_mae import MultiSceneMAE

DS = {"ip": ("Indian_pines_corrected.mat", "Indian_pines_gt.mat",
             "indian_pines_corrected", "indian_pines_gt", 16),
      "paviau": ("PaviaU.mat", "PaviaU_gt.mat", "paviaU", "paviaU_gt", 9),
      "salinas": ("Salinas_corrected.mat", "Salinas_gt.mat",
                  "salinas_corrected", "salinas_gt", 16)}
ap = argparse.ArgumentParser()
ap.add_argument("--scenes", default="ip,paviau,salinas")
ap.add_argument("--epochs", type=int, default=300)
ap.add_argument("--tile", type=int, default=64)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--tiles_per_epoch", type=int, default=64, help="per scene")
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--spatial_ratio", type=float, default=0.6)
ap.add_argument("--spectral_ratio", type=float, default=0.3)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
torch.manual_seed(a.seed); np.random.seed(a.seed)
DEV = "cuda"; torch.set_float32_matmul_precision("high")

scenes = [s.strip() for s in a.scenes.split(",")]
cubes, total = {}, 0
for name in scenes:
    cf = DS[name]
    sc = load_scene(Path.home() / "hsi_data",
                    SimpleNamespace(name=name, cube_file=cf[0], gt_file=cf[1],
                                    cube_key=cf[2], gt_key=cf[3], num_classes=cf[4]))
    cubes[name] = torch.from_numpy(sc.cube).permute(2, 0, 1).contiguous()
    B, H, W = cubes[name].shape; total += H * W
    print(f"{name}: {B} bands, {H}x{W} = {H*W:,} px")
print(f"TOTAL unlabeled pixels: {total:,}")

bands = {k: v.shape[0] for k, v in cubes.items()}
mc = OmegaConf.load("configs/model/dense_hmt.yaml")
model = MultiSceneMAE(bands, mc).to(DEV)
print(f"params: {sum(p.numel() for p in model.parameters()):,}")
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
rng = np.random.default_rng(a.seed)
steps = max(1, a.tiles_per_epoch // a.batch)
t0 = time.time()
for ep in range(1, a.epochs + 1):
    model.train(); per = {k: [] for k in scenes}
    order = list(scenes); rng.shuffle(order)
    for name in order:
        cube = cubes[name]; B, H, W = cube.shape
        for _ in range(steps):
            tiles = []
            for _ in range(a.batch):
                y = int(rng.integers(0, max(H - a.tile, 1)))
                x = int(rng.integers(0, max(W - a.tile, 1)))
                tiles.append(cube[:, y:y + a.tile, x:x + a.tile])
            clean = torch.stack(tiles).to(DEV)
            noisy, m = corrupt(clean, a.spatial_ratio, a.spectral_ratio)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = model(noisy, name)
                loss = (((pred - clean) ** 2) * m).sum() / (m.sum() * B + 1e-8)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            per[name].append(loss.item())
    sched.step()
    if ep % 25 == 0 or ep == 1:
        msg = "  ".join(f"{k} {np.mean(v):.4f}" for k, v in per.items())
        print(f"ep {ep:4d}/{a.epochs}  {msg}   {time.time()-t0:.0f}s")

out = Path.home() / "hsi_runs" / "ssl"; out.mkdir(parents=True, exist_ok=True)
for name in scenes:
    p = out / f"multi_{name}_mae_seed{a.seed}.pt"
    torch.save({"encoder": model.encoder_state_dict(name),
                "args": vars(a), "scenes": scenes}, p)
    print(f"saved -> {p}")
print(f"({time.time()-t0:.0f}s total)")
