"""Generate + FREEZE all split artifacts. Run once; artifacts are then versioned.

Produces per dataset:
  <name>_standard10_seed{0,1,2}.npz   (10% stratified, benchmark protocol)
  <name>_disjoint.npz                 (spatially disjoint blocks + guard band)
  <name>_budget{5,10,20,30}_seed{0,1,2}.npz
Prints class coverage for every artifact — check that no class has 0 train pixels
(disjoint split may need a different seed; the script retries automatically).
"""
import argparse
from pathlib import Path
from types import SimpleNamespace

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hsiseg.data.scenes import load_scene
from hsiseg.data import splits as S

DATASETS = {
    "ip": SimpleNamespace(name="ip", cube_file="Indian_pines_corrected.mat",
                          gt_file="Indian_pines_gt.mat", cube_key="indian_pines_corrected",
                          gt_key="indian_pines_gt", num_classes=16),
    "paviau": SimpleNamespace(name="paviau", cube_file="PaviaU.mat", gt_file="PaviaU_gt.mat",
                              cube_key="paviaU", gt_key="paviaU_gt", num_classes=9),
    "salinas": SimpleNamespace(name="salinas", cube_file="Salinas_corrected.mat",
                               gt_file="Salinas_gt.mat", cube_key="salinas_corrected",
                               gt_key="salinas_gt", num_classes=16),
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="~/hsi_data")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    root = Path(args.root).expanduser()
    out = Path(args.out).expanduser() if args.out else root / "splits"

    for key, cfg in DATASETS.items():
        scene = load_scene(root, cfg)
        for seed in (0, 1, 2):
            tr, te = S.stratified_split(scene.labels, 0.10, seed)
            S.save_split(out / f"{key}_standard10_seed{seed}.npz", tr, te, "standard10", seed)
            for n in (5, 10, 20, 30):
                tr, te = S.budget_split(scene.labels, n, seed)
                S.save_split(out / f"{key}_budget{n}_seed{seed}.npz", tr, te, f"budget{n}", seed)
        # disjoint: retry seeds until every class is covered in train
        for seed in range(20):
            tr, te = S.disjoint_block_split(scene.labels, block=16, train_frac=0.3,
                                            buffer=4, seed=seed)
            cov = S.class_coverage(scene.labels, tr)
            if all(v > 0 for v in cov.values()):
                S.save_split(out / f"{key}_disjoint.npz", tr, te, "disjoint", seed)
                print(f"{key} disjoint seed={seed} coverage={cov}")
                break
        else:
            print(f"WARNING: {key} disjoint split could not cover all classes; "
                  f"tune block/train_frac")
        print(f"{key}: artifacts written to {out}")
