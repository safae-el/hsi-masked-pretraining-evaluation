"""Budget splits for Pavia University at 2/3/5 labels per class, seeds 0-19, with the same rule as the
Indian Pines 20-seed splits (hsiseg.data.splits.budget_split). Self-checks before writing anything:
(1) regenerating ip_budget{2,3,5}_seed{0..19} reproduces the existing files exactly;
(2) regenerating the existing paviau_budget5_seed{0,1,2} reproduces them exactly.
Existing files are never overwritten. Afterwards the new splits are checked to be nested (2 in 3 in 5)."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_splits import DATASETS
from hsiseg.data.scenes import load_scene
from hsiseg.data import splits as S

root = Path("~/hsi_data").expanduser(); out = root / "splits"
def same(path, tr, te):
    a, b = S.load_split(path)
    return np.array_equal(np.sort(a), tr) and np.array_equal(np.sort(b), te)

ip = load_scene(root, DATASETS["ip"]); pu = load_scene(root, DATASETS["paviau"])
ok_ip = sum(same(out / f"ip_budget{n}_seed{s}.npz", *S.budget_split(ip.labels, n, s)) for n in (2, 3, 5) for s in range(20))
print(f"check 1: {ok_ip}/60 Indian Pines splits reproduced exactly")
ok_pu = sum(same(out / f"paviau_budget5_seed{s}.npz", *S.budget_split(pu.labels, 5, s)) for s in range(3))
print(f"check 2: {ok_pu}/3 existing Pavia University 5-label splits reproduced exactly")
if ok_ip != 60 or ok_pu != 3:
    sys.exit("STOP: the rule does not reproduce the existing splits; nothing written")
new = 0
for s in range(20):
    for n in (2, 3, 5):
        p = out / f"paviau_budget{n}_seed{s}.npz"
        if not p.exists():
            tr, te = S.budget_split(pu.labels, n, s); S.save_split(p, tr, te, f"budget{n}", s); new += 1
nested = all(set(S.load_split(out / f"paviau_budget2_seed{s}.npz")[0]) <= set(S.load_split(out / f"paviau_budget3_seed{s}.npz")[0])
             <= set(S.load_split(out / f"paviau_budget5_seed{s}.npz")[0]) for s in range(20))
sizes = [len(S.load_split(out / f"paviau_budget{n}_seed0.npz")[0]) for n in (2, 3, 5)]
print(f"written {new} new files; nested 2 in 3 in 5 for all 20 seeds: {nested}; training pixels at seed 0: {sizes} (9 classes)")
