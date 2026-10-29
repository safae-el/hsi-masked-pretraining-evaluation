"""Pavia University seed-count sensitivity for checkpoint 0 and for the checkpoint average (Table 1).

Usage: python analysis/pu_avg_seed_sensitivity.py results/ssl_paviau.csv results/ssl_paviau_w2.csv results/ssl_paviau_w3.csv
Same selection rules, random-number seeds and power calculation as pu_analysis.py. For the checkpoint
average, the three pretrained outcomes are averaged within each downstream seed before the shared
scratch outcome is subtracted. The checkpoint-0 lines reproduce the values reported by pu_analysis.py.
"""
import sys
import numpy as np, pandas as pd
from scipy import stats

NB = 200_000
d = pd.concat([pd.read_csv(f).assign(src=f) for f in sys.argv[1:]], ignore_index=True)
d["row"] = np.arange(len(d)); d["tag"] = d.ssl_tag.fillna("").replace("none", "")
d = d[(d.dataset == "paviau") & (d.freeze == 0) & d.enc_lr_mult.notna() & np.isclose(d.enc_lr_mult, 0.1) & d.miou.notna()].copy()
d["miou"] *= 100


def arm(b, tag):
    g = d[(d.budget == b) & (d.tag == tag)].sort_values("row").drop_duplicates("seed", keep="first").set_index("seed")
    return g["miou"].reindex(range(20))


def power(sd, delta, n):
    df = n - 1; nc = delta / (sd / np.sqrt(n)); tc = stats.t.ppf(0.975, df)
    return 1 - stats.nct.cdf(tc, df, nc) + stats.nct.cdf(-tc, df, nc)


CK = ["paviau_mae_seed0", "paviau_mae_seed99", "paviau_mae_seed7"]
rows = [("checkpoint 0", lambda b: (arm(b, CK[0]) - arm(b, "")).values),
        ("checkpoint average", lambda b: np.mean([arm(b, t).values for t in CK], 0) - arm(b, "").values)]
for label, diff in rows:
    for b in [2, 3, 5]:
        x = diff(b)
        assert not np.isnan(x).any(), f"missing runs for {label}, {b} labels per class"
        m = x.mean(); lo, hi = stats.t.interval(0.95, 19, m, stats.sem(x)); sd = x.std(ddof=1)
        mde = next(v for v in np.arange(0.1, 20, 0.01) if power(sd, v, 20) >= .8)
        e = (x - m)[np.random.default_rng([12345, b, 3]).integers(0, 20, (NB, 3))].mean(1)
        print(f"{label:18s} {b}/class diff {m:+.2f} [{lo:+.2f},{hi:+.2f}] | paired SD {sd:.2f} | "
              f"MDE(20 seeds, 80%) {mde:.1f} | zero-centred 3-seed >= +2: {100 * (e >= 2).mean():.1f}%")
