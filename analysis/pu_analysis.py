"""Pavia University replication (see PREREGISTRATION_paviau.md). Usage:
python analysis/pu_analysis.py results/ssl_paviau.csv results/ssl_paviau_w2.csv results/ssl_paviau_w3.csv"""
import sys
import numpy as np, pandas as pd
from scipy import stats
NB = 200_000
d = pd.concat([pd.read_csv(f).assign(src=f) for f in sys.argv[1:]], ignore_index=True)
d["row"] = np.arange(len(d)); d["tag"] = d.ssl_tag.fillna("").replace("none", "")
d = d[(d.dataset == "paviau") & (d.freeze == 0) & d.enc_lr_mult.notna() & np.isclose(d.enc_lr_mult, 0.1) & d.miou.notna()].copy()
for c in ["miou", "oa", "aa", "kappa"]: d[c] *= 100
def arm(b, tag, col="miou"):
    g = d[(d.budget == b) & (d.tag == tag)].sort_values("row").drop_duplicates("seed", keep="first").set_index("seed")
    return g[col].reindex(range(20))
def tci(x):
    lo, hi = stats.t.interval(0.95, len(x) - 1, x.mean(), stats.sem(x)); return x.mean(), lo, hi, stats.ttest_1samp(x, 0).pvalue
def tost(x, M):
    n = len(x); se = x.std(ddof=1) / np.sqrt(n)
    return max(stats.t.sf((x.mean() + M) / se, n - 1), stats.t.sf((M - x.mean()) / se, n - 1))
def holm(ps):
    o = np.argsort(ps); adj = np.empty(len(ps)); run = 0
    for r, i in enumerate(o): run = max(run, (len(ps) - r) * ps[i]); adj[i] = min(1, run)
    return adj
def power(sd, delta, n):
    df = n - 1; nc = delta / (sd / np.sqrt(n)); tc = stats.t.ppf(0.975, df)
    return 1 - stats.nct.cdf(tc, df, nc) + stats.nct.cdf(-tc, df, nc)
CK = ["paviau_mae_seed0", "paviau_mae_seed99", "paviau_mae_seed7"]
print("== completeness (runs per arm and budget, out of 20)")
for t in [""] + CK:
    print(f"  {t or 'scratch':18s} " + "  ".join(f"b{b}: {arm(b, t).notna().sum():2d}" for b in [2, 3, 5]))
stageA = all(arm(b, t).notna().sum() == 20 for b in [2, 3, 5] for t in ["", CK[0]])
print("stage A complete:", stageA)
if not stageA:
    sys.exit("stage A incomplete: per the plan, nothing is reported yet")
print("\n== primary: pretrained (seed 0) - scratch, matched x0.1, mIoU")
ps = []
for b in [2, 3, 5]:
    S, P = arm(b, ""), arm(b, CK[0]); x = (P - S).values; m, lo, hi, p = tci(x); ps.append(p)
    bs = x[np.random.default_rng([3030, b]).integers(0, 20, (NB, 20))].mean(1)
    print(f"{b}/class scratch {S.mean():.2f}+-{S.std():.2f} pretrained {P.mean():.2f}+-{P.std():.2f} | diff {m:+.2f} [{lo:+.2f},{hi:+.2f}] t p={p:.3f}"
          f" | Wilcoxon p={stats.wilcoxon(x).pvalue:.3f} | boot95 [{np.percentile(bs,2.5):+.2f},{np.percentile(bs,97.5):+.2f}]"
          f" | TOST +-2 p={tost(x,2):.3f} +-3 p={tost(x,3):.3f} | pretrained higher in {(x>0).sum()}/20")
print("Holm-adjusted:", np.round(holm(np.array(ps)), 3))
print("\n== other metrics (pretrained - scratch)")
for b in [2, 3, 5]:
    for c in ["oa", "aa", "kappa"]:
        x = (arm(b, CK[0], c) - arm(b, "", c)).values; m, lo, hi, p = tci(x); f = 100 if c == "kappa" else 1
        print(f"{b}/class {c:5s} {m/f:+.3f} [{lo/f:+.3f},{hi/f:+.3f}] p={p:.4f}")
print("\n== seed sensitivity")
for b in [2, 3, 5]:
    x = (arm(b, CK[0]) - arm(b, "")).values; sd = x.std(ddof=1); x0 = x - x.mean()
    mde = next(v for v in np.arange(0.1, 20, 0.01) if power(sd, v, 20) >= .8)
    e = x0[np.random.default_rng([12345, b, 3]).integers(0, 20, (NB, 3))].mean(1)
    print(f"{b}/class paired SD {sd:.2f} | MDE(20 seeds, 80%) {mde:.1f} | zero-centred 3-seed >= +2: {100*(e>=2).mean():.1f}%")
done = [t for t in CK if all(arm(b, t).notna().sum() == 20 for b in [2, 3, 5])]
print("\n== checkpoints complete:", done)
if len(done) >= 2:
    for b in [2, 3, 5]:
        X = np.array([(arm(b, t) - arm(b, "")).values for t in done])        # checkpoints x seeds
        k, n = X.shape; msb = n * X.mean(1).var(ddof=1); msw = X.var(1, ddof=1).mean()
        s2c = max(0.0, (msb - msw) / n); pF = stats.f.sf(msb / msw, k - 1, k * (n - 1))
        cm = X.mean(1); lo, hi = stats.t.interval(0.95, k - 1, cm.mean(), stats.sem(cm)) if k > 1 else (np.nan, np.nan)
        print(f"{b}/class per-checkpoint diffs " + ", ".join(f"{t[-6:]}: {v:+.2f}" for t, v in zip(done, cm)) +
              f" | mean over checkpoints {cm.mean():+.2f} [{lo:+.2f},{hi:+.2f}] (df={k-1}) | checkpoint SD {np.sqrt(s2c):.2f}"
              f" vs seed SD {np.sqrt(msw):.2f} | F-test p={pF:.3f}")
