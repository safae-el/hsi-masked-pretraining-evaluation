"""Indian Pines analyses: robustness checks, other metrics, class-wise IoU, seed-count bootstrap and power,
label equivalents, the preliminary backbone-multiplier exploration, and the 2x2 initialization-by-schedule control.

Usage: python revision_analysis.py --budgets budgets.csv --common ssl_phase0c.csv --x1 ssl_pretrained_x1.csv
Selection rules are the same as make_figures.py (explicit enc_lr_mult, keep-first duplicates,
ssl_tag NaN/'none' = scratch or in-domain, common test set = test_budget == 5 in ssl_phase0c.csv).
All resampling uses fixed seeds, so the output is identical on every run.
"""
import argparse
import numpy as np, pandas as pd
from scipy import stats

ap = argparse.ArgumentParser()
ap.add_argument("--budgets", default="budgets.csv")
ap.add_argument("--common", default="ssl_phase0c.csv")
ap.add_argument("--x1", default="ssl_pretrained_x1.csv")
a = ap.parse_args()
NB = 200_000
NAMES = ["Alfalfa", "Corn-notill", "Corn-mintill", "Corn", "Grass-pasture", "Grass-trees", "Grass-pasture-mowed",
         "Hay-windrowed", "Oats", "Soybean-notill", "Soybean-mintill", "Soybean-clean", "Wheat", "Woods",
         "Buildings-Grass-Trees-Drives", "Stone-Steel-Towers"]
SIZES = [46, 1428, 830, 237, 483, 730, 28, 478, 20, 972, 2455, 593, 205, 1265, 386, 93]

def load(path):
    d = pd.read_csv(path); d["row"] = np.arange(len(d))
    d["tag"] = d.ssl_tag.fillna("").replace("none", "")
    if "test_budget" not in d: d["test_budget"] = 0
    d["test_budget"] = d.test_budget.fillna(0)
    d = d[(d.freeze == 0) & d.enc_lr_mult.notna() & d.miou.notna()].copy()
    for c in ["miou", "oa", "aa", "kappa"]: d[c] = d[c] * 100
    return d
B, C = load(a.budgets), load(a.common)
def arm(b, ssl, enc, tag="", src=None, tb=0, col="miou"):
    src = B if src is None else src
    g = src[(src.budget == b) & (src.ssl == ssl) & np.isclose(src.enc_lr_mult, enc) & (src.tag == tag)
            & (src.test_budget == tb)].sort_values("row").drop_duplicates("seed", keep="first").set_index("seed")
    g = g.loc[[s for s in range(20) if s in g.index]]
    return g if col is None else g[col]
def tci(d):
    lo, hi = stats.t.interval(0.95, len(d) - 1, d.mean(), stats.sem(d)); return d.mean(), lo, hi, stats.ttest_1samp(d, 0).pvalue
def boot_means(d, seed, n):
    rng = np.random.default_rng(seed); return d[rng.integers(0, len(d), (NB, n))].mean(1)
def power(sd, delta, n):
    df = n - 1; nc = delta / (sd / np.sqrt(n)); tc = stats.t.ppf(0.975, df)
    return 1 - stats.nct.cdf(tc, df, nc) + stats.nct.cdf(-tc, df, nc)

print("== Robustness of paired comparisons (Secs. 3.1, 3.2)")
comps = {"matched": lambda b: arm(b, 1, .1) - arm(b, 0, .1), "apparent": lambda b: arm(b, 1, .1) - arm(b, 0, 1.0),
         "schedule": lambda b: arm(b, 0, .1) - arm(b, 0, 1.0)}
for k, (name, f) in enumerate(comps.items()):
    for b in [2, 3, 5]:
        d = f(b).dropna().values; m, lo, hi, p = tci(d)
        rng = np.random.default_rng([2026, k, b])
        flip = (np.abs((rng.choice([-1, 1], (NB, len(d))) * d).mean(1)) >= abs(d.mean()) - 1e-12).mean()
        bs = boot_means(d, [2027, k, b], len(d)); l95, h95 = np.percentile(bs, [2.5, 97.5]); l90, h90 = np.percentile(bs, [5, 95])
        print(f"{name:9s} {b}/class: {m:+.2f} [{lo:+.2f},{hi:+.2f}] t p={p:.3f} | Wilcoxon p={stats.wilcoxon(d).pvalue:.3f}"
              f" | sign-flip p={flip:.3f} | boot95 [{l95:+.2f},{h95:+.2f}] | boot90 [{l90:+.2f},{h90:+.2f}]")

print("\n== Other metrics, matched in-domain - scratch (Table metrics)")
for b in [2, 3, 5]:
    S, M = arm(b, 0, .1, col=None), arm(b, 1, .1, col=None)
    for c in ["oa", "aa", "kappa"]:
        m, lo, hi, p = tci((M[c] - S[c]).values); f = 100 if c == "kappa" else 1
        print(f"{b}/class {c:5s}: {m/f:+.3f} [{lo/f:+.3f},{hi/f:+.3f}] p={p:.4f}")
    print(f"{b}/class AA default - matched scratch: {(arm(b,0,1.0,col=None).aa - S.aa).mean():+.2f};"
          f" OA: {(arm(b,0,1.0,col=None).oa - S.oa).mean():+.2f}")

print("\n== Class-wise IoU (Table perclass) and 13-class mIoU")
pc = lambda g: np.array([[float(x) for x in s.split(";")] for s in g.per_class_iou]) * 100
for b in [2, 3, 5]:
    ps, pm = pc(arm(b, 0, .1, col=None)), pc(arm(b, 1, .1, col=None)); d = pm - ps
    print(f"{b}/class Alfalfa {d[:,0].mean():+.2f} (contribution {d[:,0].mean()/16:+.2f}), Hay-windrowed {d[:,7].mean():+.2f}")
    keep = [i for i in range(16) if i not in (0, 6, 8)]
    m, lo, hi, p = tci(pm[:, keep].mean(1) - ps[:, keep].mean(1))
    print(f"   13-class mIoU difference {m:+.2f} [{lo:+.2f},{hi:+.2f}] p={p:.2f}")

print("\n== Seed count (Sec. 3.3, Fig. 3, Table power): bootstrap with seed [12345, budget, n]")
for b in [2, 3, 5]:
    d = (arm(b, 1, .1) - arm(b, 0, .1)).values; sd = d.std(ddof=1); row = []
    for n in [3, 5, 10]:
        e = boot_means(d, [12345, b, n], n); row.append(f"n={n}: >=+2 {100*(e>=2).mean():.1f}%, <0 {100*(e<0).mean():.1f}%")
    need = [next(n for n in range(3, 1000) if power(sd, x, n) >= .8) for x in (1, 2)]
    mde = next(x for x in np.arange(0.1, 10, 0.01) if power(sd, x, 20) >= .8)
    print(f"{b}/class SD={sd:.2f} | " + " | ".join(row) + f" | seeds for 1/2 points {need[0]}/{need[1]} | MDE(20 seeds) {mde:.1f}")

print("\n== Margins and label-equivalent gains (Secs. 2.7, 3.5)")
sc = {b: arm(b, 0, .1, src=C, tb=5) for b in [2, 3, 5]}
g23, g35 = (sc[3] - sc[2]).mean(), (sc[5] - sc[3]).mean() / 2
for b in [2, 3, 5]:
    sd = arm(b, 0, .1).std(); print(f"{b}/class scratch SD {sd:.2f}: +-2 = {2/sd:.2f} SD, +-3 = {3/sd:.2f} SD")
print(f"+-2 = {2/g23:.2f} labels/class ({32/g23:.1f} px), +-3 = {3/g23:.2f} labels/class ({48/g23:.1f} px)")
for b, slope in [(2, g23), (3, g35), (5, g35)]:
    m, lo, hi, p = tci((arm(b, 1, .1) - arm(b, 0, .1)).values)
    print(f"{b}/class {m:+.2f} -> {m/slope:.2f} labels/class ({16*m/slope:.1f} px); upper CI {hi:+.2f} -> {hi/slope:.2f} ({16*hi/slope:.1f} px)")

print("\n== Preliminary backbone-multiplier exploration, 5/class, test pixels (Sec. 2.5)")
raw = pd.read_csv(a.budgets); raw["tag"] = raw.ssl_tag.fillna("").replace("none", "")
x = raw[(raw.budget == 5) & (raw.freeze == 0) & (raw.tag == "") & raw.enc_lr_mult.isin([0.05, 0.3])]
for (ssl, m), g in x.groupby(["ssl", "enc_lr_mult"]):
    print(f"{'pretrained' if ssl else 'scratch':10s} x{m}: {len(g)} runs, seeds {sorted(g.seed.unique().tolist())}, mean mIoU {100*g.miou.mean():.2f}")

print("\n== Pretrained x1.0 control: 2x2 initialization x schedule (Sec. 3.2), seed-0 checkpoint")
X = load(a.x1)
for col in ["miou", "oa", "aa", "kappa"]:
    f = 100 if col == "kappa" else 1
    for b in [2, 3, 5]:
        S1, S01, P01 = arm(b, 0, 1.0, col=col), arm(b, 0, .1, col=col), arm(b, 1, .1, col=col)
        P1 = arm(b, 1, 1.0, tag="ip_mae_seed0", src=X, col=col)
        assert len(S1) == len(S01) == len(P01) == len(P1) == 20
        if col == "miou": print(f"{b}/class pretrained x1.0 mean {P1.mean():.2f} +- {P1.std():.2f}")
        for k, (name, d) in enumerate([("pretrain@x1.0 (P1-S1)", P1 - S1), ("sched pretrained (P01-P1)", P01 - P1),
                                       ("interaction", (P01 - P1) - (S01 - S1))]):
            if col != "miou" and k > 0: continue
            d = d.values; m, lo, hi, p = tci(d)
            extra = ""
            if col == "miou":
                bs = boot_means(d, [2028, k, b], len(d)); l95, h95 = np.percentile(bs, [2.5, 97.5])
                extra = f" | Wilcoxon p={stats.wilcoxon(d).pvalue:.3f} | boot95 [{l95:+.2f},{h95:+.2f}] | higher in {int((d > 0).sum())}/20"
            print(f"  {col:5s} {b}/class {name:26s} {m/f:+.3f} [{lo/f:+.3f},{hi/f:+.3f}] t p={p:.3f}{extra}")
for b in [2, 3, 5]:
    S1, S01, P01 = arm(b, 0, 1.0), arm(b, 0, .1), arm(b, 1, .1); P1 = arm(b, 1, 1.0, tag="ip_mae_seed0", src=X)
    print(f"{b}/class apparent {(P01-S1).mean():+.2f} = scratch path: schedule {(S01-S1).mean():+.2f} + pretraining {(P01-S01).mean():+.2f}"
          f" | pretrained path: pretraining {(P1-S1).mean():+.2f} + schedule {(P01-P1).mean():+.2f}")

print("\n== Reviewer additions (no new runs)")
def holm(ps):
    o = np.argsort(ps); adj = np.empty(len(ps)); run = 0
    for r, i in enumerate(o):
        run = max(run, (len(ps) - r) * ps[i]); adj[i] = min(1, run)
    return adj
for col in ["miou", "oa", "kappa", "aa"]:
    ps = [tci((arm(b, 1, .1, col=col) - arm(b, 0, .1, col=col)).values)[3] for b in [2, 3, 5]]
    print(f"Holm, matched {col:5s}: raw " + ", ".join(f"{p:.4f}" for p in ps) + " | adjusted " + ", ".join(f"{p:.4f}" for p in holm(np.array(ps))))
print("Seed count, null-centred bootstrap (differences minus their mean), seed [12345, budget, n]")
for b in [2, 3, 5]:
    d = (arm(b, 1, .1) - arm(b, 0, .1)).values; d0 = d - d.mean(); row = []
    for n in [3, 5, 10]:
        e = boot_means(d0, [12345, b, n], n); row.append(f"n={n}: >=+2 {100*(e>=2).mean():.1f}%, >=+1 {100*(e>=1).mean():.1f}%")
    print(f"{b}/class " + " | ".join(row))
print("2x2 main effects (average of simple effects) and interaction, mIoU / OA")
for col in ["miou", "oa"]:
    for b in [2, 3, 5]:
        S1, S01, P01 = arm(b, 0, 1.0, col=col), arm(b, 0, .1, col=col), arm(b, 1, .1, col=col)
        P1 = arm(b, 1, 1.0, tag="ip_mae_seed0", src=X, col=col)
        for name, d in [("pretraining main", ((P1 - S1) + (P01 - S01)) / 2), ("schedule main", ((P01 - P1) + (S01 - S1)) / 2),
                        ("interaction", (P01 - P1) - (S01 - S1))]:
            m, lo, hi, p = tci(d.values); print(f"  {col:4s} {b}/class {name:16s} {m:+.2f} [{lo:+.2f},{hi:+.2f}] p={p:.3f}")
print("Pilot multipliers: seeds and paired pretraining difference")
for m in [0.05, 0.3]:
    g = raw[(raw.budget == 5) & (raw.freeze == 0) & (raw.tag == "") & np.isclose(raw.enc_lr_mult, m)]
    sc = g[g.ssl == 0].drop_duplicates("seed").set_index("seed").miou * 100; pr = g[g.ssl == 1].drop_duplicates("seed").set_index("seed").miou * 100
    common = sorted(set(sc.index) & set(pr.index)); d = pr.loc[common] - sc.loc[common]
    print(f"  x{m}: seeds {common}, scratch {sc.loc[common].mean():.2f}, pretrained {pr.loc[common].mean():.2f}, paired diff {d.mean():+.2f}")
print("Paired-difference SD by budget (matched mIoU):", ", ".join(f"{b}: {(arm(b,1,.1)-arm(b,0,.1)).std():.2f}" for b in [2,3,5]))

print("\n== Injected-effect check (Secs. 2.7, 3.3): zero-centered paired differences + known effect, 20,000 resamples")
def tost_ok(m, se, df, M):
    t = stats.t.ppf(0.95, df); return (m - t * se > -M) & (m + t * se < M)
INJ = []
for b in [2, 3, 5]:
    d = (arm(b, 1, .1) - arm(b, 0, .1)).values; d0 = d - d.mean()
    for delta in [0, 1, 2, 3]:
        for n in [3, 5, 10, 20]:
            x = d0[np.random.default_rng([4242, b, n, delta]).integers(0, 20, (20000, n))] + delta
            m, se = x.mean(1), x.std(1, ddof=1) / np.sqrt(n)
            with np.errstate(divide="ignore", invalid="ignore"):
                p = np.where(se > 0, 2 * stats.t.sf(np.abs(m / se), n - 1), 1.0)   # identical draws: no test
            INJ.append((b, delta, n, ((p < 0.05) & (m > 0)).mean(), (tost_ok(m, se, n - 1, 2) & (se > 0)).mean()))
            print(f"{b}/class effect +{delta} n={n:2d}: detected {100*INJ[-1][3]:5.1f}%  declared equivalent (+-2) {100*INJ[-1][4]:5.1f}%")
Cc = load(a.common); scc = {b: arm(b, 0, .1, src=Cc, tb=5) for b in [2, 3, 5]}
for lo, hi in ((2, 3), (3, 5)):
    g = (scc[hi] - scc[lo]).values
    print(f"natural positive control, labels {lo}->{hi}: {g.mean():+.2f} points, positive in {(g > 0).sum()}/20 seeds, t p={stats.ttest_1samp(g, 0).pvalue:.1e}")
