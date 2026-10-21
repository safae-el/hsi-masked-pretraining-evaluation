"""Builds the supplementary tables (LaTeX) and Fig. S1 from the existing result files. No training.
Usage: python make_supplement.py --budgets budgets.csv --x1 ssl_pretrained_x1.csv --outdir .
Selection rules identical to revision_analysis.py."""
import argparse, numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ap = argparse.ArgumentParser(); ap.add_argument("--budgets", default="budgets.csv")
ap.add_argument("--x1", default="ssl_pretrained_x1.csv"); ap.add_argument("--outdir", default="."); a = ap.parse_args()
NAMES = ["Alfalfa", "Corn-notill", "Corn-mintill", "Corn", "Grass-pasture", "Grass-trees", "Grass-pasture-mowed",
         "Hay-windrowed", "Oats", "Soybean-notill", "Soybean-mintill", "Soybean-clean", "Wheat", "Woods",
         "Buildings-Grass-Trees-Drives", "Stone-Steel-Towers"]
SIZES = [46, 1428, 830, 237, 483, 730, 28, 478, 20, 972, 2455, 593, 205, 1265, 386, 93]
NB = 200_000
def load(path):
    d = pd.read_csv(path); d["row"] = np.arange(len(d)); d["tag"] = d.ssl_tag.fillna("").replace("none", "")
    if "test_budget" not in d: d["test_budget"] = 0
    d["test_budget"] = d.test_budget.fillna(0)
    d = d[(d.freeze == 0) & d.enc_lr_mult.notna() & d.miou.notna()].copy()
    for c in ["miou", "oa", "aa", "kappa"]: d[c] = d[c] * 100
    return d
B, X = load(a.budgets), load(a.x1)
def arm(b, ssl, enc, tag="", src=None, col="miou"):
    src = B if src is None else src
    g = src[(src.budget == b) & (src.ssl == ssl) & np.isclose(src.enc_lr_mult, enc) & (src.tag == tag) & (src.test_budget == 0)]
    g = g.sort_values("row").drop_duplicates("seed", keep="first").set_index("seed"); g = g.loc[[s for s in range(20) if s in g.index]]
    assert len(g) == 20; return g if col is None else g[col]
def tci(d):
    d = np.asarray(d); lo, hi = stats.t.interval(0.95, len(d) - 1, d.mean(), stats.sem(d)); return d.mean(), lo, hi, stats.ttest_1samp(d, 0).pvalue
f2 = lambda x: f"${x:+.2f}$"; pf = lambda p: "$<0.001$" if p < 0.001 else f"{p:.3f}"
R = r"\rule[-1ex]{0pt}{3.5ex} "
out = []
# Table S1/S2: class-wise IoU at 2 and 3 labels
pc = lambda g: np.array([[float(x) for x in s.split(";")] for s in g.per_class_iou]) * 100
for k, b in enumerate([2, 3], 1):
    ps, pm = pc(arm(b, 0, .1, col=None)), pc(arm(b, 1, .1, col=None)); d = pm - ps
    rows = []
    for i, n in enumerate(NAMES):
        m, lo, hi, _ = tci(d[:, i])
        rows.append(f"{R}{n} & {SIZES[i]-b} & {ps[:, i].mean():.2f} & {pm[:, i].mean():.2f} & {f2(m)} [{f2(lo)}, {f2(hi)}] \\\\\n\\hline")
    out.append(rf"""\begin{{table}}[ht]
\caption{{Class-wise IoU (\%) at {b} labeled pixels per class under the matched schedule, mean over 20 seeds, with the paired difference (pretrained $-$ scratch) and its 95\% CI. Test pixels, labeled pixels of the class minus the {b} training pixels.}}
\label{{tab:S{k}}}
\begin{{center}}\small
\begin{{tabular}}{{|l|c|c|c|c|}}
\hline
{R}Class & Test pixels & Scratch & Pretrained & Difference [95\% CI] \\
\hline\hline
""" + "\n".join(rows) + "\n\\end{tabular}\n\\end{center}\n\\end{table}\n")
# Table S3: 2x2 condition means
conds = [("Scratch, $\\times1.0$", lambda b, c: arm(b, 0, 1.0, col=c)), ("Scratch, $\\times0.1$", lambda b, c: arm(b, 0, .1, col=c)),
         ("Pretrained, $\\times1.0$", lambda b, c: arm(b, 1, 1.0, tag="ip_mae_seed0", src=X, col=c)), ("Pretrained, $\\times0.1$", lambda b, c: arm(b, 1, .1, col=c))]
rows = []
for b in [2, 3, 5]:
    for j, (name, f) in enumerate(conds):
        cells = []
        for c in ["miou", "oa", "aa", "kappa"]:
            v = f(b, c); s = 100 if c == "kappa" else 1; dig = 3 if c == "kappa" else 2
            cells.append(f"{v.mean()/s:.{dig}f} $\\pm$ {v.std()/s:.{dig}f}")
        rows.append(f"{R}{b if j == 0 else ''} & {name} & " + " & ".join(cells) + " \\\\\n" + ("\\hline\\hline" if j == 3 and b != 5 else "\\hline"))
out.append(r"""\begin{table}[ht]
\caption{Condition means ($\pm$ SD over 20 seeds) of the $2\times2$ initialization-by-schedule design. Pretrained, in-domain pretraining seed 0; $\times0.1$ and $\times1.0$, backbone learning-rate multiplier.}
\label{tab:S3}
\begin{center}\small
\begin{tabular}{|c|l|c|c|c|c|}
\hline
""" + R + r"""Labels/class & Condition & mIoU (\%) & OA (\%) & AA (\%) & Kappa \\
\hline\hline
""" + "\n".join(rows) + "\n\\end{tabular}\n\\end{center}\n\\end{table}\n")
# Table S4: effects, mIoU and OA
rows = []
for b in [2, 3, 5]:
    E = {}
    for c in ["miou", "oa"]:
        S1, S01 = arm(b, 0, 1.0, col=c), arm(b, 0, .1, col=c); P1 = arm(b, 1, 1.0, tag="ip_mae_seed0", src=X, col=c); P01 = arm(b, 1, .1, col=c)
        E[c] = [("Pretraining at $\\times0.1$ (matched)", P01 - S01), ("Pretraining at $\\times1.0$", P1 - S1), ("Schedule, scratch", S01 - S1),
                ("Schedule, pretrained", P01 - P1), ("Apparent (P$\\times0.1$ $-$ S$\\times1.0$)", P01 - S1),
                ("Pretraining main effect", ((P1 - S1) + (P01 - S01)) / 2), ("Schedule main effect", ((P01 - P1) + (S01 - S1)) / 2),
                ("Interaction", (P01 - P1) - (S01 - S1))]
    for j in range(8):
        name = E["miou"][j][0]; cells = []
        for c in ["miou", "oa"]:
            m, lo, hi, p = tci(E[c][j][1].values); cells.append(f"{f2(m)} [{f2(lo)}, {f2(hi)}] & {pf(p)}")
        rows.append(f"{R}{b if j == 0 else ''} & {name} & " + " & ".join(cells) + " \\\\\n" + ("\\hline\\hline" if j == 7 and b != 5 else "\\hline"))
out.append(r"""\begin{table}[ht]
\caption{Paired effects in the $2\times2$ design, mean [95\% CI] and paired $t$-test $p$-value over 20 seeds, in mIoU and OA points. Main effects average the two simple effects of a factor; the interaction is the schedule effect in the pretrained arm minus that in the scratch arm. All comparisons except the matched pretraining contrast are exploratory, and $p$-values are unadjusted.}
\label{tab:S4}
\begin{center}\footnotesize\setlength{\tabcolsep}{3pt}
\begin{tabular}{|c|l|c|c|c|c|}
\hline
""" + R + r"""Labels/class & Effect & mIoU [95\% CI] & $p$ & OA [95\% CI] & $p$ \\
\hline\hline
""" + "\n".join(rows) + "\n\\end{tabular}\n\\end{center}\n\\end{table}\n")
# Table S5: seed-count bootstrap, observed- vs zero-centred
rows = []
for b in [2, 3, 5]:
    d = (arm(b, 1, .1) - arm(b, 0, .1)).values; d0 = d - d.mean()
    for n in [3, 5, 10]:
        cells = []
        for dd in (d, d0):
            e = dd[np.random.default_rng([12345, b, n]).integers(0, 20, (NB, n))].mean(1)
            cells += [f"{100*(e >= 2).mean():.1f}", f"{100*(e < 0).mean():.1f}" if dd is d else f"{100*(e >= 1).mean():.1f}"]
        rows.append(f"{R}{b if n == 3 else ''} & {n} & " + " & ".join(cells) + " \\\\\n" + ("\\hline\\hline" if n == 10 and b != 5 else "\\hline"))
out.append(r"""\begin{table}[ht]
\caption{Percentage of bootstrap estimates of the matched mIoU difference from $n$ resampled seeds (200,000 resamples). Observed, resampling of the 20 paired differences; zero-centered, the same differences minus their mean, representing no true pretraining effect.}
\label{tab:S5}
\begin{center}\small
\begin{tabular}{|c|c|c|c|c|c|}
\hline
""" + R + r"""Labels/class & $n$ & Observed: $\geq+2$ & Observed: $<0$ & Zero-centered: $\geq+2$ & Zero-centered: $\geq+1$ \\
\hline\hline
""" + "\n".join(rows) + "\n\\end{tabular}\n\\end{center}\n\\end{table}\n")
# Table S6: Holm
def holm(ps):
    o = np.argsort(ps); adj = np.empty(len(ps)); run = 0
    for r_, i in enumerate(o): run = max(run, (len(ps) - r_) * ps[i]); adj[i] = min(1, run)
    return adj
rows = []
for c, lab in [("miou", "mIoU"), ("oa", "OA"), ("aa", "AA"), ("kappa", "Kappa")]:
    ps = np.array([tci((arm(b, 1, .1, col=c) - arm(b, 0, .1, col=c)).values)[3] for b in [2, 3, 5]]); ad = holm(ps)
    rows.append(f"{R}{lab} & " + " & ".join(f"{pf(p)} / {pf(q)}" for p, q in zip(ps, ad)) + " \\\\\n\\hline")
out.append(r"""\begin{table}[ht]
\caption{Matched pretraining contrast: unadjusted / Holm-adjusted paired $t$-test $p$-values across the three label budgets.}
\label{tab:S6}
\begin{center}\small
\begin{tabular}{|l|c|c|c|}
\hline
""" + R + r"""Metric & 2 labels/class & 3 labels/class & 5 labels/class \\
\hline\hline
""" + "\n".join(rows) + "\n\\end{tabular}\n\\end{center}\n\\end{table}\n")

# Table S7: injected-effect check (zero-centred paired differences + known effect)
def tost_ok(m, se, df, M):
    t = stats.t.ppf(0.95, df); return (m - t * se > -M) & (m + t * se < M)
rows = []
for b in [2, 3, 5]:
    d = (arm(b, 1, .1) - arm(b, 0, .1)).values; d0 = d - d.mean()
    for n in [3, 5, 10, 20]:
        det, eq = [], []
        for delta in [0, 1, 2, 3]:
            x = d0[np.random.default_rng([4242, b, n, delta]).integers(0, 20, (20000, n))] + delta
            m, se = x.mean(1), x.std(1, ddof=1) / np.sqrt(n)
            with np.errstate(divide="ignore", invalid="ignore"):
                p = np.where(se > 0, 2 * stats.t.sf(np.abs(m / se), n - 1), 1.0)
            det.append(100 * ((p < 0.05) & (m > 0)).mean()); eq.append(100 * (tost_ok(m, se, n - 1, 2) & (se > 0)).mean())
        rows.append(f"{R}{b if n == 3 else ''} & {n} & " + " & ".join(f"{v:.1f}" for v in det) + " & " + " & ".join(f"{v:.1f}" for v in eq) + " \\\\\n" + ("\\hline\\hline" if n == 20 and b != 5 else "\\hline"))
out.append(r"""\begin{table}[ht]
\caption{Injected-effect check. Known effects of 0 to 3 mIoU points were added to the zero-centered matched paired differences, and $n$ seeds were resampled with replacement (20,000 samples per setting). Detected, percentage of samples with a positive mean and paired $t$-test $p<0.05$; equivalent, percentage in which two one-sided tests established equivalence within $\pm2$ points.}
\label{tab:S7}
\begin{center}\footnotesize\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{|c|c|c|c|c|c|c|c|c|c|}
\hline
""" + R + r"""Labels/class & $n$ & \multicolumn{4}{c|}{Detected (\%), by effect size} & \multicolumn{4}{c|}{Equivalent within $\pm2$ (\%), by effect size} \\
\hline
\rule[-1ex]{0pt}{3.5ex}  &  & $+0$ & $+1$ & $+2$ & $+3$ & $+0$ & $+1$ & $+2$ & $+3$ \\
\hline\hline
""" + "\n".join(rows) + "\n\\end{tabular}\n\\end{center}\n\\end{table}\n")
open(f"{a.outdir}/supp_tables.tex", "w").write("\n".join(out))
# Fig. S1: per-seed paired differences, mIoU and OA
plt.rcParams.update({"font.size": 9})
fig, axs = plt.subplots(1, 2, figsize=(6.75, 2.9))
rng = np.random.default_rng(0)
for ax, (c, lab) in zip(axs, [("miou", "mIoU difference (points)"), ("oa", "OA difference (points)")]):
    for k, b in enumerate([2, 3, 5]):
        d = (arm(b, 1, .1, col=c) - arm(b, 0, .1, col=c)).values; m, lo, hi, _ = tci(d)
        x = k + rng.uniform(-0.12, 0.12, len(d))
        ax.scatter(x, d, s=14, facecolors="white", edgecolors="black", linewidths=0.7, zorder=3)
        ax.errorbar(k + 0.3, m, yerr=[[m - lo], [hi - m]], fmt="s", color="black", ms=4, capsize=3, zorder=4)
    ax.axhline(0, color="0.4", ls="--", lw=0.8); ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["2", "3", "5"])
    ax.set_xlabel("Labeled pixels per class"); ax.set_ylabel(lab); ax.set_xlim(-0.5, 2.6)
axs[0].set_title("(a) mIoU", fontsize=9); axs[1].set_title("(b) OA", fontsize=9)
fig.tight_layout(); fig.savefig(f"{a.outdir}/figS1_per_seed.pdf"); fig.savefig(f"{a.outdir}/figS1_per_seed.png", dpi=200)
print("written")
