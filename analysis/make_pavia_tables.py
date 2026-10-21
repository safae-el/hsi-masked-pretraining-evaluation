"""Pavia University replication tables (Stage A, checkpoint seed 0), same selection rules as pu_analysis.py.
Usage: python analysis/make_pavia_tables.py results tables
Writes tables/tab_cross_scene.tex (Table 6 values) and tables/tab_pavia_supp.tex (Tables S8-S12)."""
import numpy as np, pandas as pd, os
from scipy import stats
import sys
IN = sys.argv[1] if len(sys.argv) > 1 else "results"     # folder with ssl_paviau*.csv
OUTD = sys.argv[2] if len(sys.argv) > 2 else "tables"  # output folder for the LaTeX tables
os.makedirs(OUTD, exist_ok=True); NB = 200_000
d = pd.concat([pd.read_csv(os.path.join(IN, f)) for f in ["ssl_paviau.csv", "ssl_paviau_w2.csv", "ssl_paviau_w3.csv"]], ignore_index=True)
d["row"] = np.arange(len(d)); d["tag"] = d.ssl_tag.fillna("").replace("none", "")
d = d[(d.dataset == "paviau") & (d.freeze == 0) & d.enc_lr_mult.notna() & np.isclose(d.enc_lr_mult, 0.1) & d.miou.notna()].copy()
for c in ["miou", "oa", "aa"]: d[c] *= 100
NAMES = ["Asphalt", "Meadows", "Gravel", "Trees", "Painted metal sheets", "Bare soil", "Bitumen", "Self-blocking bricks", "Shadows"]
SIZES = [6631, 18649, 2099, 3064, 1345, 5029, 1330, 3682, 947]
CK = "paviau_mae_seed0"; B = [2, 3, 5]
def arm(b, t):
    g = d[(d.budget == b) & (d.tag == t)].sort_values("row").drop_duplicates("seed", keep="first").set_index("seed").reindex(range(20))
    assert g.miou.notna().all(); return g
def ci(x):
    m = x.mean(); lo, hi = stats.t.interval(0.95, len(x) - 1, m, stats.sem(x)); return m, lo, hi, stats.ttest_1samp(x, 0).pvalue
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
s = lambda v, k=2: f"${0:.{k}f}$" if abs(v) < 0.5 * 10 ** -k else (f"$+{v:.{k}f}$" if v >= 0 else f"$-{abs(v):.{k}f}$")
pf = lambda p: "$<0.001$" if p < 0.001 else (f"{p:.3f}" if p < 0.1 else f"{p:.2f}")
R = "\\rule[-1ex]{0pt}{3.5ex} "
X = {b: (arm(b, CK).miou - arm(b, "").miou).values for b in B}
# ---------- numbers
res = {}
for b in B:
    x = X[b]; m, lo, hi, p = ci(x)
    flip = (np.abs((np.random.default_rng([2026, 10, b]).choice([-1, 1], (NB, 20)) * x).mean(1)) >= abs(x.mean()) - 1e-12).mean()
    bs = x[np.random.default_rng([3030, b]).integers(0, 20, (NB, 20))].mean(1)
    sd = x.std(ddof=1); mde = next(v for v in np.arange(0.1, 20, 0.01) if power(sd, v, 20) >= .8)
    res[b] = dict(m=m, lo=lo, hi=hi, p=p, w=stats.wilcoxon(x).pvalue, flip=flip, b95=np.percentile(bs, [2.5, 97.5]),
                  t2=tost(x, 2), t3=tost(x, 3), up=(x > 0).sum(), sd=sd, mde=mde)
hp = holm(np.array([res[b]["p"] for b in B]))
for b, h in zip(B, hp): res[b]["holm"] = h
for b in B:
    r = res[b]; print(f"{b}: {r['m']:+.2f} [{r['lo']:+.2f},{r['hi']:+.2f}] p={r['p']:.3f} holm={r['holm']:.3f} W={r['w']:.3f} flip={r['flip']:.3f} "
                      f"boot=[{r['b95'][0]:+.2f},{r['b95'][1]:+.2f}] tost2={r['t2']:.3f} tost3={r['t3']:.3f} up={r['up']} sd={r['sd']:.2f} mde={r['mde']:.1f}")
# ---------- main text: cross-scene table (Indian Pines values from the main analysis, Tables 3-4 and Sec. 3.3)
zc = {b: 100 * ((X[b] - X[b].mean())[np.random.default_rng([12345, b, 3]).integers(0, 20, (NB, 3))].mean(1) >= 2).mean() for b in B}
# Table 6 (cross-scene summary). Indian Pines values come from the Indian Pines analyses
# (revision_analysis.py, make_figures.py); Pavia University values are computed here.
CKS_ALL = ["paviau_mae_seed0", "paviau_mae_seed99", "paviau_mae_seed7"]
avg = {b: (np.array([arm(b, t_).miou.values for t_ in CKS_ALL]).mean(0) - arm(b, "").miou.values).mean() for b in B}
allpos = all((arm(b, t_).miou - arm(b, "").miou).mean() > 0 for b in B for t_ in CKS_ALL)
oapos = all((arm(b, t_).oa - arm(b, "").oa).mean() > 0 for b in B for t_ in CKS_ALL)
rng_ck = [(arm(b, t_).miou - arm(b, "").miou).mean() for b in B for t_ in CKS_ALL]
pc5 = [np.array([list(map(float, v.split(";"))) for v in g.per_class_iou]) * 100 for g in (arm(5, ""), arm(5, CK))]
npos5 = int(((pc5[1] - pc5[0]).mean(0) > 0).sum())
L_ = r"\rule[-1ex]{0pt}{3.5ex}"
T = [r"\small\begin{tabular}{|l|p{4.4cm}|p{4.4cm}|}", r"\hline",
     L_ + r"Finding & Indian Pines & Pavia University \\", r"\hline\hline",
     L_ + r"$\Delta$mIoU, checkpoint 0 & $+0.77$ / $+0.94$ / $+0.43$ & " + " / ".join(s(res[b]["m"]) for b in B) + r" \\", r"\hline",
     L_ + r"$\Delta$mIoU, mean of 3 checkpoints & Not evaluated & " + " / ".join(s(avg[b]) for b in B) + r" \\", r"\hline",
     L_ + r"Significant after Holm adjustment & None & None \\", r"\hline",
     L_ + r"Equivalence within $\pm2$ points & 5 labels only & None \\", r"\hline",
     L_ + r"OA difference & Lower at 3 and 5 labels & No decrease; positive in all 9 \\" if oapos else L_ + r"OA difference & Lower at 3 and 5 labels & CHECK \\", r"\hline",
     L_ + rf"Source of the 5-label mIoU gain & Mainly Alfalfa (rare class) & {npos5} of 9 classes improved \\", r"\hline",
     L_ + r"Zero-centered 3-seed estimates $\geq+2$ (\%) & 20.9 / 19.2 / 10.8 & " + " / ".join(f"{zc[b]:.1f}" for b in B) + r" \\", r"\hline",
     L_ + r"Effect with 80\% power, 20 seeds & 2.7 / 2.6 / 1.9 & " + " / ".join(f"{res[b]['mde']:.1f}" for b in B) + r" \\", r"\hline",
     L_ + r"Across evaluated pretraining checkpoints & 2 checkpoints, 5 labels: $+0.46$ / $-0.50$ & 3 checkpoints: "
        + ("all 9 positive, " if allpos else "") + f"{s(min(rng_ck))} to {s(max(rng_ck))}" + r" \\", r"\hline",
     r"\end{tabular}"]
open(os.path.join(OUTD, "tab_cross_scene.tex"), "w").write("\n".join(T) + "\n")
# ---------- supplement
out = []
# S8 detailed primary comparison
rows = []
for b in B:
    S, P = arm(b, ""), arm(b, CK); r = res[b]
    rows += [R + f"{b} & {s(r['m'])} [{s(r['lo'])}, {s(r['hi'])}] & "
             f"{pf(r['p'])} / {pf(r['holm'])} & {pf(r['w'])} & {pf(r['flip'])} & [{s(r['b95'][0])}, {s(r['b95'][1])}] & {pf(r['t2'])} / {pf(r['t3'])} & {r['up']}/20 \\\\", "\\hline"]
out.append(r"""\begin{table}[ht]
\caption{Pavia University replication: matched comparison of in-domain masked pretraining (Pavia University only, pretraining seed 0) and scratch training, backbone multiplier $\times0.1$, mIoU over 20 paired seeds. $t$ / Holm, paired $t$-test $p$-value, unadjusted and Holm-adjusted across the three budgets; sign-flip, paired permutation test (200,000 permutations); bootstrap, percentile 95\% CI (200,000 resamples); TOST, $p$-values of two one-sided tests at $\pm2$ / $\pm3$ points; higher, seeds in which the pretrained model had higher mIoU. Condition means are given in Table~\ref{tab:S9}.}
\label{tab:S8}
\begin{center}\footnotesize\setlength{\tabcolsep}{3pt}
\begin{tabular}{|c|c|c|c|c|c|c|c|}
\hline
""" + R + r"""Labels/class & $\Delta$ [95\% CI] & $t$ / Holm & Wilcoxon & Sign-flip & Bootstrap 95\% CI & TOST $\pm2$ / $\pm3$ & Higher \\
\hline\hline
""" + "\n".join(rows) + "\n\\end{tabular}\n\\end{center}\n\\end{table}\n")
# S9 metrics
M = []
for b in B:
    S, P = arm(b, ""), arm(b, CK)
    for name, g in [("Scratch", S), ("Pretrained", P)]:
        M += [R + f"{b if name == 'Scratch' else ''} & {name} & {g.oa.mean():.2f} $\\pm$ {g.oa.std():.2f} & {g.aa.mean():.2f} $\\pm$ {g.aa.std():.2f} & {g.kappa.mean():.3f} $\\pm$ {g.kappa.std():.3f} & {g.miou.mean():.2f} $\\pm$ {g.miou.std():.2f} \\\\", "\\hline"]
    cells = []
    for c, k in [("oa", 2), ("aa", 2), ("kappa", 3), ("miou", 2)]:
        m, lo, hi, p = ci((P[c] - S[c]).values); cells.append(f"{s(m,k)} [{s(lo,k)}, {s(hi,k)}]")
    M += [R + " & Difference & " + " & ".join(cells) + " \\\\", "\\hline\\hline" if b < 5 else "\\hline"]
out.append(r"""\begin{table}[ht]
\caption{Pavia University replication: OA, AA, Cohen's kappa, and mIoU for matched scratch training and in-domain pretraining, mean $\pm$ SD over 20 seeds. Difference, paired difference (pretrained $-$ scratch) with its 95\% CI; OA, AA, and mIoU in points, kappa in absolute units. AA and mIoU are averaged over the 9 classes.}
\label{tab:S9}
\begin{center}\footnotesize\setlength{\tabcolsep}{3pt}
\begin{tabular}{|c|l|c|c|c|c|}
\hline
""" + R + r"""Labels/class & Condition & OA (\%) & AA (\%) & Kappa & mIoU (\%) \\
\hline\hline
""" + "\n".join(M) + "\n\\end{tabular}\n\\end{center}\n\\end{table}\n")
# S10 class-wise
pc = {}
for b in B:
    S, P = arm(b, ""), arm(b, CK)
    pc[b] = [np.array([list(map(float, v.split(";"))) for v in g.per_class_iou]) * 100 for g in (S, P)]
    assert np.allclose(pc[b][0].mean(1), S.miou, atol=0.01) and np.allclose(pc[b][1].mean(1), P.miou, atol=0.01)
C = []
for k in range(9):
    d2 = (pc[2][1][:, k] - pc[2][0][:, k]).mean(); d3 = (pc[3][1][:, k] - pc[3][0][:, k]).mean()
    m, lo, hi, p = ci(pc[5][1][:, k] - pc[5][0][:, k])
    C += [R + f"{NAMES[k]} & {SIZES[k]:,} & {s(d2)} & {s(d3)} & {pc[5][0][:, k].mean():.2f} & {pc[5][1][:, k].mean():.2f} & {s(m)} [{s(lo)}, {s(hi)}] \\\\", "\\hline"]
out.append(r"""\begin{table}[ht]
\caption{Pavia University replication: class-wise IoU (\%) under the matched schedule. $\Delta$, mean paired difference (pretrained $-$ scratch) over 20 seeds at 2, 3, and 5 labeled pixels per class; scratch and pretrained means and the 95\% CI of the difference are given at 5 labels per class. Labeled pixels, ground-truth pixels of the class; test pixels are these minus the training pixels. Class-wise intervals are descriptive and are not adjusted for multiple comparisons.}
\label{tab:S10}
\begin{center}\footnotesize\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{|l|c|c|c|c|c|c|}
\hline
""" + R + r"""Class & Labeled pixels & $\Delta$, 2 labels & $\Delta$, 3 labels & Scratch, 5 & Pretrained, 5 & $\Delta$, 5 labels [95\% CI] \\
\hline\hline
""" + "\n".join(C) + "\n\\end{tabular}\n\\end{center}\n\\end{table}\n")
# S11 seed-count bootstrap (same construction as Table S5)
rows = []
for b in B:
    x = X[b]; x0 = x - x.mean()
    for n in [3, 5, 10]:
        cells = []
        for dd in (x, x0):
            e = dd[np.random.default_rng([12345, b, n]).integers(0, 20, (NB, n))].mean(1)
            cells += [f"{100*(e >= 2).mean():.1f}", f"{100*(e < 0).mean():.1f}" if dd is x else f"{100*(e >= 1).mean():.1f}"]
        rows.append(f"{R}{b if n == 3 else ''} & {n} & " + " & ".join(cells) + " \\\\\n" + ("\\hline\\hline" if n == 10 and b != 5 else "\\hline"))
    print(b, "zero-centred 3-seed >=2:", f"{zc[b]:.1f}")
out.append(r"""\begin{table}[ht]
\caption{Pavia University replication: percentage of bootstrap estimates of the matched mIoU difference from $n$ resampled seeds (200,000 resamples), constructed as in Table~\ref{tab:S5}. Observed, resampling of the 20 paired differences; zero-centered, the same differences minus their mean, representing no true pretraining effect.}
\label{tab:S11}
\begin{center}\small
\begin{tabular}{|c|c|c|c|c|c|}
\hline
""" + R + r"""Labels/class & $n$ & Observed: $\geq+2$ & Observed: $<0$ & Zero-centered: $\geq+2$ & Zero-centered: $\geq+1$ \\
\hline\hline
""" + "\n".join(rows) + "\n\\end{tabular}\n\\end{center}\n\\end{table}\n")
open(os.path.join(OUTD, "tab_pavia_supp.tex"), "w").write("\n".join(out))
print("\n".join(T))

# ---------- Stages B/C: three pretraining checkpoints (Table S12)
CKS = [("paviau_mae_seed0", "Checkpoint 0"), ("paviau_mae_seed99", "Checkpoint 99"), ("paviau_mae_seed7", "Checkpoint 7")]
S12 = []; avg_p = []
print("\n== checkpoints")
for b in B:
    Sg = arm(b, "")
    Xm = np.array([(arm(b, t_).miou - Sg.miou).values for t_, _ in CKS]); Xo = np.array([(arm(b, t_).oa - Sg.oa).values for t_, _ in CKS])
    k_, n_ = Xm.shape; msb = n_ * Xm.mean(1).var(ddof=1); msw = Xm.var(1, ddof=1).mean()
    pF = stats.f.sf(msb / msw, k_ - 1, k_ * (n_ - 1)); s2c = max(0.0, (msb - msw) / n_)
    lines = [(lab, Xm[k], Xo[k]) for k, (_, lab) in enumerate(CKS)] + [("Average", Xm.mean(0), Xo.mean(0))]
    for r_, (lab, xm, xo) in enumerate(lines):
        mm, lm, hm, pm = ci(xm); mo, lo_, ho, po = ci(xo)
        if lab == "Average":
            avg_p.append(pm)
            print(f"{b} avg miou {mm:+.2f} [{lm:+.2f},{hm:+.2f}] p={pm:.3f} TOST2={tost(xm,2):.3f} TOST3={tost(xm,3):.3f} | oa {mo:+.2f} p={po:.3f} | ckSD={np.sqrt(s2c):.2f} F p={pF:.3f}")
        else:
            print(f"{b} {lab}: miou {mm:+.2f} p={pm:.3f} | oa {mo:+.2f} p={po:.3f}")
        S12 += [R + (f"{b}" if r_ == 0 else "") + f" & {lab} & {s(mm)} [{s(lm)}, {s(hm)}] & {pf(pm)} & {s(mo)} [{s(lo_)}, {s(ho)}] & {pf(po)} \\\\",
                ("\\hline\\hline" if b < 5 else "\\hline") if lab == "Average" else ("\\cline{2-6}" if r_ == 2 else "\\cline{2-6}")]
    res[b].update(pF=pF)
ha = holm(np.array(avg_p)); print("Holm (average over checkpoints):", np.round(ha, 3))
VC = "; ".join(f"$p={res[b]['pF']:.2f}$ at {b} labels" for b in B)
out.append(r"""\begin{table}[ht]
\caption{Pavia University replication with three pretraining checkpoints (pretraining seeds 0, 99, and 7), each fine-tuned with the same 20 downstream seeds and compared with the same scratch runs. Entries are paired differences (pretrained $-$ scratch) in points with 95\% CI and paired $t$-test $p$-value. Average denotes the three pretrained outcomes averaged within each downstream seed minus the shared scratch outcome, with a paired confidence interval over the 20 downstream seeds. Holm-adjusted $p$-values of the averaged mIoU differences are """ + ", ".join(f"{h:.2f}" for h in ha) + r""" at 2, 3, and 5 labels per class. The analysis plan specified reporting the mean over checkpoints; this within-seed averaging rule for its interval was fixed after the runs were complete. It preserves the shared scratch runs and downstream-seed pairing and avoids treating checkpoint-specific outcomes as independent samples. The resulting intervals quantify downstream-seed variability for the average of the three evaluated checkpoints.}
\label{tab:S12}
\begin{center}\footnotesize
\begin{tabular}{|c|l|c|c|c|c|}
\hline
""" + R + r"""Labels/class & Pretraining & $\Delta$mIoU [95\% CI] & $p$ & $\Delta$OA [95\% CI] & $p$ \\
\hline\hline
""" + "\n".join(S12) + "\n\\end{tabular}\n\\end{center}\n\\end{table}\n")
open(os.path.join(OUTD, "tab_pavia_supp.tex"), "w").write("\n".join(out))
