"""Figures 1-5 for the masked-pretraining evaluation study (Fig. 3 uses bootstrap resampling).
In the Remote Sensing Letters manuscript, Fig. 3 here is Figure 1 and Fig. 2 here is Figure 2.

Figure rules: text >= 8 pt, no transparency, grayscale-distinguishable, panel labels,
widths 3.3125 in (single column) / 6.75 in (double column), PDF + 600 dpi PNG.

Selection rules (they reproduce Tables 3 and 4 and Secs. 3.1-3.5 of the manuscript):
  * Backbone multiplier must be recorded explicitly. Early runs with no enc_lr_mult
    (budget 5, seeds 0-2) are NOT default-schedule runs and are excluded.
  * ssl_tag NaN or 'none' = scratch / in-domain checkpoint; 'multi_ip_mae_seed0' = multi-scene.
  * Duplicate (budget, arm, seed) rows: keep the first row in file order.
  * Fig. 5 label gains use the common five-label test set (ssl_phase0c.csv, test_budget == 5).
  * Fig. 4 uses the 5-label replication runs (downstream seeds 0-9): ssl_phase0b.csv (scratch,
    in-domain and multi-scene, pretraining seeds 0 and 99), ssl_phase0c.csv (compute-matched seed 0)
    and ssl_phase0d.csv (compute-matched seed 99).

Usage: python make_figures.py --budgets budgets.csv --common ssl_phase0c.csv \
           --replication ssl_phase0b.csv --cm99 ssl_phase0d.csv --out figures
"""
import argparse, os
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy import stats

ap = argparse.ArgumentParser()
ap.add_argument("--budgets", default="budgets.csv")
ap.add_argument("--common", default="ssl_phase0c.csv")
ap.add_argument("--replication", default="ssl_phase0b.csv")
ap.add_argument("--cm99", default="ssl_phase0d.csv")
ap.add_argument("--out", default="figures")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

def load(path):
    d = pd.read_csv(path)
    d["row"] = range(len(d))
    d["tag"] = d.ssl_tag.fillna("").replace("none", "")
    d["m"] = d.miou * 100
    if "test_budget" not in d:
        d["test_budget"] = 0
    return d[(d.freeze == 0) & d.enc_lr_mult.notna() & d.miou.notna()]

FT, COMMON = load(a.budgets), load(a.common)
REP = pd.concat([load(a.replication), COMMON, load(a.cm99)])
REP = REP[(REP.budget == 5) & (REP.test_budget == 0)]

def arm(b, ssl, enc, tag, src=None, test_budget=0):
    src = FT if src is None else src
    g = src[(src.budget == b) & (src.ssl == ssl) & np.isclose(src.enc_lr_mult, enc)
            & (src.tag == tag) & (src.test_budget == test_budget)].sort_values("row")
    s = g.drop_duplicates("seed", keep="first").set_index("seed").m
    assert len(s) > 0, (b, ssl, enc, tag, test_budget)
    return s

def pdiff(x, y):
    s = sorted(set(x.index) & set(y.index)); d = (x[s] - y[s]).values; n = len(d)
    h = stats.t.ppf(.975, n - 1) * d.std(ddof=1) / np.sqrt(n)
    return d.mean(), d.mean() - h, d.mean() + h, d

B = [2, 3, 5]; W1, W2 = 3.3125, 6.75
plt.rcParams.update({"font.size": 8, "legend.fontsize": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "font.family": "DejaVu Sans", "axes.spines.top": False,
    "axes.spines.right": False, "pdf.fonttype": 42, "lines.linewidth": 1.0, "axes.linewidth": 0.6,
    "savefig.facecolor": "white"})
def save(fig, name):
    p = os.path.join(a.out, name)
    fig.savefig(p + ".pdf"); fig.savefig(p + ".png", dpi=600); plt.close(fig)

SCR  = lambda b: arm(b, 0, 0.1, "")                    # scratch, matched schedule
DEF  = lambda b: arm(b, 0, 1.0, "")                    # scratch, default schedule
IND  = lambda b: arm(b, 1, 0.1, "")                    # in-domain pretraining
MUL  = lambda b: arm(b, 1, 0.1, "multi_ip_mae_seed0")  # multi-scene pretraining
SCRc = lambda b: arm(b, 0, 0.1, "", COMMON, 5)         # scratch, common 5-label test set

# ---- Fig 1: accuracy vs budget ------------------------------------------------
fig, ax = plt.subplots(figsize=(W1, 3.5))
spec = [("Scratch", SCR, "#000000", "-", "o", -.09), ("In-domain pretraining", IND, "#0072B2", "-", "s", -.03),
        ("Multi-scene pretraining", MUL, "#D55E00", "-.", "^", .03),
        ("Scratch, default schedule", DEF, "#777777", "--", "D", .09)]
F1 = {}
for lab, f, c, ls, mk, off in spec:
    mu, ci = [], []
    for b in B:
        x = f(b).values; n = len(x)
        mu.append(x.mean()); ci.append(stats.t.ppf(.975, n - 1) * x.std(ddof=1) / np.sqrt(n))
        F1[(lab, b)] = (x.mean(), x.std(ddof=1), n)
    ax.errorbar(np.array(B) + off, mu, yerr=ci, color=c, ls=ls, marker=mk, ms=4, capsize=2, label=lab,
                mfc="white" if mk in "sD" else c)
ax.set_xlabel("Labeled pixels per class"); ax.set_ylabel("mIoU (%)"); ax.set_xticks(B)
ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(.5, 1.0), ncol=1, handlelength=2.6)
fig.tight_layout(); save(fig, "fig1_budget_curves")

# ---- Fig 2: decomposition (Eq. 1) ----------------------------------------------
fig, ax = plt.subplots(figsize=(W1, 2.7)); w = .26
parts = [("Apparent gain", "#009E73", ""), ("Baseline-schedule contribution", "#E69F00", "///"),
         ("Matched pretraining difference", "#0072B2", "...")]
F2 = {}
for i, b in enumerate(B):
    vals = [pdiff(IND(b), DEF(b)), pdiff(SCR(b), DEF(b)), pdiff(IND(b), SCR(b))]
    F2[b] = vals
    for j, (v, (lab, c, h)) in enumerate(zip(vals, parts)):
        ax.bar(i + (j - 1) * w, v[0], w, color=c, hatch=h, edgecolor="black", lw=.5, label=lab if i == 0 else None)
        ax.errorbar(i + (j - 1) * w, v[0], yerr=[[v[0] - v[1]], [v[2] - v[0]]], color="k", lw=.7, capsize=2)
ax.axhline(0, color="k", lw=.6); ax.set_xticks(range(3)); ax.set_xticklabels([f"{b} per class" for b in B])
ax.set_ylabel("Paired difference in mIoU")
ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(.5, 1.0), ncol=1, handlelength=1.6)
fig.tight_layout(); save(fig, "fig2_decomposition")

# ---- Fig 3: seed-count sensitivity (bootstrap: n seeds drawn with replacement) --
# One seeded generator per (budget, n) so the text, the power table and this figure
# report identical bootstrap numbers.
NB = 200_000
def boot_means(d, b, n):
    rng = np.random.default_rng([12345, b, n])
    return d[rng.integers(0, len(d), (NB, n))].mean(1)
fig, axs = plt.subplots(1, 3, figsize=(W2, 2.75), sharey=True)
F3 = {}
for k, (ax, b) in enumerate(zip(axs, B)):
    d = pdiff(IND(b), SCR(b))[3]; assert len(d) == 20
    ns = np.arange(2, 21); P = []; F3[b] = {}
    for n in ns:
        e = boot_means(d, b, n)
        P.append(np.percentile(e, [2.5, 25, 75, 97.5]))
        if n in (3, 5, 10): F3[b][n] = ((e >= 2).mean(), (e < 0).mean())
    P = np.array(P)
    ax.fill_between(ns, P[:, 0], P[:, 3], color="#C6DBEF", lw=0)
    ax.fill_between(ns, P[:, 1], P[:, 2], color="#6BAED6", lw=0)
    ax.plot(ns, P[:, 0], color="#08519C", lw=.5, ls=":"); ax.plot(ns, P[:, 3], color="#08519C", lw=.5, ls=":")
    ax.axhline(d.mean(), color="#08306B", lw=1.2)
    ax.text(19.8, d.mean() + .55, f"{d.mean():+.2f}", ha="right", va="bottom", color="#08306B")
    ax.axhline(0, color="k", lw=.6, ls="--"); ax.axvline(3, color="k", lw=.6, ls="-.")
    ax.set_title(f"({'abc'[k]}) {b} labeled pixels per class", loc="left")
    ax.set_xlabel("Number of seeds"); ax.set_xticks([2, 5, 10, 15, 20])
axs[0].set_ylabel("Estimated pretraining benefit\n(mIoU)")
h = [Patch(facecolor="#C6DBEF", edgecolor="#08519C", lw=.5, ls=":"), Patch(facecolor="#6BAED6", edgecolor="none"),
     Line2D([], [], color="#08306B", lw=1.2)]
fig.tight_layout(rect=(0, .1, 1, 1))
fig.legend(h, ["Central 95% of resampled means", "Central 50% of resampled means", "20-seed estimate"],
           loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(.5, 0))
save(fig, "fig3_seed_sensitivity")

# ---- Fig 4: pretraining corpus vs pretraining length (5/class, seeds 0-9) ------
def rep(tag):
    g = REP[REP.tag == tag]
    assert g.seed.nunique() == len(g) == 10, (tag, len(g))
    return g.set_index("seed").m
fig, ax = plt.subplots(figsize=(W1, 2.6))
grp = [("In-domain\n(2,400)", "ip_mae_seed"), ("In-domain\n(7,200)", "ip_mae_cm_seed"),
       ("Multi-scene\n(7,200)", "multi_ip_mae_seed")]
F4 = {}
for k, (_, pre) in enumerate(grp):
    for ps, off, mk, mfc in [(0, -.12, "o", "#000000"), (99, .12, "s", "white")]:
        v = pdiff(rep(f"{pre}{ps}"), rep("")); F4[(pre, ps)] = v
        ax.errorbar(k + off, v[0], yerr=[[v[0] - v[1]], [v[2] - v[0]]], fmt=mk, color="#000000", mfc=mfc,
                    ms=5, capsize=3, label=f"Pretraining seed {ps}" if k == 0 else None)
ax.axhline(0, color="k", lw=.6); ax.set_xticks(range(3)); ax.set_xticklabels([g[0] for g in grp])
ax.set_xlim(-.5, 2.5); ax.set_xlabel("Pretraining corpus (pretraining updates)"); ax.set_ylabel("Paired difference from\nscratch in mIoU")
ax.legend(frameon=False, loc="upper right")
fig.tight_layout(); save(fig, "fig4_corpus_length")
F4x = {"cm-multi": [pdiff(rep(f"ip_mae_cm_seed{ps}"), rep(f"multi_ip_mae_seed{ps}")) for ps in (0, 99)]}

# ---- Fig 5: additional labels (common test set) vs matched pretraining ---------
fig, ax = plt.subplots(figsize=(W1, 2.7))
lab = [pdiff(SCRc(3), SCRc(2)), pdiff(SCRc(5), SCRc(3))]
pre = [pdiff(IND(2), SCR(2)), pdiff(IND(3), SCR(3))]
for k in range(2):
    for v, x, c, mk, mfc, l in [(lab[k], k - .15, "#000000", "o", "#000000", "Additional labels"),
                                (pre[k], k + .15, "#0072B2", "s", "white", "Masked pretraining")]:
        ax.errorbar(x, v[0], yerr=[[v[0] - v[1]], [v[2] - v[0]]], fmt=mk, color=c, mfc=mfc, ms=5, capsize=3,
                    label=l if k == 0 else None)
ax.axhline(0, color="k", lw=.6); ax.set_xticks([0, 1]); ax.set_xlim(-.6, 1.6)
ax.set_xticklabels(["Starting at\n2 per class", "Starting at\n3 per class"])
ax.set_ylabel("Paired gain in mIoU"); ax.legend(frameon=False, loc="center right")
fig.tight_layout(); save(fig, "fig5_labels_vs_pretraining")

# ---- Self-check against the manuscript (stops if any value drifts) -------------
def chk(name, got, want, tol=0.006):
    ok = abs(got - want) <= tol
    print(f"{'OK  ' if ok else 'FAIL'} {name:46s} got {got:8.3f}  manuscript {want:8.3f}")
    return ok
r = []
for (lab_, b), want in {("Scratch", 2): 35.87, ("Scratch", 3): 42.40, ("Scratch", 5): 51.00,
                        ("In-domain pretraining", 2): 36.64, ("In-domain pretraining", 3): 43.33,
                        ("In-domain pretraining", 5): 51.42}.items():
    r.append(chk(f"Table 3 mean, {lab_}, {b}/class", F1[(lab_, b)][0], want))
r.append(chk("Default scratch mean, 5/class", F1[("Scratch, default schedule", 5)][0], 49.93))
r.append(chk("Default scratch SD, 5/class", F1[("Scratch, default schedule", 5)][1], 3.26))
for b, (ap_, sc, mt) in {2: (0.20, -0.57, 0.77), 3: (1.50, 0.57, 0.94), 5: (1.49, 1.07, 0.43)}.items():
    r += [chk(f"Fig 2 apparent, {b}/class", F2[b][0][0], ap_), chk(f"Fig 2 schedule, {b}/class", F2[b][1][0], sc),
          chk(f"Fig 2 matched, {b}/class", F2[b][2][0], mt)]
r += [chk("Fig 2 apparent CI low, 5/class", F2[5][0][1], -0.09), chk("Fig 2 apparent CI high, 5/class", F2[5][0][2], 3.08),
      chk("Fig 2 schedule CI low, 5/class", F2[5][1][1], -0.27), chk("Fig 2 schedule CI high, 5/class", F2[5][1][2], 2.41)]
r += [chk("Fig 5 +1 label (2->3), common test", lab[0][0], 6.59), chk("  CI low", lab[0][1], 5.19), chk("  CI high", lab[0][2], 8.00),
      chk("Fig 5 +2 labels (3->5), common test", lab[1][0], 8.93), chk("  CI low", lab[1][1], 7.48), chk("  CI high", lab[1][2], 10.39)]
for b, vals in {2: {3: (.312, .361), 5: (.255, .328), 10: (.169, .269)},
                3: {3: (.320, .333), 5: (.270, .292), 10: (.194, .222)},
                5: {3: (.160, .390), 5: (.100, .364), 10: (.037, .313)}}.items():
    for n, (ge2, neg) in vals.items():
        g, ng = F3[b][n]
        r += [chk(f"Fig 3 bootstrap n={n} share >= +2, {b}/class", g, ge2, 0.0006),
              chk(f"Fig 3 bootstrap n={n} share < 0, {b}/class", ng, neg, 0.0006)]
for (pre, ps), (m, lo, hi) in {("ip_mae_seed", 0): (0.46, -2.26, 3.18), ("ip_mae_seed", 99): (-0.50, -2.60, 1.60),
        ("ip_mae_cm_seed", 0): (-3.76, -6.12, -1.40), ("ip_mae_cm_seed", 99): (-4.25, -7.33, -1.18),
        ("multi_ip_mae_seed", 0): (-4.86, -8.86, -0.87), ("multi_ip_mae_seed", 99): (-4.04, -7.26, -0.82)}.items():
    v = F4[(pre, ps)]
    r += [chk(f"Fig 4 {pre}{ps} - scratch", v[0], m), chk("  CI low", v[1], lo), chk("  CI high", v[2], hi)]
r += [chk("Table 4b compute-matched - multi, seed 0", F4x["cm-multi"][0][0], 1.10),
      chk("Table 4b compute-matched - multi, seed 99", F4x["cm-multi"][1][0], -0.22)]
print(f"\n{sum(r)}/{len(r)} checks passed; figures written to {a.out}/")
