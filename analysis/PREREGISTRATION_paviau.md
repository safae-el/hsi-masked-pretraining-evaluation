# Pavia University replication: analysis plan fixed before any result was seen

Written 21 October 2026, about 01:00, after the runs started and before any Pavia University
fine-tuning result had been inspected (only the single test run, scratch, 5 labels, seed 0, mIoU 76.34,
was seen as a pipeline check). Decision: option A, the results are included in the JARS submission
whatever they show.

## Design (unchanged from the Indian Pines protocol)
- PaviaU-only masked pretraining, same settings (300 epochs, 2,400 updates, masking 0.6 / 0.3, 8x8 blocks),
  pretraining seeds 0 (stage A), 99 (stage B), 7 (stage C).
- Matched fine-tuning (backbone x0.1) for scratch and pretrained models, budgets 2/3/5 labels per class,
  downstream seeds 0-19, final-epoch test metrics, same split rule and label smoothing.

## What will be reported (all of it, in the paper and the supplement)
1. Primary: pretrained (checkpoint seed 0) minus scratch, mIoU, at 2/3/5 labels: mean paired difference,
   95% t CI, paired t-test, Holm across the three budgets, Wilcoxon, TOST at +-2 and +-3.
2. OA, AA, kappa differences with 95% CIs.
3. Seed sensitivity: paired SD, 80%-power detectable effect with 20 seeds, zero-centred three-seed rate
   of estimates >= +2.
4. If stages B and C are complete: per-checkpoint differences; checkpoint vs seed variance components
   (one-way random-effects decomposition of the paired differences); mean over checkpoints.

## Rules fixed now
- Stage A is reported only when all 120 runs (2 arms x 3 budgets x 20 seeds) are complete; no partial report.
- Stages B/C are reported if complete at submission time; otherwise they go to the revision, stated as such.
- Same selection rules as Indian Pines (explicit enc_lr_mult, first row per seed if duplicated).
- Wording by outcome: consistent small/non-significant -> "replicated on Pavia University";
  clear positive gain -> state it plainly and scope the small-gain result to Indian Pines;
  negative -> report as is. The abstract and conclusion are updated to match whichever holds.

## Addendum, 21 October 2026, about 13:45 (after stages B and C completed, all 240 runs)
- Item 4 is reported as planned: per-checkpoint differences and the one-way random-effects
  checkpoint-vs-seed decomposition (pu_analysis.py), plus the mean over checkpoints.
- Deviation, stated in the supplement: pu_analysis.py printed a t interval over the three checkpoint
  means (2 df). That interval treats the checkpoint means as independent although all three share the
  same scratch runs and downstream seeds, so it understates uncertainty (at 5 labels it gives
  [+0.18, +2.41]). The paper instead reports the per-seed average of the three paired differences with a
  20-seed t interval ([-0.08, +2.67] at 5 labels), the more conservative of the two.
- The primary analysis remains checkpoint seed 0, as fixed above.
