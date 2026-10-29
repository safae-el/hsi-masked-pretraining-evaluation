# Reliable evaluation of masked pretraining in label-scarce hyperspectral classification

Code, labelled-pixel splits, per-run results, and analysis scripts for

> S. El Jarai and A. Elhassouny, "Reliable Evaluation of Masked Pretraining in Label-Scarce
> Hyperspectral Classification," manuscript submitted to *Remote Sensing Letters*, 2026.

The study quantifies how downstream seed count, baseline fine-tuning schedule, evaluation metric,
and pretraining length affect the estimated benefit of in-domain masked pretraining at 2, 3, and 5
labelled pixels per class, using 20 paired downstream seeds. Indian Pines is the full controlled
analysis; Pavia University replicates the main comparison and seed analysis with three pretraining
checkpoints.

## Repository contents

| Path | Content |
|---|---|
| `hsiseg/` | Python package: dense hybrid Mamba–Transformer network, masked pretraining (`hsiseg/ssl`), scene loading, label-budget splits, training and evaluation loop |
| `configs/` | Dataset and model configuration files |
| `scripts/pretrain_ssl.py` | In-domain masked pretraining |
| `scripts/pretrain_ssl_multi.py` | Target-inclusive multi-scene pretraining (Indian Pines, Pavia University, Salinas) |
| `scripts/run_budget.py` | Fine-tuning and test-set evaluation at a label budget; appends one row per run to a CSV |
| `scripts/run_ssl_*.sh` | The command sequences used for each experiment |
| `scripts/make_budget_splits_paviau.py` | Pavia University split generation, with self-checks against the Indian Pines splits |
| `splits/` | Labeled-pixel splits used in the paper: `ip_budget{2,3,5}_seed{0..19}.npz`, `paviau_budget{2,3,5}_seed{0..19}.npz` |
| `results/` | One row per fine-tuning run: settings, code commit, OA, AA, kappa, mIoU, and class-wise IoU |
| `analysis/` | Statistical analyses, tables, and figures reported in the manuscript |
| `analysis/PREREGISTRATION_paviau.md` | Pavia University analysis plan, with a dated addendum |

## Result files and where they are used

| File | Runs | Manuscript |
|---|---|---|
| `results/budgets.csv` | Indian Pines main 20-seed arms (scratch at ×0.1 and ×1.0, in-domain and multi-scene pretraining) | Secs. 3.1–3.5; Tables 1–2; Figures 1–2 |
| `results/ssl_phase0.csv` | Classifier-only probe | Sec. 3.6 |
| `results/ssl_phase0b.csv` | Pretraining-seed replication (seed 99), 5 labels per class, downstream seeds 0–9 | Sec. 3.5 |
| `results/ssl_phase0c.csv` | Compute-matched in-domain pretraining (seed 0); scratch on the common five-label test set | Secs. 3.5–3.6 |
| `results/ssl_phase0d.csv` | Compute-matched in-domain pretraining (seed 99) | Sec. 3.5 |
| `results/ssl_pretrained_x1.csv` | In-domain pretraining fine-tuned with the default ×1.0 backbone multiplier (2×2 design) | Sec. 3.3 |
| `results/ssl_paviau*.csv` | Pavia University: scratch and three pretraining checkpoints (written by three parallel workers) | Secs. 3.1, 3.2, 3.4; Table 1 |

## Reproducing the statistics, tables, and figures (no GPU needed)

```bash
pip install numpy scipy pandas matplotlib
python analysis/revision_analysis.py --budgets results/budgets.csv --common results/ssl_phase0c.csv --x1 results/ssl_pretrained_x1.csv
python analysis/make_figures.py --budgets results/budgets.csv --common results/ssl_phase0c.csv \
       --replication results/ssl_phase0b.csv --cm99 results/ssl_phase0d.csv --out figures
python analysis/make_supplement.py --budgets results/budgets.csv --x1 results/ssl_pretrained_x1.csv --outdir supplement
python analysis/pu_analysis.py results/ssl_paviau.csv results/ssl_paviau_w2.csv results/ssl_paviau_w3.csv
python analysis/make_pavia_tables.py results tables
python analysis/pu_avg_seed_sensitivity.py results/ssl_paviau.csv results/ssl_paviau_w2.csv results/ssl_paviau_w3.csv
```

`make_figures.py` writes Figures 1 and 2 of the manuscript (`fig3_seed_sensitivity` and `fig2_decomposition`)
and compares 65 values with those reported, stopping if any differ.
`pu_avg_seed_sensitivity.py` gives the Pavia University seed-count and power columns of Table 1.
All bootstrap and permutation analyses use fixed random-number seeds.

## Re-running the experiments

Requirements: Python 3.12, PyTorch 2.5.1 with CUDA 12.1, and `mamba_ssm` 2.3.2.post1
(`requirements.txt`; the full environment used is listed in `environment_freeze.txt`).
Experiments were run on a single NVIDIA GeForce RTX 4080 Laptop GPU with bfloat16 autocast.

Data are not redistributed here. Download the Indian Pines (corrected), Pavia University, and
Salinas (corrected) scenes and ground truth from the hyperspectral scene collection of the
University of the Basque Country
(<https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes>); Indian Pines is
also available from the Purdue University Research Repository (doi:10.4231/R7RX991C). By
default, scenes are read from `~/hsi_data` and splits from `~/hsi_data/splits` (see `configs/`);
copy the files in `splits/` there.

Typical commands (the exact sequences used are in `scripts/run_ssl_*.sh`):

```bash
python scripts/pretrain_ssl.py --dataset paviau --seed 0                       # masked pretraining
python scripts/run_budget.py --dataset paviau --budget 5 --seed 0 --enc_lr_mult 0.1 --out results/example.csv
python scripts/run_budget.py --dataset paviau --budget 5 --seed 0 --enc_lr_mult 0.1 \
       --ssl <checkpoint.pt> --out results/example.csv                         # fine-tune a pretrained model
```

Pretrained checkpoints are not included; they can be regenerated with the pretraining scripts.

## License

Code: MIT License (see `LICENSE`). The hyperspectral scenes are subject to the terms of their providers.
