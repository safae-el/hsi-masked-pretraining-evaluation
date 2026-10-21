#!/usr/bin/env bash
# Phase 0: NaN-fix confirmation (budget 2) + random-vs-MAE linear probe (budget 5).
cd /mnt/c/Users/Safae/thesis
source ~/mamba_env312/bin/activate
CK=~/hsi_runs/ssl/ip_mae_seed0.pt
OUT=results/ssl_phase0.csv
while tmux has-session -t =day2 2>/dev/null; do sleep 60; done
run () { echo "### START $* $(date +%T)"
         PYTHONUNBUFFERED=1 python -u scripts/run_budget.py --dataset ip --out $OUT "$@"
         echo "### DONE $(date +%T)"; }
for S in 0 1 2; do
  run --budget 2 --seed $S --enc_lr_mult 0.1
  run --budget 2 --seed $S --enc_lr_mult 0.1 --ssl $CK
done
for S in 0 1 2 3 4; do
  run --budget 5 --seed $S --freeze
  run --budget 5 --seed $S --freeze --ssl $CK
done
