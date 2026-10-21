#!/usr/bin/env bash
# Phase 0d: compute-matched IP-only pretraining, second pretraining seed.
cd /mnt/c/Users/Safae/thesis
source ~/mamba_env312/bin/activate
while tmux has-session -t =s1 2>/dev/null; do sleep 60; done
echo "### START compute-matched pretraining seed99 $(date +%T)"
python -u scripts/pretrain_ssl.py --dataset ip --epochs 300 --tiles_per_epoch 192 --seed 99 --tag _cm
echo "### DONE pretraining $(date +%T)"
test -f ~/hsi_runs/ssl/ip_mae_cm_seed99.pt || { echo "### CHECKPOINT MISSING"; exit 1; }
for S in 0 1 2 3 4 5 6 7 8 9; do
  echo "### START --seed $S $(date +%T)"
  PYTHONUNBUFFERED=1 python -u scripts/run_budget.py --dataset ip --enc_lr_mult 0.1 \
    --budget 5 --seed $S --ssl ~/hsi_runs/ssl/ip_mae_cm_seed99.pt --out results/ssl_phase0d.csv
  echo "### DONE $(date +%T)"
done
