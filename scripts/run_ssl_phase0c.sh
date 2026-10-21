#!/usr/bin/env bash
# Phase 0c: (F4) compute-matched IP-only pretraining; (F5) common-test-set label gains.
cd /mnt/c/Users/Safae/thesis
source ~/mamba_env312/bin/activate
OUT=results/ssl_phase0c.csv
while tmux has-session -t =day2 2>/dev/null || tmux has-session -t =p0 2>/dev/null \
   || tmux has-session -t =p0b 2>/dev/null; do sleep 60; done
run () { echo "### START $* $(date +%T)"
         PYTHONUNBUFFERED=1 python -u scripts/run_budget.py --dataset ip --enc_lr_mult 0.1 --out $OUT "$@"
         echo "### DONE $(date +%T)"; }
echo "### START compute-matched pretraining $(date +%T)"
python -u scripts/pretrain_ssl.py --dataset ip --epochs 300 --tiles_per_epoch 192 --seed 0 --tag _cm
echo "### DONE pretraining $(date +%T)"
for S in 0 1 2 3 4 5 6 7 8 9; do
  run --budget 5 --seed $S --ssl ~/hsi_runs/ssl/ip_mae_cm_seed0.pt
done
for S in $(seq 0 19); do
  for B in 2 3 5; do run --budget $B --seed $S --test_budget 5; done
done
