#!/usr/bin/env bash
# Phase 0b: pretraining-seed replication at 5 labels/class, 10 downstream seeds.
cd /mnt/c/Users/Safae/thesis
source ~/mamba_env312/bin/activate
SSL=~/hsi_runs/ssl
OUT=results/ssl_phase0b.csv
while tmux has-session -t =day2 2>/dev/null || tmux has-session -t =p0 2>/dev/null; do sleep 60; done
echo "### START multi-scene pretraining seed 99 $(date +%T)"
python -u scripts/pretrain_ssl_multi.py --epochs 300 --seed 99
echo "### DONE pretraining $(date +%T)"
run () { echo "### START $* $(date +%T)"
         PYTHONUNBUFFERED=1 python -u scripts/run_budget.py --dataset ip --budget 5 \
             --enc_lr_mult 0.1 --out $OUT "$@"
         echo "### DONE $(date +%T)"; }
for S in 0 1 2 3 4 5 6 7 8 9; do
  run --seed $S
  run --seed $S --ssl $SSL/ip_mae_seed0.pt
  run --seed $S --ssl $SSL/ip_mae_seed99.pt
  run --seed $S --ssl $SSL/multi_ip_mae_seed0.pt
  run --seed $S --ssl $SSL/multi_ip_mae_seed99.pt
done
