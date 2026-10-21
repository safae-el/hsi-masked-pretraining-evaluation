#!/usr/bin/env bash
cd /mnt/c/Users/Safae/thesis
source ~/mamba_env312/bin/activate
OUT=results/ssl_pretrained_x1.csv
CKPT=~/hsi_runs/ssl/ip_mae_seed0.pt
test -f $CKPT || { echo "### CHECKPOINT MISSING: $CKPT"; exit 1; }
while tmux has-session -t =e2 2>/dev/null; do echo "waiting for e2 to finish $(date +%T)"; sleep 300; done
done_already () {
  python - "$OUT" "$1" "$2" << 'PY'
import csv, sys
out, b, s = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    rows = list(csv.DictReader(open(out)))
except FileNotFoundError:
    sys.exit(1)
ok = any(r["budget"] == b and r["seed"] == s and r["ssl"] == "1" and r.get("freeze", "0") == "0"
         and r.get("enc_lr_mult") not in (None, "") and abs(float(r["enc_lr_mult"]) - 1.0) < 1e-9
         for r in rows)
sys.exit(0 if ok else 1)
PY
}
for S in $(seq 0 19); do
  for B in 2 3 5; do
    if done_already $B $S; then echo "skip budget $B seed $S"; continue; fi
    echo "### START budget $B seed $S $(date +%T)"
    PYTHONUNBUFFERED=1 python -u scripts/run_budget.py --dataset ip --budget $B --seed $S \
      --ssl $CKPT --enc_lr_mult 1.0 --out $OUT
    echo "### DONE $(date +%T)"
  done
done
echo "### ALL DONE $(date +%T)"
