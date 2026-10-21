#!/usr/bin/env bash
# SSL paper, revision item 1: Pavia University replication of the matched comparison.
# PaviaU-only masked pretraining with the Indian Pines settings (300 epochs, 2,400 updates), pretraining
# seeds 0, 99, 7; matched fine-tuning (backbone x0.1) for scratch and for each checkpoint; budgets 2/3/5;
# downstream seeds 0-19. Order: stage A = pretraining seed 0 + scratch (the replication itself),
# stage B = checkpoint 99, stage C = checkpoint 7 (pretraining-checkpoint x downstream-seed crossing).
# Restart-safe: existing checkpoints and finished runs are skipped.
# Parallel use: run two workers on disjoint seeds with separate CSVs, e.g.
#   SEEDS="0 2 4 ... 18" OUT=results/ssl_paviau.csv   and   SEEDS="1 3 ... 19" OUT=results/ssl_paviau_w2.csv
# Pretraining is serialised with a lock, so only one worker trains each checkpoint.
set -u
cd /mnt/c/Users/Safae/thesis || exit 1
source ~/mamba_env312/bin/activate
DS=paviau
OUT=${OUT:-results/ssl_paviau.csv}
SEEDS=${SEEDS:-$(seq 0 19)}
CK=~/hsi_runs/ssl

done_already () {  # $1 budget, $2 seed, $3 checkpoint tag ("" = scratch)
  python - "$OUT" "$1" "$2" "$3" << 'PY'
import csv, sys
out, b, s, tag = sys.argv[1:5]
try:
    rows = list(csv.DictReader(open(out)))
except FileNotFoundError:
    sys.exit(1)
def ok(r):
    if r.get("budget") != b or r.get("seed") != s or r.get("freeze", "0") not in ("0", ""):
        return False
    if r.get("enc_lr_mult") in (None, "") or abs(float(r["enc_lr_mult"]) - 0.1) > 1e-9:
        return False
    t = (r.get("ssl_tag") or "").strip()
    return (r.get("ssl") == "0" and t in ("", "none")) if tag == "" else (r.get("ssl") == "1" and t == tag)
sys.exit(0 if any(ok(r) for r in rows) else 1)
PY
}

pretrain () {  # $1 pretraining seed
  local f="$CK/${DS}_mae_seed$1.pt"
  (
    flock 9
    if [ ! -f "$f" ]; then
      echo "### PRETRAIN $DS seed $1 $(date +%T)"
      PYTHONUNBUFFERED=1 python -u scripts/pretrain_ssl.py --dataset "$DS" --seed "$1" || exit 1
      echo "### PRETRAIN DONE seed $1 $(date +%T)"
    fi
  ) 9>"/tmp/pu_pretrain_seed$1.lock" || { echo "### PRETRAIN FAILED seed $1"; exit 1; }
  [ -f "$f" ] || { echo "### checkpoint $f not found after pretraining"; exit 1; }
}

ft () {  # $1 budget, $2 downstream seed, $3 pretraining seed ("" = scratch)
  local tag="" extra=()
  if [ -n "$3" ]; then tag="${DS}_mae_seed$3"; extra=(--ssl "$CK/$tag.pt"); fi
  if done_already "$1" "$2" "$tag"; then return; fi
  echo "### START b$1 s$2 ${tag:-scratch} $(date +%T)"
  PYTHONUNBUFFERED=1 python -u scripts/run_budget.py --dataset "$DS" --budget "$1" --seed "$2" "${extra[@]}" \
    --enc_lr_mult 0.1 --out "$OUT" || echo "### FAILED b$1 s$2 ${tag:-scratch}"
  echo "### DONE $(date +%T)"
}

pretrain 0
for S in $SEEDS; do for B in 2 3 5; do ft $B $S ""; ft $B $S 0; done; done
echo "### STAGE A DONE $(date +%T)"
pretrain 99
for S in $SEEDS; do for B in 2 3 5; do ft $B $S 99; done; done
echo "### STAGE B DONE $(date +%T)"
pretrain 7
for S in $SEEDS; do for B in 2 3 5; do ft $B $S 7; done; done
echo "### ALL DONE $(date +%T)"
