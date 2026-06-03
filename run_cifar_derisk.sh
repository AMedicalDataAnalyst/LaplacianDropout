#!/usr/bin/env bash
# CIFAR de-risk: 2 runs on CIFAR-10 (baseline + band_drop_all+augmix), seed 0,
# to get a binary signal on whether band_drop_all helps CIFAR-C before
# committing to the 12-day full sweep.
# Total: ~3h baseline + ~16h combo + ~1h eval = ~20h.
set -e
cd "$(dirname "$0")"

DATA=/mnt/c/Users/JoyToy/Documents/Projects/data
SNAP=./cifar_snapshots
RESULTS=./cifar_results
mkdir -p "$SNAP" "$RESULTS"

WRN_ARGS="--model wrn --layers 28 --widen-factor 10 --epochs 100 --batch-size 128 --learning-rate 0.1 --num-workers 2"
EVAL_ARGS="--layers 28 --widen-factor 10 --c-subsample 1000 --pgd-subsample 2000"

run_one() {
  local METHOD=$1 DATASET=$2 SEED=$3
  local TAG="${METHOD//+/_p_}"
  local SAVE="$SNAP/${DATASET}_${TAG}_seed${SEED}"
  local CKPT="$SAVE/checkpoint.pth.tar"
  local OUT="$RESULTS/cifar_eval_${DATASET}_${TAG}_seed${SEED}.json"

  if [[ -f "$OUT" ]]; then
    echo "[skip] $OUT already exists"; return 0
  fi

  echo "========================================================================"
  echo "  $(date -Iseconds)  dataset=$DATASET  method=$METHOD  seed=$SEED"
  echo "========================================================================"
  mkdir -p "$SAVE"

  python3 -u augmix/cifar.py \
    --dataset "$DATASET" $WRN_ARGS \
    --data-path "$DATA" \
    --save "$SAVE" --aug-method "$METHOD" --band-levels 4 --seed "$SEED"

  echo "--- eval ---"
  local CDIR
  if [[ "$DATASET" == "cifar10" ]]; then CDIR="$DATA/CIFAR-10-C"; else CDIR="$DATA/CIFAR-100-C"; fi
  python3 -u cifar_eval.py --checkpoint "$CKPT" \
    --dataset "$DATASET" $EVAL_ARGS \
    --data-path "$DATA" --c-dir "$CDIR" --out "$OUT"
  python3 -u cifar_aggregate.py || true
}

# Baseline first (mCE denominator).
run_one "baseline" "cifar10" "0"
run_one "band_drop_all+augmix" "cifar10" "0"

echo "CIFAR_DERISK_DONE  $(date -Iseconds)"
