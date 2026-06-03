#!/usr/bin/env bash
# CIFAR tier-2 sweep: 5 methods × 2 datasets × 3 seeds = 30 runs.
# Each run: train (~3-14h depending on method) → cifar_eval (~30 min).
# Total ETA: ~12 days at WRN-28-10 on this GPU. Run as:
#   nohup bash run_cifar_sweep.sh > run_cifar_sweep.log 2>&1 &
set -e
cd "$(dirname "$0")"

DATA=/mnt/c/Users/JoyToy/Documents/Projects/data
SNAP=./cifar_snapshots
RESULTS=./cifar_results
mkdir -p "$SNAP" "$RESULTS"

# WRN-28-10 + 100 epochs + batch 128 + SGD cosine + LR 0.1 (with grad clip)
WRN_ARGS="--model wrn --layers 28 --widen-factor 10 --epochs 100 --batch-size 128 --learning-rate 0.1 --num-workers 2"
EVAL_ARGS="--layers 28 --widen-factor 10 --c-subsample 1000 --pgd-subsample 2000"

# Mixing pool for the (faithful-fractals-pending) PixMix runs.
# Tier-1 used Imagenette train as the fallback for fractals; same here.
MIX=/mnt/c/Users/JoyToy/Documents/Projects/data/imagenette2-320/train

run_one() {
  local METHOD=$1 DATASET=$2 SEED=$3
  local TAG="${METHOD//+/_p_}"   # filename-friendly: 'band_drop_all+augmix' -> 'band_drop_all_p_augmix'
  local SAVE="$SNAP/${DATASET}_${TAG}_seed${SEED}"
  local CKPT="$SAVE/checkpoint.pth.tar"
  local OUT="$RESULTS/cifar_eval_${DATASET}_${TAG}_seed${SEED}.json"

  if [[ -f "$OUT" ]]; then
    echo "[skip] $OUT already exists"
    return 0
  fi

  echo "========================================================================"
  echo "  $(date -Iseconds)  dataset=$DATASET  method=$METHOD  seed=$SEED  -> $OUT"
  echo "========================================================================"
  mkdir -p "$SAVE"

  if [[ "$METHOD" == "pixmix" ]]; then
    # PixMix uses its own harness; --mixing-set required.
    python3 -u pixmix/cifar.py \
      --dataset "$DATASET" $WRN_ARGS --droprate 0.0 \
      --data-path "$DATA" --mixing-set "$MIX" \
      --save "$SAVE" --aug-method pixmix --seed "$SEED"
  else
    # All other methods through augmix/cifar.py.
    python3 -u augmix/cifar.py \
      --dataset "$DATASET" $WRN_ARGS \
      --data-path "$DATA" \
      --save "$SAVE" --aug-method "$METHOD" --band-levels 4 --seed "$SEED"
  fi

  echo "--- eval ---"
  local CDIR
  if [[ "$DATASET" == "cifar10" ]]; then CDIR="$DATA/CIFAR-10-C"; else CDIR="$DATA/CIFAR-100-C"; fi
  python3 -u cifar_eval.py --checkpoint "$CKPT" \
    --dataset "$DATASET" $EVAL_ARGS \
    --data-path "$DATA" --c-dir "$CDIR" --out "$OUT"
  python3 -u cifar_aggregate.py || true
}

# Run order: do baseline first (mCE normalization needs it). All baseline
# CIFAR-10 then all baseline CIFAR-100. Then the rest, smallest run first.
for SEED in 0 1 2; do
  for DS in cifar10 cifar100; do
    run_one "baseline" "$DS" "$SEED"
  done
done
for SEED in 0 1 2; do
  for DS in cifar10 cifar100; do
    run_one "augmix" "$DS" "$SEED"
  done
done
for SEED in 0 1 2; do
  for DS in cifar10 cifar100; do
    run_one "pixmix" "$DS" "$SEED"
  done
done
for SEED in 0 1 2; do
  for DS in cifar10 cifar100; do
    run_one "band_drop_all" "$DS" "$SEED"
  done
done
for SEED in 0 1 2; do
  for DS in cifar10 cifar100; do
    run_one "band_drop_all+augmix" "$DS" "$SEED"
  done
done

echo "CIFAR_SWEEP_DONE  $(date -Iseconds)"
