#!/usr/bin/env bash
# Reviewer-defense control runs (see docs/REVIEWER_DEFENSE.md, Control A).
# Tests whether random multi-band masking is doing the work, vs operator-family
# memorization. Order: Imagenette runs first (~2h each, faster signal), then
# the slower CIFAR runs (~16h each).
# Total: ~36h sequential.
set -e
cd "$(dirname "$0")"

DATA=/mnt/c/Users/JoyToy/Documents/Projects/data
SNAP=./cifar_snapshots
RESULTS=./cifar_results
mkdir -p "$SNAP" "$RESULTS"

WRN_ARGS="--model wrn --layers 28 --widen-factor 10 --epochs 100 --batch-size 128 --learning-rate 0.1 --num-workers 2"
EVAL_ARGS="--layers 28 --widen-factor 10 --c-subsample 1000 --pgd-subsample 2000"

stamp() { date -Iseconds; }

# ------- Imagenette runs first (fast: ~2h each, where Phase-1 win was largest) -------

for MASK in hf_only lf_only; do
  OUT="bandmask_${MASK}_224.json"
  if [[ -f "$OUT" ]]; then
    echo "[skip $(stamp)] $OUT already exists"; continue
  fi
  echo "======================================================================"
  echo "  $(stamp)  Imagenette-224 band_drop_all+JSD --band-mask $MASK"
  echo "======================================================================"
  python3 -u compare_methods.py \
    --methods band_drop_all \
    --runs 1 --max-epochs 50 --use-jsd \
    --batch-size 96 --corruption-eval fast \
    --resolution 224 --pad 16 \
    --band-mask "$MASK" \
    --out "$OUT"
done

# ------- CIFAR runs SKIPPED (de-risk showed Pareto LOSS on CIFAR-32) -------
# 2026-06-03 decision: redirect CIFAR's 32h of compute to 3-seed Imagenette
# hf_only / lf_only runs. CIFAR transfer failure is documented in
# docs/PHASE2_STATUS.md (Gate A: combo mCE 163 vs baseline 100).

# ------- 3-seed Imagenette hf_only / lf_only -------
# Repeats seed 0 inside --runs 3 (compare_methods doesn't expose a start seed).
# The single-seed seed-0 JSONs from above remain on disk for cross-check.
# Total: ~6h (3h per mask).

for MASK in hf_only lf_only; do
  OUT="bandmask_${MASK}_224_3seed.json"
  if [[ -f "$OUT" ]]; then
    echo "[skip $(stamp)] $OUT already exists"; continue
  fi
  echo "======================================================================"
  echo "  $(stamp)  Imagenette-224 band_drop_all+JSD --band-mask $MASK  (3 seeds)"
  echo "======================================================================"
  python3 -u compare_methods.py \
    --methods band_drop_all \
    --runs 3 --max-epochs 50 --use-jsd \
    --batch-size 96 --corruption-eval fast \
    --resolution 224 --pad 16 \
    --band-mask "$MASK" \
    --out "$OUT"
done

echo "REVIEWER_DEFENSE_DONE  $(stamp)"
