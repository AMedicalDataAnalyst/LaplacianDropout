#!/usr/bin/env bash
# Phase 2 weight-EMA comparison at Imagenette-224.
#
# Question: does a simple weight-EMA (Polyak averaging) baseline match or
# beat the band_drop_all augmentation for corruption robustness?
#
# 4 methods × 3 seeds, JSD on (auto-off for baseline), fast corruption eval,
# reference 224² protocol (same COMMON flags as run_subtractive_ablation.sh).
# Each run passively tracks EMA shadows at decay 0.99 and 0.999; the raw
# training trajectory is untouched, so the raw rows double as a same-seed
# control against results/phase1_comparison/comparison_224_main.json:
#   baseline                              0.566 corr (3-seed mean)
#   augmix+JSD                            0.701
#   band_drop_all+JSD                     0.806
#   band_drop_all+augmix+JSD (HEADLINE)   0.826
#
# Decay grid: 0.99 (τ ≈ 1 epoch) and 0.999 (τ ≈ 10 epochs). Slower decays
# (0.9995+) are excluded because with init-from-weights EMA and only ~4900
# steps the random init would retain ≥8% weight at the end of training.
#
# Resumeable: each method writes one JSON; skips any that already exist.
set -e -o pipefail
cd "$(dirname "$0")"
mkdir -p results/phase2_ema logs/phase2

# The 16 GB 5070 Ti OOMed from allocator fragmentation on the 3rd+ run
# within one process (the ~3.5 GB fp16 train cache is re-allocated per run).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY=.venv/bin/python
COMMON="--runs 3 --max-epochs 50 --use-jsd --batch-size 96 --corruption-eval fast --resolution 224 --pad 16 --seed-offset 0 --ema-decays 0.99 0.999"

# Most interesting first: the direct hypothesis test (baseline+EMA vs the
# checked-in band_drop_all rows), then prior-art aug + EMA, then whether
# EMA stacks with the headline, then band_drop_all alone + EMA.
declare -a JOBS=(
  "baseline"
  "augmix"
  "band_drop_all+augmix"
  "band_drop_all"
)

for m in "${JOBS[@]}"; do
  suffix="${m//+/_}"
  out="results/phase2_ema/ema_${suffix}.json"
  if [[ -f "$out" ]]; then
    echo "[$(date -Iseconds)] skip $m — $out exists"
    continue
  fi
  log="logs/phase2/run_ema_${suffix}.txt"
  echo "[$(date -Iseconds)] start $m -> $out"
  if $PY -u compare_methods.py --methods "$m" $COMMON --out "$out" 2>&1 | tee "$log"; then
    echo "[$(date -Iseconds)] done $m"
  else
    echo "[$(date -Iseconds)] FAILED $m (no JSON written; re-run this script to retry)"
  fi
done
echo "[$(date -Iseconds)] EMA comparison sweep complete"
