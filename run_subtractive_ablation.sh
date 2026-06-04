#!/usr/bin/env bash
# Phase 2 subtractive-ablation sweep at Imagenette-224.
#
# 9 methods × 3 seeds, JSD on, fast corruption eval. ~3 h per method ≈ ~27 h
# total sequential. Resumeable: each method writes one JSON; skips any that
# already exist.
#
# Order: most-interesting-first — full stack (the publishable headline
# candidate), then progressive deletions to attribute the contribution.
#
# Reference points (already on disk, NOT re-run here):
#   baseline                              0.566 corr (3-seed mean)
#   augmix+JSD                            0.701
#   band_drop_all+JSD                     0.806
#   band_drop_all+augmix+JSD (HEADLINE)   0.826
# in results/phase1_comparison/comparison_224_main.json
set -e
cd "$(dirname "$0")"
mkdir -p results/phase2_subtractive

# Common flags
COMMON="--runs 3 --max-epochs 50 --use-jsd --batch-size 96 --corruption-eval fast --resolution 224 --pad 16 --seed-offset 0"

stamp() { date -Iseconds; }

# Format: method-name : output-file-suffix
declare -a JOBS=(
  # Most interesting first
  "band_drop_all+augmix+bit_depth+pca_color"
  "band_drop_all+bit_depth+pca_color"
  "band_drop_all+augmix+bit_depth"
  "band_drop_all+augmix+pca_color"
  "band_drop_all+bit_depth"
  "band_drop_all+pca_color"
  "bit_depth+pca_color"
  "bit_depth"
  "pca_color"
)

for METHOD in "${JOBS[@]}"; do
  TAG="${METHOD//+/_p_}"
  OUT="results/phase2_subtractive/${TAG}_224_3seed.json"
  if [[ -f "$OUT" ]]; then
    echo "[skip $(stamp)] $OUT exists"
    continue
  fi
  echo "============================================================"
  echo "  $(stamp)  $METHOD  (3 seeds)"
  echo "============================================================"
  python3 -u compare_methods.py \
    --methods "$METHOD" \
    $COMMON \
    --out "$OUT"
done

echo "SUBTRACTIVE_ABLATION_DONE  $(stamp)"
