#!/usr/bin/env bash
# Band-count ablation @ 224. For each levels in {1..6} (= bands 2..7) train
# band_drop_all + JSD on Imagenette, single seed, fast corruption eval. Each
# run writes its own JSON keyed by the band count.
set -e
cd "$(dirname "$0")"

for L in 1 2 3 4 5 6; do
  BANDS=$((L + 1))
  OUT="band_count_ablation_L${L}_bands${BANDS}.json"
  echo "======================================================================"
  echo "  levels=$L  (bands=$BANDS)  ->  $OUT"
  echo "======================================================================"
  python3 -u compare_methods.py \
    --methods band_drop_all \
    --runs 1 \
    --max-epochs 50 \
    --use-jsd \
    --batch-size 96 \
    --corruption-eval fast \
    --resolution 224 \
    --pad 16 \
    --band-levels "$L" \
    --out "$OUT"
done

echo "BAND_COUNT_ABLATION_DONE"
