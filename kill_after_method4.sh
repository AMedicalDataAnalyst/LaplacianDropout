#!/usr/bin/env bash
# Waits for method 4 (band_drop_all+augmix+pca_color) output to appear,
# then kills the subtractive ablation chain so methods 5-9 don't run.
# Per option-C decision 2026-06-05: stop after single-axis attribution.
set -u
cd "$(dirname "$0")"

TRIGGER=results/phase2_subtractive/band_drop_all_p_augmix_p_pca_color_224_3seed.json
TIMEOUT=43200   # 12 h max wait
WAITED=0

echo "$(date -Iseconds)  waiting for $TRIGGER"
while [[ ! -f "$TRIGGER" ]]; do
  sleep 120
  WAITED=$((WAITED + 120))
  if [[ $WAITED -ge $TIMEOUT ]]; then
    echo "$(date -Iseconds)  TIMEOUT — exiting without killing."
    exit 1
  fi
done

# Let the JSON write fully settle.
sleep 15

echo "$(date -Iseconds)  trigger seen — killing subtractive sweep chain"
pkill -TERM -f "run_subtractive_ablation.sh" 2>/dev/null || true
sleep 3
pkill -KILL -f "run_subtractive_ablation.sh" 2>/dev/null || true
# Kill any compare_methods.py currently mid-run on a subtractive method
pkill -KILL -f "compare_methods.py.*bit_depth\|compare_methods.py.*pca_color" 2>/dev/null || true

echo "$(date -Iseconds)  done"
