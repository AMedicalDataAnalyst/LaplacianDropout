#!/usr/bin/env bash
# Retrain band_drop_all (no band-mask) seed 0 with explicit seeding + save,
# then run the 7-attack band-targeted PGD sweep on it, then regenerate
# the plot + table.
set -e
cd "$(dirname "$0")"

stamp() { date -Iseconds; }

echo "===== $(stamp)  retrain band_drop_all seed 0 ====="
python3 -u compare_methods.py \
  --methods band_drop_all \
  --runs 1 --max-epochs 50 --use-jsd \
  --batch-size 96 --corruption-eval off \
  --resolution 224 --pad 16 \
  --seed-offset 0 \
  --save-checkpoints \
  --out /tmp/discard_bd_only.json

# compare_methods.py --save-checkpoints writes to
# checkpoints/{method.replace('+','_')}_r{run}.pt
test -f checkpoints/band_drop_all_r0.pt
echo "  ckpt saved at checkpoints/band_drop_all_r0.pt"

echo "===== $(stamp)  band-targeted PGD on the recovered ckpt ====="
for TARGET in none 0 1 2 3 4 5; do
  OUT="results/phase2_targeted_pgd/bd_only_target_${TARGET}.json"
  if [[ -f "$OUT" ]]; then
    echo "[skip] $OUT"; continue
  fi
  python3 -u band_targeted_pgd.py \
    --checkpoint checkpoints/band_drop_all_r0.pt \
    --band-target "$TARGET" \
    --eps 4 --steps 20 --subsample 1000 \
    --out "$OUT" 2>&1 | grep -E "adv acc|target:"
done

echo "===== $(stamp)  regenerate plot + table ====="
python3 plot_band_targeted_pgd.py

echo "BD_ONLY_RECOVERY_DONE $(stamp)"
