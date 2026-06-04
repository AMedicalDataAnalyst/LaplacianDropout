#!/usr/bin/env bash
# Sweep band-targeted PGD across 5 models × 7 attacks (standard + 6 band targets).
# Subsample 1000 / eps 4 / 20 steps; ~30s per attack → ~18 min total.
set -e
cd "$(dirname "$0")"

mkdir -p results/phase2_targeted_pgd

declare -A CKPTS=(
  [baseline]=checkpoints/baseline_r0.pt
  [bd_augmix]=checkpoints/band_drop_all_augmix_r0.pt
  [bd_only]=checkpoints/band_drop_all_r0.pt
  [hf_only]=checkpoints/band_drop_all_hf_only_seed0.pt
  [lf_only]=checkpoints/band_drop_all_lf_only_seed0.pt
)

for NAME in baseline bd_augmix bd_only hf_only lf_only; do
  CKPT="${CKPTS[$NAME]}"
  for TARGET in none 0 1 2 3 4 5; do
    OUT="results/phase2_targeted_pgd/${NAME}_target_${TARGET}.json"
    if [[ -f "$OUT" ]]; then
      echo "[skip] $OUT"; continue
    fi
    echo "===== $(date -Iseconds)  $NAME  target=$TARGET ====="
    python3 -u band_targeted_pgd.py \
      --checkpoint "$CKPT" \
      --band-target "$TARGET" \
      --eps 4 --steps 20 --subsample 1000 \
      --out "$OUT" 2>&1 | grep -E "adv acc|target:"
  done
done

echo "BAND_TARGETED_PGD_DONE $(date -Iseconds)"
