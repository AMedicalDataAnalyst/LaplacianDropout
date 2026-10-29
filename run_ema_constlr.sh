#!/usr/bin/env bash
# Phase 2 weight-EMA comparison, constant-LR arm, at Imagenette-224.
#
# Follow-up to run_ema_comparison.sh: under the cosine schedule EMA's
# mid-training lead is fully absorbed by the LR tail. This arm gives EMA
# its best case — warmup then FLAT LR, so end-of-training annealing can
# only come from the EMA shadow. Same protocol otherwise (3 seeds, JSD,
# fast corruption eval). The EMA rows here are compared against the
# cosine-trained reference numbers (band_drop_all 0.806, headline 0.826).
#
# Note: plateau early-stop never fires at constant LR, so every run uses
# the full 50-epoch budget (matched or larger than the cosine arm's).
set -e -o pipefail
cd "$(dirname "$0")"
mkdir -p results/phase2_ema logs/phase2

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY=.venv/bin/python
COMMON="--runs 3 --max-epochs 50 --use-jsd --batch-size 96 --corruption-eval fast --resolution 224 --pad 16 --seed-offset 0 --schedule constant --ema-decays 0.99 0.999"

declare -a JOBS=(
  "baseline"
  "augmix"
  "band_drop_all+augmix"
)

for m in "${JOBS[@]}"; do
  suffix="${m//+/_}"
  out="results/phase2_ema/ema_constlr_${suffix}.json"
  if [[ -f "$out" ]]; then
    echo "[$(date -Iseconds)] skip $m — $out exists"
    continue
  fi
  log="logs/phase2/run_ema_constlr_${suffix}.txt"
  echo "[$(date -Iseconds)] start $m -> $out"
  if $PY -u compare_methods.py --methods "$m" $COMMON --out "$out" 2>&1 | tee "$log"; then
    echo "[$(date -Iseconds)] done $m"
  else
    echo "[$(date -Iseconds)] FAILED $m (no JSON written; re-run this script to retry)"
  fi
done
echo "[$(date -Iseconds)] constant-LR EMA sweep complete"
