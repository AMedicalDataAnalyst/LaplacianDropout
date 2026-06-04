#!/usr/bin/env bash
# Re-run seed 0 of hf_only and lf_only with --save-checkpoints, then
# full canonical Imagenette-C eval (15 corruptions × {1,3,5} × full val)
# on each saved checkpoint. The original 3-seed runs didn't save .pt files
# so we have to re-train; with --seed-offset 0 and explicit seeding we get
# a representative seed for the per-corruption table.
set -e
cd "$(dirname "$0")"

stamp() { date -Iseconds; }

for MASK in hf_only lf_only; do
  echo "============================================================"
  echo "  $(stamp)  TRAIN seed 0 with --band-mask $MASK + save ckpt"
  echo "============================================================"
  python3 -u compare_methods.py \
    --methods band_drop_all \
    --runs 1 --max-epochs 50 --use-jsd \
    --batch-size 96 --corruption-eval off \
    --resolution 224 --pad 16 \
    --band-mask "$MASK" \
    --seed-offset 0 \
    --save-checkpoints \
    --out "/tmp/discard_$MASK.json"

  CKPT="checkpoints/band_drop_all_r0.pt"
  EVAL_CKPT="checkpoints/band_drop_all_${MASK}_seed0.pt"
  mv "$CKPT" "$EVAL_CKPT"
  echo "  saved -> $EVAL_CKPT"

  echo "============================================================"
  echo "  $(stamp)  FULL eval on $EVAL_CKPT"
  echo "============================================================"
  python3 -u imagenette_curriculum.py \
    --eval-only --load-checkpoint "$EVAL_CKPT" \
    --corruption-eval full --resolution 224 --pad 16 \
    --out "results/phase2_reviewer_defense/imagenette_results_FULL_224_bandmask_${MASK}_seed0.json"
done

echo "BANDMASK_FULLEVAL_DONE  $(stamp)"
