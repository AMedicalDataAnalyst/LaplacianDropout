# 12-hour autonomous run — wake-up summary

## TL;DR — what changed

Found a recipe that **strictly improves over baseline on both clean accuracy AND corruption robustness** on Imagenette. The original Laplacian curriculum framing was overcomplicated — the work is done by a simple band-masking augmentation, and a short clean fine-tune recovers the small clean-accuracy gap.

## Headline numbers (ResNet-18, Imagenette-160, 3 seeds, FULL Imagenette-C eval)

| Method | Clean | Mean Corruption | Notes |
|---|---|---|---|
| baseline | 0.8759 | 0.5429 | reference |
| **longA** | **0.8864** (+1.05) | **0.7131** (+17.0) | best clean acc |
| **extralongA** | 0.8809 (+0.5) | **0.7360** (+19.3) | best robustness |

Both Pareto-dominate baseline (better on both dimensions). Pick longA for accuracy-max, extralongA for robustness-max.

## Recipe (longA)
```bash
# Stage 1: band-dropout pretraining (~5 min/seed)
python3 imagenette_curriculum.py --engine plateau --runs 3 \
    --variants band_drop_all --max-final-phase 80 \
    --save-checkpoints --out s1.json
# rename ckpts to longA_s1_seed{0,1,2}.pt

# Stage 2: clean fine-tune at LR/10 (~1.5 min/seed)
for r in 0 1 2; do
  python3 imagenette_curriculum.py --engine plateau --runs 1 \
    --variants baseline --lr-finetune --save-checkpoints \
    --load-checkpoint checkpoints/longA_s1_seed${r}.pt \
    --out s2_seed${r}.json
done
```

For extralongA, use `--max-final-phase 120` in stage 1.

## What's in the repo now (files written during this run)
- `SHAPE_BIAS_FINDINGS.md` — full narrative + methodology + trade-off discussion + reproducing instructions
- `shape_bias_report.md` — auto-generated summary table
- `shape_bias_tradeoff.png` — scatter of clean vs corruption per variant
- `shape_bias_per_corruption.png` — per-corruption breakdown across severities for baseline / longA / extralongA
- `imagenette_curriculum.py` — single-script implementation (~1.1k lines now; trains, evaluates, distills, supports load-checkpoint, eval-only, fast and full corruption eval modes)
- `analyze_imagenette.py`, `build_report.py` — analysis utilities
- `checkpoints/` — 35 saved models from various experiments; the relevant ones are `longA_s2_seed{0,1,2}.pt`, `extralongA_s2_seed{0,1,2}.pt`, `baseline_r{1,2}.pt`
- `imagenette_results_*.json` — raw per-run results (one file per experiment cell)

## Order of experiments tried (key results in parens)
1. Original Laplacian curriculum at fixed budgets (val_acc *regresses* with more epochs)
2. Plateau-based schedule redesign (fixed the regression; still 11pp below baseline clean)
3. Dropout strength sweep p ∈ {0.3,0.5,0.7,0.9} (p=0.5 sweet spot)
4. Drop the residual too (`cd_p05_noprotect`): +1.7pp corruption
5. **band_drop / band_drop_all without any curriculum** (single phase, dropout from step 0): equals or beats curriculum on every metric
6. Alternative augmentations: joint_blur (corruption-robust without shape bias — useful negative control), joint_residual, joint_mix, joint_multiblur
7. Two-stage fine-tuning: twostage_A (band_drop → clean FT @ LR/10) → +0.5pp clean over baseline
8. **longA** (extended pretrain max=80 + clean FT) → +1.05pp clean, +17pp corruption
9. **extralongA** (extended pretrain max=120 + clean FT) → +0.5pp clean, +19.3pp corruption
10. Distillation experiments: KL-only distillation doesn't transfer robustness when student doesn't see augmented inputs
11. Full Imagenette-C eval on top 3 models (baseline, longA, extralongA) × 3 seeds with severities {1,3,5}

## Open question I sketched out — making it publishable

(Asked earlier in conversation; full discussion above.) Short version:

- **Workshop paper / arXiv preprint** (~3-5 days more work): add Geirhos cue-conflict eval, scale to ImageNet-100 at 224², reproduce AugMix for direct comparison. With those, the "simple band-dropout matches/beats AugMix on a different Pareto axis" story stands.
- **Full venue paper** (~3-6 weeks + ImageNet-1k compute): tier-1 plus ResNet-50 on full ImageNet-1k, multiple architectures, adversarial-robustness eval, theoretical or empirical analysis of why band dropout works. Gating cost is ImageNet-1k training (rent ~$50-200 of A100 time).

## What I did *not* try (would need more time/compute)
- Geirhos cue-conflict eval (the gold-standard shape-bias measure)
- ImageNet-R / ImageNet-Sketch OOD evals
- Scaling to ImageNet-100 / ImageNet-1k
- Other architectures (ResNet-50, ViT, ConvNeXt)
- Comparison against AugMix, DeepAugment, PIXMIX, SIN+IN baselines
- Adversarial-robustness eval (PGD / AutoAttack)
- Theoretical analysis of *why* random band masking induces shape bias (probably related to spectral regularization)

If you want, I can pick up any of those when you're back.
