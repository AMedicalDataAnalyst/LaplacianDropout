# Phase 1 — Imagenette / tier-1 (COMPLETE)

The full table of numbers lives in `TIER1_FINDINGS.md`; this is the
high-altitude summary.

## What we set out to test

User's original spec (`docs/historical/INSTRUCTIONS_CURRICULUM.md`):
"Assess whether a Laplacian frequency curriculum achieves better
performance than baseline on CIFAR-10. Care more about final accuracy and
beating baseline than speed; longer runs allowed if performance keeps
increasing; double the epochs; avoid playing with other hyperparameters."

## What we actually established (in chronological order)

1. **Curriculum doesn't beat baseline on CIFAR clean accuracy.** Across
   1× — 64× epoch budgets, the curriculum and curriculum-dropout variants
   lost to vanilla baseline by 1–4 pp clean accuracy on CIFAR-10. CIFAR-10
   was too saturated for the technique to show.
2. **Pivot to Imagenette** at 160² and then 224². Same pattern: curriculum
   never beat baseline on *clean accuracy*. But we discovered that
   per-sample random `band_drop_all` augmentation — the actual mechanism
   inside the curriculum — *did* yield major corruption-robustness gains,
   independent of any curriculum schedule.
3. **`band_drop_all` is the mechanism**, not the curriculum. Per training
   image, decompose into 6 Laplacian bands (5 bandpass at σ ∈ {1, 2, 4, 8,
   16} + residual) and drop each independently with p=0.5. One line, no
   schedule, applied from step 0.
4. **Combination beats either alone.** `band_drop_all` applied on top of
   AugMix (with the JSD consistency loss) is the Pareto winner: stronger
   corruption robustness than AugMix alone *and* AugMix's clean accuracy.

## Headline results

**Imagenette-224 full eval, canonical 15 corruptions × {sev 1,3,5} × full
3925-image val set, single seed, normalized to the 224 baseline:**

| Method | Clean | Mean Corr (15) | mCE (rel) |
|---|---|---|---|
| baseline (no aug) | 0.896 | 0.573 | **100.0** |
| **band_drop_all + AugMix + JSD** | **0.903** | **0.820** | **47.2** |

- **52.8% relative reduction in mean corruption error.**
- Strict Pareto: both clean (+0.7 pp) and corruption (+24.7 pp) improved.
- Wins on every one of 15 corruptions. Biggest gains: contrast +51 pp,
  defocus_blur +45 pp, fog +36 pp, glass_blur +34 pp.

The corresponding *per-corruption* breakdown matches a clean spectral
story: methods win biggest on spectrally-narrow corruptions
(blur/contrast/fog/noise), and barely move on broadband corruptions
(jpeg_compression +0.7 pp).

## What we also learned (decided but not headline)

- **Band count saturates at 5–6 bands.** Going from 2 to 4 bands adds
  +13 pp corruption acc; 4 to 6 adds +1.8 pp; 6 to 7 adds nothing. Pareto
  knee is 4 bands; current default of 6 is reasonable but slightly
  over-engineered.
- **The "no all-zero" guard hurts very slightly.** Implementing `min_kept=1`
  (the always-keep-one-band variant) costs ~0.5–0.8 pp clean *and*
  corruption. The all-zero training samples (~1.6 % at p=0.5, 6 bands)
  appear to act as an implicit label-smoothing regularizer. Keep the
  original implementation (option exists for re-use; default off).
- **JSD consistency loss is load-bearing.** Without it the combination
  loses ~3 pp corruption. Worth its 3× per-step compute cost.

## Code that came out of Phase 1 (still in use)

- `imagenette_curriculum.py` — the original training pipeline; still the
  reference for the in-GPU `_band_dropout(p, drop_residual, min_kept)` /
  `laplacian_bands` / `gaussian_blur` (separable) implementations.
- `compare_methods.py` — multi-method comparison harness, JSD loss,
  per-method `apply_method_aug` dispatcher. Used for the
  band-count ablation via `--band-levels`.
- `compute_mce.py` — mCE normalized to local baseline. Phase 2 will swap
  to AlexNet normalization for ImageNet, keep this for CIFAR/Imagenette.
- `eval_glass_blur_224.py` — parallel-CPU corruption eval for the slow
  glass_blur corruption.

These are all Phase-1 vintage and stable. Phase 2 added a per-image
`band_dropout_transform.BandDropAll` callable (see `PHASE2_STATUS.md`)
that's faithful to `_band_dropout(p=0.5, drop_residual=True)` for use
inside torchvision dataset pipelines.

## What's NOT in Phase 1

- No CIFAR corruption robustness (we only ran CIFAR clean accuracy).
- No ImageNet-scale anything.
- No architecture diversity (just ResNet-18 throughout).
- No calibration / adversarial / OOD evals.
- No Geirhos shape-bias measurement.

Those are all Phase 2.
