# Phase 2 — Subtractive-axes ablation (bit-depth + PCA-color)

Tests whether two *subtractive* augmentation axes orthogonal to
Laplacian-band dropout — bit-depth reduction and PCA color-plane
dropping (`subtractive_transforms.py`) — add anything on top of the
`band_drop_all+AugMix` headline recipe, and what each is worth alone.

All runs: Imagenette-224, ResNet-18, 3 seeds, JSD on, fast corruption
eval (sev 3, 1000-img subsample), 50 epochs. Subtractive parameters:
`bit_depth bits=(2,5) apply_p=1.0`, `pca p=0.5`.

Reference points (3-seed means from
`../phase1_comparison/comparison_224_main.json`, NOT re-run here):
baseline 0.566 · augmix+JSD 0.701 · band_drop_all+JSD 0.806 ·
**band_drop_all+augmix+JSD (HEADLINE) 0.826**.

## Results (3-seed mean corruption acc)

| File | Method | Clean | Corr | vs reference |
|---|---|---|---|---|
| `bit_depth_224_3seed.json` | bit_depth+JSD | 0.897 | 0.642 | +7.6 pp vs baseline |
| `pca_color_224_3seed.json` | pca_color+JSD | 0.888 | 0.610 | +4.4 pp vs baseline |
| `band_drop_all_p_augmix_p_bit_depth_p_pca_color_224_3seed.json` | full quad stack | 0.902 | 0.781 | **−4.6 pp vs HEADLINE** |
| `band_drop_all_p_bit_depth_p_pca_color_224_3seed.json` | bd + both subtractive | 0.898 | 0.774 | −3.2 pp vs bd-only+aug HEADLINE |

## Findings (2026-06-05, commit `0edc0e8`)

1. **Both axes work standalone** — modest corruption gains over baseline
   (+8 / +4 pp) with no clean-accuracy cost.
2. **The two axes are mechanistically complementary** — anti-correlated
   per-corruption specialisations:
   - `bit_depth`: noise specialist (gaussian +35, impulse +38, shot +36 pp
     vs baseline) but actively *hurts* on contrast/fog.
   - `pca_color`: low-frequency specialist (contrast +44, fog +28,
     frost +14) but actively *hurts* on noise (gaussian −23, impulse −21,
     shot −23).
3. **Stacking interferes.** Adding both axes to the headline recipe is
   4.6 pp *worse* on corruption acc despite unchanged clean acc —
   subtractive saturation, not additivity.

**Decision: `band_drop_all+AugMix+JSD` remains the recipe.** The
standalone numbers become a "related axes" paragraph in the paper; the
interference result is the evidence that the headline is not
under-tuned.

## Why only 4 of the 9 planned methods exist

`run_subtractive_ablation.sh` queued 9 methods (full stack + progressive
deletions). After the first two combo runs showed interference and the
two standalone runs (run separately, `run_subtractive_alone.log`)
completed the single-axis attribution, the remaining pairwise
combinations could not change the decision — the sweep was deliberately
stopped ("option C", 2026-06-05; see `kill_after_method4.sh` at repo
root). This is a documented early stop, not missing data.

## Reproduction

```bash
python compare_methods.py --methods <METHOD> \
  --runs 3 --max-epochs 50 --use-jsd --batch-size 96 \
  --corruption-eval fast --resolution 224 --pad 16 --seed-offset 0 \
  --out results/phase2_subtractive/<method>_224_3seed.json
```

File schema: standard `compare_methods.py` output records (see
`../phase1_comparison/README.md`).
