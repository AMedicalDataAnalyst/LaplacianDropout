# Phase 1 — Full Imagenette-C eval per checkpoint (HEADLINE numbers)

This is where the publishable tier-1 numbers live. Each file is the
output of `eval_glass_blur_224.py` or the original full-eval path —
**14 or 15 corruptions × {sev 1, 3, 5} × full 3925-image Imagenette val
set** evaluated against a single trained checkpoint.

## What was tested

The "headline question": does `band_drop_all + AugMix + JSD` (and the
longA two-stage variant) actually outperform AugMix / PixMix / IPMix /
the baseline on full canonical Imagenette-C, at scale, on a fair
side-by-side eval?

## File mapping

| File | Checkpoint method | Resolution |
|---|---|---|
| `imagenette_results_FULL_baseline_r{1,2}.json` | unaugmented baseline (seeds 1, 2 — r0 was overwritten by a checkpoint-name collision) | 160 |
| `imagenette_results_FULL_augmix_r0.json` | AugMix + JSD | 160 |
| `imagenette_results_FULL_pixmix_r0.json` | PixMix + JSD | 160 |
| `imagenette_results_FULL_ipmix_r0.json` | IPMix + JSD | 160 |
| `imagenette_results_FULL_bd_augmix_s1_r0.json` | band_drop_all + AugMix + JSD (single-stage, 80 ep) | 160 |
| `imagenette_results_FULL_bd_pixmix_r0.json` | band_drop_all + PixMix + JSD | 160 |
| `imagenette_results_FULL_longA_seed{0,1,2}.json` | longA recipe (3 seeds) | 160 |
| `imagenette_results_FULL_extralongA_seed{0,1,2}.json` | extralongA recipe (3 seeds) | 160 |
| `imagenette_results_FULL_longA_bd_aug_r0.json` | **longA bd+aug → clean FT (best 160 recipe)** | 160 |
| `imagenette_results_FULL_224_baseline_r0.json` | baseline at 224 | **224** |
| `imagenette_results_FULL_224_bd_augmix_r0.json` | **band_drop_all + AugMix + JSD at 224 (HEADLINE)** | **224** |

The `glass_blur` corruption was added to the 224 files via a separate
parallel-CPU eval (`eval_glass_blur_224.py`) because it's ~150× slower
than the other corruptions; it was merged into the existing JSONs in
place.

## File schema

Each file is a list of length 1 (one run) with:
```python
{
  "variant" / "checkpoint" / "method": ...,
  "val_acc": clean accuracy,
  "val_acc_residual_at_test": shape-bias proxy,
  "val_acc_blur_at_test": σ=2 blur eval,
  "corruption": {
      "gaussian_noise": {"1": acc, "3": acc, "5": acc},
      ...
      "_mean": mean over corruption × severity,
      "_n_corruptions": 14 or 15,
      "_source": "..." for 224 files
  }
}
```

## Headline result (the row that goes in the paper)

From `imagenette_results_FULL_224_bd_augmix_r0.json` + `..._baseline_r0.json`:

| Method (224², full 15 corruptions) | Clean | Mean Corr | mCE (rel) |
|---|---|---|---|
| baseline (no aug) | 0.896 | 0.573 | 100.0 |
| **band_drop_all + AugMix + JSD** | **0.903** | **0.820** | **47.2** |

**52.8 % relative reduction in mean corruption error** — and the
combination beats baseline on clean too (strict Pareto win).
Per-corruption breakdown and the full 160² comparison live in
`docs/TIER1_FINDINGS.md`.

## How to consume

```bash
# Recompute the mCE table (uses these files + the AlexNet table is
# not in scope for Imagenette — falls back to relative-to-baseline)
python compute_mce.py

# Per-corruption ablation table
python -c "
import json, numpy as np
b = json.load(open('imagenette_results_FULL_224_baseline_r0.json'))[0]['corruption']
w = json.load(open('imagenette_results_FULL_224_bd_augmix_r0.json'))[0]['corruption']
for c in sorted(b):
    if c.startswith('_'): continue
    bm = np.mean(list(b[c].values())); wm = np.mean(list(w[c].values()))
    print(f'{c:20s}  base {bm:.3f}  win {wm:.3f}  +{100*(wm-bm):.1f}pp')
"
```

## What's NOT in this folder

- ImageNet-1k or ImageNet-C anything (Phase 2 work, blocked on data).
- Reviewer-defense `hf_only`/`lf_only` runs — those are in
  `../phase2_reviewer_defense/`.
- The band-count and `min_kept` ablations — `../phase1_ablations/`.
