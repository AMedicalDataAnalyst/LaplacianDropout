# Phase 1 — Multi-method comparison runs

Output of `compare_methods.py`. Each file contains multiple training runs
(typically 3 seeds × multiple methods) with **fast corruption eval**
(sev 3 only, 1000-image subset) attached at the end of each run.

## What was tested

Fair side-by-side comparison of all augmentation methods under one
training harness (`compare_methods.py`):
- baseline (no aug)
- AugMix (with JSD)
- PixMix
- IPMix (re-implementation — see Phase 1 lessons)
- band_drop_all (with JSD)
- band_drop_all + AugMix (combo, with JSD)
- band_drop_all + PixMix
- augmix + band_drop_all (the *reverse* order)

All methods through the same training loop, optimizer, schedule, and
fast corruption eval — so the comparison is apples-to-apples.

## File mapping

| File | Resolution | Methods compared | Seeds | Notes |
|---|---|---|---|---|
| `comparison_jsd_results.json` | 160 | baseline, augmix, pixmix, band_drop_all, band_drop_all+augmix, band_drop_all+pixmix | 3 each | JSD on; the main tier-1 comparison at 160 |
| `comparison_ipmix.json` | 160 | ipmix, band_drop_all+ipmix | 3 each | added after the main run when IPMix was implemented |
| `comparison_longA_bd_augmix_s1.json` | 160 | bd_augmix stage-1 only | 1 seed | first half of longA |
| `comparison_longA_bd_augmix_s2_r{0,1,2}.json` | 160 | bd_augmix stage-2 (FT) | 1 each | second half — per-seed for the 3-seed longA recipe |
| `comparison_224_main.json` | **224** | baseline, augmix, band_drop_all, band_drop_all+augmix | 3 each | the 3-seed 224 reference table — **cited heavily** in Phase 2 |

## File schema

JSON arrays of run records. Each record:
```python
{
  "method": "augmix" | "band_drop_all" | ...,
  "run": 0..2,
  "val_acc": clean test acc,
  "val_acc_residual_at_test": ...,
  "val_acc_blur_at_test": ...,
  "corruption": {<corruption>: {<sev>: acc}, "_mean": float, "_subsample": 1000},
  "epochs_done": ...,
  "checkpoint": "checkpoints/<name>.pt",
  "band_levels": int,     # added in Phase 1.5
  "resolution": int,      # added in Phase 1.5
  "history": {"step_loss": [...], "epoch_val_acc": [...]}  # if --log-per-epoch
}
```

## Key 224 numbers (from `comparison_224_main.json`, 3-seed means)

These are the reference table that Phase 2 reviewer-defense controls are
compared against:

| Method (224, fast corruption eval) | Clean | σ=2 blur | Residual-only | Corruption |
|---|---|---|---|---|
| baseline | 0.898 | — | 0.134 | 0.566 |
| AugMix + JSD | 0.908 | — | 0.143 | 0.701 |
| **`band_drop_all` + JSD** | 0.888 | 0.882 | 0.702 | **0.806** |
| **`band_drop_all + AugMix` + JSD** | 0.900 | — | 0.697 | **0.826** |

The two `band_drop_all*` rows have very tight per-seed std (corr std ≈
0.0004) — useful when judging whether tier-2 control numbers are
significantly different.

## How to consume

```bash
# Aggregate via the official tier-1 reporter
python build_report.py
# -> docs/tier1_report.md + figures/tier1_tradeoff.png

# Or hand-aggregate one file
python -c "
import json, statistics as s
ref = json.load(open('comparison_224_main.json'))
for m in sorted({r['method'] for r in ref}):
    rs = [r for r in ref if r['method'] == m]
    print(f'{m:25s} clean {s.mean(r[\"val_acc\"] for r in rs):.4f}  '
          f'corr {s.mean(r[\"corruption\"][\"_mean\"] for r in rs):.4f}')
"
```
