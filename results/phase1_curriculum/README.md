# Phase 1 — Curriculum experiments on CIFAR-10

The original starting point of the project: test whether a Laplacian
frequency curriculum (progressively unlocking high-frequency bands during
training) improves CIFAR-10 accuracy. Source: user's spec in
`docs/historical/INSTRUCTIONS_CURRICULUM.md`.

## What was tested

Across an epoch-budget sweep (1× → 64× the baseline 100-epoch CIFAR-10
recipe), we compared:
- `baseline` — no Laplacian decomposition
- `decompose_identity` — Laplacian decomposition then re-sum (sanity check)
- `curriculum` — progressive band unlock during training
- `curriculum_dropout` — curriculum + random per-band dropout
- `residual_only` — only the low-frequency residual ever shown

All runs used the same WideResNet, optimizer, and schedule; only the
input transform varied.

## Result files

| File | Budget | Variants present | n seeds |
|---|---|---|---|
| `curriculum_results.json` | 1× (100 ep) | baseline + curriculum × 2 + decompose_identity + residual_only | 3 each |
| `curriculum_results_2x.json` | 2× | same 5 | 3 each |
| `curriculum_results_4x.json` | 4× | baseline + curriculum × 2 | 3 each |
| `curriculum_results_8x.json` | 8× | baseline + curriculum × 2 | 3 each |
| `curriculum_results_16x.json` | 16× | baseline + curriculum × 2 | 3 each |
| `curriculum_results_64x.json` | 64× | curriculum only | 3 |

Records contain `variant`, `val_acc`, `val_acc_residual_at_test` (eval on
residual-only inputs as a shape-bias proxy), plus run metadata.

## Key findings — see `docs/PHASE1_SUMMARY.md` for the full table

- **Curriculum never beat baseline on clean accuracy.** Best result at any
  budget: `curriculum` at 16× gives 0.9304 vs baseline 0.9415 — a 1.1 pp
  loss. Even 64× of curriculum (0.9365) doesn't catch the 16× baseline.
- **Curriculum_dropout is worse than plain curriculum.** Adding stronger
  regularization made things worse on CIFAR's saturated benchmark.
- **`residual_only` collapses to ~13 % accuracy** (CIFAR-10 random = 10 %).
  Expected — the residual at CIFAR-32 contains too little structure to
  classify from.
- **We never measured corruption robustness in this phase** (only clean
  accuracy on the CIFAR-10 test set). The corruption story emerged later
  on Imagenette.

## Implication that drove the project pivot

CIFAR-10 clean accuracy is too saturated for a frequency-based
augmentation to show gains. The work moved to Imagenette in
`results/phase1_imagenette/` — where `band_drop_all` (the dropout
mechanism extracted from the curriculum) became the actual headline.

## How to re-read these files

Plain JSON arrays of run records:
```python
import json
runs = json.load(open('curriculum_results_16x.json'))
for r in runs:
    print(r['variant'], r['val_acc'])
```
Or aggregate in bulk with `python analyze_imagenette.py
results/phase1_curriculum/*.json` (uses the same record schema).
