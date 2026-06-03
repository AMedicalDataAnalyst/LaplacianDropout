# Phase 1 — Imagenette screening + multi-seed runs

The "second-attempt" runs after pivoting from CIFAR. These are the
exploration data — screening sweeps over schedule/dropout variants,
two-stage recipes, and multi-seed confirmations — but **not** the full
Imagenette-C corruption evals (those live in
`../phase1_imagenette_full/`).

## What was tested

This folder captures the iterative search for the right augmentation
recipe on Imagenette at 160 resolution. Notable phases of work:

1. **Screening** (`imagenette_results_screening{1,2,3}.json`,
   `imagenette_results_1x.json`, `imagenette_results_4x.json`) — wide
   sweep over band-dropout variants, curriculum variants, and "joint"
   methods (joint_blur, joint_residual, joint_multiblur, joint_mix). Each
   variant trained 3× at the relevant epoch budget.
2. **Plateau training** (`imagenette_results_plateau.json`) — tested a
   plateau-based scheduler instead of fixed epochs.
3. **Multi-seed confirmation** (`imagenette_results_multiseed.json`) — 3
   seeds of the most-promising variants from screening, including
   `band_drop_all` (the eventual headline mechanism).
4. **"longA" recipe** (`imagenette_results_longA_*.json`,
   `imagenette_results_longA_s{1,2}_seed*.json`) — the two-stage recipe
   where stage 1 trains with aug + JSD, stage 2 fine-tunes on clean
   images at LR/10. Discovered as a way to recover clean-acc cost.
5. **"extralongA" recipe** (`imagenette_results_extralongA_*.json`) —
   longA with even longer stage-1 training to test saturation.
6. **Two-stage A/B sanity** (`imagenette_results_twostage_*.json`) —
   tested band_drop_all and baseline through the same two-stage pipeline
   for like-for-like comparison.
7. **Distillation experiments** (`imagenette_results_distill_*.json`) —
   tested whether a band-dropout model could be distilled into a
   baseline model. Did not become headline.
8. **"final_train"** (`imagenette_results_final_train_*.json`) — the
   3-seed final-recipe runs that were then full-evaluated and ended up
   in `../phase1_imagenette_full/`.
9. **"A_fullLR"** (`imagenette_results_A_fullLR_*.json`) — recipe A with
   full LR (no stage-2 LR/10), 3 seeds.

## File schemas

All records are JSON arrays of run dicts. Common keys:
- `variant` or `method` — augmentation identifier
- `val_acc` — clean test accuracy
- `val_acc_residual_at_test` — accuracy when eval'd on Laplacian residual
  only (shape-bias proxy)
- `val_acc_blur_at_test` — accuracy when eval'd on σ=2 blurred images
- `corruption` — if present, dict `{corruption_name: {sev: acc}, _mean}`.
  Most files here used **fast eval** (sev 3 only, 1000-image subset).
  Full-eval variants are in `../phase1_imagenette_full/`.
- `epochs_done`, `time_seconds`, `total_epochs` — run metadata
- `history` — per-epoch loss/acc trajectories (only when
  `--log-per-epoch` was set)

## Key takeaways (from this folder's data)

These are the screening/exploration results that *informed* the headline,
not the headline itself.

- `band_drop_all` (drop each of 6 bands with p=0.5, **including** the
  residual) consistently outperformed the curriculum variants on
  corruption robustness while costing little clean accuracy.
- The two-stage "longA" recipe (band_drop_all+aug stage 1 → clean
  fine-tune stage 2) was the original Pareto winner. (At 224, single
  stage suffices — see `../phase1_imagenette_full/`.)
- Distillation experiments did **not** yield a "no clean cost" version
  of band_drop_all and were abandoned.

For the consolidated tier-1 numbers, look at:
- `docs/TIER1_FINDINGS.md` — the headline doc
- `docs/PHASE1_SUMMARY.md` — high-altitude summary
- `../phase1_imagenette_full/` — full Imagenette-C eval per checkpoint
- `../phase1_comparison/` — multi-method comparison runs

## How to consume

```bash
# Auto-build the Phase-1 tier-1 report (also reads ../phase1_comparison/)
python build_report.py

# Or hand-aggregate:
python analyze_imagenette.py results/phase1_imagenette/imagenette_results_*.json --out report.png
```
