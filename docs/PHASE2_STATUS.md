# Phase 2 — CIFAR + ImageNet scaling (paused at the ImageNet compute gate)

## What Phase 2 is

Take the Phase-1 `band_drop_all` mechanism and answer three questions:
1. Does the corruption-robustness win transfer to **CIFAR-10/-100** (smaller
   images, fewer frequency octaves)?
2. Does it transfer to **ImageNet-1k** at the canonical scale where AugMix,
   PixMix, IPMix etc. are benchmarked, with the AlexNet-normalized mCE?
3. Does it survive **architecture changes** (ResNet-50, ConvNeXt-T, ViT-B/16)?

Plus a robustness battery (calibration, adversarial, OOD, shape-bias).

## Status of every Phase-2 task

| # | Task | State |
|---|---|---|
| 1 | Acquire datasets | Partial — CIFAR-10-C, CIFAR-100-C, fractals **extracted**. ImageNet-1k/-C/-R/-A/-Sketch still pending download. |
| 2 | `band_dropout_transform.py` per-image module | **DONE** (CORE review item) |
| 3 | Patch `augmix/imagenet.py` for band_drop_all | DONE |
| 4 | De-risk on ImageNet-100/200 | Blocked on ImageNet data |
| 5 | Train ImageNet-1k RN-50 × 4 methods | Blocked on ImageNet data + GPU-days |
| 6 | ConvNeXt-T + ViT-B recipe + runs | Recipe done (`--optimizer adamw --schedule cosine`); runs blocked |
| 7 | Patch `pixmix/imagenet.py` for band_drop_all | DONE |
| 8 | Calibration adapter (`calibration_eval.py`) | DONE |
| 9 | OOD eval (`ood_eval.py`) for -R/-A/-Sketch | DONE |
| 10 | Adversarial wrapper (`adversarial_eval.py`) | DONE |
| 11 | Mechanism / theory writeup | **3 of 4 figures DONE at 224²** (band-count, gradient spectrum, band-targeted PGD — see `THEORY_OUTLINE.md`); shape-bias figure needs trained 1k model |
| 12 | Geirhos cue-conflict runner (`geirhos_shape_bias.py`) | **DONE + validated 2026-07-18** — stimuli staged, class map checked in, harness reproduces published RN-50 shape bias (0.222). Science number needs 1k model; see `results/phase2_geirhos/README.md` |
| 13 | Tier-2 paper writeup | Outline + skeleton in `PAPER_OUTLINE.md`; sections blocked |
| 14 | Band-count ablation at 224² | DONE (Phase 1.5 — sweep result in `band_count_ablation_L*.json`) |
| 15 | min_kept=1 ablation | DONE (no improvement — keep original) |
| 16 | Patch `augmix/cifar.py` | DONE |
| 17 | Patch `pixmix/cifar.py` | DONE |
| 18 | `cifar_eval.py` | DONE; now uses reference CIFAR-C via `--c-dir` |
| 19 | CIFAR smoke test | DONE |
| 20 | CIFAR full sweep (5 methods × 2 datasets × 3 seeds) | **GATED on #22 result** |
| 21 | `cifar_aggregate.py` results aggregator | DONE; writes `docs/CIFAR_RESULTS.md` |
| 22 | CIFAR de-risk (baseline + combo, 1 seed) | DONE — Gate A **failed** (see below); CIFAR sweep permanently skipped |
| 23 | Extract CIFAR-C / fractals tarballs | DONE |
| 24 | Wire cifar_eval to reference -C | DONE |
| 25 | Re-eval baseline with reference -C | DONE (mean corr acc = 0.7706, comparable to published) |
| 26 | Repo tidy + Phase 1/2 doc split | DONE (this doc) |
| 27 | Reviewer-defense hf_only/lf_only, 3 seeds each | **DONE 2026-06-04** — both strictly worse than random `all` (see `results/phase2_reviewer_defense/README.md`) |
| 28 | Band-targeted PGD sweep (5 models × 7 attacks) | **DONE 2026-06-04** — spectral vulnerability profile (see `results/phase2_targeted_pgd/README.md`) |
| 29 | Gradient-spectrum mechanism analysis | **DONE 2026-06-04** — HF concentration 5.6 → 3.2 (see `THEORY_OUTLINE.md` exp 3) |
| 30 | Subtractive-axes ablation (bit_depth, pca_color) | **DONE 2026-06-05** — standalone gains, but stacking interferes; headline recipe unchanged (see `results/phase2_subtractive/README.md`) |

## De-risk OUTCOME (2026-06-03 07:47) — Gate A FAILED on CIFAR

CIFAR-10 single-seed comparison (full -C eval via reference Hendrycks data):

| Metric | Baseline | `band_drop_all+augmix` | Δ |
|---|---|---|---|
| Clean acc | 0.9575 | 0.9514 | −0.6 pp |
| Mean corruption acc | 0.7706 | **0.6673** | **−10.3 pp** |
| mCE (rel) | 100.0 | **163.3** | +63 % worse |
| PGD@2/255 | 0.048 | 0.110 | +6.1 pp |

**The CIFAR transfer failed.** Strict Pareto LOSS on clean *and* corruption.
Below the pre-stated Gate A threshold (corruption acc < 0.77). Possible causes
include band-count constraint (CIFAR-32 capped at 4 levels vs 5 on 224), JSD
over-regularization at small scale, or grad-clip side effects.

**Decision (2026-06-03):** skip the full 30-run CIFAR sweep. Re-pitch the
method as "224 px+" rather than universal. The CIFAR result becomes a
documented null in the paper rather than a comparison column.

## Currently running

Nothing. All local (Imagenette-scale) experiments concluded 2026-06-05.
**The project is paused at the ImageNet gate**: every remaining task
needs ImageNet-1k data + rented GPU compute. The step-by-step rental
walkthrough (instance specs, budget, staging, launch commands, decision
gates) is `docs/PHASE2_KICKOFF.md`; the data shopping list is
`docs/DATA_SETUP.md`.

Completed since the last version of this section (all at Imagenette-224):
- Reviewer-defense 3-seed hf_only/lf_only controls (task 27) — decisive.
- Band-targeted PGD spectral vulnerability profile (task 28).
- Gradient-spectrum analysis (task 29).
- Subtractive-axes ablation (task 30) — headline recipe unchanged.
- Geirhos shape-bias harness staged + validated (task 12).

## Decision gates and what comes after

### Gate A: de-risk result — RESOLVED 2026-06-03 (failed transfer)

Pre-stated criteria (kept for record):
- band_drop_all+augmix clean acc ≥ 0.95 AND mean corruption acc ≥ 0.80 →
  green light for full 30-run CIFAR sweep.
- Mean corruption acc 0.77–0.80 → ambiguous; consider more seeds.
- Mean corruption acc < 0.77 → CIFAR-32 doesn't transfer. Skip the sweep.

**Actual result: corr acc 0.667, below the threshold → skip the sweep.** See
the result table above and the "doesn't transfer" framing.

### Gate B: ImageNet-1k RN-50 result (after Phase 5)

- **band_drop_all+augmix mCE ≤ 65 AND clean within 1pp of baseline** →
  workshop-quality result; proceed with Phase 2 (calibration, OOD,
  adversarial, Geirhos, ConvNeXt/ViT).
- **mCE 65–75** → comparable to AugMix; pursue full battery, expect a
  "complementary not strictly better" framing.
- **mCE > 75** → headline didn't scale; rethink.

## Files Phase 2 added / depends on

### New code (all in repo root)
- `band_dropout_transform.py` — the per-image Laplacian-band-dropout
  callable. **The core review item.** Used by all four patched harnesses.
  Has a built-in self-test: `python band_dropout_transform.py`.
- `cifar_eval.py` — clean / CIFAR-C / PGD / RMS-CE for a checkpoint.
  Supports `--c-dir` (reference) and on-the-fly (fallback).
- `cifar_aggregate.py` — walks `cifar_results/`, writes
  `docs/CIFAR_RESULTS.md`.
- `ood_eval.py` — ImageNet-R/-A/-Sketch eval for a trained 1k model.
  Uses pixmix's wnid masks (no list duplication).
- `calibration_eval.py` — RMS-CE / AURRA on clean + ImageNet-C per
  corruption × severity. Uses pixmix's `calibration_tools.get_measures`.
- `adversarial_eval.py` — torchattacks wrapper (PGD/APGD/AutoAttack)
  with normalization handled internally.
- `geirhos_shape_bias.py` — Geirhos cue-conflict runner. Expects external
  stimuli + a class-map JSON (see `DATA_SETUP.md`).
- `check_data.py` — verifies the expected data directory layout.
- `run_cifar_derisk.sh` / `run_cifar_sweep.sh` — orchestrators.

### Patched third-party (small diffs — search "BandDropAll" / "aug-method")
- `augmix/imagenet.py` — added `--aug-method`, `--band-drop-p`,
  `--optimizer`, `--schedule`, `--warmup-epochs`. Fixed a pre-existing
  py2-era bug in `accuracy()`. Made `corrupted_data` optional.
- `augmix/cifar.py` — same `--aug-method` etc., plus `--seed`, `--data-path`,
  `--band-levels` (defaults to 4 for CIFAR-32 since σ=16 doesn't fit
  32 px). Added gradient clipping at `||g||₂ ≤ 5.0` (required for band
  methods at LR=0.1; no-op for baseline/AugMix).
- `pixmix/imagenet.py` — `--aug-method` for {pixmix, baseline,
  band_drop_all, band_drop_all+pixmix}.
- `pixmix/cifar.py` — same, plus seed exposure + grad clip.

## Pre-computed reference data — DO NOT RE-RUN

These exist on disk and are the source of truth for any Imagenette-224
comparisons. Cite, don't regenerate.

### 3-seed Imagenette-224 (Phase 1, `comparison_224_main.json`)
| Method | Clean (mean) | Corruption (mean) |
|---|---|---|
| baseline | 0.898 | 0.566 |
| augmix+JSD | 0.908 | 0.701 |
| band_drop_all+JSD (`band_mask='all'`) | 0.888 | **0.806** |
| band_drop_all+augmix+JSD | 0.900 | **0.826** |

### Band-count ablation (Phase 1.5, `band_count_ablation_L*_bands*.json`, single seed)
| Levels | Bands | Clean | Corr | Residual-only |
|---|---|---|---|---|
| 1 | 2 | 0.905 | 0.656 | 0.902 |
| 2 | 3 | 0.905 | 0.719 | 0.894 |
| 3 | 4 | 0.894 | 0.784 | 0.869 |
| 4 | 5 | 0.894 | 0.792 | 0.833 |
| 5 | 6 | 0.886 | 0.802 | 0.708 |
| 6 | 7 | 0.884 | 0.804 | 0.498 |
Saturates at L=5; Pareto knee at L=3.

### Reviewer-defense control — DONE 2026-06-04, 3 seeds each
| Method | Clean | Corr | Source |
|---|---|---|---|
| band_drop_all+JSD `--band-mask all` (3 seeds, ref) | 0.888 | **0.806** | `comparison_224_main.json` |
| band_drop_all+JSD `--band-mask hf_only` (3 seeds) | 0.873 | **0.747** | `bandmask_hf_only_224_3seed.json` |
| band_drop_all+JSD `--band-mask lf_only` (3 seeds) | 0.900 | **0.573** | `bandmask_lf_only_224_3seed.json` |

Both targeted variants strictly worse than full random on corruption acc
(≥30σ) — refutes the operator-family-memorization attack. Full analysis:
`results/phase2_reviewer_defense/README.md`.

## How to launch the next experiment

### Launch ImageNet-1k training (THE next step; needs rented GPUs + data)
Follow `docs/PHASE2_KICKOFF.md` top-to-bottom — it is the complete
rental-instance walkthrough (specs, budget, data staging, pre-flight,
phase-by-phase launch commands, decision gates, recovery). Dataset
sources and layout: `docs/DATA_SETUP.md`.

### CIFAR sweep — do NOT launch
Gate A failed (see above); the full 30-run CIFAR sweep was deliberately
skipped. `run_cifar_sweep.sh` is kept only for the record.

### Re-eval an existing checkpoint with the new reference -C
```
python cifar_eval.py --checkpoint <path>.pth.tar --dataset cifar10 \
  --layers 28 --widen-factor 10 \
  --data-path /mnt/c/Users/JoyToy/Documents/Projects/data \
  --c-dir /mnt/c/Users/JoyToy/Documents/Projects/data/CIFAR-10-C \
  --out cifar_results/cifar_eval_<dataset>_<method>_seed<s>.json
```
