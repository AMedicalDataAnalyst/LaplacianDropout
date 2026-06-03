# Files inventory

Every file at the repo root or in a project subdir, with phase, status,
and what it is.

Phases: **P1** = Phase 1 (Imagenette / tier-1). **P2** = Phase 2 (CIFAR + ImageNet
scaling). **Shared** = used by both. **Archive** = historical / superseded.

## Top-level docs (live)

| File | Phase | What |
|---|---|---|
| `README.md` | Shared | Entry point for fresh sessions |
| `docs/PHASE1_SUMMARY.md` | P1 | High-altitude summary of what was achieved |
| `docs/PHASE2_STATUS.md` | P2 | Current state, what's running, gating decisions |
| `docs/LESSONS_LEARNED.md` | Shared | Cross-phase findings to remember |
| `docs/FILES_INVENTORY.md` | Shared | This file |
| `docs/TIER1_FINDINGS.md` | P1 | Detailed Phase-1 results, per-corruption tables |
| `docs/CIFAR_RESULTS.md` | P2 | Auto-generated CIFAR sweep table; updated by `cifar_aggregate.py` |
| `docs/DATA_SETUP.md` | P2 | Canonical dataset list + launch commands |
| `docs/PAPER_OUTLINE.md` | P2 | Paper skeleton with current numbers slotted in |
| `docs/THEORY_OUTLINE.md` | Shared | Mechanism hypotheses + experiment shortlist |

## Top-level docs (archived in `docs/historical/`)

| File | Phase | Why archived |
|---|---|---|
| `INSTRUCTIONS_CURRICULUM.md` | P1 | Original user spec; superseded by results but historically interesting |
| `OVERNIGHT_STATUS.md` | P2 | Snapshot from earlier overnight run; superseded by `PHASE2_STATUS.md` |
| `SHAPE_BIAS_FINDINGS.md` | P1 | Early shape-bias work; subsumed into `TIER1_FINDINGS.md` |
| `WAKEUP_SUMMARY.md` | P1 | Snapshot doc; superseded |
| `tier1_report.md` | P1 | Auto-generated comparison table; superseded by `TIER1_FINDINGS.md` |
| `shape_bias_report.md` | P1 | Auto-generated; superseded |

## Active Python code (root, in current use)

### Phase 1 (Imagenette pipeline — still imported by P2 utilities)
| File | Status | What |
|---|---|---|
| `imagenette_curriculum.py` | LIVE | Original training pipeline. Reference for `_band_dropout`, `laplacian_bands`, `gaussian_blur` (separable), `ImagenetteNet`, evaluation utilities. Used by `compare_methods.py` and `band_dropout_transform.py` self-test. |
| `compare_methods.py` | LIVE | Multi-method comparison harness with JSD loss; supports `--band-levels` for band-count ablation. |
| `compute_mce.py` | LIVE | Local-baseline-normalized mCE. Used for Imagenette/CIFAR; for ImageNet use the AlexNet table inside `augmix/imagenet.py`. |
| `eval_glass_blur_224.py` | LIVE | Parallel-CPU glass_blur eval (slow corruption, separated from the rest). |
| `build_report.py` | LIVE | Generates Imagenette `tier1_report.md` + `tier1_tradeoff.png` from result JSONs. Phase-1 era; reads JSONs at the repo root. |
| `analyze_imagenette.py` | LIVE | Older analysis script for Imagenette results. |
| `curriculum_experiment.py` | LIVE | Original CIFAR curriculum script (Phase 1). Not currently used; kept for reproducibility of historical numbers. |

### Phase 2 (CIFAR + ImageNet — current work)
| File | Status | What |
|---|---|---|
| `band_dropout_transform.py` | LIVE — **CORE REVIEW ITEM** | Per-image `BandDropAll(p, drop_residual, min_kept, levels)` transform for torchvision pipelines. Has a built-in self-test (`python band_dropout_transform.py`). |
| `cifar_eval.py` | LIVE | Clean / CIFAR-C / PGD@2/255 / RMS-CE for a checkpoint. Supports `--c-dir` (reference) and on-the-fly fallback. |
| `cifar_aggregate.py` | LIVE | Walks `cifar_results/`, computes mean ± std + mCE, writes `docs/CIFAR_RESULTS.md`. Re-runnable anytime. |
| `ood_eval.py` | LIVE | ImageNet-R / -A / -Sketch eval. Reuses pixmix's wnid masks. |
| `calibration_eval.py` | LIVE | RMS-CE / AURRA on clean + ImageNet-C. Uses pixmix `calibration_tools.get_measures`. |
| `adversarial_eval.py` | LIVE | torchattacks PGD / APGD / AutoAttack wrapper. |
| `geirhos_shape_bias.py` | LIVE | Geirhos cue-conflict shape-bias %. Needs external stimuli per `DATA_SETUP.md`. |
| `check_data.py` | LIVE | Verifies expected data dir layout. |
| `run_cifar_derisk.sh` | LIVE | 2-run de-risk orchestrator (currently running). |
| `run_cifar_sweep.sh` | LIVE | 30-run full sweep orchestrator (gated on de-risk). |
| `run_band_count_ablation.sh` | LIVE | Band-count ablation orchestrator (Phase-1.5 — already run). |

## Patched third-party

| Path | What we changed |
|---|---|
| `augmix/imagenet.py` | Added `--aug-method`, `--band-drop-p`, `--optimizer`, `--schedule`, `--warmup-epochs`. `corrupted_data` made optional. Fixed `accuracy()` `.view`→`.reshape`. |
| `augmix/cifar.py` | Same `--aug-method` plus `--seed`, `--data-path`, `--band-levels`. Added gradient clipping at 5.0. Guarded `test_c` against missing -C dir. |
| `pixmix/imagenet.py` | `--aug-method` for {pixmix, baseline, band_drop_all, band_drop_all+pixmix} + `--band-drop-p`. |
| `pixmix/cifar.py` | Same + `--seed`, `--band-levels`, gradient clipping. |

To see the exact diffs, search the files for `BandDropAll` or `aug-method`.

## Result JSONs — consolidated into `results/` (2026-06-03)

All Phase-1 result files relocated from repo root into subdirs under
`results/`. Scripts updated (`build_report.py`, `compute_mce.py`,
`eval_glass_blur_224.py`) to read from the new paths.

| Subdir | Contents |
|---|---|
| `results/phase1_curriculum/` | `curriculum_results*.json` — early CIFAR curriculum runs (Phase 1) |
| `results/phase1_imagenette/` | `imagenette_results_*.json` — Imagenette screening + multi-seed runs |
| `results/phase1_imagenette_full/` | `imagenette_results_FULL_*.json` — full Imagenette-C eval per checkpoint |
| `results/phase1_comparison/` | `comparison_*.json` — multi-method comparison runs from `compare_methods.py` |
| `results/phase1_ablations/` | `band_count_ablation_L*_bands*.json`, `band_keep1_*.json` |
| `results/phase2_reviewer_defense/` | `bandmask_*_224.json` — seed-0 hf_only / lf_only runs |
| `cifar_results/` | Phase-2 CIFAR per-(dataset, method, seed) eval JSONs |

Active 3-seed Imagenette runs write to root (`bandmask_*_224_3seed.json`)
while training, then will be moved into `results/phase2_reviewer_defense/`
when complete.

## Logs — `logs/`

| Subdir | Contents |
|---|---|
| `logs/phase1/` | Phase-1 `run_*.txt` training logs |
| `logs/phase2/` | `run_cifar_derisk.log`, `queue_after_derisk.log`, `run_band_count_ablation.log`, `run_band_keep1.log` |

`run_imagenette_3seed.log` stays at root while the 3-seed runs are active.

## Figures — `figures/`

| File | Source |
|---|---|
| `tier1_tradeoff.png` | regenerated by `build_report.py` |
| `shape_bias_per_corruption.png`, `shape_bias_tradeoff.png` | Phase-1 shape-bias plots |

## Checkpoints

- `checkpoints/` (~2.7 GB) — Phase 1 Imagenette models (longA, extralongA, bd_augmix variants, etc.). Filename collisions between 160² and 224² runs were a Phase-1 gotcha; current contents are the 224² versions for the duplicated names.
- `cifar_snapshots/` (~1.1 GB) — Phase 2 CIFAR checkpoints. Subdirs per
  (dataset, method, seed). Each contains `checkpoint.pth.tar` (latest) and
  `model_best.pth.tar` (best val acc). Currently has:
  - `cifar10_baseline_seed0/`
  - `cifar10_band_drop_all_p_augmix_seed0/` (training in progress)

## Caches

- `imagenette_cache/` (3.7 GB) — pre-loaded Imagenette tensors at 160 and 224 resolution, used by `imagenette_curriculum.ImagenetteLoader`. Phase 1; reused by `compare_methods.py`.
- `__pycache__/` — Python bytecode. Safe to ignore.
- `cifar10/` — torchvision-cached CIFAR-10 tensors (created by an early Phase-1 script). Not in current use.

## Logs

- `run_*.txt` — Phase 1 training logs. Historical reference.
- `run_*.log` — Phase 2 training logs (`run_cifar_derisk.log` is the live one).

## External data (not in repo)

| Dataset | Where | Used by |
|---|---|---|
| Imagenette 320² | `/mnt/c/Users/JoyToy/Documents/Projects/data/imagenette2-320/` | Phase 1, Phase 2 PixMix mixing-pool fallback |
| CIFAR-10 / -100 | `/mnt/c/Users/JoyToy/Documents/Projects/data/cifar-{10-batches-py, 100-python}/` | All CIFAR runs |
| CIFAR-10-C / -100-C | `/mnt/c/Users/JoyToy/Documents/Projects/data/CIFAR-{10,100}-C/` | `cifar_eval.py --c-dir` |
| Fractals mixing set | `/mnt/c/Users/JoyToy/Documents/Projects/data/fractals_and_fvis/fractals/images/` | Phase 2 PixMix (when added to sweep) |
| ImageNet-1k / -C / -R / -A / -Sketch | Not yet downloaded | Phase 2 ImageNet runs |
