# Laplacian band-dropout for corruption-robust image classification

**Project:** evaluate whether random Laplacian-band masking
(`band_drop_all`) is a useful data augmentation for improving model
robustness to image corruptions, OOD inputs, and adversarial
perturbations.

**Current state (June 2026):**
- **Phase 1** (Imagenette) complete. Headline: at 224² on the full canonical
  15-corruption Imagenette-C, `band_drop_all + AugMix + JSD` gives
  **mCE 47.2** (52.8 % relative error reduction) vs the unaugmented
  baseline, with clean accuracy also improved by 0.7 pp — a strict
  Pareto win.
- **Phase 2** (ImageNet scaling + robustness battery) infrastructure
  built. CIFAR transfer experiment ran and **failed** — the method is now
  pitched as "224 px+ only." 3-seed Imagenette reviewer-defense
  controls in progress. ImageNet-1k runs are blocked on data + GPU
  compute.

## Repo layout

```
laplacian_schedule/
  README.md                      ← you are here
  docs/                          ← all human-written documentation
  results/                       ← all training/eval result JSONs (organised by phase)
  logs/                          ← raw training/eval stdout (organised by phase)
  figures/                       ← all .png plots
  *.py                           ← code (training, eval, aggregators)
  *.sh                           ← launcher scripts
  requirements.txt               ← pip deps (torch installed separately, see file)
  augmix/  pixmix/               ← third-party repos with our patches
  checkpoints/                   ← Phase 1 model state dicts
  cifar_results/  cifar_snapshots/   ← Phase 2 CIFAR outputs
  imagenette_cache/              ← cached Imagenette tensors
```

## Entry points — read in this order

1. **`docs/PHASE1_SUMMARY.md`** — what we did in Phase 1, what we found,
   what we kept. Read this first if you've never seen the project.
2. **`docs/PHASE2_STATUS.md`** — current state of Phase 2: what's done,
   what's running, what's blocked, all decision gates resolved so far.
3. **`docs/PHASE2_KICKOFF.md`** — step-by-step setup walkthrough for a
   freshly-rented GPU instance, with launch commands, expected outputs,
   decision gates, cost ledger template, and recovery procedures.
4. **`docs/LESSONS_LEARNED.md`** — cross-phase gotchas to remember and
   pitfalls to avoid (22 entries).
5. **`docs/REVIEWER_DEFENSE.md`** — the "is band-dropout just training on
   the test corruptions?" concern, the steelman attack, the defenses,
   and the control experiments (`--band-mask hf_only / lf_only`).
6. **`docs/THEORY_OUTLINE.md`** — mechanism hypotheses + experiment
   shortlist for the analysis section of the paper.
7. **`docs/PAPER_OUTLINE.md`** — paper skeleton with current numbers
   slotted in and `[BLOCKED]` flags for what's pending.
8. **`docs/TIER1_FINDINGS.md`** — the detailed Phase-1 results doc with
   per-corruption tables. Read after `PHASE1_SUMMARY.md` for numbers.
9. **`docs/DATA_SETUP.md`** — what data is staged, what still needs
   downloading, where it goes, and the launch commands once it's there.
10. **`docs/FILES_INVENTORY.md`** — every file in the repo classified by
    phase and purpose.

## Results — per-folder summaries

Each `results/<subdir>/README.md` documents what was tested, what's in
the files, and the key findings:

| Folder | What's inside |
|---|---|
| **`results/phase1_curriculum/`** | Original CIFAR-10 curriculum experiments (1× → 64× epoch budgets). The work that proved CIFAR was too saturated for clean-acc gains and motivated the pivot to Imagenette. |
| **`results/phase1_imagenette/`** | Imagenette screening + multi-seed runs at 160² — the iterative search that found `band_drop_all` as the load-bearing mechanism and the longA two-stage recipe. |
| **`results/phase1_imagenette_full/`** | Full Imagenette-C eval per checkpoint (14 or 15 corruptions × {sev 1,3,5} × full 3925-image val set). **Contains the headline 224² mCE 47.2 result.** |
| **`results/phase1_comparison/`** | Multi-method side-by-side runs from `compare_methods.py` — AugMix vs PixMix vs IPMix vs `band_drop_all` and combinations, at 160 and 224. The 3-seed 224 reference table lives here. |
| **`results/phase1_ablations/`** | Targeted ablations: band-count sweep (saturates at 5–6 bands) and `min_kept=1` test (no improvement, kept off). |
| **`results/phase2_reviewer_defense/`** | The `--band-mask hf_only` / `lf_only` controls that pre-empt the operator-family-memorization reviewer attack. Single-seed already shows both targeted variants strictly worse than the full random method. |
| **`cifar_results/`** | Phase-2 CIFAR per-(dataset, method, seed) eval JSONs. Aggregated by `cifar_aggregate.py` into `docs/CIFAR_RESULTS.md`. |

## Logs

- `logs/phase1/` — old Phase-1 training logs (`run_*.txt`)
- `logs/phase2/` — Phase-2 training logs (de-risk, ablations, queue wrapper)
- `run_imagenette_3seed.log` (at repo root, only while it's being written)

## Quick reference — the method itself

Per training image, decompose into 6 Laplacian bands (σ ∈ {1,2,4,8,16} +
residual) and drop each independently with p=0.5. Apply *on top of* an
AugMix-style augmentation for the best result, with the JSD consistency
loss.

The standalone per-image transform is in `band_dropout_transform.py`
(the *core review item* — run `python band_dropout_transform.py` to
execute its faithfulness self-test).

## How to pick up if this session fails

1. Read `docs/PHASE2_STATUS.md` — written to be self-contained.
2. `tail -30 run_imagenette_3seed.log` to see where the 3-seed run got to.
3. `python cifar_aggregate.py` to refresh CIFAR result table.
4. For launching anything new, follow `docs/PHASE2_KICKOFF.md`
   step-by-step.

## What this repo does NOT contain

- Trained ImageNet-1k models (none exist yet; ImageNet-1k training is
  Phase 2's biggest cost, blocked on data + GPU compute).
- ImageNet-1k data (auth-gated; see `docs/DATA_SETUP.md`).
- ImageNet-C / ImageNet-R / ImageNet-A / ImageNet-Sketch (not yet downloaded).
- Geirhos cue-conflict stimuli (the `geirhos_shape_bias.py` runner
  exists; the external stimuli need cloning per `docs/DATA_SETUP.md`).
