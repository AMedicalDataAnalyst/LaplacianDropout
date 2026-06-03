# Overnight status — tier-2 scaffolding implementation

## TL;DR

All code/scaffolding for tier-2 is written and smoke-tested. The remaining
work is **data acquisition + compute**, neither of which I can do
autonomously. The band-count ablation you asked for is running now.

## Done tonight (10 items, ~13 files changed/added)

### Core review item — read this first
- **`band_dropout_transform.py`** (NEW, ~110 lines) — the per-image
  Laplacian-band-dropout torchvision transform reused by both ImageNet
  harnesses. Self-test verifies bit-identity to the batch version at B=1,
  perfect reconstruction, and the four edge cases. Run
  `python band_dropout_transform.py` to re-verify.

### Harness patches (small diffs against the upstream repos)
- **`augmix/imagenet.py`** — added `--aug-method {augmix, baseline,
  band_drop_all, band_drop_all+augmix}`, `--band-drop-p`, `--optimizer
  {sgd,adamw}`, `--schedule {step,cosine}`, `--warmup-epochs`. Made
  `corrupted_data` optional (skip test_c if absent). Fixed a pre-existing
  py2-era bug in `accuracy()` (`.view` → `.reshape` on non-contiguous).
  Smoke-tested all 4 methods on Imagenette + ConvNeXt-T with adamw/cosine.
- **`pixmix/imagenet.py`** — added `--aug-method {pixmix, baseline,
  band_drop_all, band_drop_all+pixmix}` and `--band-drop-p`; modified
  `PixMixDataset.__getitem__` to support the four paths. Smoke-tested all
  4 methods (PixMixDataset instantiates and produces valid normalized
  tensors).

### Standalone eval scripts (drop-in, no harness coupling)
- **`ood_eval.py`** — ImageNet-R / -A / -Sketch evaluation. Reuses pixmix's
  wnid masks rather than duplicating the 200-class subset lists. Smoke-tested.
- **`calibration_eval.py`** — RMS-CE / AURRA on clean + ImageNet-C per
  corruption × severity. Re-uses pixmix's `calibration_tools.get_measures`.
- **`adversarial_eval.py`** — torchattacks PGD / APGD / AutoAttack
  wrapper. ε in [0,1] pixel space, model normalization handled internally.
  Installed `torchattacks 3.5.1`. Smoke-tested PGD path.
- **`geirhos_shape_bias.py`** — Geirhos cue-conflict shape-bias %. Expects
  the rgeirhos/texture-vs-shape stimuli + a `geirhos_classes_to_imagenet.json`
  map (the runner is written; the external repo clone + JSON map is in
  `DATA_SETUP.md`).

### Data + writing
- **`DATA_SETUP.md`** — expected directory layout, where each dataset
  comes from (by name; no fabricated URLs), and the launch commands for
  every tier-2 run once data is staged.
- **`check_data.py`** — sanity check that prints OK / MISSING for every
  expected path. Run anytime after staging.
- **`THEORY_OUTLINE.md`** — four hypotheses for why band-masking helps and
  a six-experiment shortlist that discriminates between them (the analysis
  section that turns this from "an aug" into a paper).
- **`PAPER_OUTLINE.md`** — tier-2 paper skeleton with current numbers
  slotted in; remaining sections marked `[BLOCKED-ON-DATA]`.

## Running now (overnight)

- **Band-count ablation** (your request) — sweeps Laplacian levels ∈ {1..6}
  (band counts 2..7) for `band_drop_all + JSD` on Imagenette-224, single
  seed each, fast corruption eval. Outputs:
  `band_count_ablation_L{1..6}_bands*.json`. Log:
  `run_band_count_ablation.log`. Estimated ~3-5 h total. I'll incorporate
  results into TIER1_FINDINGS / PAPER_OUTLINE on completion.

## Blocked on you / external resources

These are the things I can't move forward on autonomously:

### Data acquisition (#1)
Cannot download:
- **ImageNet-1k** (image-net.org auth required, ~150 GB)
- **ImageNet-C** (Hendrycks Zenodo tarball, ~70 GB)
- **ImageNet-R/A/Sketch** (Hendrycks / Wang github releases)
- **PixMix fractals mixing set** (PixMix release, ~2 GB)
- **Geirhos texture-vs-shape repo + 16-class mapping JSON**

`DATA_SETUP.md` has the names + expected paths. Run `python check_data.py`
to see status.

### Tier-2 training runs (#4, #5, #6, #7-runs)
All depend on the above data + multi-GPU compute (1-2 days per run × ~16
runs across architectures × methods + faithful PixMix). The launch commands
are in `DATA_SETUP.md` and the harnesses are smoke-tested.

### Tier-2 evals (#8-runs, #9-runs, #10-runs, #12-runs)
Need at least one trained ImageNet-1k checkpoint each.

## Tasks state

| # | Subject | State |
|---|---|---|
| 1 | Acquire ImageNet data | completed (scaffolding) — actual downloads on you |
| 2 | band_drop_all transform module | **completed** ★ review-anchor file |
| 3 | Patch augmix harness | completed |
| 4 | De-risk ImageNet-100/200 | pending — needs data |
| 5 | Train ImageNet-1k RN-50 | pending — needs data + GPU-days |
| 6 | ConvNeXt-T + ViT-B | completed (recipe flags); runs pending |
| 7 | Faithful PixMix-with-fractals | completed (patch); runs pending |
| 8 | Calibration adapter | completed |
| 9 | OOD eval script | completed |
| 10 | Adversarial wrapper | completed |
| 11 | Theory analysis | completed (outline + experiment shortlist) |
| 12 | Geirhos runner | completed |
| 13 | Tier-2 writeup | in_progress (outline + slotted numbers) |
| 14 | Band-count ablation | **running now** |

## What you do tomorrow morning

1. Read `band_dropout_transform.py` (the review-anchor file). The two
   judgment calls I baked in are documented in the module docstring and
   summarized in my prior message (normalized-tensor domain; no all-zero
   guard).
2. Skim the augmix/pixmix patches in `git diff` (use `git diff
   augmix/imagenet.py pixmix/imagenet.py` once you `git init` if needed —
   the repo isn't a git repo currently).
3. Check the band-count ablation log: `tail run_band_count_ablation.log`
   and `ls band_count_ablation_*.json`. If I'm still working on it I'll
   have a follow-up message with the results table.
4. Start the data acquisition for `DATA_SETUP.md` — ImageNet-1k is the
   slow one; everything else is fast.
5. Pick which architectures to run first (ResNet-50 alone for workshop,
   add ConvNeXt or ViT for full venue).
