# Phase 2 kickoff checklist

Walk through this top-to-bottom on a freshly-rented GPU instance. Each
step has its launch commands, expected outputs, wall-clock estimate,
and decision gate (if any). Total budget for full venue: **~$2,600 spot,
~3 weeks compute**. Workshop-only path stops after Phase B: **~$800, ~1 week**.

Notation:
- `$REPO` = path to this repo on the instance (e.g. `/home/ubuntu/laplacian_schedule`)
- `$DATA` = data root (e.g. `/data/datasets`)
- `$SNAP` = checkpoint root (e.g. `$REPO/snapshots`)

---

## Prerequisites

- **GPU**: 8× A100 80GB recommended (or 8× H100 80GB). 40 GB variants force
  smaller batches and slow wall-clock; avoid unless community spot price <
  $0.50/GPU·h.
- **Disk**: ≥500 GB NVMe (350 GB for data, 50 GB for checkpoints, headroom for OS).
- **CUDA / PyTorch**: PyTorch 2.x with CUDA 12.x. Verify: `python -c "import
  torch; print(torch.cuda.is_available(), torch.version.cuda)"`.
- **tmux or screen** — every long-running training command must be in a
  detachable session.

---

## Step 1 — Instance setup (~10 min)

```bash
# 1a. Clone the repo to the instance
git clone <your-repo-url> $REPO
cd $REPO

# 1b. Install Python deps
#     torch+torchvision must match the instance's CUDA version. For
#     CUDA 12.4 (8x A100 80GB):
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 \
  --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt    # numpy, scipy, matplotlib, Pillow,
                                    # scikit-image, imagecorruptions, torchattacks

# 1c. (One-time) Verify band_dropout_transform self-test passes
python band_dropout_transform.py
# Expect: "All faithfulness checks passed."

# 1d. Set environment for convenience
export REPO=$(pwd)
export DATA=/data/datasets   # adjust to your mount
export SNAP=$REPO/snapshots
mkdir -p $SNAP
```

---

## Step 2 — Stage data (~hours-to-days, mostly bandwidth-bound)

Download each into `$DATA/`. See `docs/DATA_SETUP.md` for full list and sources
(no fabricated URLs; find by name on the listed sites).

### Tier 1 — required for headline (download these first)

```bash
# ILSVRC2012 train + val (~150 GB). Use the standard valprep.sh to sort val into class dirs.
$DATA/imagenet/train/<wnid>/*.JPEG
$DATA/imagenet/val/<wnid>/*.JPEG

# ImageNet-C (~70 GB) — Hendrycks Zenodo "ImageNet-C"
$DATA/imagenet_c/<corruption>/<sev>/<wnid>/*.JPEG

# ImageNet-R / -A / -Sketch (~20 GB combined)
$DATA/imagenet_r/<wnid>/*.jpg
$DATA/imagenet_a/<wnid>/*.jpg
$DATA/imagenet_sketch/<wnid>/*.JPEG
```

### Tier 2 — only if going beyond workshop
```bash
# Hendrycks Zenodo: ImageNet-C-bar, ImageNet-P, ImageNet-O
# rgeirhos/texture-vs-shape (small, git clone)
# bethgelab/stylize-datasets (val only is enough)
```

### Useful to rsync from the local box (optional)
- `imagenette2-320` — only needed if you want to re-run Phase 1 sanity
  checks. Not required for Phase 2.
- `fractals_and_fvis/` — needed for the faithful PixMix run in Phase D
  (~2 GB, can also be re-downloaded on the instance).
- **NOT NEEDED**: `cifar-10-batches-py`, `cifar-100-python`, `CIFAR-10-C`,
  `CIFAR-100-C`. The CIFAR transfer experiment was Phase 1.5 and failed
  (see `docs/PHASE2_STATUS.md` Gate A). Phase 2 is ImageNet-only.

### Verify staging
```bash
python check_data.py --root $DATA
# Tier-1 items must show [OK]. Tier-2/3 can be MISSING for now.
```

---

## Step 3 — Phase 0: pre-flight (~1 h, ~$15)

A 1-2 epoch end-to-end run of each method through the patched augmix
harness, on Imagenette (which is in repo via rsync) or a tiny ImageNet
subset. Catches setup bugs before Phase A.

```bash
# Easiest: use Imagenette directly (already in ImageFolder layout)
for METHOD in baseline augmix band_drop_all band_drop_all+augmix; do
  python augmix/imagenet.py $DATA/imagenette2-320 \
    --model resnet50 --epochs 2 --batch-size 256 --num-workers 8 \
    --aug-method "$METHOD" --save /tmp/preflight_$METHOD
done
# Pass criteria: each finishes without error; final test acc > 0.10 (not random).
```

If any method errors out: **stop**. Common causes:
- OOM at batch 256 → drop to 128 or 64; investigate before going further
- Missing `corrupted_data` arg → already optional in our patched harness
- AlexNet table missing → confirm `augmix/imagenet.py` line 148+ has `ALEXNET_ERR`

---

## Step 4 — Phase A: ImageNet-100 de-risk (~1 day, ~$100)

The real go/no-go. If RN-50 mCE on the 100-class subset is competitive
with AugMix, escalate to B. If not, debug here for $100 not $700.

### 4a. Create the 100-class subset (~5 min)
```bash
# Sorted wnid first-100 — reproducible. Symlinks save disk.
SUBSET=$DATA/imagenet_100
mkdir -p $SUBSET/{train,val}
for split in train val; do
  ls $DATA/imagenet/$split | sort | head -100 | while read wnid; do
    ln -s $DATA/imagenet/$split/$wnid $SUBSET/$split/$wnid
  done
done
# For ImageNet-C subset (use only the same 100 wnids):
SUBSET_C=$DATA/imagenet_c_100
mkdir -p $SUBSET_C
for c in $(ls $DATA/imagenet_c); do
  for s in 1 2 3 4 5; do
    mkdir -p $SUBSET_C/$c/$s
    ls $DATA/imagenet/train | sort | head -100 | while read wnid; do
      ln -s $DATA/imagenet_c/$c/$s/$wnid $SUBSET_C/$c/$s/$wnid 2>/dev/null
    done
  done
done
```

### 4b. Launch 4 methods (parallel if you have multi-GPU; sequential is ~24 h)
```bash
tmux new -s phase_a
for METHOD in baseline augmix band_drop_all band_drop_all+augmix; do
  python augmix/imagenet.py $SUBSET $SUBSET_C \
    --model resnet50 --epochs 90 --batch-size 256 \
    --aug-method "$METHOD" \
    --save $SNAP/phaseA_rn50_${METHOD//+/_p_} \
    2>&1 | tee $REPO/logs/phaseA_${METHOD//+/_p_}.log
done
```
Wall-clock: ~6 h per method on 8× A100 → ~24 h sequential, or ~6 h if
parallelized across 4 instances. For just-de-risk, sequential is fine.

### 4c. Gate A decision
The augmix harness prints `mCE (normalized by AlexNet): <number>` at the
end of training. Compare:

| Outcome | Decision |
|---|---|
| `band_drop_all+augmix` mCE ≤ AugMix mCE − 3 AND clean within 1 pp of baseline | **GREEN — escalate to Phase B** |
| `band_drop_all+augmix` mCE ≈ AugMix mCE | **YELLOW — proceed to Phase B but adjust paper pitch to "complementary, simpler"** |
| `band_drop_all+augmix` mCE > AugMix mCE OR clean drops > 2 pp | **RED — STOP. Debug.** Possible causes: ImageNet recipe mismatch, data preprocessing bug, AugMix combination order issue. Re-check `docs/LESSONS_LEARNED.md`. |

---

## Step 5 — Phase B: ImageNet-1k RN-50 headline (~3-5 days, ~$700)

The publishable result. Canonical AlexNet-normalized mCE comes free at
end of training.

```bash
tmux new -s phase_b
for METHOD in baseline augmix band_drop_all band_drop_all+augmix; do
  python augmix/imagenet.py $DATA/imagenet $DATA/imagenet_c \
    --model resnet50 --epochs 90 --batch-size 256 --num-workers 16 \
    --aug-method "$METHOD" \
    --save $SNAP/phaseB_rn50_${METHOD//+/_p_} \
    2>&1 | tee $REPO/logs/phaseB_${METHOD//+/_p_}.log
done
```

Each run: ~16 h on 8× A100 (baseline ~8 h, JSD ones ~16 h). Total ~3 days
sequential. **Save best.pth.tar** — Phase C evals load these.

### Gate B decision
| Outcome | Decision |
|---|---|
| `band_drop_all+augmix` mCE ≤ 65 AND clean within 1 pp of baseline | **Workshop publishable as-is. Run Phase C+D.** |
| mCE 65-75 | **Comparable to AugMix. Run Phase C+D, expect "complementary" framing.** |
| mCE > 75 | **Headline didn't scale.** Stop and rethink pitch. |

Reference points: AugMix 65.3, PixMix ~56, IPMix 63, DeepAugment 60.4, PRIME 55.5.

---

## Step 6 — Phase C: Robustness battery (~1 day, ~$25 total)

All cheap, single-GPU. Run on Phase B's 4 checkpoints.

```bash
CKPT=$SNAP/phaseB_rn50_band_drop_all_p_augmix/model_best.pth.tar

# 6a. OOD generalization
python ood_eval.py --checkpoint $CKPT --arch resnet50 \
  --r-dir $DATA/imagenet_r --a-dir $DATA/imagenet_a \
  --sketch-dir $DATA/imagenet_sketch --out results/ood_combo.json

# 6b. Calibration
python calibration_eval.py --checkpoint $CKPT --arch resnet50 \
  --val-dir $DATA/imagenet/val --c-dir $DATA/imagenet_c \
  --out results/calib_combo.json

# 6c. Adversarial (PGD; skip AutoAttack unless a reviewer asks)
python adversarial_eval.py --checkpoint $CKPT --arch resnet50 \
  --val-dir $DATA/imagenet/val --attack pgd --eps 4 --subsample 5000 \
  --out results/adv_combo.json

# 6d. Geirhos shape-bias % — needs external stimuli + class map first
git clone https://github.com/rgeirhos/texture-vs-shape $DATA/texture-vs-shape
# Then derive geirhos_classes_to_imagenet.json from their mapping file; see docs/DATA_SETUP.md
python geirhos_shape_bias.py --checkpoint $CKPT --arch resnet50 \
  --stimuli-dir $DATA/texture-vs-shape/data-cue-conflict \
  --class-map-json geirhos_classes_to_imagenet.json \
  --out results/shape_bias_combo.json
```

Repeat for the other 3 Phase-B checkpoints (baseline, augmix, band_drop_all)
to populate the comparison table. **Saves the entire robustness table for
the paper in 1 day.**

### Reviewer-defense control on ImageNet (~2 extra training runs, ~$200)
Critical for the paper. Train `band_drop_all+augmix` with `--band-mask
hf_only` and `--band-mask lf_only`. See `docs/REVIEWER_DEFENSE.md`.
```bash
for MASK in hf_only lf_only; do
  python augmix/imagenet.py $DATA/imagenet $DATA/imagenet_c \
    --model resnet50 --epochs 90 --batch-size 256 \
    --aug-method band_drop_all+augmix --band-mask $MASK \
    --save $SNAP/phaseC_rn50_combo_$MASK \
    2>&1 | tee $REPO/logs/phaseC_combo_$MASK.log
done
# Expected: both lose to band_drop_all+augmix --band-mask all on mCE.
# That refutes the operator-family-memorization attack.
```

---

## Step 7 — Phase D: Faithful PixMix with fractals (~3 days, ~$400)

```bash
tmux new -s phase_d
for METHOD in pixmix baseline band_drop_all+pixmix; do
  python pixmix/imagenet.py \
    --data-standard $DATA/imagenet/train --data-val $DATA/imagenet/val \
    --imagenet-r-dir $DATA/imagenet_r --imagenet-c-dir $DATA/imagenet_c \
    --mixing-set $DATA/fractals_and_fvis/fractals/images \
    --num-classes 1000 \
    --aug-method "$METHOD" \
    --save $SNAP/phaseD_pixmix_${METHOD//+/_p_} \
    2>&1 | tee $REPO/logs/phaseD_${METHOD//+/_p_}.log
done
```
The pixmix harness has -C eval built in; mCE prints at end of training.

### Gate D decision
If `band_drop_all+pixmix` doesn't beat pixmix-alone by ≥2 pp on mCE, the
band-dropout signal is "complementary to AugMix but not PixMix" —
interesting and reportable but adjusts paper positioning.

---

## Step 8 — Phase E: Architecture diversity (~6-8 days, ~$1,250)

Run **ONLY** if Phase B is strong. Each arch ~$400-700.

### ConvNeXt-T (~4 days)
```bash
for METHOD in baseline augmix band_drop_all band_drop_all+augmix; do
  python augmix/imagenet.py $DATA/imagenet $DATA/imagenet_c \
    --model convnext_tiny --epochs 90 --batch-size 256 --num-workers 16 \
    --optimizer adamw --schedule cosine --learning-rate 5e-4 \
    --decay 0.05 --warmup-epochs 5 \
    --aug-method "$METHOD" \
    --save $SNAP/phaseE_convnext_${METHOD//+/_p_}
done
```

### ViT-B/16 (~3 days)
ViT recipes are touchier. Start with these defaults and adjust if loss
plateaus high:
```bash
for METHOD in baseline augmix band_drop_all band_drop_all+augmix; do
  python augmix/imagenet.py $DATA/imagenet $DATA/imagenet_c \
    --model vit_b_16 --epochs 90 --batch-size 128 --num-workers 16 \
    --optimizer adamw --schedule cosine --learning-rate 3e-4 \
    --decay 0.05 --warmup-epochs 10 \
    --aug-method "$METHOD" \
    --save $SNAP/phaseE_vit_${METHOD//+/_p_}
done
```

Run Phase C evals on each as completed.

---

## Step 9 — Phase F: Mechanism analysis (~3 days intermittent, ~$150)

The figures for the paper's "Why does this work?" section. Each is small;
can run on a single rented GPU between other work.

```bash
# 9a. Band-count ablation (already done on Imagenette; redo on IN-100 if you want scale match)
# 9b. Per-band-only ablation (train 6 models dropping each band individually)
# 9c. Gradient-spectrum analysis (eval-time on baseline + combo)
# 9d. Band-targeted PGD (small extension to adversarial_eval.py)
# 9e. Matched-budget Gaussian noise control
```

See `docs/THEORY_OUTLINE.md` Section "Concrete experiments" for exact
specs. These mostly produce one figure each; F4 (band-targeted PGD) is the
strongest mechanistic figure.

---

## Step 10 — Phase G: Writeup integration

Aggregate everything:
```bash
# Phase B canonical mCE — printed at end of training, captured in your tee logs.
grep "mCE" $REPO/logs/phaseB_*.log

# Phase C battery — in results/*.json
ls -la results/

# Format the headline table:
python -c "
import json, glob
for f in sorted(glob.glob('results/*.json')):
    print(f, json.load(open(f)))
"
```

Slot everything into `docs/PAPER_OUTLINE.md`'s `[BLOCKED]` rows. Then move
to drafting `docs/PHASE2_FINAL.md` (you'll write this as you go).

---

## Decision gates summary

| After | Spent | Gate | If pass | If fail |
|---|---|---|---|---|
| Phase 0 | ~$15 | pipeline works | A | debug |
| Phase A | ~$115 | RN-50 method shows on IN-100 | B | stop, $115 sunk |
| Phase B | ~$815 | IN-1k mCE ≤ 65 (workshop) | C+D | stop, $815 sunk, rethink pitch |
| Phase C | ~$840 | robustness battery looks coherent | D, E | "complementary" framing |
| Phase D | ~$1,240 | combo beats PixMix | E | "complementary to AugMix only" framing |
| Phase E | ~$2,490 | ConvNeXt/ViT show similar wins | submit | "CNN only" framing |

The two clearest stop-points are after **A** ($115 sunk) and after
**B** ($815 sunk). Both give full info to decide whether escalating is
worth it.

---

## Common failure modes (from `docs/LESSONS_LEARNED.md`)

- **`accuracy()` view bug** — we already patched `.view(-1)` → `.reshape(-1)`
  in `augmix/imagenet.py:201`. Fresh clones of upstream augmix repo will
  re-introduce this; verify the file matches our patched version.
- **`corrupted_data` positional arg required** — our patch made it
  optional via `nargs='?'`. Don't merge an upstream cleanup that removes
  this.
- **OOM on 40 GB GPUs at batch 256** — drop to 128 for ViT, 192 for
  ConvNeXt, 256 still fits for RN-50.
- **NaN loss in band methods** — the augmix CIFAR harness needed grad clip
  at 5.0; the ImageNet harness does NOT (we tested — no NaN at RN-50/
  ConvNeXt batch 256 LR 0.1 SGD). If you hit NaN, first try LR/2.
- **JSD 3× forward pass blows up memory** — JSD concatenates 3 views into
  one batch; effective forward batch = 3× nominal. Use `--batch-size 128`
  (= 384 effective) for ViT-B if 256 OOMs.
- **DataParallel state-dict prefix** — our eval scripts strip `module.`
  prefix on load; don't worry about it.

---

## Cost ledger (running tally template)

| Item | Started | Finished | Cost | mCE / metric |
|---|---|---|---|---|
| Phase 0 | | | | n/a |
| Phase A: baseline | | | | |
| Phase A: augmix | | | | |
| Phase A: band_drop_all | | | | |
| Phase A: combo | | | | |
| **Gate A decision** | | | | |
| Phase B: baseline | | | | |
| ... | | | | |

Keep this table updated as you go. Sets up the paper's experimental
section directly.

---

## If something breaks mid-Phase

- **Don't restart from scratch.** Each method's training writes a
  checkpoint per epoch (`checkpoint.pth.tar`) and best (`model_best.pth.tar`).
  Resume with `--resume <path>` on the augmix harness (line 414).
- **Lost progress notes** — every long-running launch is wrapped in
  `tee logs/<name>.log`. Even if tmux dies, the log is on disk.
- **Spot instance preemption** — Lambda/RunPod sometimes preempt. Set up
  a cron that rsyncs `$SNAP/` and `$REPO/results/` to persistent storage
  every hour. The cost of re-running an 8-hour epoch sequence is
  ~$96 in compute alone.

---

## Final reminder before launching

Update `docs/PHASE2_STATUS.md` after each Gate decision and at the end of
each Phase. A fresh Claude session reading the repo should be able to
look at PHASE2_STATUS and know exactly what's done, what's running, and
what's next. The same discipline that saved this project from session
crashes during Phase 1 will save it from session crashes during Phase 2.
