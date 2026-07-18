# Phase 2 — Geirhos cue-conflict shape bias (harness validated, awaiting 1k models)

Mechanism experiment #5 from `docs/THEORY_OUTLINE.md`: the shape-vs-
texture bias measurement of Geirhos et al. (ICLR 2019). Prediction (H2):
`band_drop_all`-trained models should be more shape-biased than
baseline/AugMix.

## Status (2026-07-18)

**The harness is validated end-to-end and ready; the science number is
blocked on ImageNet-1k training** (Phase 2 A/B, see
`docs/PHASE2_KICKOFF.md`).

Why it cannot run on the existing Imagenette checkpoints: the 1280
cue-conflict stimuli span Geirhos's 16 entry-level categories, and a
model must be able to name *both* the shape class and the texture class
for a stimulus to count. Only 2 of the 16 categories (dog, truck)
exist in Imagenette-10, leaving exactly **10 usable stimuli** — far too
few to estimate shape bias. This eval is inherently an ImageNet-1k-head
measurement.

## Harness validation — torchvision pretrained ResNet-50

`shape_bias_rn50_torchvision_pretrained.json` — torchvision
`ResNet50_Weights.IMAGENET1K_V1` run through our
`geirhos_shape_bias.py`:

| Metric | Ours | Published |
|---|---|---|
| Cue-conflict stimuli scored | 1200 | 1200 (1280 minus 80 shape==texture) |
| shape_correct / texture_correct / neither | 179 / 627 / 394 | — |
| **shape_bias** | **0.222** | ~0.21–0.22 for vanilla ResNet-50 |

Matching the published value confirms stimuli parsing, the 16-class
mapping, and the decision rule are all faithful. Note the decision rule
uses the **mean** probability over each category's ImageNet classes,
matching Geirhos's canonical `probabilities_to_decision.py` (an earlier
draft of our runner summed, which biases toward large categories — fixed
2026-07-18 before any science numbers were produced).

## Inputs (staged, see `docs/DATA_SETUP.md` Tier 3)

- Stimuli: `texture-vs-shape/stimuli/style-transfer-preprocessed-512/`
  (clone of `rgeirhos/texture-vs-shape` in the local data root; rsync or
  re-clone onto the GPU instance).
- Class map: `geirhos_classes_to_imagenet.json` at repo root, generated
  by `make_geirhos_class_map.py` directly from Geirhos's own
  `human_categories.py` (16 categories, 207 ImageNet indices).

## To run on the Phase-2 ImageNet-1k checkpoints

```bash
python geirhos_shape_bias.py \
  --checkpoint snapshots/rn50_<method>/model_best.pth.tar --arch resnet50 \
  --stimuli-dir $DATA/texture-vs-shape/stimuli/style-transfer-preprocessed-512 \
  --class-map-json geirhos_classes_to_imagenet.json \
  --out results/phase2_geirhos/shape_bias_rn50_<method>.json
```

Expected outcome that would support H2: baseline ≈ 0.22 (validated
here), AugMix ≈ 0.28–0.32 (published AugMix shape-bias gain), and
`band_drop_all(+AugMix)` above AugMix.
