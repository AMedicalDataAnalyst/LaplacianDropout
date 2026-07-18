# Phase 2 data setup

Single source of truth for what data we need, where it lives, and what
it's used for. Run `python check_data.py` after staging to verify.

## Currently staged on the local box (June 2026)

`/mnt/c/Users/JoyToy/Documents/Projects/data/`

| Dataset | Path | Used by | Notes |
|---|---|---|---|
| Imagenette 320² | `imagenette2-320/` | Phase 1 | already in `train/<wnid>/*.JPEG` layout |
| **PixMix fractals** | `fractals_and_fvis/fractals/images/` | Phase 2 Phase-D faithful PixMix | 14 248 fractal JPEGs — rsync to the GPU instance |
| **Geirhos texture-vs-shape** | `texture-vs-shape/` | `geirhos_shape_bias.py` | staged 2026-07-18; cue-conflict stimuli in `stimuli/style-transfer-preprocessed-512/` (1280 PNGs). Harness validated against pretrained RN-50 — see `results/phase2_geirhos/README.md` |

### Phase 1.5 historical (CIFAR transfer experiment, FAILED — not needed for Phase 2)

| Dataset | Path | Status |
|---|---|---|
| CIFAR-10 | `cifar-10-batches-py/` | not required for Phase 2 (CIFAR transfer failed; see `PHASE2_STATUS.md` Gate A) |
| CIFAR-100 | `cifar-100-python/` | same |
| CIFAR-10-C / -100-C | `CIFAR-{10,100}-C/` | same |

## Still to download — Tier 1 (required for headline)

| Dataset | Approx size | Source | Used for |
|---|---|---|---|
| **ILSVRC2012** train + val | ~150 GB | image-net.org (registration required) | All ImageNet training |
| **ImageNet-C** | ~70 GB | Hendrycks Zenodo "ImageNet-C" | Canonical mCE (AlexNet-normalized) |
| **ImageNet-R** | ~10 GB | github `hendrycks/imagenet-r` releases | Rendition robustness |
| **ImageNet-A** | ~3 GB | github `hendrycks/natural-adv-examples` releases | Natural adversarial |
| **ImageNet-Sketch** | ~7 GB | github `HaohanWang/ImageNet-Sketch` | Sketch OOD generalization |

## Still to download — Tier 2 (for fuller IPMix-grade eval)

| Dataset | Approx size | Source | Used for |
|---|---|---|---|
| **ImageNet-C̄** | ~70 GB | Hendrycks Zenodo "ImageNet-C-bar" | Alt-corruption mCE |
| **ImageNet-P** | ~120 GB | Hendrycks Zenodo "ImageNet-P" | Consistency on ImageNet |
| **ImageNet-O** | ~1 GB | github `hendrycks/natural-adv-examples` (-O subdir) | OOD detection (AUPR) |

## Optional — Tier 3 (generality / Geirhos)

| Dataset | Approx size | Source | Used for |
|---|---|---|---|
| **Stylized-ImageNet val** | ~7 GB | github `bethgelab/stylize-datasets` | Direct shape-bias test |
| **rgeirhos/texture-vs-shape** | small | `git clone rgeirhos/texture-vs-shape` | **STAGED locally** (see table above); rsync or re-clone on the instance |
| **Geirhos 16-class map** | small | `python make_geirhos_class_map.py` | **DONE** — `geirhos_classes_to_imagenet.json` checked in at repo root |
| **DTD (Textures)** | ~0.7 GB | torchvision built-in / vgg/dtd | Anomaly OOD |
| **LSUN / Places365 / SVHN** | varies | torchvision built-in | Anomaly OOD |

## Reference code to clone (don't reimplement)

| Repo | What's in it for us |
|---|---|
| `hendrycks/robustness` | Canonical mCE script, ImageNet-P mFR/mT5D eval |
| `hendrycks/imagenet-r` | -R loader + 200-class subset mask |
| `hendrycks/natural-adv-examples` | -A / -O loaders + AUPR for -O |
| `hendrycks/imagenet-c-bar` | -C̄ eval |
| `rgeirhos/texture-vs-shape` | Cue-conflict stimuli + ImageNet-class mapping |
| `Jingkang50/OpenOOD` *(optional)* | Unified anomaly-detection benchmark |
| `augmix` (already cloned at `augmix/`) | AlexNet-normalized `compute_mce` |
| `pixmix` (already cloned at `pixmix/`) | `calibration_tools.py`, ImageNet-R loader |

## Expected directory layout (final)

```
$DATA/    # the data root on your GPU instance, e.g. /data/datasets/
  imagenet/
    train/<wnid>/*.JPEG             [TODO]
    val/<wnid>/*.JPEG               [TODO]
  imagenet_c/<corruption>/<sev>/<wnid>/*.JPEG   [TODO]
  imagenet_r/<wnid>/*.jpg           [TODO]
  imagenet_a/<wnid>/*.jpg           [TODO]
  imagenet_sketch/<wnid>/*.JPEG     [TODO]
  fractals_and_fvis/                [rsync from local box where it's staged]
  imagenet_o/<wnid>/*.JPEG          [TODO Tier 2]
  imagenet_c_bar/...                [TODO Tier 2]
  imagenet_p/<perturbation>/...     [TODO Tier 2]
  texture-vs-shape/stimuli/style-transfer-preprocessed-512/<geirhos_class>/*.png   [staged locally; rsync]
  stylized-imagenet/val/...         [TODO Tier 3]

# At the repo root (NOT in $DATA):
geirhos_classes_to_imagenet.json    [DONE — checked in]
```

## Launch commands (post-staging, ImageNet)

```bash
# Phase 2 #5 — ResNet-50 × 4 methods (CIFAR analogue already wired via run_cifar_sweep.sh)
for METHOD in baseline augmix band_drop_all band_drop_all+augmix; do
  python augmix/imagenet.py data/imagenet data/imagenet_c \
    --model resnet50 --epochs 90 --batch-size 256 \
    --aug-method "$METHOD" --save snapshots/rn50_$METHOD
done

# Phase 2 #6 — ConvNeXt-T (same methods, AdamW + cosine recipe)
for METHOD in baseline augmix band_drop_all band_drop_all+augmix; do
  python augmix/imagenet.py data/imagenet data/imagenet_c \
    --model convnext_tiny --epochs 90 --batch-size 256 \
    --optimizer adamw --schedule cosine --learning-rate 5e-4 \
    --decay 0.05 --warmup-epochs 5 \
    --aug-method "$METHOD" --save snapshots/cnt_$METHOD
done

# Phase 2 #6 — ViT-B/16
for METHOD in baseline augmix band_drop_all band_drop_all+augmix; do
  python augmix/imagenet.py data/imagenet data/imagenet_c \
    --model vit_b_16 --epochs 90 --batch-size 128 \
    --optimizer adamw --schedule cosine --learning-rate 3e-4 \
    --decay 0.05 --warmup-epochs 10 \
    --aug-method "$METHOD" --save snapshots/vit_$METHOD
done

# Phase 2 #7 — Faithful PixMix with fractals (now that fractals are staged)
for METHOD in pixmix baseline band_drop_all+pixmix; do
  python pixmix/imagenet.py \
    --data-standard data/imagenet/train --data-val data/imagenet/val \
    --imagenet-r-dir data/imagenet_r --imagenet-c-dir data/imagenet_c \
    --mixing-set /mnt/c/Users/JoyToy/Documents/Projects/data/fractals_and_fvis/fractals/images \
    --num-classes 1000 --aug-method "$METHOD" --save checkpoints/$METHOD
done

# Phase 2 #8 — Per-trained-checkpoint eval battery
python ood_eval.py --checkpoint snapshots/rn50_band_drop_all+augmix/model_best.pth.tar \
  --arch resnet50 --r-dir data/imagenet_r --a-dir data/imagenet_a \
  --sketch-dir data/imagenet_sketch --out ood_rn50_combo.json

python calibration_eval.py --checkpoint snapshots/rn50_band_drop_all+augmix/model_best.pth.tar \
  --arch resnet50 --val-dir data/imagenet/val --c-dir data/imagenet_c \
  --out calib_rn50_combo.json

python adversarial_eval.py --checkpoint snapshots/rn50_band_drop_all+augmix/model_best.pth.tar \
  --arch resnet50 --val-dir data/imagenet/val --attack pgd --eps 4 \
  --subsample 5000 --out adv_rn50_combo.json

python geirhos_shape_bias.py --checkpoint snapshots/rn50_band_drop_all+augmix/model_best.pth.tar \
  --arch resnet50 --stimuli-dir data/texture-vs-shape/stimuli/style-transfer-preprocessed-512 \
  --class-map-json geirhos_classes_to_imagenet.json --out shape_bias_rn50_combo.json
```
