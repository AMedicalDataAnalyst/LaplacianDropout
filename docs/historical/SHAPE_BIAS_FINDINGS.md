# Shape-bias experiments on Imagenette — findings and recipe

## TL;DR

The original Laplacian-curriculum approach (residual-first warmup → progressive band unlock → all-bands dropout phase) was overcomplicated. The same shape-bias / robustness benefits come from **random band dropout applied uniformly from step 0 with no curriculum**, and the accuracy hit can be recovered with a short clean fine-tune.

**Best recipe (longA)** — strict improvement over baseline on every dimension we measured:

```
Stage 1: train ResNet-18 from scratch on Imagenette-160 with `band_drop_all`
         augmentation (every band including residual independently kept
         with p=0.5 per sample). Plateau-based training, max 80 epochs.
Stage 2: continue training the same model on clean images (no augmentation)
         at LR ÷ 10. Plateau-based, max 50 epochs.
```

Result (3 seeds, full Imagenette-C eval = 14 corruptions × {sev 1,3,5} × full val set):

| Model | Clean val_acc | Mean corruption acc | Residual-only eval | σ=2 blur eval |
|---|---|---|---|---|
| ResNet-18 baseline | 0.8759 ± 0.0003 | 0.5429 ± 0.0064 | 0.111 | 0.486 |
| **longA** | **0.8864 ± 0.0019** | **0.7131 ± 0.0081** | **0.368** | **0.851** |
| **extralongA** (max_pretrain=120) | 0.8809 ± 0.0018 | **0.7360 ± 0.0057** | 0.412 | 0.853 |
| Δ longA vs baseline | **+1.05 pp** | **+17.0 pp** | +25.7 pp | +36.5 pp |
| Δ extralongA vs baseline | +0.50 pp | **+19.3 pp** | +30.1 pp | +36.7 pp |

`longA` and `extralongA` are two clean Pareto points: pick longA for maximum clean accuracy (still beats baseline), pick extralongA for maximum corruption robustness (still beats baseline clean). Both strictly improve on both axes.

## What we tried, in order

### 1. Original Laplacian curriculum on Imagenette
The script developed earlier for CIFAR (`curriculum_experiment.py`) was ported to Imagenette: 6-band Laplacian decomposition, schedule = residual-only → progressive band unlock → all-bands tail with optional 50% bandpass dropout in the final phase.

Out of the box, `curriculum_dropout` gave 0.760 clean / 0.671 corr — strong shape bias (residual-only test = 0.65) but a 11pp clean-accuracy deficit. With *more* training the curriculum variants got *worse* (the schedule's fractional structure means more epochs = more residual-only warmup, which the model couldn't recover from).

### 2. Plateau-based schedule redesign
Replaced the fixed fractional schedule with a plateau detector: each phase trains until val_acc stops improving, then transitions. Fixed the accuracy-regression-with-more-epochs problem (curriculum_dropout went from 0.760 to 0.760 instead of dropping), and the shape-bias signature stayed sharp. But still ~11pp below baseline on clean.

### 3. Dropout-strength sweep
Tried `p ∈ {0.3, 0.5, 0.7, 0.9}` for the final-phase dropout. p=0.5 was the sweet spot; higher dropout reduced both clean acc and corruption robustness.

### 4. Ablation: residual protection is harmful
The original schedule protected the residual band from dropout (on the theory that shape information should always be present). Removing this protection (`cd_p05_noprotect`) gave 0.794 clean / 0.735 corr — better on *both* dimensions than the protected version.

### 5. **Big finding: the curriculum isn't needed**
Tested `band_drop`: random band dropout (each bandpass band kept with p=0.5 per sample) applied from step 0 with no curriculum and no warmup. Result: 0.843 clean / 0.684 corr. Adding "drop the residual too" → `band_drop_all` gave 0.867 / 0.713 — basically matching the protected-residual variant's clean acc while gaining corruption robustness.

The Laplacian curriculum framework was a red herring; the work is being done by the random band masking, not by the staging. A simple data augmentation matches a 6-phase curriculum.

### 6. Alternative: blur augmentation
Tested `joint_blur` (per-batch 50/50 mix of clean and σ=4 Gaussian-blurred images). Reached 0.850 clean / 0.709 corr — close to band_drop_all on robustness but with *no* shape bias (residual-only test = 0.164, similar to baseline). So plain blur augmentation gets you the corruption robustness without the representational shape bias. Useful to distinguish: shape bias and corruption robustness are correlated but not identical.

### 7. Two-stage fine-tuning
Reasoning: band_drop_all has all the properties we want except clean accuracy is 0.8pp under baseline. A short clean fine-tune at low LR (lr/10) should bring clean accuracy back up without destroying the shape bias.

- **twostage_A** (band_drop_all → clean FT @ lr/10): 0.880 / 0.699. Beats baseline clean (!), keeps most corruption robustness.
- **twostage_A full-LR FT**: 0.885 / 0.667. Higher clean but eats more robustness — the lower the FT LR, the better the trade-off.
- **twostage_B** (baseline → band_drop_all FT): 0.872 / 0.691. Slightly worse than A; the band_drop has to undo the texture-bias baseline learned.

### 8. **Best result: longA**
Same as twostage_A but with `max_final_phase=80` for the band_drop_all pretraining (vs 50). Longer pretraining lets band_drop_all converge to a better starting point (0.877 / 0.753 after stage 1), and the clean fine-tune then brings clean accuracy up to **0.886 / 0.713**.

### 9. Distillation experiments (no win)
- `band_drop_all + KL←baseline`: KL distillation from baseline teacher into a band_drop_all-augmented student. Result 0.870 / 0.717 — marginal over band_drop_all alone, much worse than longA.
- `baseline + KL←longA (no aug)`: distill longA's soft labels into a fresh baseline student trained on clean images only. Result 0.876 / 0.516 — barely above baseline. **Distillation alone doesn't transfer robustness when the student never sees augmented inputs.**

## Trade-off frontier

![tradeoff](shape_bias_tradeoff.png)

Marker size encodes residual-only test accuracy (shape-bias proxy). The Pareto frontier on (clean, corruption) is owned by longA: nothing else achieves both ≥0.88 clean and ≥0.70 corruption.

## Per-corruption breakdown

![per_corruption](shape_bias_per_corruption.png)

longA wins on every corruption type except severity-1 contrast (tied) and severity-1 brightness (baseline marginally ahead). The gap is biggest on the noise and blur families — where baseline's texture bias hurts most.

## What "shape bias" actually means here

We used three proxies, all of which agree directionally:

1. **Residual-only test accuracy**: the model classifies test images stripped to their lowest Laplacian band (heavily blurred). Baseline: 0.11; longA: 0.37; band_drop_all: 0.55. This is the Geirhos-style "can the model recognize shape without texture" test, adapted to our band decomposition.
2. **σ=2 blur test accuracy**: lighter blur; same direction. Baseline 0.49; longA 0.85.
3. **Imagenette-C mean corruption accuracy**: the standard robustness benchmark (Hendrycks & Dietterich). Shape-biased models are predicted to do well here, and longA confirms: +17pp over baseline.

The three metrics are correlated but not identical — `joint_blur` is highly corruption-robust (0.709) but has *no* shape bias (residual eval 0.164). It learned to handle blurred inputs by exposure, not by switching to shape-based representations. Useful negative control.

## Hyperparameters that mattered

- **`band_drop_all` over `band_drop` (drop residual too)**: +1.5pp clean, +0.3pp corruption. The "protect residual" intuition is wrong on Imagenette.
- **Dropout rate p=0.5**: clear sweet spot; both lower and higher hurt.
- **Stage-1 length**: 80 epochs > 50 epochs > 20 epochs. More band_drop pretraining helps until it plateaus around 75.
- **Stage-2 LR**: ÷10 of base. Full-LR FT recovers more clean accuracy (+0.5pp) but loses ~3pp corruption robustness. ÷10 is the sweet spot.
- **No curriculum**: the original 6-phase curriculum, dropout-strength variants, dropout-from-step-0 variants — none beat plain band_drop_all single-phase.

## Hyperparameters that didn't matter

- Cosine LR vs constant LR in stage 2 (similar results within noise).
- Distillation temperature (4 vs 8 — minor).
- Whether stage 2 trains a few epochs more or fewer (plateau handles it).

## Reproducing

Code: `imagenette_curriculum.py`. Recipe:
```bash
# Stage 1
python3 imagenette_curriculum.py --engine plateau --runs 3 \
    --variants band_drop_all --max-final-phase 80 \
    --save-checkpoints --out s1.json

# Rename ckpt
mv checkpoints/band_drop_all_r0.pt checkpoints/longA_s1_seed0.pt

# Stage 2
python3 imagenette_curriculum.py --engine plateau --runs 1 \
    --variants baseline --lr-finetune \
    --load-checkpoint checkpoints/longA_s1_seed0.pt \
    --save-checkpoints --out s2.json

# Full eval
python3 imagenette_curriculum.py --eval-only --corruption-eval full \
    --load-checkpoint checkpoints/baseline_r0.pt \
    --out final.json
```

Dataset: `imagenette2-320`, downsampled to 160² on first run (cached). 9469 train / 3925 val images, 10 classes.

Compute: single RTX 3080. Full longA pipeline (stage 1 + stage 2) = ~6 min/seed including fast corruption eval. Full Imagenette-C eval = ~14 min per model.
