# Tier-2 paper outline

A short structural skeleton with current numbers slotted in. Replace
placeholders [BLOCKED-ON-DATA: …] as the ImageNet-1k runs land.

## Working title

"Random Laplacian-band masking: a one-line augmentation that complements
AugMix and improves spectral robustness on ImageNet."

## Pitch (one paragraph)

We propose `band_drop_all`: per training image, decompose into 6 Laplacian
bands and independently drop each with probability 0.5. Applied as a per-sample
augmentation with no curriculum, it produces models that are more robust to
ImageNet-C corruptions — particularly to spectrally-narrow corruptions like
blur, fog, and contrast — and combines with AugMix's JSD consistency loss
for a strict Pareto improvement on clean and corrupted accuracy. On
Imagenette-160/224 with ResNet-18, the combination reduces mean corruption
error (mCE, relative to the unaugmented baseline) to **47.2 over the canonical 15
corruptions at 224²** — a 52.8% relative error reduction — while improving
clean accuracy by 0.7 pp.

## Section structure

1. **Introduction** — robustness gap; existing aug families; the open
   question of whether a *spectrally structured* aug helps.

2. **Method** — band_drop_all in pseudocode (3 lines), one figure showing
   the 6 bands of an example image and 4 random-mask realizations. Note
   the composition with AugMix (band-drop on the augmented view) and the
   JSD term.

3. **Experiments — Imagenette (proof of concept)**
   - Tier-1 result table (160² and 224²). [DONE — see TIER1_FINDINGS.md]
   - 224² canonical 15-corruption mCE: 47.2, strict Pareto win. [DONE]
   - Band-count ablation [DONE]: corruption robustness saturates at L=5
     (6 bands); the σ=16/32 bandpass terms add < 1 pp; the Pareto knee is
     at L=3 (4 bands) with no clean-acc cost.

4. **Experiments — ImageNet-1k (the main result)**
   - 4-method × 3-arch table (ResNet-50, ConvNeXt-T, ViT-B). [BLOCKED]
   - Canonical AlexNet-normalized mCE comparing to AugMix 65.3,
     PixMix ~56, IPMix 63, DeepAugment 60.4, PRIME 55.5. [BLOCKED]
   - Faithful PixMix-with-fractals comparison + our combo. [BLOCKED]

5. **Robustness battery (Phase-2 evals on trained 1k models)**
   - OOD: ImageNet-R / ImageNet-A / ImageNet-Sketch. [BLOCKED]
   - Calibration: RMS-CE on clean + corrupted. [BLOCKED]
   - Adversarial: PGD@ε=4/255. [BLOCKED]
   - Shape-bias: Geirhos cue-conflict %. [BLOCKED]

6. **Mechanism / analysis**
   - Per-corruption breakdown (spectral selectivity of wins). [DONE @ 224²]
   - Band-count ablation conclusion. [PENDING]
   - Gradient spectrum / shape-bias / band-targeted PGD. [PENDING — see
     THEORY_OUTLINE.md for the four-figure story.]

7. **Related work** — AugMix, PixMix, IPMix, PRIME, DeepAugment;
   frequency-domain augmentation lit (FreqDrop, AmplitudeMix, etc.);
   shape-vs-texture lit (Geirhos, stylized ImageNet).

8. **Limitations** — single-image domain; doesn't address dataset shift
   beyond what's captured by the corruption/OOD suite; the JSD term roughly
   3× the per-step compute (true of AugMix too).

## Venue targeting

- **Workshop strong**: ICLR/NeurIPS robustness workshop with ImageNet-100 +
  ResNet-50 + Imagenette canonical mCE = ready to submit.
- **Full venue**: needs ImageNet-1k + 2 architectures + Geirhos + calibration
  + one of (adversarial or OOD) + theory section. All scaffolding now in
  place; gated on compute (~1-2 weeks of multi-GPU training).

## Figure inventory

1. Method figure (band decomposition + dropped variants — 1 col).
2. Pareto scatter at ImageNet-1k (clean vs mCE), our combo dominates.
3. Per-corruption bar chart (Δ vs baseline, our combo).
4. Band-count ablation (mCE vs #bands).
5. Robustness battery table (clean / mCE / R / A / Sketch / ECE / PGD / shape-bias).
6. Mechanism analysis figure (gradient spectra OR band-targeted PGD heatmap).
