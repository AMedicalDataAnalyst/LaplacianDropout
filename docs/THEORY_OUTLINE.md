# Why does random Laplacian-band dropout help corruption robustness?

**Status: outline + experiment shortlist.** This is the analysis section that
turns a one-line augmentation into a paper. The point is not to "prove the
right mechanism" but to characterize *what* the augmentation induces in
trained models, ideally in a way that distinguishes it from existing
explanations (data aug regularization, noise injection, shape bias).

## Candidate mechanisms

### H1 — Spectral robustness via random subspace masking
The Laplacian pyramid is a near-orthonormal multiresolution decomposition.
Independently zeroing bands per sample = projecting the input onto a random
subset of the 6 frequency subspaces, then training the model to classify
across all 2^6 = 64 such projections. The model is forced to be invariant
to band-presence — equivalently, robust to any perturbation whose spectrum
is concentrated in a single band.

ImageNet-C corruptions are spectrally structured (defocus_blur attacks
high-freq bands; fog/contrast attacks the low-freq residual; noise spans
broadband). A model that classifies correctly with any random subset of
bands removed should — by construction — be robust to corruption types that
selectively suppress or distort one band.

**Predicts:** wins should be largest where the corruption has the most
spectral selectivity. Smallest where corruption is broadband (jpeg). Per-corruption
results: contrast +51, defocus_blur +45, fog +36 (narrowband-spectrum
corruptions, big wins); jpeg +0.7 (broadband, no gain). **Consistent.**

### H2 — Shape bias (texture suppression)
Dropping high-freq bands removes texture; the residual / low-freq bands
carry shape. Training with frequent texture-suppression nudges the model
toward shape features (Geirhos et al. framing).

**Predicts:** residual-only-eval accuracy should be much higher than baseline
(measures how much the model can do from low-freq alone). At 224: residual
acc = 0.697 (vs baseline 0.134). **Strongly consistent.**

### H3 — Generic data augmentation / regularization
Any aug that diversifies inputs reduces overfit. Band-dropout might just be a
particularly effective aug whose specific structure isn't load-bearing —
random Gaussian noise of similar magnitude could match.

**Test:** compare band_drop_all vs (i) per-pixel Gaussian noise at matched ℓ2
budget, (ii) random low-pass/high-pass filter, (iii) cutout in frequency
space (block out a random rectangle in the FFT). If the structured Laplacian
basis matters, (i)-(iii) should underperform band_drop_all.

### H4 — JSD consistency × spectral diversity
The JSD loss enforces stable predictions across the three input views. With
band-dropout, those views span a much wider distribution than AugMix's PIL
ops, so JSD enforces a stronger functional smoothness.

**Test:** band_drop_all WITHOUT JSD vs WITH JSD vs AugMix WITH JSD.
Decomposes the effect of view diversity from the JSD itself.

## Concrete experiments (cheap, in order)

1. **Band-count ablation** — sweep levels ∈ {1..6}. Tells us how many bands
   are needed for the benefit; thin band counts test whether *any*
   structured-subspace dropout helps or whether spectral granularity matters.
   *(In progress now.)*

2. **Per-band-only ablation** — train with each band dropped individually
   (not random). Identifies which bands contribute most to the robustness
   gain.

3. **Frequency-spectrum of saliency** — for trained baseline vs trained
   band_drop_all model, compute the average power spectrum of input
   gradients (∂L/∂x) over the val set. Predicts (H1): band_drop_all
   gradients should have flatter spectrum (broader frequency reliance).

4. **Match-budget noise control** — Gaussian noise injection at the ℓ2
   budget of an average band-dropped image. If H3 dominates, this matches
   band_drop_all; if H1 dominates, it doesn't.

5. **Shape-bias % (Geirhos cue-conflict)** — direct measurement of H2.
   Predict band_drop_all model has higher shape-bias than baseline/AugMix.

6. **Spectrally-targeted adversarial attacks** — PGD with the perturbation
   constrained to a single Laplacian band. If H1 holds, band_drop_all should
   be robust to any single-band attack while baseline collapses.

## What the writeup looks like

A "Mechanism" section presenting (1)+(5)+(3)+(6) as the four-figure story:
robustness scales with band count, model is shape-biased, the gradient
spectrum is flat, and the model resists band-targeted adversarial attacks.
That's a concrete, testable mechanistic story, not just "this aug helps."
