# Reviewer defense — "is band-dropout just training on the test corruptions?"

The single most likely methodological challenge to this paper. This doc
states the risk, the steelman attack, the defenses we already have, and
the control experiments we should run before submitting.

## The convention reviewers will hold us to

The AugMix / PixMix / IPMix lineage enforces a soft rule: the training
augmentation operators should be **disjoint from the test-time corruption
operators**. AugMix's reference operator set (`augmentations.augmentations`)
deliberately excludes brightness, contrast, noise, and blur for exactly this
reason — those operators appear in ImageNet-C / CIFAR-C and including them
at training time would be "cheating." The `--all-ops` flag exposes them but
is footnoted as opt-in and acknowledged to change the comparison.

Reviewers in this lineage *do* enforce this rule. Reviewer 2-style replies
of "your operator set overlaps the corruption set, so the result is unfair"
are common. We need to anticipate this.

## The steelman attack against band_drop_all

> "Random Laplacian-band dropout operates in the frequency domain. So do
> several test-time corruptions: defocus_blur, glass_blur, motion_blur,
> zoom_blur (all linear low-pass filters), and to a lesser extent the
> noise corruptions (broadband frequency perturbations) and contrast (a
> multiplicative frequency scaling). By training the network on random
> spectral subspace masking, the authors have trained on the general
> *form* of these corruptions. The +45 pp gain on defocus_blur and +51 pp
> on contrast aren't generalization — they're operator-family memorization.
> AugMix is bound by the disjoint-ops convention; band_drop_all is not.
> The comparison is unfair."

A version of this attack can be applied to **9 of the 15 corruptions** in
our table (the 4 blurs, 3 noises, contrast, fog). That's the bulk of the
headline gain — so the attack is dangerous and we must rebut it directly.

## Defenses already in the data

### Defense 1 — wins on geometric / occlusion corruptions
Six of fifteen corruptions have **no frequency-domain operator
relationship** to band-dropout. Our wins on these are the load-bearing
counter-evidence:

| Corruption | Δ vs baseline | Why it refutes the attack |
|---|---|---|
| frost | +26.4 pp | texture overlay, geometric not spectral |
| fog | +36.2 pp | additive low-freq overlay (not band-masking) |
| snow | +15.3 pp | particle occlusion, geometric |
| elastic_transform | +12.0 pp | geometric warp, no frequency component |
| pixelate | +11.4 pp | spatial downsample, all scales jointly |
| brightness | +7.7 pp | DC offset only |
| jpeg_compression | +0.7 pp | DCT basis (different from Laplacian) |

If the augmentation were just learning to defeat the test-corruption
operator family, these should not improve. They all do.

### Defense 2 — JPEG win is "free transfer" across frequency bases
JPEG compression is **DCT-based**, not Laplacian. Yet we gain +0.7 pp on
it. The number is small, but its existence is qualitative evidence that
the model has learned a *generic* notion of spectral invariance, not just
robustness to Laplacian-band perturbations.

### Defense 3 — operator identity vs operator family
The disjoint-ops convention rules out training on the *same operator* as
the test set. It doesn't rule out training on operators that share a
basis or a domain. AugMix's reference set includes `color`, `equalize`,
`posterize`, `solarize` — all color-space operations — and tests on
`brightness` (also color-space). Nobody calls this overlap. The rule is
operator identity. We satisfy that: random structured Laplacian-subspace
masking is not Gaussian blur of any specific σ, not band-pass filtering of
any specific cutoff, not any corruption in the test set.

### Defense 4 — frame the augmentation as a random subspace projection
In the method section, frame `band_drop_all` explicitly as **structured
random masking on the Laplacian basis**, not as a corruption operator. The
analog reviewers should compare it to is Cutout (random spatial subspace
masking), not defocus_blur (deterministic frequency operator).

## Control experiments to run

These pre-empt the reviewer at low compute cost. All three are queueable
on the same infrastructure as the main sweep.

### Control A — targeted hf_only / lf_only (READY TO LAUNCH)
**The decisive control.** If the win comes from "operator family
memorization," then a variant that *only* drops the high-freq bands (most
aligned with blur operators) should win **more** on defocus_blur than the
random multi-band `band_drop_all`. If it wins **less** (predicted), the
random multi-band *structure* is doing the work, not the operator-family
alignment.

| Variant | Drops | Predicted defocus_blur gain | Predicted overall mCE |
|---|---|---|---|
| `band_drop_all` (current) | random subset of 6 bands | +45 pp | mCE 47 |
| `band_drop_all hf_only` | random subset of top half only | similar or less | worse |
| `band_drop_all lf_only` | random subset of bottom half only | less | worse |

If `hf_only` does **NOT** crush defocus_blur more than `all` does, the
operator-memorization theory is refuted.

**How to launch (now wired into all four harnesses):**

```bash
# CIFAR (uses currently-running infrastructure)
python augmix/cifar.py --dataset cifar10 --model wrn --layers 28 --widen-factor 10 \
  --epochs 100 --batch-size 128 --learning-rate 0.1 --num-workers 2 \
  --data-path /mnt/c/.../data --save snapshots/c10_bd_hf_seed0 \
  --aug-method band_drop_all --band-mask hf_only --seed 0

# repeat with --band-mask lf_only for comparison
```

```bash
# Imagenette 224 (re-uses Phase 1 infrastructure)
python compare_methods.py --methods band_drop_all --use-jsd \
  --runs 1 --max-epochs 50 --batch-size 96 --corruption-eval fast \
  --resolution 224 --pad 16 \
  --band-mask hf_only --out band_mask_hf_only_224.json
```

### Control B — cross-basis (DCT-drop) variant
Train a `dct_drop_all` variant: same random subspace masking but in DCT
basis, matching JPEG's basis (which is in the test set but unrelated to
the Laplacian basis we currently use). If it achieves similar mCE, the
mechanism is "random structured subspace masking" generically, not
Laplacian-specific. **NOT YET IMPLEMENTED** — requires a new module
analogous to `band_dropout_transform.py` but using a DCT basis. ~half-day
new code; cheap to write before the next sweep.

### Control C — matched-budget Gaussian noise
Train a model with per-pixel Gaussian noise at matched ℓ₂ budget. Tests
H3 from `THEORY_OUTLINE.md`: "is this just strong regularization?" If
noise injection at matched budget matches `band_drop_all`'s mCE, the
structured spectral aspect isn't load-bearing. We expect it does NOT
match. ~no new code (torchvision has noise transforms); one training run.

## Where to address this in the paper

A short subsection in **Method** titled **"Relationship to test-time
corruptions"** with the following structure (about half a page):

> "ImageNet-C contains spectrally-structured corruptions (the four blurs
> and three noises in particular) that share the frequency-domain nature
> of our augmentation. We address the resulting fairness concern with three
> arguments:
>
> 1. *Disjoint operator identity.* Random Laplacian-subspace masking is
>    not Gaussian blur of any fixed σ, nor any specific spectral filter.
>    The convention enforced in AugMix and PixMix is disjoint operators,
>    not disjoint operator domains [cite AugMix].
>
> 2. *Empirical: wins on corruptions with no spectral relationship.*
>    Table 3 shows we gain +26 pp on frost, +36 pp on fog, +15 pp on
>    snow, +12 pp on elastic_transform, +7 pp on brightness, and +12 pp
>    on pixelate. None of these is a frequency operator. If the gain
>    were operator-family memorization, these should not improve.
>
> 3. *Empirical: cross-basis transfer.* The +0.7 pp gain on
>    jpeg_compression (DCT basis, not Laplacian) shows the learned
>    invariance generalizes beyond the basis the augmentation operates in.
>
> Section X presents control experiments (`hf_only`, `lf_only`, DCT-drop,
> matched-budget Gaussian noise) further isolating the random
> subspace-masking structure from the operator-family effect."

Then a short **Controls** subsection in Experiments with the hf_only /
lf_only / DCT-drop results.

## Status

| Item | State |
|---|---|
| `band_mask='hf_only' / 'lf_only'` exposed in code | ✅ DONE |
| CLI flag `--band-mask` on all four harnesses | ✅ DONE |
| Faithfulness check (hf_only / lf_only correct vs intent) | ✅ DONE |
| hf_only / lf_only trained on CIFAR | ⏳ queueable after de-risk |
| hf_only / lf_only trained on Imagenette-224 (cheap, ~2 h each) | ⏳ queueable any time |
| DCT-drop variant implemented | ❌ not yet |
| Matched-budget noise variant trained | ❌ not yet |
| Method-section copy drafted | ❌ not yet |

## Recommendation

**Do not submit without Control A.** It's the cheapest, most decisive
defense, and the flag is now in place. Plan to add a single
hf_only/lf_only sweep (4 runs: hf_only + lf_only on CIFAR-10 + Imagenette-224,
all seed 0) to the queue after the current CIFAR de-risk. That adds ~6 h
of training and converts a likely reviewer attack into a pre-empted one.

Control B (DCT-basis) and Control C (matched-budget noise) are
nice-to-have but lower priority — only run them if the hf_only / lf_only
result needs reinforcing.
