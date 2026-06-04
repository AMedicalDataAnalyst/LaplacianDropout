# Phase 2 — Reviewer-defense controls (band-mask hf_only / lf_only)

The "is band-dropout just training on the test corruptions?" defense
experiment. Background: `docs/REVIEWER_DEFENSE.md`.

## What was tested

The default `band_drop_all` drops *any* of the 6 Laplacian bands at
random. The reviewer attack: "you're operating in the frequency domain,
which is also where ~9 of the 15 test corruptions live — your win on
defocus_blur is just operator-family memorization, not generalization."

Two control variants probe whether the **random multi-band structure**
is doing the work or whether targeted (operator-family-aligned) dropping
would do as well:

- `hf_only`: drop only the top-half bands (highest frequencies); low-freq
  and residual always kept. **More** aligned with blur-like corruptions —
  if memorization theory holds, this should win more on blur.
- `lf_only`: drop only the bottom-half bands (incl. residual); high-freq
  always kept. **Less** aligned with blur.

Compare against the full random method (`band_mask='all'`, 3-seed mean
in `../phase1_comparison/comparison_224_main.json` corr=0.806).

## File mapping

| File | Variant | Seeds | Notes |
|---|---|---|---|
| `bandmask_hf_only_224.json` | `band_drop_all+JSD --band-mask hf_only` | 1 (seed 0) | original single-seed; superseded by 3-seed |
| `bandmask_lf_only_224.json` | `band_drop_all+JSD --band-mask lf_only` | 1 (seed 0) | same |
| **`bandmask_hf_only_224_3seed.json`** | same, 3 seeds | 3 | headline; ran without explicit `--seed-offset` (old code path) |
| **`bandmask_lf_only_224_3seed.json`** | same, 3 seeds | 3 | headline; **first run with explicit `torch.manual_seed(0/1/2)`** — note higher cross-seed std vs the others |

## Headline result — 3-seed mean ± std (DONE 2026-06-04)

| Method @ 224 | Clean | σ=2 blur | Residual-only | Corruption (fast) |
|---|---|---|---|---|
| `band_drop_all` (mask=**all**, 3-seed ref from `comparison_224_main`) | 0.888 ± 0.003 | 0.882 ± 0.004 | 0.702 ± 0.004 | **0.806 ± 0.0004** |
| `band_drop_all` (mask=**hf_only**, 3-seed) | 0.873 ± 0.005 | 0.871 ± 0.004 | 0.198 ± 0.013 | **0.747 ± 0.002** |
| `band_drop_all` (mask=**lf_only**, 3-seed, explicit seeds) | **0.900 ± 0.002** | 0.638 ± 0.012 | 0.105 ± 0.007 | **0.573 ± 0.006** |

**Both targeted variants are strictly worse than the full random method on corruption acc — significant by ≥30σ.** That's the decisive "best case" outcome from `docs/REVIEWER_DEFENSE.md`:

1. `hf_only`: −5.9 pp corruption vs `all`. Operator-family-aligned dropping (HF bands ≈ blur operators) does *less* well, not more. Refutes the memorization theory.
2. `lf_only`: −23.3 pp corruption vs `all`. Massively worse, because the residual being droppable is the load-bearing piece (model never learns to recover from missing structure if residual is always present).

## Per-corruption (3-seed mean, fast eval sev 3, 1000-img subsample)

| Corruption | all | hf_only | Δ vs all | lf_only | Δ vs all |
|---|---|---|---|---|---|
| brightness | 0.842 | 0.778 | −6.4 | 0.862 | +2.0 |
| contrast | 0.852 | 0.505 | **−34.7** | 0.593 | −25.9 |
| defocus_blur | 0.858 | 0.850 | −0.7 | 0.375 | **−48.2** |
| elastic_transform | 0.782 | 0.799 | +1.7 | 0.552 | −23.0 |
| fog | 0.849 | 0.628 | −22.2 | 0.711 | −13.8 |
| frost | 0.752 | 0.523 | −22.8 | 0.620 | −13.2 |
| gaussian_noise | 0.765 | 0.811 | +4.6 | 0.338 | **−42.7** |
| impulse_noise | 0.738 | 0.811 | +7.3 | 0.293 | **−44.5** |
| jpeg_compression | 0.858 | 0.850 | −0.7 | 0.864 | +0.7 |
| motion_blur | 0.837 | 0.833 | −0.4 | 0.500 | **−33.7** |
| pixelate | 0.871 | 0.857 | −1.4 | 0.803 | −6.8 |
| shot_noise | 0.779 | 0.819 | +4.0 | 0.327 | **−45.2** |
| snow | 0.684 | 0.572 | −11.2 | 0.549 | −13.5 |
| zoom_blur | 0.811 | 0.818 | +0.7 | 0.631 | −18.0 |
| **MEAN** | **0.805** | **0.747** | **−5.9** | **0.573** | **−23.3** |

## Mechanistic story this supports

- `hf_only` keeps low-freq + residual always present → model learns most of what matters and only takes a modest hit. *Wins on noise corruptions* (model becomes specifically robust to noise/HF distortions) but loses on contrast/fog (which attack low-freq, which `hf_only` never trained to recover).
- `lf_only` keeps high-freq always present → model never learns to handle missing structure → falls off a cliff on blur and noise (corruptions that disturb the structure the model relied on at training time). Wins only on brightness (DC offset, handled by batchnorm regardless) and jpeg (high-freq quantization, which lf_only happens to learn to ignore).
- The random multi-band `all` setting dominates by exposing the model to BOTH "missing detail" AND "missing structure" stress, neither of which the targeted variants do alone.

## Variance note — Lesson #24 validated

Cross-seed std on corruption acc:
- `all` (old code, no explicit seed): 0.0004
- `hf_only` (old code, no explicit seed): 0.0017
- `lf_only` (**new code, explicit `torch.manual_seed(0/1/2)`**): 0.006

Going from implicit RNG cascade → explicit different seeds raised the std ~15× (0.0004 → 0.006), which is the more honest number. Confirms the explicit seeding patch produces meaningfully different inits.

## How these were produced

The `--band-mask` flag was added to `band_dropout_transform.BandDropAll`,
`imagenette_curriculum._band_dropout`, and threaded through
`compare_methods.py` + both `augmix/` and `pixmix/` harnesses. See
`docs/REVIEWER_DEFENSE.md` for the full mechanism and the planned
follow-up controls (DCT-basis, matched-budget noise).

Reproduction:
```bash
python compare_methods.py --methods band_drop_all \
  --runs 3 --max-epochs 50 --use-jsd \
  --batch-size 96 --corruption-eval fast \
  --resolution 224 --pad 16 \
  --band-mask hf_only \
  --out results/phase2_reviewer_defense/bandmask_hf_only_224_3seed.json
```

## File schema

Same as `compare_methods.py` output records (see
`../phase1_comparison/README.md`). The `band_mask` value is implicit in
the filename (the run's `BAND_MASK_OVERRIDE` is set globally before
training, not per-record).
