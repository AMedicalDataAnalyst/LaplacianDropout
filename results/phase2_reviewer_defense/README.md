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

## File mapping (current — more pending)

| File | Variant | Seeds | Status |
|---|---|---|---|
| `bandmask_hf_only_224.json` | `band_drop_all+JSD --band-mask hf_only` | 1 (seed 0) | ✅ done |
| `bandmask_lf_only_224.json` | `band_drop_all+JSD --band-mask lf_only` | 1 (seed 0) | ✅ done |
| **`bandmask_hf_only_224_3seed.json`** | same, 3 seeds | 3 | ⏳ in progress at repo root, will be moved here |
| **`bandmask_lf_only_224_3seed.json`** | same, 3 seeds | 3 | ⏳ queued after hf_only |

## Single-seed result so far (seed 0)

| Method @ 224 | Clean | σ=2 blur | Residual-only | Corruption (fast) |
|---|---|---|---|---|
| `band_drop_all` (mask=all, 3-seed ref) | 0.888 | 0.882 | 0.702 | **0.806** |
| `band_drop_all` (mask=hf_only, seed 0) | 0.875 | 0.872 | 0.195 | **0.749** |
| `band_drop_all` (mask=lf_only, seed 0) | 0.907 | 0.661 | 0.104 | **0.589** |

**Both targeted variants are strictly worse than the full random method.**
That's the "best case" outcome from `docs/REVIEWER_DEFENSE.md`:

1. `hf_only`: −5.7 pp corruption vs `all` → operator-family-aligned
   dropping does *less* well, not more. Refutes the memorization theory.
2. `lf_only`: −21.7 pp corruption vs `all` → much worse, because the
   residual being droppable is the load-bearing piece (model never learns
   to recover from missing structure if residual is always there).

Once the 3-seed runs land, mean ± std will replace the single-seed
numbers above and these get formally cited in the paper's "Relationship
to test-time corruptions" section.

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
