# Phase 1.5 — Ablations

Targeted ablations after the headline `band_drop_all + AugMix + JSD`
recipe was established. Both ablations were single-seed at 224
(corroboration of the 3-seed reference in `../phase1_comparison/`).

## What was tested

### 1. Band-count ablation — how many bands do we actually need?

Files: `band_count_ablation_L{1,2,3,4,5,6}_bands{2,3,4,5,6,7}.json`

For each Laplacian-level count L ∈ {1..6} (= 2..7 bands), train
`band_drop_all + JSD` on Imagenette-224 for one seed and measure clean +
fast-corruption + residual + blur acc.

| L | Bands (σ + residual) | Clean | Residual-only | σ=2 blur | Corruption |
|---|---|---|---|---|---|
| 1 | 2 (σ=1, res) | 0.905 | 0.902 | 0.827 | 0.656 |
| 2 | 3 (σ=1,2, res) | 0.905 | 0.894 | 0.897 | 0.719 |
| 3 | 4 (σ=1,2,4, res) | 0.894 | 0.869 | 0.888 | 0.784 |
| 4 | 5 (σ=1,2,4,8, res) | 0.894 | 0.833 | 0.885 | 0.792 |
| **5** | **6 (default; σ=1,2,4,8,16, res)** | 0.886 | 0.708 | 0.880 | **0.802** |
| 6 | 7 (σ=1..32, res) | 0.884 | 0.498 | 0.876 | 0.804 |

**Saturates at 5–6 bands.** The σ=32 band at L=6 is physically
meaningless at 224² and adds nothing. Most of the corruption gain comes
from the σ=2,4,8 mid-frequency bands.

The residual-only drop is a test-difficulty artifact: with more levels,
the residual is more aggressively low-passed, so the test is harder by
construction.

### 2. min_kept=1 — does guaranteeing at least one band kept help?

File: `band_keep1_224.json`

The default `band_drop_all` allows all bands to be dropped simultaneously
(~1.6 % of samples at p=0.5, 6 bands → all-zero training image). Tested
whether enforcing `min_kept=1` (rejection-resample any all-zero draw)
improves things.

| Variant @ 224 | Clean | Corr |
|---|---|---|
| `band_drop_all` (min_kept=0, 3-seed ref) | 0.888 | 0.806 |
| `band_drop_all` (min_kept=0, 1-seed) | 0.886 | 0.802 |
| **`band_drop_all_keep1`** (min_kept=1, 1-seed) | **0.881** | **0.798** |

**No improvement — slightly worse.** Removing the all-zero samples costs
~0.5–0.8 pp on both clean and corruption. Interpretation: the all-zero
training cases act as implicit label-smoothing (model forced to predict
prior on blank input). The flag exists in the code for re-use but
defaults off.

## How these were produced

```bash
# Band-count ablation (one entry per L)
for L in 1 2 3 4 5 6; do
  python compare_methods.py --methods band_drop_all --runs 1 \
    --use-jsd --max-epochs 50 --batch-size 96 --corruption-eval fast \
    --resolution 224 --pad 16 --band-levels $L \
    --out results/phase1_ablations/band_count_ablation_L${L}_bands$((L+1)).json
done

# min_kept=1 single run
python compare_methods.py --methods band_drop_all_keep1 --runs 1 \
  --use-jsd --max-epochs 50 --batch-size 96 --corruption-eval fast \
  --resolution 224 --pad 16 \
  --out results/phase1_ablations/band_keep1_224.json
```

## How to re-read

Each band_count file is a single-record JSON array with the
`compare_methods.py` schema. The `band_levels` and `resolution` fields
are populated. The fast corruption eval is sev 3 only, 1000-image
subsample.
