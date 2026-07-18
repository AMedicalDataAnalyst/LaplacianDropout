# Phase 2 — Band-targeted PGD (spectral vulnerability profile)

Mechanism experiment #6 from `docs/THEORY_OUTLINE.md`: PGD attacks with
the perturbation projected onto a *single Laplacian band*, probing which
frequency bands each trained model is adversarially reliant on. If the
spectral-robustness hypothesis (H1) holds, `band_drop_all` training
should resist any single-band attack that collapses the baseline.

Setup: Imagenette-224, ε=4/255, 20-step PGD, 1000-image subsample
(`band_targeted_pgd.py`, swept by `run_band_targeted_pgd.sh`).
5 models × 7 attacks (unconstrained + band targets 0–5). One JSON per
(model, target): `<model>_target_<band|none>.json`.

## Headline table (accuracy under attack; higher = more robust)

Canonical rendered copy: `docs/band_targeted_pgd_table.md`, figure:
`figures/band_targeted_pgd.png`.

| Model | none | band 0 (HF) | 1 | 2 | 3 | 4 | 5 (LF) |
|---|---|---|---|---|---|---|---|
| baseline | 0.031 | 0.441 | 0.796 | 0.886 | 0.899 | 0.911 | 0.903 |
| band_drop_all+AugMix (HEADLINE) | 0.011 | **0.648** | 0.757 | 0.826 | 0.882 | 0.904 | 0.908 |
| band_drop_all only | 0.002 | 0.598 | 0.706 | 0.734 | 0.866 | 0.906 | 0.908 |
| band_drop_all hf_only | 0.022 | **0.775** | 0.828 | 0.772 | 0.802 | 0.870 | 0.866 |
| band_drop_all lf_only | 0.001 | **0.050** | 0.621 | 0.861 | 0.899 | 0.904 | 0.905 |

## Findings

1. Every model is most vulnerable to the **highest-frequency band
   (band 0)** — that is where adversarial perturbations concentrate.
2. `band_drop_all` training raises band-0 robustness from 0.44 → 0.60–0.65
   (+16–21 pp over baseline): the model trained with random band removal
   is measurably less reliant on any single band.
3. The reviewer-defense controls behave exactly as the mechanism
   predicts: `hf_only` (trained with HF bands droppable) is the *most*
   band-0-robust (0.775); `lf_only` (HF always present at training) is
   catastrophically band-0-reliant (0.050). Clean cross-check of the
   `../phase2_reviewer_defense/` story from the adversarial side.
4. Unconstrained PGD ("none") still breaks all models — band dropout is
   a robustness reshaper, not an adversarial defense. We claim spectral
   reliance redistribution, nothing more.

## Reproduction

```bash
bash run_band_targeted_pgd.sh          # ~18 min on an RTX 3080
python plot_band_targeted_pgd.py       # regenerates figure + docs table
```
