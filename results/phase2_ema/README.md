# Phase 2 — Weight-EMA comparison (does simple EMA beat band-dropout?)

**Question:** does a simple weight-EMA baseline (Polyak averaging of model
weights during training) match or beat the `band_drop_all` augmentation for
corruption robustness? Tested in two arms: (1) EMA under the established
cosine-LR recipe, and (2) EMA at its best case — constant LR, where the
EMA shadow provides *all* of the end-of-training annealing.

**Answer (2026-08-21): EMA is not a substitute for band-dropout in either
arm, but at constant LR it is a real (small) complement.**

1. **Not a substitute.** Even at its best case (constant LR, decay 0.999),
   baseline+EMA reaches corruption acc **0.599 ±0.008** vs `band_drop_all`'s
   0.807 — EMA closes ~5 of the 26 pp gap. EMA also never moves the
   residual-band probe (~0.15 vs ~0.70), i.e. it does not create the
   frequency-robust features band-dropout trains.
2. **Under cosine LR, EMA is neutral everywhere** (all end-of-training
   deltas within ~1σ of seed noise). Per-epoch histories show EMA 0.99
   leading raw val acc by up to ~7 pp mid-training, fully absorbed by the
   cosine tail — the textbook "EMA substitutes for LR decay, doesn't add
   to it" interaction.
3. **Under constant LR, EMA-as-annealer adds real robustness** on every
   method: baseline +5.1 pp corr (0.548 → 0.599), augmix +2.8 pp
   (0.701 → 0.729), and on the headline recipe it produces the **best
   corruption number recorded on this benchmark**:
   `band_drop_all+augmix (const-LR) [ema 0.999]` = **0.8376 ±0.0037** vs
   the cosine headline's 0.8296 ±0.0021 (+0.8 pp, ~2σ), at the cost of
   −0.4 pp clean (0.895 vs 0.899) and a lower residual probe (0.66 vs
   0.70). A complement worth noting, not a strict Pareto win.

## What was run

Two launchers → `compare_methods.py`, both at the reference 224² protocol
(3 seeds, `--seed-offset 0`, JSD on, batch 96, fast corruption eval:
14 corruptions × sev 3 × 1000-image deterministic subsample), EMA decays
0.99 and 0.999 tracked passively in every run:

- `run_ema_comparison.sh` — cosine LR (the established recipe):
  baseline, augmix, band_drop_all+augmix, band_drop_all.
- `run_ema_constlr.sh` — `--schedule constant` (2-epoch warmup then flat
  3e-3; plateau early-stop can never fire, so all runs use the full
  50-epoch budget): baseline, augmix, band_drop_all+augmix.

EMA implementation (`ModelEMA` in `compare_methods.py`): shadow deep-copied
from the live weights at run start (no zero-init bias), updated after every
optimizer step with a **fixed** decay; float buffers (BN running stats)
averaged, integer buffers copied. The shadow is passive — it consumes no
RNG and never influences the LR schedule or early stopping, so the raw rows
of each cosine run double as same-seed controls (they reproduce the Phase-1
reference table within seed noise). Slower decays than 0.999 were excluded:
at ~4900 steps the random init would retain ≥ 8 % weight in the average.

## Key numbers (3-seed means ± std, fast corruption eval)

| Method (224²) | Clean | Corruption | Residual |
|---|---|---|---|
| baseline [raw] | 0.8949 ±0.0046 | 0.5479 ±0.0139 | 0.147 |
| baseline [ema 0.999] | 0.8931 ±0.0027 | 0.5463 ±0.0111 | 0.152 |
| baseline const-LR [raw] | 0.8363 ±0.0209 | 0.5224 ±0.0443 | 0.136 |
| baseline const-LR [ema 0.999] | 0.8979 ±0.0015 | 0.5990 ±0.0083 | 0.140 |
| augmix [raw] | 0.9082 ±0.0004 | 0.7008 ±0.0069 | 0.159 |
| augmix [ema 0.999] | 0.9075 ±0.0043 | 0.6972 ±0.0064 | 0.154 |
| augmix const-LR [raw] | 0.8589 ±0.0237 | 0.6583 ±0.0304 | 0.158 |
| augmix const-LR [ema 0.999] | 0.9089 ±0.0009 | 0.7287 ±0.0094 | 0.148 |
| band_drop_all [raw] | 0.8893 ±0.0050 | 0.8066 ±0.0032 | 0.703 |
| band_drop_all [ema 0.999] | 0.8897 ±0.0013 | 0.8059 ±0.0015 | 0.692 |
| band_drop_all+augmix [raw] | 0.8991 ±0.0018 | 0.8296 ±0.0021 | 0.701 |
| band_drop_all+augmix [ema 0.999] | 0.8986 ±0.0015 | 0.8284 ±0.0023 | 0.688 |
| band_drop_all+augmix const-LR [raw] | 0.8659 ±0.0040 | 0.7942 ±0.0039 | 0.538 |
| band_drop_all+augmix const-LR [ema 0.99] | 0.8941 ±0.0012 | 0.8301 ±0.0060 | 0.653 |
| **band_drop_all+augmix const-LR [ema 0.999]** | 0.8951 ±0.0027 | **0.8376 ±0.0037** | 0.664 |

(ema 0.99 rows omitted where they tell the same story; full table via
`python ema_aggregate.py`.)

## Interpretation

- The user-hypothesis ("simple EMA may beat the augmentation") is refuted:
  EMA is an *optimization* smoother, not a robustness mechanism — its gains
  appear exactly where the schedule leaves annealing on the table, and cap
  out ~20 pp short of band-dropout's corruption acc.
- The constant-LR+EMA headline number (0.8376) is a candidate recipe tweak
  for the ImageNet phase: EMA is standard practice at that scale anyway,
  and the const-LR arm suggests band-dropout benefits mildly from
  averaging-based annealing. Worth a control cell in the ImageNet sweep,
  not a change to the headline claim (the gain is +0.8 pp corr / −0.4 pp
  clean — complementary, non-Pareto).
- Caveats: fast corruption eval (sev 3, 1000-image subsample); one LR
  (3e-3) for the constant arm — no LR retune; decay grid only {0.99,
  0.999} (slower decays need a longer run or a debiased/warmup EMA).

## Files

- Cosine arm: `ema_baseline.json`, `ema_augmix.json`,
  `ema_band_drop_all.json`, `ema_band_drop_all_augmix.json`
- Constant-LR arm: `ema_constlr_baseline.json`, `ema_constlr_augmix.json`,
  `ema_constlr_band_drop_all_augmix.json`

Standard run-record schema plus:
- `schedule`: "cosine" | "constant"
- `ema_decays`: [0.99, 0.999]
- `ema.<decay>.{val_acc, val_acc_residual_at_test, val_acc_blur_at_test,
  corruption}` — all evaluated on the EMA shadow weights
- `history[*].val_acc_ema.<decay>` — per-epoch EMA val accuracy

## How to consume

```bash
python ema_aggregate.py          # full table incl. Phase-1 reference rows
```

## Operational notes (2026-08-20/21)

- Run on the RTX 5070 Ti (16 GB) with a rebuilt env (`.venv`, torch
  2.13.0+cu130); `imagecorruptions` needs the NumPy-2 patch
  (`np.float_` → `np.float64` in site-packages/imagecorruptions/corruptions.py).
- One CUDA OOM from allocator fragmentation when a single process trains
  many runs back-to-back at 224² (the ~3.5 GB fp16 train cache is
  re-allocated per run). Fixed in both launchers with
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; launchers are
  resumable (skip methods whose JSON exists) and detect failures via
  pipefail.
- Wall-clock: ~10 min per JSD run, ~4 min per baseline run on this GPU;
  each arm took ≈ 1.5–2 h.
