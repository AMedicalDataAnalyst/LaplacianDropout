# Claude Code instructions — Laplacian curriculum integration experiment

## Goal
Run a comparison of five training variants on CIFAR-10 to assess whether a
Laplacian frequency curriculum induces a measurable shift in what the network
relies on. This is an integration test of the curriculum into the speedrun
pipeline, not a benchmark.

## Environment
- Windows, RTX 3080
- Python with `torch`, `torchvision`, CUDA build matching the driver
- CIFAR-10 raw batches at:
  `C:\Users\JoyToy\Documents\Projects\data\cifar-10-batches-py`

## What's being tested

Five variants, 3 runs each (15 total runs):

| Variant | Purpose |
|---|---|
| `baseline` | Unmodified pipeline, ceiling reference |
| `decompose_identity` | Decompose then sum all bands. **Sanity check** — should match baseline within noise. If it doesn't, the decomposition itself is the confound. |
| `curriculum` | Residual-only → progressive band addition, no dropout |
| `curriculum_dropout` | Same schedule, 50% band dropout in final phase (residual protected) |
| `residual_only` | Train and eval on residual only — accuracy ceiling for shape-only |

Two metrics per run:
- `val_acc` — standard test accuracy
- `val_acc_residual_at_test` — accuracy when test inputs are stripped to residual at inference. The falsifiable prediction is that curriculum variants will show a *smaller gap* between these two than baseline, indicating reliance on lower-frequency features.

## Steps

1. Place `curriculum_experiment.py` in a working directory. It will create
   `cifar10/` next to itself and cache `train.pt` / `test.pt` on first run.

2. Verify CIFAR raw batches exist:
   ```
   dir "C:\Users\JoyToy\Documents\Projects\data\cifar-10-batches-py"
   ```
   Should show `data_batch_1`...`data_batch_5`, `test_batch`, `batches.meta`.

3. Run:
   ```
   python curriculum_experiment.py --runs 3
   ```

   Optional flags:
   - `--variants baseline curriculum` to run a subset
   - `--out my_results.json` to change the output file

4. Expected timeline (rough estimates):
   - Warmup (one-time `torch.compile`): 30–120 seconds
   - Each run: ~10–15 seconds (the original speedrun is ~5s; curriculum
     adds decomposition cost per batch)
   - Total: ~3–5 minutes for the full 15-run sweep

   Note: please double the number of epochs

## What to look for

A summary table prints at the end. Roughly what to expect:
- `baseline` and `decompose_identity` `val_acc` should be within ~0.5pp of each
  other and around 0.92–0.94. **If they diverge, stop and report.**
- `curriculum` and `curriculum_dropout` `val_acc` likely a few points lower
  than baseline — CIFAR doesn't have much spectrum to exploit.
- `residual_only` `val_acc` significantly lower, probably 0.55–0.75 range,
  showing the shape-only ceiling.
- **The interesting column is `shape_gap`** — `val_acc` minus
  `val_acc_residual_at_test`. Smaller gap = the model relies less on
  high-frequency information. If `curriculum` and `curriculum_dropout`
  show meaningfully smaller gaps than baseline, the curriculum is doing
  what it's supposed to do.

## What to report back

1. The full final summary table (the section after the `=====` divider)
2. Total wall-clock time
3. Any errors in full
4. The contents of `curriculum_results.json`

## What to do if it fails

- **`FileNotFoundError` re CIFAR**: `CIFAR_RAW_ROOT` at the top of the
  script is wrong; update it to the parent of `cifar-10-batches-py`.
- **`CUDA out of memory`**: reduce `training_batch_size` in `run_one()`
  from 1536 to 512.
- **Compile hangs >5 min**: change `mode="max-autotune"` to default in both
  `@torch.compile` decorators.
- **`decompose_identity` accuracy far below `baseline`**: this would be a
  real finding (decomposition is leaking signal). Report it, don't fix it.

## Do NOT
- Do not modify the schedule logic, dropout rates, or band count.
- Do not change the optimizer hyperparameters.
- Do not interpret single-run differences as significant — 3 runs per variant
  gives very noisy std estimates. The shape_gap metric is what we're after.

