# Lessons learned — cross-phase

Things worth remembering, in roughly the order they were discovered.

## Methodological

1. **CIFAR-10 clean accuracy is too saturated to show frequency-augmentation
   gains.** Don't use CIFAR clean acc as the first benchmark; pivot to a
   richer dataset early. Phase 1 wasted ~2 days on CIFAR clean accuracy
   before this was obvious.
2. **`imagecorruptions` library applies severity parameters in absolute
   pixel units, calibrated for ImageNet (224 px).** Applied to CIFAR
   (32 px) it produces ~30 pp lower baseline corruption accuracy than the
   released CIFAR-10-C tarballs. ALWAYS download the released
   `CIFAR-10-C.tar` / `CIFAR-100-C.tar` for canonical comparison; the
   library is for on-the-fly fallback only.
3. **mCE normalization matters for cross-paper comparison.** Our Phase-1
   mCE is normalized to *our* baseline (relative-CE, =100 by construction),
   not to AlexNet (the canonical reference). For Phase-2 ImageNet, the
   augmix repo's `compute_mce` already uses the AlexNet error table — use
   that for any published-comparison number.
4. **3 seeds is the bar.** Single-seed results draw methodological
   complaints. Phase 1 mostly ran 3 seeds; Phase 2 should maintain.
5. **Augmentation operations must be disjoint from the corruption set.**
   AugMix's default `augmentations` set excludes brightness/contrast/noise/
   blur to avoid corruption overlap. Band-dropout is *spectrally adjacent*
   to blur but operationally distinct (random subspace projection vs
   deterministic Gaussian); flag this explicitly in the paper.

## Mechanism

6. **The curriculum was a distraction.** The per-sample `band_drop_all`
   aug *inside* the curriculum was doing the work; the unlock schedule was
   redundant. Removing the schedule simplified the method to one line and
   improved results.
7. **JSD consistency loss is load-bearing.** Without it the
   `band_drop_all+augmix` combo loses ~3 pp corruption acc. Pay the 3×
   per-step forward cost.
8. **Operating domain matters: apply band-dropout on the *normalized*
   tensor**, not raw [0,1] pixels. The Phase 1 GPU pipeline did this; the
   Phase 2 per-image transform replicates it (drops in after
   `ToTensor`+`Normalize`).
9. **Band count saturates fast.** 4 bands captures ~83 % of the
   robustness gain; 6 captures the rest at a ~1 pp clean cost; 7+ adds
   nothing. The σ=2/4/8 mid-frequency bands are the load-bearing ones.
10. **The "all-zero training image" is not a bug; it's a feature.** When
    `drop_residual=True`, all 6 bands can be dropped (~1.6 % of samples
    at p=0.5). Implementing a `min_kept=1` guard *hurts* clean and
    corruption accuracy by ~0.5–0.8 pp. The all-zero samples act as an
    implicit label-smoothing regularizer.

## Engineering

11. **WRN-28-10 at JSD on a 10 GB GPU is ~14 h per 100-epoch run.** Plan
    accordingly. Initial estimates of ~3 h were optimistic.
12. **CIFAR-32 needs `--band-levels 4` (= 5 bands).** σ=16 would require a
    48-pixel Gaussian kernel, which doesn't fit a 32-pixel image. Caught
    only at training time as a `RuntimeError: Padding size...`. The per-image
    `BandDropAll(levels=...)` is parametrized for this.
13. **Band-drop methods need gradient clipping at LR=0.1 on
    WRN-28-10/CIFAR.** Without it, NaN loss within 1 epoch. Clip to 5.0
    (no-op for baseline/AugMix; saves band methods). Phase-2 cifar.py
    patches include this.
14. **The `augmix/cifar.py` repo has a `.view(-1)` on non-contiguous
    tensor bug** that fails on modern PyTorch. The `imagenet.py` variant
    has the same. Both fixed to `.reshape(-1)` in our patches.
15. **pixmix `imagenet.py` parses argv at module *import* (not just in
    main).** If you `import pixmix.imagenet`, you must fake `sys.argv` to
    satisfy `--mixing-set --num-classes`. `ood_eval.py` does this dance.
16. **DataParallel checkpoints have `module.` key prefix.** All our
    loaders strip it before `load_state_dict` to keep compatibility with
    both single-GPU and DP saves.

## Workflow

17. **Smoke test the FULL pipeline with 1-epoch runs before committing
    to multi-day sweeps.** Phase 2 caught two issues this way (band-level
    crash, gradient explosion). Both would have wasted ~12 days.
18. **De-risk gates before expensive runs.** Phase 2 added a 2-run CIFAR
    de-risk (~19 h) before committing to the 30-run sweep (~12 days).
    Cheap insurance.
19. **Checkpoint filenames must encode all axes (method, dataset,
    seed) explicitly** — silent overwrites by colliding names cost a
    re-eval round in Phase 1 (the 160 vs 224 checkpoints collided in
    `checkpoints/`).
20. **Auto-aggregator scripts that re-read JSONs every run are
    invaluable.** `cifar_aggregate.py` re-reads `cifar_results/` and
    rebuilds `docs/CIFAR_RESULTS.md`. No partial-result loss.

23. **(2026-06-03) Explicit per-run seed setting is required for
    methodological cleanliness.** Phase-1 `compare_methods.py` had *no*
    `torch.manual_seed` call anywhere — runs were "different" only because
    `model.reset()` consumed the global RNG between iterations. This
    produced:
    - genuinely different inits *within* one invocation (confirmed:
      epoch-1 val_acc spread of 10 pp across the 3-seed runs in
      `comparison_224_main.json`)
    - identical run-0 across separate invocations (not reproducible)
    - per-seed records labelled "run 0/1/2" with no formal seed value
    Fix: added `--seed-offset` to `compare_methods.py`; each run now
    calls `torch.manual_seed(offset + run_idx)` + the cuda + numpy
    equivalents, and the record carries an explicit `seed` field. **All
    Phase-2 runs use this from 2026-06-03 onward.** Phase-1
    `comparison_*.json` files predate the fix; their cross-run variance
    is real but their seed labels are nominal.

24. **Corruption-acc cross-seed variance can be misleadingly low** when
    the eval is on a *deterministically seeded subsample* (the case for
    our fast eval — `rng = np.random.default_rng(0)` in
    `evaluate_corruption_suite`). The eval is identical across runs;
    only the model varies. Combined with a method that converges
    consistently (heavy regularization, small dataset), the run-to-run
    corr std can drop to ~0.0004.

    **Don't fix this with eval-set bootstrap CIs — that's not standard
    practice in the AugMix/PixMix/IPMix lineage.** The convention is
    N-seed mean ± std on the *full* canonical val set (no subsample).
    Bootstrap-over-images answers "if I had a different 50k…" which no
    reader is asking — the benchmark is fixed. Cross-seed std captures
    the method-noise readers do care about ("is the improvement robust
    to lucky training?").

    **Right thing to do:** for headline numbers, use the full
    50k × 15 × 5 = 3.75M-evaluation pass (no subsample → eval noise is
    negligible) and report 3-seed mean ± std. For ablation tables,
    deterministic subsample is fine since those aren't headline numbers.
    If 3-seed std looks suspiciously low on a headline number, run 1–2
    more seeds for a sanity check rather than reaching for bootstrap.

## Repo hygiene

21. **Result JSONs in repo root accumulate fast.** Phase 1 left ~50
    `imagenette_results_*.json` and `curriculum_results_*.json` at the
    top level. Consider archiving into `results/phase1/` after Phase 2
    completes (deferred — would break `build_report.py` paths until
    `build_report.py` is updated).
22. **Don't move .py files during a tidy.** They have import
    dependencies and `sys.path` insertions that aren't always obvious.
    Move docs; document the code locations.
