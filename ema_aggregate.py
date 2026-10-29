"""Aggregate the Phase-2 weight-EMA comparison (results/phase2_ema/).

Prints a per-method table of 3-seed mean ± std for clean val acc and fast
corruption acc, for the raw weights and each tracked EMA decay, next to the
Phase-1 224² reference rows (results/phase1_comparison/comparison_224_main.json).

Usage:  python ema_aggregate.py [--dir results/phase2_ema]
"""

import argparse
import glob
import json
import os
import statistics as stats


def _rows_from_runs(runs, tag_prefix=""):
    """Collapse a list of run records into {variant_label: aggregates}."""
    out = {}
    by_method = {}
    for r in runs:
        sched = r.get("schedule", "cosine")
        label = r["method"] if sched == "cosine" else f"{r['method']} ({sched}-lr)"
        by_method.setdefault(label, []).append(r)
    for method, rs in sorted(by_method.items()):
        variants = {"raw": [(r["val_acc"], r["corruption"]["_mean"],
                             r["val_acc_residual_at_test"]) for r in rs]}
        for r in rs:
            for k, e in r.get("ema", {}).items():
                variants.setdefault(f"ema {k}", []).append(
                    (e["val_acc"], e["corruption"]["_mean"],
                     e["val_acc_residual_at_test"]))
        for var, vals in variants.items():
            clean = [v[0] for v in vals]
            corr = [v[1] for v in vals]
            res = [v[2] for v in vals]
            label = f"{tag_prefix}{method} [{var}]"
            out[label] = {
                "n": len(vals),
                "clean_mean": stats.mean(clean),
                "clean_std": stats.stdev(clean) if len(clean) > 1 else 0.0,
                "corr_mean": stats.mean(corr),
                "corr_std": stats.stdev(corr) if len(corr) > 1 else 0.0,
                "res_mean": stats.mean(res),
            }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/phase2_ema")
    ap.add_argument("--reference",
                    default="results/phase1_comparison/comparison_224_main.json")
    args = ap.parse_args()

    runs = []
    for f in sorted(glob.glob(os.path.join(args.dir, "*.json"))):
        runs.extend(json.load(open(f)))
    rows = _rows_from_runs(runs)

    ref_rows = {}
    if os.path.exists(args.reference):
        ref_rows = _rows_from_runs(json.load(open(args.reference)),
                                   tag_prefix="ref: ")

    header = (f"{'variant':<44} {'n':>2} {'clean':>15} {'corr':>15} {'res':>7}")
    print(header)
    print("-" * len(header))
    for label, r in {**ref_rows, **rows}.items():
        print(f"{label:<44} {r['n']:>2} "
              f"{r['clean_mean']:.4f} ±{r['clean_std']:.4f} "
              f"{r['corr_mean']:.4f} ±{r['corr_std']:.4f} "
              f"{r['res_mean']:>7.4f}")


if __name__ == "__main__":
    main()
