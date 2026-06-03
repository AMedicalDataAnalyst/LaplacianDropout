"""Analysis + plots for the Imagenette curriculum experiments.

Reads one or more `imagenette_results_*.json` files and produces:
- A summary table: variant × {val_acc, res_test, blur_test, corr_mean (if present)}
- A trade-off scatter: clean val_acc vs mean corruption acc
- Per-corruption bars: relative robustness vs baseline

Usage:
    python3 analyze_imagenette.py results/phase1_imagenette/imagenette_results_*.json --out report.png
"""

import argparse
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def aggregate(results, name_override=None):
    """Group by variant; return dict variant -> dict of stats lists.

    If name_override is given, all results map to that name (used when a
    file represents a single named experiment regardless of internal variant).
    """
    by_variant = defaultdict(lambda: defaultdict(list))
    corr_by_variant = defaultdict(list)
    for r in results:
        v = name_override or r["variant"]
        by_variant[v]["val_acc"].append(r["val_acc"])
        by_variant[v]["val_acc_residual_at_test"].append(r["val_acc_residual_at_test"])
        by_variant[v]["val_acc_blur_at_test"].append(r["val_acc_blur_at_test"])
        if "total_epochs" in r:
            by_variant[v]["epochs"].append(r["total_epochs"])
        if "corruption" in r:
            by_variant[v]["corr_mean"].append(r["corruption"]["_mean"])
            corr_by_variant[v].append(r["corruption"])
    return by_variant, corr_by_variant


def load_grouped(file_patterns):
    """Load files from a dict {label: [glob patterns]} and merge.

    Returns same shape as aggregate() but unions across labels.
    """
    import glob
    out_by, out_corr = defaultdict(lambda: defaultdict(list)), defaultdict(list)
    for label, patterns in file_patterns.items():
        files = []
        for p in patterns:
            files.extend(glob.glob(p))
        results = []
        for f in files:
            with open(f) as fh:
                results.extend(json.load(fh))
        by_v, corr_v = aggregate(results, name_override=label)
        for k, v in by_v.items():
            for kk, vv in v.items():
                out_by[k][kk].extend(vv)
        for k, v in corr_v.items():
            out_corr[k].extend(v)
    return out_by, out_corr


def fmt_mean_std(xs):
    if not xs:
        return "—"
    a = np.asarray(xs)
    if len(a) == 1:
        return f"{a.mean():.4f}"
    return f"{a.mean():.4f} ± {a.std():.4f}"


def print_summary(by_variant):
    cols = ["val_acc", "val_acc_residual_at_test", "val_acc_blur_at_test", "corr_mean", "epochs"]
    head = ["variant"] + cols
    widths = [22, 22, 22, 22, 22, 12]
    print("  ".join(f"{h:<{w}}" for h, w in zip(head, widths)))
    print("-" * 130)
    for v, stats in by_variant.items():
        row = [v] + [fmt_mean_std(stats.get(c, [])) for c in cols]
        print("  ".join(f"{val:<{w}}" for val, w in zip(row, widths)))


def plot_tradeoff(by_variant, out_path):
    fig, ax = plt.subplots(figsize=(9, 6))
    for v, stats in by_variant.items():
        if not stats.get("corr_mean"):
            continue
        clean = np.mean(stats["val_acc"])
        corr = np.mean(stats["corr_mean"])
        res = np.mean(stats["val_acc_residual_at_test"])
        # marker size encodes residual-eval acc (shape bias proxy)
        s = 80 + 1000 * res
        ax.scatter(clean, corr, s=s, alpha=0.7,
                   label=f"{v} (res={res:.2f})")
        ax.annotate(v, (clean, corr), xytext=(5, 5), textcoords="offset points",
                    fontsize=8)
    ax.set_xlabel("Clean val accuracy")
    ax.set_ylabel("Mean corruption accuracy (Imagenette-C)")
    ax.set_title("Clean accuracy vs corruption robustness\n"
                 "(marker size ∝ residual-only test accuracy = shape-bias proxy)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Wrote tradeoff plot to {out_path}")


def plot_per_corruption(corr_by_variant, out_path, baseline_name="baseline"):
    """One subplot per corruption type, severity on x, acc on y, line per variant."""
    if not corr_by_variant:
        return
    # Collect all corruption names
    sample = next(iter(corr_by_variant.values()))[0]
    corruption_names = [k for k in sample.keys() if not k.startswith("_")]
    n_corr = len(corruption_names)
    cols = 5
    rows = (n_corr + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 2.6),
                              sharey=True)
    axes = np.array(axes).reshape(-1)
    for idx, cname in enumerate(corruption_names):
        ax = axes[idx]
        for v, runs in corr_by_variant.items():
            # average across runs
            sevs = sorted({s for r in runs for s in r[cname].keys()},
                          key=lambda x: int(x))
            accs = []
            for s in sevs:
                vals = [r[cname][s] for r in runs if s in r[cname]]
                accs.append(np.mean(vals))
            ax.plot([int(s) for s in sevs], accs, marker="o", label=v)
        ax.set_title(cname, fontsize=9)
        ax.set_xlabel("severity")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
    for idx in range(n_corr, len(axes)):
        axes[idx].axis("off")
    axes[0].legend(loc="lower left", fontsize=7)
    fig.suptitle("Per-corruption accuracy by severity")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"Wrote per-corruption plot to {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="results JSONs to combine")
    ap.add_argument("--out-prefix", default="imagenette_report")
    args = ap.parse_args()

    all_results = []
    for path in args.files:
        with open(path) as f:
            all_results.extend(json.load(f))

    by_variant, corr_by_variant = aggregate(all_results)
    print(f"\n# Aggregate over {len(args.files)} files, {len(all_results)} runs\n")
    print_summary(by_variant)

    if any(stats.get("corr_mean") for stats in by_variant.values()):
        plot_tradeoff(by_variant, args.out_prefix + "_tradeoff.png")
        plot_per_corruption(corr_by_variant, args.out_prefix + "_per_corruption.png")


if __name__ == "__main__":
    main()
