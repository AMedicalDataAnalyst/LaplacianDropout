"""Aggregate the band-targeted PGD sweep results into a single table + figure.

Input: results/phase2_targeted_pgd/<model>_target_<t>.json (35 files)
Output:
  - figures/band_targeted_pgd.png      grouped bar chart
  - docs/band_targeted_pgd_table.md    headline numbers
"""

import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
RES_DIR = os.path.join(ROOT, "results", "phase2_targeted_pgd")

MODEL_ORDER = ["baseline", "bd_augmix", "bd_only", "hf_only", "lf_only"]
TARGET_ORDER = ["none", "0", "1", "2", "3", "4", "5"]
TARGET_LABELS = {
    "none": "all bands\n(standard PGD)",
    "0": "band 0\n(σ=1, HF)",
    "1": "band 1\n(σ=2)",
    "2": "band 2\n(σ=4)",
    "3": "band 3\n(σ=8)",
    "4": "band 4\n(σ=16)",
    "5": "residual\n(low-pass)",
}
MODEL_LABELS = {
    "baseline":   "baseline (no aug)",
    "bd_augmix":  "band_drop_all + AugMix (HEADLINE)",
    "bd_only":    "band_drop_all only",
    "hf_only":    "band_drop_all hf_only",
    "lf_only":    "band_drop_all lf_only",
}


def load():
    """Return {(model, target): adv_acc}."""
    out = {}
    for f in sorted(glob.glob(os.path.join(RES_DIR, "*.json"))):
        base = os.path.basename(f).replace(".json", "")
        # filename = <model>_target_<t>.json
        for m in MODEL_ORDER:
            pre = f"{m}_target_"
            if base.startswith(pre):
                target = base[len(pre):]
                with open(f) as fh:
                    out[(m, target)] = json.load(fh)["adv_acc"]
                break
    return out


def main():
    data = load()
    if not data:
        print("(no results yet)"); return

    # Build a table: model rows × target columns
    print(f"\n{'model':<24}", end="")
    for t in TARGET_ORDER:
        print(f"{t:>10}", end="")
    print()
    print("-" * 24 + "-" * 70)
    for m in MODEL_ORDER:
        if all((m, t) not in data for t in TARGET_ORDER):
            continue
        print(f"{m:<24}", end="")
        for t in TARGET_ORDER:
            v = data.get((m, t))
            print(f"{v:>10.3f}" if v is not None else f"{'—':>10}", end="")
        print()

    # Markdown table
    lines = ["# Band-targeted PGD vulnerability profile\n",
             "Eval @ Imagenette-224, ε=4/255, 20-step PGD, 1000-image subsample.\n",
             "Accuracy under the attack (higher = more robust).\n",
             ""]
    lines.append("| Model | " + " | ".join(TARGET_ORDER) + " |")
    lines.append("|" + "---|" * (len(TARGET_ORDER) + 1))
    for m in MODEL_ORDER:
        row = [MODEL_LABELS.get(m, m)]
        for t in TARGET_ORDER:
            v = data.get((m, t))
            row.append(f"{v:.3f}" if v is not None else "—")
        lines.append("| " + " | ".join(row) + " |")
    with open(os.path.join(ROOT, "docs", "band_targeted_pgd_table.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nWrote docs/band_targeted_pgd_table.md")

    # Grouped bar chart: x-axis = target band, one group per model.
    fig, ax = plt.subplots(figsize=(13, 6))
    n_targets = len(TARGET_ORDER)
    n_models  = len(MODEL_ORDER)
    width = 0.82 / n_models
    x = np.arange(n_targets)
    colors = ["#555", "#1f77b4", "#9ecae1", "#2ca02c", "#d62728"]
    for i, m in enumerate(MODEL_ORDER):
        ys = [data.get((m, t), np.nan) for t in TARGET_ORDER]
        ax.bar(x + (i - (n_models - 1) / 2) * width, ys, width=width,
               label=MODEL_LABELS.get(m, m), color=colors[i % len(colors)])
    ax.set_xticks(x)
    ax.set_xticklabels([TARGET_LABELS[t] for t in TARGET_ORDER], fontsize=9)
    ax.set_ylabel("Accuracy under PGD attack (higher = more robust)")
    ax.set_title("Band-targeted PGD vulnerability profile (Imagenette-224, ε=4/255, 20 steps)\n"
                 "Each group = one target band; bar height = model's accuracy when only that band can be attacked.\n"
                 "Flat profile = spectral invariance. Spiky profile (low bar on one band) = model relies on that band.")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
    ax.set_ylim(0, 1.0)
    fig.tight_layout()
    out_png = os.path.join(ROOT, "figures", "band_targeted_pgd.png")
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
