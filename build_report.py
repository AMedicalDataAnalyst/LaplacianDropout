"""Final shape-bias report — tier-1 version with AugMix/PixMix/IPMix comparisons."""

import json
import glob
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))


def _resolve_path(pat):
    """Map a result-filename pattern to its new subdir under results/.
    Backwards-compat: if not found in the expected subdir, falls back to root."""
    prefix_to_subdir = [
        ('curriculum_results',         'results/phase1_curriculum'),
        ('imagenette_results_FULL',    'results/phase1_imagenette_full'),
        ('imagenette_results_',        'results/phase1_imagenette'),
        ('comparison_',                'results/phase1_comparison'),
        ('band_count_ablation_',       'results/phase1_ablations'),
        ('band_keep1_',                'results/phase1_ablations'),
        ('bandmask_',                  'results/phase2_reviewer_defense'),
    ]
    for prefix, subdir in prefix_to_subdir:
        if pat.startswith(prefix):
            return os.path.join(ROOT, subdir, pat)
    return os.path.join(ROOT, pat)  # fallback (e.g. for new patterns)


GROUPS = [
    # ===== Original Imagenette pipeline (no JSD) =====
    ("baseline (no JSD)",
     ["imagenette_results_multiseed.json"],
     "baseline"),
    ("baseline (FULL eval)",
     ["imagenette_results_FULL_baseline_*.json"],
     "eval_only"),
    ("curriculum_dropout (orig)",
     ["imagenette_results_multiseed.json"],
     "curriculum_dropout"),
    ("band_drop_all (no JSD)",
     ["imagenette_results_multiseed.json"],
     "band_drop_all"),
    ("joint_blur (no JSD)",
     ["imagenette_results_multiseed.json"],
     "joint_blur"),
    ("longA (no JSD, FULL eval)",
     ["imagenette_results_FULL_longA_seed*.json"],
     "eval_only"),
    ("extralongA (no JSD, FULL eval)",
     ["imagenette_results_FULL_extralongA_seed*.json"],
     "eval_only"),
    # ===== Tier-1 comparisons (with JSD) =====
    ("AugMix + JSD",
     ["comparison_jsd_results.json"],
     "augmix"),
    ("PixMix + JSD",
     ["comparison_jsd_results.json"],
     "pixmix"),
    ("band_drop_all + JSD",
     ["comparison_jsd_results.json"],
     "band_drop_all"),
    ("band_drop_all + AugMix + JSD",
     ["comparison_jsd_results.json"],
     "band_drop_all+augmix"),
    ("band_drop_all + PixMix + JSD",
     ["comparison_jsd_results.json"],
     "band_drop_all+pixmix"),
    ("IPMix + JSD",
     ["comparison_ipmix.json"],
     "ipmix"),
    ("band_drop_all + IPMix + JSD",
     ["comparison_ipmix.json"],
     "band_drop_all+ipmix"),
    ("longA (bd_aug + AugMix → FT) + JSD",
     ["comparison_longA_bd_augmix_s2_r*.json"],
     "baseline"),
    # FULL evals
    ("AugMix+JSD (FULL eval)",
     ["imagenette_results_FULL_augmix_r0.json"],
     "eval_only"),
    ("PixMix+JSD (FULL eval)",
     ["imagenette_results_FULL_pixmix_r0.json"],
     "eval_only"),
    ("IPMix+JSD (FULL eval)",
     ["imagenette_results_FULL_ipmix_r0.json"],
     "eval_only"),
    ("band_drop_all+PixMix+JSD (FULL eval)",
     ["imagenette_results_FULL_bd_pixmix_r0.json"],
     "eval_only"),
    ("band_drop_all+AugMix+JSD (FULL eval, 80ep)",
     ["imagenette_results_FULL_bd_augmix_s1_r0.json"],
     "eval_only"),
    ("longA bd+aug→FT (FULL eval)",
     ["imagenette_results_FULL_longA_bd_aug_r0.json"],
     "eval_only"),
]


def load_group(patterns, filt):
    rs = []
    for pat in patterns:
        for f in sorted(glob.glob(_resolve_path(pat))):
            with open(f) as fh:
                rs.extend(json.load(fh))
    if filt is not None:
        rs = [r for r in rs
              if r.get("variant", r.get("method")) == filt]
    return rs


def aggregate(rs):
    clean = np.array([r["val_acc"] for r in rs]) if rs else np.array([])
    res = np.array([r["val_acc_residual_at_test"] for r in rs]) if rs else np.array([])
    blur = np.array([r["val_acc_blur_at_test"] for r in rs]) if rs else np.array([])
    corr = np.array([r["corruption"]["_mean"] for r in rs if "corruption" in r])
    return clean, res, blur, corr


def fmt(arr):
    if len(arr) == 0:
        return "—"
    if len(arr) == 1:
        return f"{arr[0]:.4f}"
    return f"{arr.mean():.4f} ± {arr.std():.4f}"


def main():
    rows = []
    for label, pats, filt in GROUPS:
        rs = load_group(pats, filt)
        clean, res, blur, corr = aggregate(rs)
        rows.append((label, len(rs), clean, res, blur, corr))

    md = ["# Imagenette tier-1 comparison: ours vs AugMix vs PixMix vs IPMix\n"]
    md.append("Methodology: ResNet-18 on Imagenette-160, AMP+AdamW, plateau-based "
              "scheduler. The 'no JSD' rows are from the original baselines using the "
              "single-forward training loop. The '+ JSD' rows use the AugMix-style "
              "Jensen-Shannon consistency loss (3 forward passes per batch, λ=12). "
              "Corruption metric is mean accuracy across 14 Imagenette-C corruptions "
              "(glass_blur excluded), severity 3 (fast eval, 1000-image subset) or "
              "{1,3,5} (FULL eval, full 3925-image val set).\n")
    md.append("| Variant | n | Clean val_acc | Residual-only (shape proxy) | σ=2 blur eval | Mean corruption acc |")
    md.append("|---|---|---|---|---|---|")
    for label, n, clean, res, blur, corr in rows:
        md.append(f"| {label} | {n} | {fmt(clean)} | {fmt(res)} | {fmt(blur)} | {fmt(corr)} |")
    md_text = "\n".join(md) + "\n"
    out_md = os.path.join(ROOT, "docs", "tier1_report.md")
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w") as fh:
        fh.write(md_text)
    print(md_text)

    fig, ax = plt.subplots(figsize=(12, 7))
    palette = plt.cm.tab20(np.linspace(0, 1, len(rows)))
    for (label, n, clean, res, blur, corr), c in zip(rows, palette):
        if len(corr) == 0 or len(clean) == 0:
            continue
        cm_v, cs = clean.mean(), clean.std() if len(clean) > 1 else 0.0
        rm = res.mean() if len(res) else 0.0
        krm, krs = corr.mean(), corr.std() if len(corr) > 1 else 0.0
        s = 80 + 1200 * rm
        ax.scatter(cm_v, krm, s=s, alpha=0.78, color=c,
                   edgecolors="black", linewidths=0.7)
        if cs > 0 or krs > 0:
            ax.errorbar(cm_v, krm, xerr=cs, yerr=krs, fmt="none",
                        ecolor=c, alpha=0.55, capsize=3)
        ax.annotate(label, (cm_v, krm), xytext=(7, 6),
                    textcoords="offset points", fontsize=7)
    ax.set_xlabel("Clean validation accuracy")
    ax.set_ylabel("Mean corruption accuracy")
    ax.set_title("Imagenette tier-1 comparison: clean vs corruption robustness\n"
                 "Marker size ∝ residual-only-eval accuracy (shape-bias proxy)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_png = os.path.join(ROOT, "figures", "tier1_tradeoff.png")
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f"\nWrote {out_md} and {out_png}")


if __name__ == "__main__":
    main()
