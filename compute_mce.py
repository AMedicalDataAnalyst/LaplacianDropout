"""Compute mCE (mean Corruption Error) normalized to our own baseline.

The Hendrycks ImageNet-C mCE is normalized to AlexNet's per-corruption errors.
For Imagenette there's no canonical reference; we normalize to our own
unaugmented ResNet-18 baseline. This gives a "relative-CE" that's comparable
across methods on this benchmark but not directly to published ImageNet mCE.

mCE = (1/N_c) Σ_c (avg_severity_error_method[c] / avg_severity_error_baseline[c])

Lower is better. mCE=100 → equal to baseline. mCE<100 → more robust.
"""

import json
import glob
import os
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))


def _resolve_path(pat):
    """Map a result-filename pattern to its new subdir under results/.
    Backwards-compat: falls back to root if no known prefix matches."""
    prefix_to_subdir = [
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
    return os.path.join(ROOT, pat)


def per_corruption_means(json_files):
    """Return {corruption: [acc, acc, ...]} pooled across runs."""
    out = defaultdict(list)
    for fp in json_files:
        with open(fp) as fh:
            data = json.load(fh)
        for r in data:
            if "corruption" not in r:
                continue
            for c, sevs in r["corruption"].items():
                if c.startswith("_"):
                    continue
                # average across severities for this run
                accs = [sevs[s] for s in sevs]
                out[c].append(np.mean(accs))
    return {c: np.mean(v) for c, v in out.items()}


def errors_from(accs: dict) -> dict:
    return {c: 1.0 - a for c, a in accs.items()}


def mce(method_errors: dict, ref_errors: dict) -> float:
    cs = sorted(set(method_errors) & set(ref_errors))
    ratios = [method_errors[c] / max(ref_errors[c], 1e-6) for c in cs]
    return 100.0 * np.mean(ratios)


GROUPS = {
    "baseline (FULL)": ["imagenette_results_FULL_baseline_*.json"],
    "longA (FULL)": ["imagenette_results_FULL_longA_seed*.json"],
    "extralongA (FULL)": ["imagenette_results_FULL_extralongA_seed*.json"],
    "AugMix+JSD (FULL)": ["imagenette_results_FULL_augmix_r0.json"],
    "PixMix+JSD (FULL)": ["imagenette_results_FULL_pixmix_r0.json"],
    "IPMix+JSD (FULL)": ["imagenette_results_FULL_ipmix_r0.json"],
    "band_drop_all+AugMix+JSD (FULL)": ["imagenette_results_FULL_bd_augmix_r0.json"],
    "band_drop_all+PixMix+JSD (FULL)": ["imagenette_results_FULL_bd_pixmix_r0.json"],
    "longA bd+aug → FT (FULL)": ["imagenette_results_FULL_longA_bd_aug_r0.json"],
}


def main():
    # First, compute baseline errors as the reference.
    baseline_files = []
    for pat in GROUPS["baseline (FULL)"]:
        baseline_files.extend(glob.glob(_resolve_path(pat)))
    if not baseline_files:
        print("No FULL baseline eval found, cannot compute mCE.")
        return
    baseline_accs = per_corruption_means(baseline_files)
    baseline_errs = errors_from(baseline_accs)

    print("=== Per-corruption accuracy by method (FULL eval) ===")
    cs = sorted(baseline_errs.keys())
    header = ["method"] + cs + ["mean_corr_acc", "mCE_rel"]
    print("  ".join(f"{h[:18]:>18}" for h in header))
    print("-" * (20 * len(header)))

    for method, patterns in GROUPS.items():
        files = []
        for pat in patterns:
            files.extend(glob.glob(_resolve_path(pat)))
        if not files:
            continue
        accs = per_corruption_means(files)
        errs = errors_from(accs)
        common = [c for c in cs if c in accs]
        if not common:
            continue
        mean_acc = np.mean([accs[c] for c in common])
        m = mce({c: errs[c] for c in common},
                {c: baseline_errs[c] for c in common})
        cells = [f"{method[:18]:>18}"]
        for c in cs:
            v = accs.get(c, np.nan)
            cells.append(f"{v:>18.3f}" if not np.isnan(v) else "                 —")
        cells.append(f"{mean_acc:>18.3f}")
        cells.append(f"{m:>18.1f}")
        print("  ".join(cells))


if __name__ == "__main__":
    main()
