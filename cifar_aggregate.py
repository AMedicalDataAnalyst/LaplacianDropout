"""Live results aggregator for the CIFAR sweep.

Walks ./cifar_results/*.json (per-run outputs from cifar_eval.py) and prints
a mean ± std table grouped by (dataset, method). Safe to run anytime — picks
up new files as they appear.

Also computes mCE (mean Corruption Error) normalized to the per-(dataset)
'baseline' method, when those runs exist. Writes a markdown table to
CIFAR_RESULTS.md.

Usage:  python cifar_aggregate.py
"""

import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT, 'cifar_results')

# Filename convention: cifar_eval_<dataset>_<method-mangled>_seed<s>.json
# (method strings like 'band_drop_all+augmix' become 'band_drop_all_p_augmix'
#  in filenames; we un-mangle here).
FN_RE = re.compile(
    r'cifar_eval_(?P<dataset>cifar10|cifar100)_(?P<method>[a-z0-9_]+?)_seed(?P<seed>\d+)\.json')


def unmangle(m):
    return m.replace('_p_', '+')


def collect():
    runs = defaultdict(list)  # (dataset, method) -> [run dicts]
    for fp in sorted(glob.glob(os.path.join(RESULTS_DIR, 'cifar_eval_*.json'))):
        m = FN_RE.match(os.path.basename(fp))
        if not m:
            continue
        with open(fp) as f:
            d = json.load(f)
        key = (m.group('dataset'), unmangle(m.group('method')))
        d['_seed'] = int(m.group('seed'))
        runs[key].append(d)
    return runs


def agg(values):
    a = np.array(values, dtype=float)
    if len(a) == 0:
        return None, None, 0
    if len(a) == 1:
        return float(a[0]), 0.0, 1
    return float(a.mean()), float(a.std()), len(a)


def per_corruption_means(run):
    """Return {corruption: mean_acc_across_severities} for one run."""
    corr = run.get('corruption', {})
    out = {}
    for c, sevs in corr.items():
        if c.startswith('_') or not isinstance(sevs, dict):
            continue
        out[c] = float(np.mean(list(sevs.values())))
    return out


def main():
    runs = collect()
    if not runs:
        print("(no result files yet)")
        return

    rows = []
    by_ds_baseline_errs = {}

    # First pass: compute per-dataset baseline per-corruption errors.
    for (ds, method), rs in runs.items():
        if method != 'baseline':
            continue
        per_corr = defaultdict(list)
        for r in rs:
            for c, acc in per_corruption_means(r).items():
                per_corr[c].append(1.0 - acc)
        by_ds_baseline_errs[ds] = {c: float(np.mean(v)) for c, v in per_corr.items()}

    # Second pass: per-(dataset, method) summary.
    for (ds, method), rs in sorted(runs.items()):
        seeds = sorted(r['_seed'] for r in rs)
        clean_acc, clean_std, n = agg([r['clean']['acc'] for r in rs])
        rms_ce, _, _ = agg([r['clean']['rms_ce'] for r in rs])
        if 'corruption' in rs[0]:
            corr_acc, corr_std, _ = agg([r['corruption']['_mean'] for r in rs])
        else:
            corr_acc, corr_std = None, None
        if 'adversarial' in rs[0]:
            pgd_acc, pgd_std, _ = agg([r['adversarial']['pgd_acc'] for r in rs])
        else:
            pgd_acc, pgd_std = None, None
        # mCE relative to this dataset's baseline (if both available)
        mce = None
        if corr_acc is not None and ds in by_ds_baseline_errs:
            bref = by_ds_baseline_errs[ds]
            ratios = []
            for r in rs:
                per_c = per_corruption_means(r)
                run_ratios = [(1.0 - per_c[c]) / max(bref[c], 1e-6)
                              for c in per_c if c in bref]
                if run_ratios:
                    ratios.append(float(np.mean(run_ratios)))
            if ratios:
                mce = 100.0 * float(np.mean(ratios))
        rows.append((ds, method, n, seeds, clean_acc, clean_std,
                     corr_acc, corr_std, mce, pgd_acc, pgd_std, rms_ce))

    # Print table.
    print(f"\n{'dataset':<10} {'method':<28} {'n':>2} {'seeds':<8} "
          f"{'clean':>16} {'corr':>16} {'mCE':>6} {'PGD':>14} {'RMS-CE':>7}")
    print('-' * 118)
    for (ds, m, n, sds, ca, cs, ka, ks, mce, pa, ps, rms) in rows:
        clean_s = f"{ca:.4f}±{cs:.4f}" if ca is not None else '—'
        corr_s = f"{ka:.4f}±{ks:.4f}" if ka is not None else '—'
        mce_s = f"{mce:.1f}" if mce is not None else '—'
        pgd_s = f"{pa:.4f}±{ps:.4f}" if pa is not None else '—'
        rms_s = f"{100*rms:.2f}" if rms is not None else '—'
        seeds_s = ','.join(str(s) for s in sds)
        print(f"{ds:<10} {m:<28} {n:>2} {seeds_s:<8} {clean_s:>16} "
              f"{corr_s:>16} {mce_s:>6} {pgd_s:>14} {rms_s:>7}")

    # Markdown summary.
    md = ["# CIFAR sweep results (auto-aggregated)\n",
          f"Aggregated from {len(runs)} (dataset, method) combinations, "
          f"{sum(len(r) for r in runs.values())} total runs.\n",
          "Mean ± std over seeds. mCE normalized to the per-dataset `baseline` runs "
          "(=100 by construction; lower is better).\n",
          "| Dataset | Method | n | Clean | Corruption | mCE | PGD@2/255 | RMS-CE clean |",
          "|---|---|---|---|---|---|---|---|"]
    for (ds, m, n, sds, ca, cs, ka, ks, mce, pa, ps, rms) in rows:
        clean_s = f"{ca:.4f} ± {cs:.4f}" if ca is not None else '—'
        corr_s = f"{ka:.4f} ± {ks:.4f}" if ka is not None else '—'
        mce_s = f"{mce:.1f}" if mce is not None else '—'
        pgd_s = f"{pa:.4f} ± {ps:.4f}" if pa is not None else '—'
        rms_s = f"{100*rms:.2f}" if rms is not None else '—'
        md.append(f"| {ds} | `{m}` | {n} | {clean_s} | {corr_s} | {mce_s} | {pgd_s} | {rms_s} |")
    md.append("")
    out_md = os.path.join(ROOT, 'docs', 'CIFAR_RESULTS.md')
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, 'w') as f:
        f.write('\n'.join(md))
    print(f"\nWrote {out_md} ({len(rows)} rows)")


if __name__ == '__main__':
    main()
