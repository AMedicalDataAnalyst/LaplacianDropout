"""Morning report for run_pretrained_night.sh -> results/phase3_pretrained/REPORT.md.

Pre-registered (written before the queue started, 2026-11-22):
  metric   mean corruption accuracy (14 corruptions, severity 3, 1000 images),
           weight-EMA 0.99 shadow, warm-up + constant LR
  primary  band_drop_all+augmix minus augmix (both with JSD), pretrained, lr 3e-4,
           paired by seed, per dataset
  second   band_drop_all minus baseline (neither with JSD), same cells
  guard    clean accuracy must not drop by more than 1 pp
  WIN      mean paired difference > 2 SE on a dataset
Runs are short (~2000 steps), which is why the 0.99 shadow is the primary
read-out rather than 0.999; both and the raw weights are tabulated.
"""
import glob
import json
import math
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results", "phase3_pretrained")
PAIRS = (("band_drop_all+augmix", "augmix", "primary: band drop on top of AugMix (JSD)"),
         ("band_drop_all", "baseline", "secondary: band drop alone (no JSD)"))
WEIGHTS = ("0.99", "0.999", "raw")


def read(r, w):
    src = r if w == "raw" else r["ema"][w]
    return 100 * src["corruption"]["_mean"], 100 * src["val_acc"]


def mean_se(xs):
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, float("nan")
    return m, math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1) / n)


def fmt(xs):
    m, se = mean_se(xs)
    return f"{m:+.2f} ± {se:.2f}" if len(xs) > 1 else f"{m:+.2f} (n=1)"


def main():
    cells = defaultdict(dict)   # (dataset, init, lr) -> seed -> {method: result}
    for p in sorted(glob.glob(os.path.join(RES, "*.json"))):
        for r in json.load(open(p)):
            cells[(r["dataset"], r["init"], r["lr"])].setdefault(r["seed"], {})[r["method"]] = r
    out = ["# Band dropout on a pretrained backbone (night of 2026-11-22)", ""]
    if not cells:
        out.append("No finished cells.")
    out += ["Corruption accuracy = mean over 14 corruptions at severity 3 (1000 validation images). "
            "All numbers in percentage points; differences are paired by seed, ± is the standard error.",
            "", "## Paired differences (weight-EMA 0.99, the pre-registered read-out)", "",
            "| dataset | init | lr | comparison | seeds | Δ corruption | Δ clean | verdict |",
            "|---|---|---|---|---|---|---|---|"]
    for key in sorted(cells):
        for a, b, label in PAIRS:
            dc, dv = [], []
            for seed, arms in sorted(cells[key].items()):
                if a in arms and b in arms:
                    ca, va = read(arms[a], "0.99")
                    cb, vb = read(arms[b], "0.99")
                    dc.append(ca - cb)
                    dv.append(va - vb)
            if not dc:
                continue
            m, se = mean_se(dc)
            mv, _ = mean_se(dv)
            if len(dc) < 2:
                verdict = "n=1"
            elif mv < -1.0:
                verdict = "clean guard broken"
            elif m > 2 * se:
                verdict = "WIN"
            elif m < -2 * se:
                verdict = "worse"
            else:
                verdict = "no clear effect"
            out.append(f"| {key[0]} | {key[1]} | {key[2]:g} | {label} | {len(dc)} | "
                       f"{fmt(dc)} | {fmt(dv)} | {verdict} |")
    out += ["", "## Absolute numbers (mean over seeds; corruption / clean)", ""]
    for key in sorted(cells):
        seeds = sorted(cells[key])
        out += [f"### {key[0]}, {key[1]}, lr {key[2]:g} (seeds {seeds})", "",
                "| arm | " + " | ".join(f"EMA {w}" if w != "raw" else "raw weights" for w in WEIGHTS) + " |",
                "|---|" + "---|" * len(WEIGHTS)]
        for method in ("baseline", "band_drop_all", "augmix", "band_drop_all+augmix"):
            row = []
            for w in WEIGHTS:
                vals = [read(cells[key][s][method], w) for s in seeds if method in cells[key][s]]
                row.append(f"{sum(v[0] for v in vals) / len(vals):.1f} / "
                           f"{sum(v[1] for v in vals) / len(vals):.1f}" if vals else "-")
            out.append(f"| {method} | " + " | ".join(row) + " |")
        out.append("")
    log = os.path.join(HERE, "logs", "phase3", "night_1122.log")
    if os.path.exists(log):
        lines = [l.rstrip() for l in open(log)]
        bad = [l for l in lines if "FAILED" in l or "GUARD" in l]
        out += ["## Queue", "", f"- cells finished: {sum(len(v) for v in cells.values())}",
                f"- failed or not started: {len(bad)}"] + [f"  - `{l}`" for l in bad]
    path = os.path.join(RES, "REPORT.md")
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
