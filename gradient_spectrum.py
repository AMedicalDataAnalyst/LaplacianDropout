"""Frequency-domain saliency analysis.

For trained checkpoints, compute |∂L/∂x| averaged over a val pass, decompose
into Laplacian bands (using the same basis the training augmentation operates
on), and plot the per-band energy distribution.

Hypothesis: baseline gradients concentrate at HF (the model relies on
high-frequency content for classification). The band_drop_all model has been
forced to handle inputs with arbitrary bands missing → its gradients should be
flatter across bands (model attends to a broader frequency range).

If the prediction holds, this is a mechanism figure for the paper.

Usage:
  python gradient_spectrum.py
  -> figures/gradient_spectrum.png, results/phase1_ablations/gradient_spectrum.json
"""

import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

import imagenette_curriculum as ic

ROOT = os.path.dirname(os.path.abspath(__file__))
RES, PAD = 224, 16


def per_band_gradient_energy(model, loader, n_max=2000):
    """For each band of |∂L/∂x|, return its mean ||·||² as a fraction of total."""
    model.eval()
    pad = loader.pad
    side = loader.resolution
    N_BANDS = ic.LEVELS + 1
    energies = torch.zeros(N_BANDS, device="cuda")
    total_n = 0

    for inputs_padded, labels in loader:
        # inputs are normalized fp16 padded tensors; crop centrally
        # to match training-time geometry.
        x = inputs_padded[:, :, pad:pad + side, pad:pad + side].contiguous(
            memory_format=torch.channels_last
        ).requires_grad_(True)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits = model(x)
            loss = F.cross_entropy(logits, labels)
        g, = torch.autograd.grad(loss, x, retain_graph=False, create_graph=False)
        # |g| → Laplacian decomposition → per-band squared L2.
        g_abs = g.abs().float()  # (B, 3, H, W)
        bands = ic.laplacian_bands(g_abs, levels=ic.LEVELS, sigma0=ic.SIGMA0)
        for i, b in enumerate(bands):
            # sum over (B, C, H, W) — accumulate squared L2 energy
            energies[i] += (b ** 2).sum().item()
        total_n += g.shape[0]
        if total_n >= n_max:
            break
    return (energies / energies.sum()).cpu().tolist(), total_n


def main():
    val_loader = ic.ImagenetteLoader(
        "val", batch_size=64, resolution=RES, pad=PAD
    )

    CKPTS = {
        "baseline (r0)": "checkpoints/baseline_r0.pt",
        "band_drop_all+AugMix (r0)": "checkpoints/band_drop_all_augmix_r0.pt",
        "band_drop_all-only (r0)": "checkpoints/band_drop_all_r0.pt",
    }

    results = {}
    for name, path in CKPTS.items():
        model = ic.ImagenetteNet(num_classes=10).cuda().to(
            memory_format=torch.channels_last
        )
        ckpt = torch.load(path, map_location="cuda", weights_only=True)
        model.load_state_dict(ckpt["state_dict"])
        ic.warmup(model)
        # NOTE: warmup mutates parameters via a tiny training pass; reload
        # to ensure we measure the SAVED model, not the warmed-up one.
        model.load_state_dict(ckpt["state_dict"])

        print(f"=== {name} ===")
        frac, n = per_band_gradient_energy(model, val_loader, n_max=2000)
        n_bands = len(frac)
        labels = [f"band {i} (σ={ic.SIGMA0 * 2**i:.0f})" for i in range(n_bands - 1)] + [
            "residual"
        ]
        for lbl, f in zip(labels, frac):
            print(f"  {lbl:>22s}: {f:.4f}")
        results[name] = {
            "fractions": frac,
            "labels": labels,
            "n_samples": n,
            "checkpoint": path,
        }

    os.makedirs(os.path.join(ROOT, "results", "phase1_ablations"), exist_ok=True)
    out_json = os.path.join(ROOT, "results", "phase1_ablations", "gradient_spectrum.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_json}")

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    n_bands = len(next(iter(results.values()))["fractions"])
    x = np.arange(n_bands)
    width = 0.27
    colors = ["#888", "#1f77b4", "#2ca02c"]
    for i, (name, r) in enumerate(results.items()):
        ax.bar(x + (i - 1) * width, r["fractions"], width=width, label=name,
               color=colors[i % len(colors)])
    labels = results[list(results)[0]]["labels"]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Fraction of total gradient energy")
    ax.set_title(f"Per-band gradient energy distribution on Imagenette-{RES} val\n"
                 f"(|∂L/∂x| decomposed into {n_bands} Laplacian bands; averaged over "
                 f"~{r['n_samples']} val images)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
    out_png = os.path.join(ROOT, "figures", "gradient_spectrum.png")
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
