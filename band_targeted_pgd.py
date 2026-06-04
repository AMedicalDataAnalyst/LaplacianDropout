"""Band-targeted PGD: adversarial perturbation constrained to a single
Laplacian band.

Standard ℓ∞ PGD picks δ ∈ {||δ||∞ ≤ ε} to maximize loss. Band-targeted
PGD adds the constraint that δ must live entirely in band k of the
Laplacian decomposition (all other bands of δ are zero). This probes the
model's spectral vulnerability profile — a model that relies heavily on
band k will be especially vulnerable to band-k PGD.

The attack is *not* in any training augmentation pipeline (it's
gradient-aligned adversarial perturbation, not random masking) AND not
in any test-time corruption (corruptions are deterministic operators).
So robustness to band-targeted PGD cannot be explained by operator-family
memorization. It directly probes spectral robustness.

Mechanism: at each PGD step, project δ onto the band-k Laplacian subspace
by computing laplacian_bands(δ) and zeroing all entries except band k.
Then clip to ε. The two operations don't perfectly commute (clip can
introduce off-band components) but converge after enough steps.

Usage:
  python band_targeted_pgd.py --checkpoint <path> [--band-target N | --band-target none]
                              [--eps 4] [--steps 20] [--subsample 1000]
                              --out results/.../<file>.json

A `--band-target none` setting runs standard PGD (no band projection).
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

import imagenette_curriculum as ic


def project_to_band(delta: torch.Tensor, band: int, levels: int) -> torch.Tensor:
    """Zero out all bands of `delta` except band `band` (0..levels = residual)."""
    bands = ic.laplacian_bands(delta, levels=levels, sigma0=ic.SIGMA0)
    return bands[band]


def pgd_attack(model, x_clean, y, eps, steps, band_target, levels):
    """ℓ∞-PGD with optional band-target constraint on δ.
    x_clean: normalized fp16 batch (B,3,H,W); y: (B,) long.
    Returns x_adv (still normalized).
    Random start within [-eps, +eps]."""
    alpha = eps / 4.0
    delta = torch.empty_like(x_clean, dtype=torch.float32).uniform_(-eps, eps)
    if band_target is not None:
        delta = project_to_band(delta, band_target, levels)
    delta = delta.clamp_(-eps, eps)
    delta.requires_grad_(True)

    for _ in range(steps):
        x_adv = (x_clean.float() + delta).to(x_clean.dtype)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            loss = F.cross_entropy(model(x_adv), y)
        g, = torch.autograd.grad(loss, delta, retain_graph=False, create_graph=False)
        delta = delta.detach() + alpha * g.sign()
        if band_target is not None:
            delta = project_to_band(delta, band_target, levels)
        delta = delta.clamp_(-eps, eps)
        delta.requires_grad_(True)

    return (x_clean.float() + delta.detach()).to(x_clean.dtype)


def eval_attack(model, loader, eps, steps, band_target, levels, subsample):
    """Returns accuracy under the attack across a (sub)sample."""
    pad = loader.pad
    side = loader.resolution
    correct = total = 0
    seen = 0
    for x_padded, y in loader:
        x = x_padded[:, :, pad:pad + side, pad:pad + side].contiguous(
            memory_format=torch.channels_last
        )
        n = x.shape[0]
        if subsample is not None and seen + n > subsample:
            x = x[: subsample - seen]
            y = y[: subsample - seen]
            n = x.shape[0]
        x_adv = pgd_attack(model, x, y, eps, steps, band_target, levels)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
            preds = model(x_adv).argmax(1)
        correct += (preds == y).sum().item()
        total += n
        seen += n
        if subsample is not None and seen >= subsample:
            break
    return correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--band-target", default=None,
                    help="0..levels (residual) for band-targeted PGD; "
                    "'none' or omit for standard PGD on δ unrestricted")
    ap.add_argument("--eps", type=float, default=4.0,
                    help="ε in 0-255 units; will be divided by 255 and scaled "
                    "by 1/IMAGENET_STD because inputs are normalized")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--subsample", type=int, default=1000)
    ap.add_argument("--resolution", type=int, default=224)
    ap.add_argument("--pad", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # ε is specified in pixel-space 0-255 units. Our inputs are normalized
    # by IMAGENET_STD. So in normalized space, ε must be scaled by
    # 1 / (IMAGENET_STD * 255).
    # Use a scalar mean of IMAGENET_STD across channels (close enough for the
    # ε budget — this is also what torchattacks does internally).
    std_mean = float(ic.IMAGENET_STD.mean())
    eps_norm = (args.eps / 255.0) / std_mean

    band_target = None
    if args.band_target is not None and args.band_target.lower() not in ("none", ""):
        band_target = int(args.band_target)

    levels = ic.LEVELS  # default 5 → 6 bands (0..levels with residual at index `levels`)

    print(f"  checkpoint: {args.checkpoint}")
    print(f"  band-target: {band_target if band_target is not None else 'none (standard PGD)'}")
    print(f"  eps={args.eps}/255 ({eps_norm:.4f} in normalized space)  steps={args.steps}  "
          f"subsample={args.subsample}")

    model = ic.ImagenetteNet(num_classes=10).cuda().to(memory_format=torch.channels_last)
    ckpt = torch.load(args.checkpoint, map_location="cuda", weights_only=True)
    model.load_state_dict(ckpt["state_dict"])
    ic.warmup(model)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    loader = ic.ImagenetteLoader("val", batch_size=64,
                                  resolution=args.resolution, pad=args.pad)

    acc = eval_attack(model, loader, eps_norm, args.steps, band_target,
                       levels, args.subsample)
    print(f"  adv acc = {acc:.4f}")

    out = {
        "checkpoint": args.checkpoint,
        "band_target": band_target if band_target is not None else "standard_pgd",
        "eps_255": args.eps,
        "eps_normalized": eps_norm,
        "steps": args.steps,
        "subsample": args.subsample,
        "adv_acc": acc,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
