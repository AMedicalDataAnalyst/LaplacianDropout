"""Tier-1 comparison of band_drop_all (ours) vs AugMix vs PixMix on Imagenette.

Methods implemented faithfully (with JSD consistency loss where applicable),
trained at matched compute, evaluated on the same Imagenette-C suite.

Also supports combinations:
    band_drop_all + AugMix → both augmentations applied in sequence per sample
    band_drop_all + PixMix → same
"""

import os
import json
import argparse
from math import ceil
from typing import Callable, Optional

import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as tvm
from torchvision.transforms.v2 import AugMix
from PIL import Image
import numpy as np

# Reuse infrastructure from the existing script.
import imagenette_curriculum as ic
import subtractive_transforms as st

# Set by main() from --band-mask. Read by apply_method_aug for band_drop_all variants.
BAND_MASK_OVERRIDE = 'all'

# Subtractive-transform overrides; set by main() from CLI flags.
BIT_DEPTH_RANGE_OVERRIDE = (2, 5)
BIT_DEPTH_APPLY_P_OVERRIDE = 1.0
PCA_DROP_P_OVERRIDE = 0.5
PCA_MIN_KEPT_OVERRIDE = 1
PCA_COMPONENT_MASK_OVERRIDE = 'all'
PCA_APPLY_P_OVERRIDE = 1.0


def _bit_depth(inputs):
    return st._bit_depth_drop(inputs,
                              bits_range=BIT_DEPTH_RANGE_OVERRIDE,
                              apply_p=BIT_DEPTH_APPLY_P_OVERRIDE)


def _pca_color(inputs):
    return st._pca_color_drop(inputs,
                              p=PCA_DROP_P_OVERRIDE,
                              min_kept=PCA_MIN_KEPT_OVERRIDE,
                              component_mask=PCA_COMPONENT_MASK_OVERRIDE,
                              apply_p=PCA_APPLY_P_OVERRIDE)


def _band_drop(inputs):
    return ic._band_dropout(inputs, p=0.5, drop_residual=True,
                            band_mask=BAND_MASK_OVERRIDE)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


############################################
#       Augmentation building blocks       #
############################################

# torchvision.transforms.v2.AugMix runs on uint8 (or float) tensors on GPU.
# We'll wrap it to operate on a normalized fp16 batch by denorm → uint8 → AugMix
# → uint8 → norm fp16, all on-GPU.

_AUGMIX = AugMix(severity=3, mixture_width=3, chain_depth=-1, alpha=1.0,
                 all_ops=True)


def _denorm_to_uint8(x_norm: torch.Tensor) -> torch.Tensor:
    """fp16 normalized (mean/std) channels-last → uint8 contiguous (B, 3, H, W)."""
    mean = ic.IMAGENET_MEAN.view(1, 3, 1, 1).to(x_norm.device, x_norm.dtype)
    std = ic.IMAGENET_STD.view(1, 3, 1, 1).to(x_norm.device, x_norm.dtype)
    x = x_norm.contiguous() * std + mean
    x = (x.float() * 255).clamp(0, 255).to(torch.uint8)
    return x


def _norm_from_uint8(x_uint8: torch.Tensor) -> torch.Tensor:
    """uint8 (B, 3, H, W) → fp16 normalized channels-last."""
    mean = ic.IMAGENET_MEAN.view(1, 3, 1, 1).cuda().half()
    std = ic.IMAGENET_STD.view(1, 3, 1, 1).cuda().half()
    x = x_uint8.half() / 255.0
    x = (x - mean) / std
    return x.contiguous(memory_format=torch.channels_last)


def apply_augmix_batch(x_norm: torch.Tensor) -> torch.Tensor:
    """Apply AugMix to a normalized fp16 batch, return normalized fp16."""
    x_u8 = _denorm_to_uint8(x_norm)
    x_aug = _AUGMIX(x_u8)
    return _norm_from_uint8(x_aug)


# PixMix mixing functions (vectorized over batch).
def _pixmix_get_ab(beta: float, B: int, device) -> tuple:
    a = torch.empty(B, 1, 1, 1, device=device)
    b = torch.empty(B, 1, 1, 1, device=device)
    half = torch.rand(B, device=device) < 0.5
    # Case 1: small-prob a ~ Beta(β,1), b ~ Beta(1,β)
    a_case1 = torch.distributions.Beta(beta, 1.0).sample((B,)).to(device)
    b_case1 = torch.distributions.Beta(1.0, beta).sample((B,)).to(device)
    # Case 2: a = 1+Beta(1,β), b = -Beta(1,β)
    a_case2 = 1.0 + torch.distributions.Beta(1.0, beta).sample((B,)).to(device)
    b_case2 = -torch.distributions.Beta(1.0, beta).sample((B,)).to(device)
    a_v = torch.where(half, a_case1, a_case2).view(B, 1, 1, 1)
    b_v = torch.where(half, b_case1, b_case2).view(B, 1, 1, 1)
    return a_v, b_v


def _pixmix_add(img1, img2, beta):
    # Run in fp32 to avoid fp16 over/underflow with Beta tails.
    a, b = _pixmix_get_ab(beta, img1.shape[0], img1.device)
    img1_f = img1.float() * 2 - 1
    img2_f = img2.float() * 2 - 1
    out = a * img1_f + b * img2_f
    return ((out + 1) / 2).clamp(0, 1)


def _pixmix_multiply(img1, img2, beta):
    a, b = _pixmix_get_ab(beta, img1.shape[0], img1.device)
    # fp32 math; clamp inputs to a value that's representable in fp16 too
    # so negative exponents don't blow up. PixMix paper uses 1e-37 in fp32.
    img1_f = (img1.float() * 2).clamp(min=1e-6)
    img2_f = (img2.float() * 2).clamp(min=1e-6)
    out = (img1_f ** a) * (img2_f ** b)
    return (out / 2).clamp(0, 1)


def apply_pixmix_batch(x_norm: torch.Tensor, mix_set_norm: torch.Tensor,
                       k: int = 4, beta: float = 3.0,
                       all_ops: bool = True) -> torch.Tensor:
    """PixMix on a normalized fp16 batch.

    Args:
        x_norm: (B, 3, H, W) normalized fp16 input.
        mix_set_norm: (M, 3, H, W) normalized fp16 mixing pool (random subset
            of training set, or fractals, etc.).
        k: max number of mixing rounds.
        beta: distribution sharpness parameter.
    """
    B = x_norm.shape[0]
    # Convert to [0,1] range (PixMix operates in image space, not normalized).
    mean = ic.IMAGENET_MEAN.view(1, 3, 1, 1).to(x_norm.device, x_norm.dtype)
    std = ic.IMAGENET_STD.view(1, 3, 1, 1).to(x_norm.device, x_norm.dtype)
    x = (x_norm * std + mean).clamp(0, 1)
    mix = (mix_set_norm * std + mean).clamp(0, 1)

    # Each sample: first decide whether to start from augmented self or self.
    # We approximate "augment_input" by applying AugMix to a copy.
    # (PixMix's augment_input picks one PIL op; AugMix-3 is a richer augmentation
    # but commonly used. For tighter fidelity to the paper, one could swap.)
    x_u8 = (x.float() * 255).clamp(0, 255).to(torch.uint8)
    x_aug_u8 = _AUGMIX(x_u8)
    x_aug = x_aug_u8.half() / 255.0

    start_aug = (torch.rand(B, 1, 1, 1, device=x.device) < 0.5).to(x.dtype)
    cur = start_aug * x_aug + (1 - start_aug) * x

    # Random per-sample number of mixing rounds in [0, k].
    n_rounds = torch.randint(0, k + 1, (B,), device=x.device)
    max_rounds = k

    # Sample mixing partners for every round.
    for r in range(max_rounds):
        active = (n_rounds > r).view(B, 1, 1, 1).to(x.dtype)
        # Half use augmented self, half use mixing-pool image.
        pick_self = (torch.rand(B, 1, 1, 1, device=x.device) < 0.5).to(x.dtype)
        # New aug of self each round.
        cur_u8 = (cur.float() * 255).clamp(0, 255).to(torch.uint8)
        cur_aug = _AUGMIX(cur_u8).half() / 255.0
        # Random mixing partner from pool.
        idx = torch.randint(0, mix.shape[0], (B,), device=x.device)
        partner_pool = mix[idx]
        partner = pick_self * cur_aug + (1 - pick_self) * partner_pool
        # Pick add or multiply, per-sample.
        use_add = (torch.rand(B, 1, 1, 1, device=x.device) < 0.5).to(x.dtype)
        out_add = _pixmix_add(cur, partner, beta)
        out_mul = _pixmix_multiply(cur, partner, beta)
        new_cur = use_add * out_add + (1 - use_add) * out_mul
        new_cur = new_cur.clamp(0, 1)
        # Only apply to active samples.
        cur = active * new_cur + (1 - active) * cur

    # Renormalize to fp16 channels-last.
    out_norm = (cur - mean) / std
    return out_norm.to(torch.float16).contiguous(memory_format=torch.channels_last)


############################################
#         Train one variant                #
############################################

def build_loaders(batch_size: int = 128, resolution: int = 160, pad: int = 8):
    train_loader = ic.ImagenetteLoader("train", batch_size=batch_size,
                                       aug={"flip": True},
                                       resolution=resolution, pad=pad)
    val_loader = ic.ImagenetteLoader("val", batch_size=256,
                                      resolution=resolution, pad=pad)
    return train_loader, val_loader


def make_mix_set(train_loader, n: int = 1000) -> torch.Tensor:
    """Random pool of `n` training images, normalized fp16 channels-last.
    Used as the PixMix mixing set (substitute for fractals)."""
    idx = torch.randperm(train_loader.n)[:n]
    out = train_loader.images[idx]
    # crop to RESOLUTION from padded.
    pad = train_loader.pad
    res = train_loader.resolution
    out = out[:, :, pad:pad + res, pad:pad + res].contiguous(memory_format=torch.channels_last)
    return out


def cosine_lr(step: int, total_steps: int, base_lr: float,
              warmup_steps: int) -> float:
    import math
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    t = min(t, 1.0)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * t))


def apply_ipmix_batch(x_norm: torch.Tensor, mix_set_norm: torch.Tensor,
                      k: int = 3, t: int = 3,
                      patch_sizes: tuple = (8, 16, 32, 64)) -> torch.Tensor:
    """IPMix (Huang et al. 2023): multi-level mixing — pixel + patch + image.

    Faithful simplification: per-sample, chain up to k mixing ops, where each
    op picks a mixing level (pixel / patch / image-level only) and a mixing
    function (add / multiply / random_pixel / random_element). Synthetic
    images come from `mix_set_norm` (substitute for fractals).
    """
    B, C, H, W = x_norm.shape
    mean = ic.IMAGENET_MEAN.view(1, 3, 1, 1).to(x_norm.device, x_norm.dtype)
    std = ic.IMAGENET_STD.view(1, 3, 1, 1).to(x_norm.device, x_norm.dtype)
    # Work in [0,1] image space.
    cur = (x_norm * std + mean).clamp(0, 1)
    mix = (mix_set_norm * std + mean).clamp(0, 1)
    beta = 3.0

    # Image-level: apply AugMix to a working copy first.
    cur_u8 = (cur.float() * 255).clamp(0, 255).to(torch.uint8)
    cur = _AUGMIX(cur_u8).half() / 255.0

    n_rounds = torch.randint(0, k + 1, (B,), device=x_norm.device)
    for r in range(k):
        active = (n_rounds > r).view(B, 1, 1, 1).to(cur.dtype)

        # Sample synthetic partner (or a fresh aug of cur)
        partner_idx = torch.randint(0, mix.shape[0], (B,), device=x_norm.device)
        partner_synth = mix[partner_idx]

        # Per-sample: 50% partner is synth, 50% is freshly-augmented cur
        cur_u8 = (cur.float() * 255).clamp(0, 255).to(torch.uint8)
        cur_aug = _AUGMIX(cur_u8).half() / 255.0
        use_synth = (torch.rand(B, 1, 1, 1, device=x_norm.device) < 0.5).to(cur.dtype)
        partner = use_synth * partner_synth + (1 - use_synth) * cur_aug

        # Pick mixing level: pixel / patch (image-level handled via the
        # AugMix calls above)
        # 0: full pixel-level via _pixmix_add
        # 1: full pixel-level via _pixmix_multiply
        # 2: per-pixel random mask
        # 3: per-element random mask
        # 4: patch-level (random rectangle gets the mix)
        level = torch.randint(0, 5, (B,), device=x_norm.device)
        # Compute all candidates, then select.
        add_out = _pixmix_add(cur, partner, beta)
        mul_out = _pixmix_multiply(cur, partner, beta)
        # Per-pixel random λ mask:
        lam_pix = torch.rand(B, 1, H, W, device=cur.device, dtype=cur.dtype)
        pixel_mask_out = (lam_pix * cur + (1 - lam_pix) * partner).clamp(0, 1)
        # Per-element random λ:
        lam_elt = torch.rand(B, C, H, W, device=cur.device, dtype=cur.dtype)
        elt_mask_out = (lam_elt * cur + (1 - lam_elt) * partner).clamp(0, 1)
        # Patch-level: random rectangle gets the partner (everywhere else stays).
        patch_out = cur.clone()
        for i in range(B):
            ps = patch_sizes[torch.randint(0, len(patch_sizes), (1,)).item()]
            ps = min(ps, H, W)
            y0 = torch.randint(0, H - ps + 1, (1,)).item()
            x0 = torch.randint(0, W - ps + 1, (1,)).item()
            lam = float(torch.distributions.Beta(beta, beta).sample().item())
            patch_out[i, :, y0:y0+ps, x0:x0+ps] = (
                lam * cur[i, :, y0:y0+ps, x0:x0+ps]
                + (1 - lam) * partner[i, :, y0:y0+ps, x0:x0+ps])

        # Select per-sample.
        sel = level.view(B, 1, 1, 1)
        new_cur = torch.where(sel == 0, add_out,
                  torch.where(sel == 1, mul_out,
                  torch.where(sel == 2, pixel_mask_out,
                  torch.where(sel == 3, elt_mask_out, patch_out))))
        new_cur = new_cur.clamp(0, 1)
        cur = active * new_cur + (1 - active) * cur

    out = (cur - mean) / std
    return out.to(torch.float16).contiguous(memory_format=torch.channels_last)


def apply_method_aug(method: str, inputs: torch.Tensor,
                     mix_set: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Per-batch augmentation pipeline for the named comparison method."""
    if method == "baseline":
        return inputs
    if method == "band_drop_all":
        return ic._band_dropout(inputs, p=0.5, drop_residual=True,
                                band_mask=BAND_MASK_OVERRIDE)
    if method == "band_drop_all_keep1":
        return ic._band_dropout(inputs, p=0.5, drop_residual=True, min_kept=1,
                                band_mask=BAND_MASK_OVERRIDE)
    if method == "augmix":
        return apply_augmix_batch(inputs)
    if method == "pixmix":
        return apply_pixmix_batch(inputs, mix_set)
    if method == "band_drop_all+augmix":
        return apply_augmix_batch(ic._band_dropout(inputs, p=0.5, drop_residual=True,
                                                   band_mask=BAND_MASK_OVERRIDE))

    # --- subtractive add-ons (BitDepthReduction, PCAColorDropout) ---
    # Canonical composition order (mirrors band_drop_all+augmix): innermost first.
    # band_drop_all -> AugMix -> bit_depth -> pca_color.
    if method == "bit_depth":
        return _bit_depth(inputs)
    if method == "pca_color":
        return _pca_color(inputs)
    if method == "bit_depth+pca_color":
        return _pca_color(_bit_depth(inputs))
    if method == "band_drop_all+bit_depth":
        return _bit_depth(_band_drop(inputs))
    if method == "band_drop_all+pca_color":
        return _pca_color(_band_drop(inputs))
    if method == "band_drop_all+bit_depth+pca_color":
        return _pca_color(_bit_depth(_band_drop(inputs)))
    if method == "band_drop_all+augmix+bit_depth":
        return _bit_depth(apply_augmix_batch(_band_drop(inputs)))
    if method == "band_drop_all+augmix+pca_color":
        return _pca_color(apply_augmix_batch(_band_drop(inputs)))
    if method == "band_drop_all+augmix+bit_depth+pca_color":
        return _pca_color(_bit_depth(apply_augmix_batch(_band_drop(inputs))))
    if method == "augmix+bit_depth":
        return _bit_depth(apply_augmix_batch(inputs))
    if method == "augmix+pca_color":
        return _pca_color(apply_augmix_batch(inputs))
    if method == "augmix+bit_depth+pca_color":
        return _pca_color(_bit_depth(apply_augmix_batch(inputs)))
    if method == "band_drop_all+pixmix":
        return apply_pixmix_batch(
            ic._band_dropout(inputs, p=0.5, drop_residual=True), mix_set)
    if method == "augmix+band_drop_all":
        return ic._band_dropout(apply_augmix_batch(inputs), p=0.5, drop_residual=True)
    if method == "ipmix":
        return apply_ipmix_batch(inputs, mix_set)
    if method == "band_drop_all+ipmix":
        return apply_ipmix_batch(ic._band_dropout(inputs, p=0.5, drop_residual=True),
                                  mix_set)
    raise ValueError(f"unknown method: {method}")


def train_one(method: str, model: ic.ImagenetteNet,
              max_epochs: int = 50, base_lr: float = 3e-3,
              weight_decay: float = 0.05, label_smoothing: float = 0.1,
              jsd_lambda: float = 12.0, use_jsd: bool = True,
              patience: int = 6, min_phase: int = 8, eps: float = 0.002,
              plateau_lr_threshold: float = 0.3,
              lr_warmup_epochs: int = 2,
              batch_size: int = 128,
              resolution: int = 160, pad: int = 8,
              skip_reset: bool = False) -> dict:
    train_loader, val_loader = build_loaders(batch_size, resolution=resolution, pad=pad)
    mix_set = (make_mix_set(train_loader)
                if any(t in method for t in ("pixmix", "ipmix")) else None)

    if not skip_reset:
        model.reset()

    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr,
                                   weight_decay=weight_decay, fused=True)
    scaler = torch.amp.GradScaler("cuda")

    steps_per_epoch = len(train_loader)
    warmup_steps = lr_warmup_epochs * steps_per_epoch
    phase_total_steps = max_epochs * steps_per_epoch

    history = []
    best_val = -float("inf")
    epochs_since_best = 0
    step = 0
    epoch = 0
    last_lr = base_lr
    plateau_hit = False

    while epoch < max_epochs:
        model.train()
        for inputs, labels in train_loader:
            lr = cosine_lr(step, phase_total_steps, base_lr, warmup_steps)
            for g in optimizer.param_groups:
                g["lr"] = lr
            last_lr = lr

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                if use_jsd and method != "baseline":
                    aug1 = apply_method_aug(method, inputs, mix_set)
                    aug2 = apply_method_aug(method, inputs, mix_set)
                    cat = torch.cat([inputs.contiguous(memory_format=torch.channels_last),
                                     aug1, aug2], dim=0)
                    logits_all = model(cat)
                    L = inputs.shape[0]
                    lc = logits_all[:L]
                    la = logits_all[L:2 * L]
                    lb = logits_all[2 * L:]
                    loss_ce = F.cross_entropy(lc, labels,
                                               label_smoothing=label_smoothing)
                    pc = F.softmax(lc, dim=1)
                    pa = F.softmax(la, dim=1)
                    pb = F.softmax(lb, dim=1)
                    pm = torch.clamp((pc + pa + pb) / 3.0, 1e-7, 1.0).log()
                    loss_jsd = (F.kl_div(pm, pc, reduction="batchmean")
                                + F.kl_div(pm, pa, reduction="batchmean")
                                + F.kl_div(pm, pb, reduction="batchmean")) / 3.0
                    loss = loss_ce + jsd_lambda * loss_jsd
                else:
                    aug_inp = apply_method_aug(method, inputs, mix_set)
                    logits = model(aug_inp.contiguous(memory_format=torch.channels_last))
                    loss = F.cross_entropy(logits, labels,
                                           label_smoothing=label_smoothing)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            step += 1

        # Eval after each epoch.
        val_acc = ic.evaluate(model, val_loader, filter_fn=None)
        history.append({"epoch": epoch, "val_acc": val_acc, "lr": last_lr})
        if val_acc > best_val + eps:
            best_val = val_acc
            epochs_since_best = 0
        else:
            epochs_since_best += 1
        epoch += 1

        lr_low = (last_lr <= plateau_lr_threshold * base_lr)
        if epoch >= min_phase and epochs_since_best >= patience and lr_low:
            plateau_hit = True
            break

    final_val = ic.evaluate(model, val_loader, filter_fn=None)
    final_res = ic.evaluate(model, val_loader, filter_fn=ic.residual_filter)
    final_blur = ic.evaluate(model, val_loader, filter_fn=ic.blur_filter)
    return {
        "method": method,
        "max_epochs": max_epochs,
        "epochs_done": epoch,
        "plateau_hit": plateau_hit,
        "use_jsd": use_jsd,
        "val_acc": final_val,
        "val_acc_residual_at_test": final_res,
        "val_acc_blur_at_test": final_blur,
        "history": history,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--max-epochs", type=int, default=50)
    ap.add_argument("--use-jsd", action="store_true")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--corruption-eval", choices=["off", "fast", "full"],
                    default="fast")
    ap.add_argument("--save-checkpoints", action="store_true")
    ap.add_argument("--load-checkpoint", type=str, default=None,
                    help="path to a starting checkpoint; --skip-reset implied")
    ap.add_argument("--lr-finetune", action="store_true",
                    help="reduce base_lr by 10x")
    ap.add_argument("--out", type=str, default="comparison_results.json")
    ap.add_argument("--resolution", type=int, default=160,
                    help="image resolution (default 160; standard is 224)")
    ap.add_argument("--pad", type=int, default=8,
                    help="random-crop padding (default 8; use 16 for 224)")
    ap.add_argument("--band-levels", type=int, default=None,
                    help="override the Laplacian level count for band_drop_all "
                    "ablation. Total bands = levels + 1 (residual). Default "
                    "keeps the module default (5 levels = 6 bands).")
    ap.add_argument("--band-mask", default='all',
                    choices=['all', 'hf_only', 'lf_only'],
                    help="reviewer-defense control: 'all' = current default; "
                    "'hf_only' = drop only the top half of bands (highest freq); "
                    "'lf_only' = drop only the bottom half. See "
                    "docs/REVIEWER_DEFENSE.md.")
    ap.add_argument("--seed-offset", type=int, default=0,
                    help="explicit base seed; run i uses seed (offset + i). "
                    "Added 2026-06-03 to fix a methodological gap — pre-fix "
                    "runs relied on RNG advancing between model.reset() calls, "
                    "which produced different inits within an invocation but "
                    "non-reproducible across invocations. Default 0 keeps "
                    "the run-0 init close to pre-fix behavior in practice.")
    # --- Subtractive-transform args (subtractive_transforms.py) ---
    ap.add_argument("--bit-depth-bits-range", nargs=2, type=int, default=[2, 5],
                    metavar=('LO', 'HI'),
                    help="bits sampled uniformly from [LO, HI] inclusive per image "
                    "for BitDepthReduction (default 2 5).")
    ap.add_argument("--bit-depth-apply-p", type=float, default=1.0,
                    help="probability of applying BitDepthReduction per image (default 1.0)")
    ap.add_argument("--pca-drop-p", type=float, default=0.5,
                    help="per-component drop probability for PCAColorDropout (default 0.5)")
    ap.add_argument("--pca-min-kept", type=int, default=1,
                    help="min components kept per image (default 1; "
                    "default 1 protects against the ~12.5%% all-dropped event "
                    "for C=3 at p=0.5)")
    ap.add_argument("--pca-component-mask", default='all',
                    choices=['all', 'leading', 'trailing'],
                    help="which PCA components are droppable; 'all' default, "
                    "'leading' = top-half (highest variance), 'trailing' = bottom-half")
    ap.add_argument("--pca-apply-p", type=float, default=1.0,
                    help="probability of applying PCAColorDropout per image (default 1.0)")
    args = ap.parse_args()

    global BAND_MASK_OVERRIDE
    global BIT_DEPTH_RANGE_OVERRIDE, BIT_DEPTH_APPLY_P_OVERRIDE
    global PCA_DROP_P_OVERRIDE, PCA_MIN_KEPT_OVERRIDE
    global PCA_COMPONENT_MASK_OVERRIDE, PCA_APPLY_P_OVERRIDE
    BAND_MASK_OVERRIDE = args.band_mask
    BIT_DEPTH_RANGE_OVERRIDE = tuple(args.bit_depth_bits_range)
    BIT_DEPTH_APPLY_P_OVERRIDE = args.bit_depth_apply_p
    PCA_DROP_P_OVERRIDE = args.pca_drop_p
    PCA_MIN_KEPT_OVERRIDE = args.pca_min_kept
    PCA_COMPONENT_MASK_OVERRIDE = args.pca_component_mask
    PCA_APPLY_P_OVERRIDE = args.pca_apply_p
    if args.band_mask != 'all':
        print(f"Overriding band_mask = {BAND_MASK_OVERRIDE!r}")
    if any(m in (','.join(args.methods)) for m in ('bit_depth', 'pca_color')):
        print(f"Subtractive: bit_depth bits={BIT_DEPTH_RANGE_OVERRIDE} apply_p={BIT_DEPTH_APPLY_P_OVERRIDE} "
              f"| pca p={PCA_DROP_P_OVERRIDE} min_kept={PCA_MIN_KEPT_OVERRIDE} "
              f"mask={PCA_COMPONENT_MASK_OVERRIDE!r} apply_p={PCA_APPLY_P_OVERRIDE}")

    if args.band_levels is not None:
        ic.LEVELS = args.band_levels
        ic.NUM_BANDS = args.band_levels + 1
        ic.RESIDUAL_IDX = args.band_levels
        ic.PER_BAND_FRACTION = (1.0 - ic.SCHEDULE_TAIL) / max(1, ic.NUM_BANDS - 1)
        print(f"Overriding ic.LEVELS = {ic.LEVELS} (band count = {ic.NUM_BANDS})")

    print(f"compare_methods | methods={args.methods} runs={args.runs} "
          f"max_epochs={args.max_epochs} jsd={args.use_jsd}")

    model = ic.ImagenetteNet(num_classes=10).cuda().to(memory_format=torch.channels_last)
    print("Warming up...")
    ic.warmup(model)
    print("Warmup complete.\n")

    corruption_loader = None
    if args.corruption_eval != "off":
        corruption_loader = ic._build_corruption_eval_loader(
            resolution=args.resolution, pad=args.pad)
    if args.save_checkpoints:
        os.makedirs("checkpoints", exist_ok=True)

    effective_lr = args.lr * (0.1 if args.lr_finetune else 1.0)

    header = (f"{'method':<24} {'run':>3} {'val_acc':>9} {'res':>9} "
              f"{'blur':>9} {'corr':>9} {'ep':>4}")
    print(header)
    print("-" * len(header))

    all_results = []
    for method in args.methods:
        for r in range(args.runs):
            seed = args.seed_offset + r
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            if args.load_checkpoint:
                ckpt = torch.load(args.load_checkpoint, map_location="cuda",
                                   weights_only=True)
                model.load_state_dict(ckpt["state_dict"])
            result = train_one(method, model,
                                max_epochs=args.max_epochs,
                                base_lr=effective_lr,
                                use_jsd=args.use_jsd,
                                batch_size=args.batch_size,
                                resolution=args.resolution, pad=args.pad,
                                skip_reset=bool(args.load_checkpoint))
            result["run"] = r
            result["seed"] = seed
            result["band_levels"] = ic.LEVELS
            result["resolution"] = args.resolution
            corr_mean = float("nan")
            if corruption_loader is not None:
                sevs = (3,) if args.corruption_eval == "fast" else (1, 3, 5)
                sub = 1000 if args.corruption_eval == "fast" else None
                corr = ic.evaluate_corruption_suite(model, corruption_loader,
                                                     severities=sevs, subsample=sub)
                result["corruption"] = corr
                corr_mean = corr["_mean"]
            if args.save_checkpoints:
                ckpt_path = os.path.join("checkpoints",
                                          f"{method.replace('+', '_')}_r{r}.pt")
                torch.save({"state_dict": model.state_dict(),
                            "method": method, "run": r,
                            "val_acc": result["val_acc"]}, ckpt_path)
                result["checkpoint"] = ckpt_path
            all_results.append(result)
            print(f"{method:<24} {r:>3} {result['val_acc']:>9.4f} "
                  f"{result['val_acc_residual_at_test']:>9.4f} "
                  f"{result['val_acc_blur_at_test']:>9.4f} "
                  f"{corr_mean:>9.4f} {result['epochs_done']:>4}")

    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
