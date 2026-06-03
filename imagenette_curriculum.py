"""Imagenette Laplacian-curriculum experiment.

Mirrors the methodology of curriculum_experiment.py (CIFAR-10) but on
Imagenette at 160 resolution with a vanilla ConvNet + AdamW + cosine LR
(not the highly-tuned CIFAR speedrun). The expectation is that the less
saturated baseline gives the frequency curriculum more headroom to help.
"""

import os
import io
import json
import argparse
from math import ceil
from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as tvm
from PIL import Image

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

IMAGENETTE_ROOT = os.environ.get(
    "IMAGENETTE_ROOT",
    "/mnt/c/Users/JoyToy/Documents/Projects/data/imagenette2-320",
)
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "imagenette_cache")
RESOLUTION = 160
PAD = 8  # cache at RESOLUTION + 2*PAD, random-crop to RESOLUTION

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)


############################################
#         Laplacian decomposition          #
############################################

NUM_BANDS = 6  # 5 bandpass + 1 residual; σ doubles from 1 → 16
LEVELS = NUM_BANDS - 1
RESIDUAL_IDX = NUM_BANDS - 1
SIGMA0 = 1.0


def _gaussian_kernel_1d(sigma: float, device, dtype) -> torch.Tensor:
    half = max(1, int(3.0 * sigma + 0.5))
    coords = torch.arange(-half, half + 1, device=device, dtype=dtype)
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    return g / g.sum()


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur — O(K) per output pixel, not O(K²).

    At 160² with σ=16 the 2D kernel would be 97×97 ≈ 9k mults per output;
    separable gives 2·97 ≈ 200, so this is the difference between a 4-minute
    epoch and a 40-second epoch when decomposition runs every batch.
    """
    if sigma <= 0:
        return x
    C = x.shape[1]
    g = _gaussian_kernel_1d(sigma, x.device, x.dtype)
    k_v = g.view(1, 1, -1, 1).expand(C, 1, -1, 1).contiguous()
    k_h = g.view(1, 1, 1, -1).expand(C, 1, 1, -1).contiguous()
    pad = (g.numel() - 1) // 2
    x = F.pad(x, (0, 0, pad, pad), mode="reflect")
    x = F.conv2d(x, k_v, padding=0, groups=C)
    x = F.pad(x, (pad, pad, 0, 0), mode="reflect")
    x = F.conv2d(x, k_h, padding=0, groups=C)
    return x


def laplacian_bands(x: torch.Tensor, levels: int = LEVELS, sigma0: float = SIGMA0):
    bands = []
    current = x
    sigma = sigma0
    for _ in range(levels):
        blurred = gaussian_blur(current, sigma=sigma)
        bands.append(current - blurred)
        current = blurred
        sigma *= 2.0
    bands.append(current)
    return bands


############################################
#            Curriculum schedule           #
############################################

# Same fractional structure as CIFAR script:
#   - last 21.6% of training: all bands active (dropout kicks in if enabled)
#   - first 78.4%: bands unlock one at a time (residual always present)
# With NUM_BANDS=6 (5 to unlock above residual), per-band fraction = 0.784 / 5.
SCHEDULE_TAIL = 0.216
PER_BAND_FRACTION = (1.0 - SCHEDULE_TAIL) / (NUM_BANDS - 1)


@dataclass(frozen=True)
class StageSpec:
    active_bands: tuple
    protect: tuple
    dropout_p: float


def schedule_baseline(step, total_steps):
    return StageSpec(active_bands=tuple(range(NUM_BANDS)),
                     protect=tuple(range(NUM_BANDS)), dropout_p=0.0)


def _make_curriculum(dropout_p_final: float) -> Callable[[int, int], StageSpec]:
    def fn(step, total_steps):
        progress = step / max(1, total_steps)
        bands_to_unlock = min(NUM_BANDS - 1, int(progress / PER_BAND_FRACTION))
        active = tuple(range(NUM_BANDS - 1 - bands_to_unlock, NUM_BANDS))
        all_unlocked = bands_to_unlock == NUM_BANDS - 1
        p = dropout_p_final if all_unlocked else 0.0
        return StageSpec(active_bands=active, protect=(RESIDUAL_IDX,), dropout_p=p)
    return fn


SCHEDULES = {
    "baseline": schedule_baseline,
    "decompose_identity": schedule_baseline,
    "curriculum": _make_curriculum(0.0),
    "curriculum_dropout": _make_curriculum(0.5),
    # Dropout-strength sweep (plateau engine only, fixed engine uses the
    # same _make_curriculum fn — schedule logic identical, the variant
    # name drives plateau phases via variant_phases below).
    "cd_p03": _make_curriculum(0.3),
    "cd_p05": _make_curriculum(0.5),
    "cd_p07": _make_curriculum(0.7),
    "cd_p09": _make_curriculum(0.9),
    # Like cd_p05 but residual is droppable too (no shape-preservation).
    "cd_p05_noprotect": _make_curriculum(0.5),
    # Like curriculum_dropout but no all-bands-without-dropout phase first
    # (dropout starts at unlock of first HF band).
    "cd_early": _make_curriculum(0.5),
    # Variation A: random band-dropout from step 0 (no curriculum, no warmup).
    # Tests whether the curriculum matters or whether dropout alone induces
    # the shape bias.
    "band_drop": _make_curriculum(0.5),
    # Band-dropout strength sweep.
    "band_drop_p03": _make_curriculum(0.3),
    "band_drop_p07": _make_curriculum(0.7),
    # Band-dropout with residual also droppable.
    "band_drop_all": _make_curriculum(0.5),
    "band_drop_all_p07": _make_curriculum(0.7),
    # Variation B: 50/50 joint training on clean image + residual-only
    # (no curriculum, no dropout). Mimics SIN+IN joint training.
    "joint_residual": _make_curriculum(0.0),
    # Variation C: 50/50 joint training on clean image + heavily-blurred
    # (σ=4) image. The blur-aug shape-bias baseline.
    "joint_blur": _make_curriculum(0.0),
    # Joint training with mixed-severity blur (σ ∈ {2, 4, 8} per sample).
    "joint_multiblur": _make_curriculum(0.0),
    # Joint training mixing clean + residual + blur. Three-way mixture.
    "joint_mix": _make_curriculum(0.0),
}


# Plateau-based engine: discrete phases, each trained until val_acc plateaus.
# Phase ordering matches the fractional schedule (residual → +bandpass, HF last).
# The final phase is the all-bands "consolidation" phase where most learning
# should happen; warmup phases are bounded tighter.
def variant_phases(variant_name: str):
    """List of (active_bands, dropout_p, protect_override) per phase.

    protect_override is None to use the default (residual protected when
    dropout_p > 0), or a tuple to override. Used for the no-protect ablation.
    """
    full = tuple(range(NUM_BANDS))
    if variant_name in ("baseline", "decompose_identity"):
        return [(full, 0.0, None)]
    if variant_name == "curriculum":
        return [(tuple(range(NUM_BANDS - 1 - k, NUM_BANDS)), 0.0, None)
                for k in range(NUM_BANDS)]
    if variant_name == "curriculum_dropout":
        phases = [(tuple(range(NUM_BANDS - 1 - k, NUM_BANDS)), 0.0, None)
                  for k in range(NUM_BANDS)]
        phases.append((full, 0.5, None))
        return phases
    # Dropout-strength sweep — same structure as curriculum_dropout, vary p.
    if variant_name in ("cd_p03", "cd_p05", "cd_p07", "cd_p09"):
        p = int(variant_name[-2:]) / 10.0
        phases = [(tuple(range(NUM_BANDS - 1 - k, NUM_BANDS)), 0.0, None)
                  for k in range(NUM_BANDS)]
        phases.append((full, p, None))
        return phases
    if variant_name == "cd_p05_noprotect":
        phases = [(tuple(range(NUM_BANDS - 1 - k, NUM_BANDS)), 0.0, None)
                  for k in range(NUM_BANDS)]
        phases.append((full, 0.5, ()))  # protect nothing → residual droppable
        return phases
    if variant_name == "cd_early":
        # Dropout starts after the FIRST HF band is unlocked (not last phase).
        # Idea: longer total dropout exposure → stronger shape bias.
        phases = []
        for k in range(NUM_BANDS):
            dp = 0.5 if k >= 1 else 0.0  # no dropout in residual-only phase
            phases.append((tuple(range(NUM_BANDS - 1 - k, NUM_BANDS)), dp, None))
        return phases
    if variant_name in AUG_VARIANTS:
        # Single phase, all bands active. Augmentation handled in train loop
        # via variant-specific input transforms.
        return [(full, 0.0, None)]
    raise ValueError(f"unknown variant: {variant_name}")


def _band_dropout(inputs, p, drop_residual, min_kept=0, band_mask='all'):
    """Random per-band, per-sample mask. residual is protected unless drop_residual.

    band_mask:
        'all'     — every band droppable subject to drop_residual (default)
        'hf_only' — only the top half (highest-frequency bands) droppable;
                    lower-frequency bands and residual always kept
        'lf_only' — only the bottom half (lowest-frequency + residual) droppable;
                    high-frequency bands always kept

    The 'hf_only' / 'lf_only' modes are reviewer-defense controls (see
    docs/REVIEWER_DEFENSE.md): they test whether the random multi-band structure
    of band_drop_all is doing the work vs whether the win is operator-family
    memorization.

    If min_kept > 0, rejection-resample any sample whose total kept-droppable
    count is below the threshold.

    min_kept=0 AND band_mask='all' preserves the original control flow and RNG
    sequence bit-for-bit (used for all Phase-1 results).
    """
    bands = laplacian_bands(inputs, levels=LEVELS, sigma0=SIGMA0)
    B = bands[0].shape[0]
    n = len(bands)
    if min_kept == 0 and band_mask == 'all':
        # Original sequential path — kept bit-identical for reproducibility.
        out = torch.zeros_like(bands[0])
        for idx in range(n):
            if idx == n - 1 and not drop_residual:
                out = out + bands[idx]
            else:
                keep = (torch.rand(B, 1, 1, 1, device=inputs.device) > p).to(bands[idx].dtype)
                out = out + keep * bands[idx]
        return out
    # General path: build the explicit droppable-index list.
    half = n // 2
    if band_mask == 'all':
        drop_idx = list(range(n)) if drop_residual else list(range(n - 1))
    elif band_mask == 'hf_only':
        drop_idx = list(range(half))               # bands 0..half-1 (highest freq)
    elif band_mask == 'lf_only':
        drop_idx = list(range(half, n))            # bands half..n-1 (lowest freq + residual)
    else:
        raise ValueError(f"unknown band_mask {band_mask!r}")
    n_drop = len(drop_idx)
    keep_mask = (torch.rand(B, n_drop, device=inputs.device) > p)
    if min_kept > 0:
        # Always-kept bands count toward min_kept; only need (min_kept - that) from droppable.
        always_kept = n - n_drop
        target = max(0, min_kept - always_kept)
        if target > 0:
            bad = keep_mask.sum(dim=1) < target
            for _ in range(8):
                if not bad.any():
                    break
                keep_mask[bad] = torch.rand(int(bad.sum()), n_drop, device=inputs.device) > p
                bad = keep_mask.sum(dim=1) < target
            if bad.any():
                bi = bad.nonzero().view(-1)
                forced = torch.randint(0, n_drop, (bi.numel(),), device=inputs.device)
                keep_mask[bi] = False
                keep_mask[bi, forced] = True
    keep_mask = keep_mask.to(bands[0].dtype)
    out = torch.zeros_like(bands[0])
    droppable_set = set(drop_idx)
    j = 0
    for idx in range(n):
        if idx in droppable_set:
            out = out + keep_mask[:, j].view(B, 1, 1, 1) * bands[idx]
            j += 1
        else:
            out = out + bands[idx]
    return out


def apply_input_aug(variant_name: str, inputs: torch.Tensor) -> torch.Tensor:
    """Per-batch input transform for the non-curriculum augmentation variants."""
    if variant_name == "band_drop":
        return _band_dropout(inputs, p=0.5, drop_residual=False)
    if variant_name == "band_drop_p03":
        return _band_dropout(inputs, p=0.3, drop_residual=False)
    if variant_name == "band_drop_p07":
        return _band_dropout(inputs, p=0.7, drop_residual=False)
    if variant_name == "band_drop_all":
        return _band_dropout(inputs, p=0.5, drop_residual=True)
    if variant_name == "band_drop_all_p07":
        return _band_dropout(inputs, p=0.7, drop_residual=True)
    if variant_name == "joint_residual":
        B = inputs.shape[0]
        bands = laplacian_bands(inputs, levels=LEVELS, sigma0=SIGMA0)
        residual = bands[-1]
        mask = (torch.rand(B, 1, 1, 1, device=inputs.device) < 0.5).to(inputs.dtype)
        return mask * residual + (1 - mask) * inputs
    if variant_name == "joint_blur":
        B = inputs.shape[0]
        blurred = gaussian_blur(inputs, sigma=4.0)
        mask = (torch.rand(B, 1, 1, 1, device=inputs.device) < 0.5).to(inputs.dtype)
        return mask * blurred + (1 - mask) * inputs
    if variant_name == "joint_multiblur":
        # Each sample: 50% clean, 50% blurred at σ ∈ {2, 4, 8}.
        B = inputs.shape[0]
        sigmas = [2.0, 4.0, 8.0]
        # Compute all blurred versions, pick per-sample randomly.
        blurred = [gaussian_blur(inputs, sigma=s) for s in sigmas]
        choice = torch.randint(0, len(sigmas), (B,), device=inputs.device)
        idx_mask = [(choice == i).view(B, 1, 1, 1).to(inputs.dtype)
                    for i in range(len(sigmas))]
        any_blur = sum(idx_mask[i] * blurred[i] for i in range(len(sigmas)))
        keep_clean = (torch.rand(B, 1, 1, 1, device=inputs.device) < 0.5).to(inputs.dtype)
        return keep_clean * inputs + (1 - keep_clean) * any_blur
    if variant_name == "joint_mix":
        # Per-sample: 33% clean, 33% blurred (σ=4), 33% residual-only.
        B = inputs.shape[0]
        bands = laplacian_bands(inputs, levels=LEVELS, sigma0=SIGMA0)
        residual = bands[-1]
        blurred = gaussian_blur(inputs, sigma=4.0)
        choice = torch.randint(0, 3, (B,), device=inputs.device)
        m_clean = (choice == 0).view(B, 1, 1, 1).to(inputs.dtype)
        m_blur = (choice == 1).view(B, 1, 1, 1).to(inputs.dtype)
        m_res = (choice == 2).view(B, 1, 1, 1).to(inputs.dtype)
        return m_clean * inputs + m_blur * blurred + m_res * residual
    return inputs


# Variants whose decomposition is handled by apply_input_aug instead of by
# the assemble_input (StageSpec) path.
AUG_VARIANTS = {"band_drop", "band_drop_p03", "band_drop_p07",
                "band_drop_all", "band_drop_all_p07",
                "joint_residual", "joint_blur", "joint_multiblur", "joint_mix"}


def assemble_input(bands, stage, use_decomposition, original):
    if not use_decomposition:
        return original
    B = bands[0].shape[0]
    device = bands[0].device
    out = torch.zeros_like(bands[0])
    for idx in stage.active_bands:
        if idx in stage.protect or stage.dropout_p == 0.0:
            out = out + bands[idx]
        else:
            keep = (torch.rand(B, 1, 1, 1, device=device) > stage.dropout_p).to(bands[idx].dtype)
            out = out + keep * bands[idx]
    return out


############################################
#                Dataset                   #
############################################

def _load_split_to_tensor(split: str, resolution: int, pad: int) -> tuple:
    """Walk class folders, resize each image to (resolution+2*pad), stack as uint8."""
    side = resolution + 2 * pad
    split_dir = os.path.join(IMAGENETTE_ROOT, "train" if split == "train" else "val")
    class_names = sorted(os.listdir(split_dir))
    images, labels = [], []
    for class_idx, cname in enumerate(class_names):
        cdir = os.path.join(split_dir, cname)
        for fname in sorted(os.listdir(cdir)):
            with Image.open(os.path.join(cdir, fname)) as img:
                img = img.convert("RGB")
                # Resize short edge to `side`, then center-crop to (side, side).
                w, h = img.size
                if w < h:
                    nw = side
                    nh = int(round(h * side / w))
                else:
                    nh = side
                    nw = int(round(w * side / h))
                img = img.resize((nw, nh), Image.BILINEAR)
                left = (nw - side) // 2
                top = (nh - side) // 2
                img = img.crop((left, top, left + side, top + side))
                images.append(torch.from_numpy(_pil_to_uint8_chw(img)))
            labels.append(class_idx)
    images = torch.stack(images)  # N, 3, side, side, uint8
    labels = torch.tensor(labels, dtype=torch.long)
    return images, labels, class_names


def _pil_to_uint8_chw(pil_img):
    import numpy as np
    arr = np.asarray(pil_img, dtype="uint8")  # H, W, 3
    return arr.transpose(2, 0, 1).copy()  # 3, H, W


def get_or_build_cache(split: str, resolution: int = RESOLUTION, pad: int = PAD) -> dict:
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{split}_{resolution}_pad{pad}.pt")
    if os.path.exists(cache_path):
        return torch.load(cache_path, map_location="cpu", weights_only=True)
    images, labels, classes = _load_split_to_tensor(split, resolution, pad)
    blob = {"images": images, "labels": labels, "classes": classes,
            "resolution": resolution, "pad": pad}
    torch.save(blob, cache_path)
    return blob


class ImagenetteLoader:
    """In-GPU, fp16, channels-last loader.

    Pre-loads padded images (uint8) to GPU once, normalizes once per epoch,
    and yields batches via random crops + flips. Matches the in-GPU loader
    style of the CIFAR speedrun script for speed.
    """

    def __init__(self, split: str, batch_size: int, aug: dict | None = None,
                 resolution: int = RESOLUTION, pad: int = PAD,
                 keep_uint8_cpu: bool = False):
        blob = get_or_build_cache(split, resolution, pad)
        self.resolution = resolution
        self.pad = pad
        self.batch_size = batch_size
        self.train = (split == "train")
        self.aug = aug or {}
        self.classes = blob["classes"]

        imgs_uint8 = blob["images"].cuda(non_blocking=True)  # N, 3, side, side
        # Normalize on GPU (fp16). Mean/std broadcast on channel dim.
        mean = IMAGENET_MEAN.view(1, 3, 1, 1).cuda().half()
        std = IMAGENET_STD.view(1, 3, 1, 1).cuda().half()
        self.images = ((imgs_uint8.half() / 255.0) - mean) / std
        self.images = self.images.to(memory_format=torch.channels_last)
        self.labels = blob["labels"].cuda()
        self.n = len(self.labels)
        # Keep an uint8 CPU copy when needed (for corruption eval — corruption
        # functions take HWC uint8 numpy arrays, and we don't want to repeatedly
        # denormalize through the fp16 cache).
        self.images_uint8_cpu = blob["images"] if keep_uint8_cpu else None

    def __len__(self):
        return self.n // self.batch_size if self.train else ceil(self.n / self.batch_size)

    def _random_crop_and_flip(self, batch_padded):
        B, C, H, W = batch_padded.shape
        r = self.pad  # max offset
        # Random crop offsets.
        y = torch.randint(0, 2 * r + 1, (B,), device=batch_padded.device)
        x = torch.randint(0, 2 * r + 1, (B,), device=batch_padded.device)
        by = torch.arange(self.resolution, device=batch_padded.device).view(1, 1, self.resolution, 1)
        bx = torch.arange(self.resolution, device=batch_padded.device).view(1, 1, 1, self.resolution)
        yy = (y.view(B, 1, 1, 1) + by).expand(B, C, self.resolution, self.resolution)
        xx = (x.view(B, 1, 1, 1) + bx).expand(B, C, self.resolution, self.resolution)
        bi = torch.arange(B, device=batch_padded.device).view(B, 1, 1, 1).expand_as(yy)
        ci = torch.arange(C, device=batch_padded.device).view(1, C, 1, 1).expand_as(yy)
        out = batch_padded[bi, ci, yy, xx]
        if self.aug.get("flip", False):
            flip = (torch.rand(B, device=out.device) < 0.5).view(B, 1, 1, 1)
            out = torch.where(flip, out.flip(-1), out)
        return out

    def _center_crop(self, batch_padded):
        return batch_padded[:, :, self.pad:self.pad + self.resolution,
                            self.pad:self.pad + self.resolution]

    def __iter__(self):
        if self.train:
            order = torch.randperm(self.n, device=self.images.device)
            for i in range(len(self)):
                idxs = order[i * self.batch_size:(i + 1) * self.batch_size]
                batch = self.images[idxs].contiguous(memory_format=torch.channels_last)
                batch = self._random_crop_and_flip(batch)
                yield batch.contiguous(memory_format=torch.channels_last), self.labels[idxs]
        else:
            for i in range(len(self)):
                idxs = slice(i * self.batch_size, (i + 1) * self.batch_size)
                batch = self.images[idxs]
                batch = self._center_crop(batch)
                yield batch.contiguous(memory_format=torch.channels_last), self.labels[idxs]


############################################
#                  Model                    #
############################################

class ImagenetteNet(nn.Module):
    """Vanilla ResNet-18 (fp32 params) — train with AMP autocast for fp16.

    Using torchvision's stock ResNet-18 because it's:
      - well-tested and a standard Imagenette baseline,
      - has a stride-2 stem so activations don't blow VRAM at 160² resolution,
      - not hand-tuned to ceiling, which is the whole point of switching away
        from the CIFAR speedrun (curriculum needs headroom to help).
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.net = tvm.resnet18(weights=None, num_classes=num_classes)
        self.to(memory_format=torch.channels_last)

    def reset(self):
        for m in self.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()

    def forward(self, x):
        return self.net(x.to(memory_format=torch.channels_last))


############################################
#               Training                   #
############################################

def cosine_lr(step: int, total_steps: int, base_lr: float, warmup_steps: int) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    import math
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * progress))


def evaluate(model, loader, filter_fn=None) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
        for imgs, labels in loader:
            x = imgs if filter_fn is None else filter_fn(imgs)
            logits = model(x.contiguous(memory_format=torch.channels_last))
            preds = logits.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.numel()
    return correct / total


# ---- Corruption evaluation (Hendrycks ImageNet-C style) ----
# 14 corruption types × chosen severities. Glass_blur is excluded because the
# imagecorruptions library implements it at ~400ms/image, making it dominate
# wall-clock — re-add at final evaluation if needed.
CORRUPTION_TYPES = (
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog",
    "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression",
)


def _build_corruption_eval_loader(resolution: int = RESOLUTION, pad: int = PAD):
    """Val loader that also keeps a CPU uint8 copy for corruption application."""
    return ImagenetteLoader("val", batch_size=128, keep_uint8_cpu=True,
                             resolution=resolution, pad=pad)


def evaluate_corruption(model, corruption_loader, corruption_name: str,
                        severity: int, batch_size: int = 128,
                        n_workers: int = 4, subsample: int | None = None) -> float:
    from imagecorruptions import corrupt
    import numpy as np
    from concurrent.futures import ProcessPoolExecutor
    model.eval()
    pad = corruption_loader.pad
    res = corruption_loader.resolution
    images_uint8 = corruption_loader.images_uint8_cpu  # (N, 3, side, side) cpu uint8
    # Center-crop to (res, res) in HWC order for the corrupt() call.
    cropped = images_uint8[:, :, pad:pad + res, pad:pad + res].permute(0, 2, 3, 1).contiguous().numpy()  # (N, H, W, 3) uint8
    labels = corruption_loader.labels.cpu().numpy()
    if subsample is not None and subsample < len(labels):
        # Deterministic subsample (stratified is not needed for screening).
        rng = np.random.default_rng(0)
        idx = rng.choice(len(labels), size=subsample, replace=False)
        idx.sort()
        cropped = cropped[idx]
        labels = labels[idx]
    n = len(labels)

    mean_g = IMAGENET_MEAN.view(1, 3, 1, 1).cuda().half()
    std_g = IMAGENET_STD.view(1, 3, 1, 1).cuda().half()

    correct = 0
    total = 0
    for i in range(0, n, batch_size):
        batch_hwc = cropped[i:i + batch_size]  # (B, H, W, 3) uint8
        # Apply corruption per image (CPU). Library expects HWC uint8.
        batch_corr = np.empty_like(batch_hwc)
        for j in range(batch_hwc.shape[0]):
            batch_corr[j] = corrupt(batch_hwc[j],
                                    corruption_name=corruption_name,
                                    severity=severity)
        # Convert to GPU normalized fp16.
        t = torch.from_numpy(batch_corr).cuda(non_blocking=True)
        t = t.permute(0, 3, 1, 2).contiguous()  # (B, 3, H, W) uint8
        t = t.half() / 255.0
        t = (t - mean_g) / std_g
        t = t.contiguous(memory_format=torch.channels_last)
        lbl = torch.from_numpy(labels[i:i + batch_size]).cuda(non_blocking=True)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
            logits = model(t)
        preds = logits.argmax(1)
        correct += (preds == lbl).sum().item()
        total += lbl.numel()
    return correct / total


def evaluate_corruption_suite(model, corruption_loader, severities=(1, 3, 5),
                              corruptions=CORRUPTION_TYPES,
                              subsample: int | None = None) -> dict:
    """Returns nested dict: {corruption_name: {severity: acc}, "_mean": float}."""
    results = {}
    all_acc = []
    for c in corruptions:
        results[c] = {}
        for s in severities:
            acc = evaluate_corruption(model, corruption_loader, c, s,
                                      subsample=subsample)
            results[c][s] = acc
            all_acc.append(acc)
    results["_mean"] = sum(all_acc) / len(all_acc)
    if subsample is not None:
        results["_subsample"] = subsample
    return results


def residual_filter(x):
    return laplacian_bands(x, levels=LEVELS, sigma0=SIGMA0)[-1]


def blur_filter(x):
    # Mild Gaussian blur — removes high-freq texture while preserving shape.
    return gaussian_blur(x, sigma=2.0)


def run_one(variant_name: str, schedule_fn: Callable, model: ImagenetteNet,
            epochs: float, batch_size: int = 128, base_lr: float = 3e-3,
            weight_decay: float = 0.05, label_smoothing: float = 0.1,
            log_per_epoch: bool = False) -> dict:
    train_loader = ImagenetteLoader("train", batch_size=batch_size,
                                     aug={"flip": True})
    val_loader = ImagenetteLoader("val", batch_size=256)

    steps_per_epoch = len(train_loader)
    total_steps = ceil(epochs * steps_per_epoch)
    warmup_steps = min(500, ceil(0.05 * total_steps))

    model.reset()
    use_decomp = variant_name != "baseline"

    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr,
                                   weight_decay=weight_decay, fused=True)
    scaler = torch.amp.GradScaler("cuda")

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    starter.record()

    step = 0
    epochs_done = 0
    loss_history = []  # per-step train loss (raw cross-entropy, fp32)
    val_history = []   # per-epoch (epoch, val_acc, val_acc_residual)
    while step < total_steps:
        model.train()
        for inputs, labels in train_loader:
            if step >= total_steps:
                break
            if use_decomp:
                bands = laplacian_bands(inputs, levels=LEVELS, sigma0=SIGMA0)
                stage = schedule_fn(step, total_steps)
                inputs_use = assemble_input(bands, stage, True, inputs)
            else:
                inputs_use = inputs

            lr = cosine_lr(step, total_steps, base_lr, warmup_steps)
            for g in optimizer.param_groups:
                g["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(inputs_use.contiguous(memory_format=torch.channels_last))
                loss = F.cross_entropy(logits, labels, label_smoothing=label_smoothing)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_history.append(loss.detach().float().item())
            step += 1
        epochs_done += 1
        if log_per_epoch:
            va = evaluate(model, val_loader, filter_fn=None)
            vr = evaluate(model, val_loader, filter_fn=residual_filter)
            val_history.append((epochs_done, va, vr))

    ender.record()
    torch.cuda.synchronize()
    time_seconds = 1e-3 * starter.elapsed_time(ender)

    val_acc = evaluate(model, val_loader, filter_fn=None)
    val_acc_residual = evaluate(model, val_loader, filter_fn=residual_filter)
    val_acc_blur = evaluate(model, val_loader, filter_fn=blur_filter)
    return {
        "variant": variant_name,
        "epochs": epochs,
        "val_acc": val_acc,
        "val_acc_residual_at_test": val_acc_residual,
        "val_acc_blur_at_test": val_acc_blur,
        "time_seconds": time_seconds,
        "total_steps": total_steps,
        "loss_history": loss_history,
        "val_history": val_history,
    }


def run_plateau(variant_name: str, model: ImagenetteNet,
                batch_size: int = 128, base_lr: float = 3e-3,
                weight_decay: float = 0.05, label_smoothing: float = 0.1,
                max_warmup_phase: int = 12, max_final_phase: int = 50,
                patience: int = 5, min_phase: int = 5, eps: float = 0.003,
                lr_warmup_epochs: int = 2,
                plateau_lr_threshold: float = 0.3,
                skip_reset: bool = False,
                teacher: nn.Module | None = None,
                distill_temperature: float = 4.0,
                distill_alpha: float = 0.7) -> dict:
    """Plateau-based phase schedule: each phase trains until val_acc plateaus.

    Phases come from variant_phases(); each phase has its own LR schedule
    (linear warmup → cosine decay over max_phase). The final phase is the
    all-bands consolidation phase (and the dropout phase for curriculum_dropout);
    warmup phases are bounded tighter.
    """
    import math
    train_loader = ImagenetteLoader("train", batch_size=batch_size,
                                     aug={"flip": True})
    val_loader = ImagenetteLoader("val", batch_size=256)

    if not skip_reset:
        model.reset()
    is_aug_variant = variant_name in AUG_VARIANTS
    # For aug variants, the input transform is handled by apply_input_aug,
    # not the StageSpec path — so skip the decomposition in the phase loop.
    use_decomp = (variant_name != "baseline") and (not is_aug_variant)
    phases = variant_phases(variant_name)

    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr,
                                   weight_decay=weight_decay, fused=True)
    scaler = torch.amp.GradScaler("cuda")

    steps_per_epoch = len(train_loader)
    history = []  # per-epoch dicts
    phase_summaries = []  # one summary per phase

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    starter.record()

    global_epoch = 0
    last_lr = base_lr

    for phase_idx, phase_tuple in enumerate(phases):
        # phase_tuple is (active_bands, dropout_p, protect_override or None).
        active_bands, dropout_p, protect_override = phase_tuple
        is_last_phase = (phase_idx == len(phases) - 1)
        max_phase = max_final_phase if is_last_phase else max_warmup_phase
        warmup_steps = lr_warmup_epochs * steps_per_epoch
        phase_total_steps = max_phase * steps_per_epoch

        if protect_override is not None:
            protect = protect_override
        else:
            protect = ((RESIDUAL_IDX,) if (use_decomp and dropout_p > 0.0)
                       else tuple(range(NUM_BANDS)))
        stage = StageSpec(active_bands=active_bands, protect=protect,
                          dropout_p=dropout_p)

        best_val = -float("inf")
        epochs_since_best = 0
        step_in_phase = 0
        epoch_in_phase = 0
        plateau_hit = False

        while epoch_in_phase < max_phase:
            model.train()
            for inputs, labels in train_loader:
                if step_in_phase < warmup_steps:
                    lr = base_lr * (step_in_phase + 1) / max(1, warmup_steps)
                else:
                    t = (step_in_phase - warmup_steps) / max(1, phase_total_steps - warmup_steps)
                    t = min(t, 1.0)
                    lr = 0.5 * base_lr * (1.0 + math.cos(math.pi * t))
                for g in optimizer.param_groups:
                    g["lr"] = lr
                last_lr = lr

                if use_decomp:
                    bands = laplacian_bands(inputs, levels=LEVELS, sigma0=SIGMA0)
                    inputs_use = assemble_input(bands, stage, True, inputs)
                elif is_aug_variant:
                    inputs_use = apply_input_aug(variant_name, inputs)
                else:
                    inputs_use = inputs

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    logits = model(inputs_use.contiguous(memory_format=torch.channels_last))
                    loss_ce = F.cross_entropy(logits, labels,
                                               label_smoothing=label_smoothing)
                    if teacher is not None:
                        with torch.no_grad():
                            teacher_logits = teacher(inputs.contiguous(
                                memory_format=torch.channels_last))
                        T = distill_temperature
                        loss_kl = F.kl_div(
                            F.log_softmax(logits / T, dim=-1),
                            F.softmax(teacher_logits / T, dim=-1),
                            reduction="batchmean") * (T * T)
                        loss = (1 - distill_alpha) * loss_ce + distill_alpha * loss_kl
                    else:
                        loss = loss_ce
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                step_in_phase += 1

            val_acc = evaluate(model, val_loader, filter_fn=None)
            history.append({
                "global_epoch": global_epoch,
                "phase_idx": phase_idx,
                "epoch_in_phase": epoch_in_phase,
                "val_acc": val_acc,
                "active_bands": list(active_bands),
                "dropout_p": dropout_p,
                "lr": last_lr,
            })

            if val_acc > best_val + eps:
                best_val = val_acc
                epochs_since_best = 0
            else:
                epochs_since_best += 1

            epoch_in_phase += 1
            global_epoch += 1

            # Only declare plateau once LR has dropped to a fraction of base —
            # otherwise val_acc noise at high LR triggers a false plateau.
            lr_low_enough = (last_lr <= plateau_lr_threshold * base_lr)
            if (epoch_in_phase >= min_phase and epochs_since_best >= patience
                    and lr_low_enough):
                plateau_hit = True
                break

        phase_summaries.append({
            "phase_idx": phase_idx,
            "active_bands": list(active_bands),
            "dropout_p": dropout_p,
            "epochs": epoch_in_phase,
            "best_val_acc": best_val,
            "plateau_hit": plateau_hit,
        })

    ender.record()
    torch.cuda.synchronize()
    time_seconds = 1e-3 * starter.elapsed_time(ender)

    val_acc = evaluate(model, val_loader, filter_fn=None)
    val_acc_residual = evaluate(model, val_loader, filter_fn=residual_filter)
    val_acc_blur = evaluate(model, val_loader, filter_fn=blur_filter)
    return {
        "variant": variant_name,
        "engine": "plateau",
        "total_epochs": global_epoch,
        "val_acc": val_acc,
        "val_acc_residual_at_test": val_acc_residual,
        "val_acc_blur_at_test": val_acc_blur,
        "time_seconds": time_seconds,
        "history": history,
        "phase_summaries": phase_summaries,
        "plateau_params": {
            "max_warmup_phase": max_warmup_phase,
            "max_final_phase": max_final_phase,
            "patience": patience,
            "min_phase": min_phase,
            "eps": eps,
            "lr_warmup_epochs": lr_warmup_epochs,
            "base_lr": base_lr,
            "plateau_lr_threshold": plateau_lr_threshold,
        },
    }


def warmup(model: ImagenetteNet):
    """One short pass to amortize any kernel autotune cost."""
    train_loader = ImagenetteLoader("train", batch_size=128, aug={"flip": True})
    val_loader = ImagenetteLoader("val", batch_size=256)
    model.reset()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=True)
    scaler = torch.amp.GradScaler("cuda")
    for i, (inputs, labels) in enumerate(train_loader):
        bands = laplacian_bands(inputs, levels=LEVELS, sigma0=SIGMA0)
        stage = _make_curriculum(0.5)(0, 100)
        inputs_use = assemble_input(bands, stage, True, inputs)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits = model(inputs_use.contiguous(memory_format=torch.channels_last))
            loss = F.cross_entropy(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if i >= 2:
            break
    evaluate(model, val_loader, filter_fn=None)
    evaluate(model, val_loader, filter_fn=residual_filter)
    evaluate(model, val_loader, filter_fn=blur_filter)


############################################
#               Experiment                 #
############################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--variants", nargs="+", default=list(SCHEDULES.keys()))
    parser.add_argument("--epochs", type=float, default=20.0,
                        help="training epochs (use multiples like 20, 80, 320 for budget sweep)")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--out", type=str, default="imagenette_results.json")
    parser.add_argument("--log-per-epoch", action="store_true",
                        help="record loss per step and val acc per epoch into the JSON")
    parser.add_argument("--engine", choices=["fixed", "plateau"], default="fixed",
                        help="fixed: progress-based schedule, --epochs sets length. "
                             "plateau: train each phase until val_acc plateaus.")
    parser.add_argument("--corruption-eval", choices=["off", "fast", "full"],
                        default="off",
                        help="off: skip; fast: severity 3 only, 1000 subsample; "
                             "full: severities 1,3,5, full val set")
    parser.add_argument("--save-checkpoints", action="store_true",
                        help="save model state dict per run to checkpoints/<variant>_r<i>.pt")
    parser.add_argument("--load-checkpoint", type=str, default=None,
                        help="path to a checkpoint .pt; loaded into model BEFORE each run "
                             "(so each run starts from this pretrained state, no model.reset())")
    parser.add_argument("--lr-finetune", action="store_true",
                        help="reduce base_lr by 10x (use after --load-checkpoint to fine-tune)")
    parser.add_argument("--max-final-phase", type=int, default=50,
                        help="cap epochs for the final/single phase of plateau training")
    parser.add_argument("--distill-teacher", type=str, default=None,
                        help="path to teacher checkpoint .pt; student trains with "
                             "KL distillation from teacher's soft labels on clean inputs.")
    parser.add_argument("--distill-temperature", type=float, default=4.0)
    parser.add_argument("--distill-alpha", type=float, default=0.7,
                        help="weight of KL term (1-alpha for CE)")
    parser.add_argument("--eval-only", action="store_true",
                        help="skip training; just eval the loaded checkpoint")
    parser.add_argument("--resolution", type=int, default=RESOLUTION,
                        help="image resolution for eval/corruption loaders (default 160)")
    parser.add_argument("--pad", type=int, default=PAD,
                        help="crop padding; cache is built at resolution + 2*pad")
    args = parser.parse_args()

    print(f"Imagenette curriculum experiment | engine={args.engine} "
          f"epochs={args.epochs} runs={args.runs} variants={args.variants}")

    model = ImagenetteNet(num_classes=10).cuda().to(memory_format=torch.channels_last)
    print("Warming up...")
    warmup(model)
    print("Warmup complete.\n")

    header = (f"{'variant':<22} {'run':>3} {'val_acc':>9} {'res_test':>9} "
              f"{'blur_test':>9} {'corr_mean':>9} {'ep':>5} {'time_s':>8}")
    print(header)
    print("-" * len(header))

    corruption_loader = None
    if args.corruption_eval != "off":
        corruption_loader = _build_corruption_eval_loader(
            resolution=args.resolution, pad=args.pad)
    if args.save_checkpoints:
        os.makedirs("checkpoints", exist_ok=True)

    effective_lr = args.lr * (0.1 if args.lr_finetune else 1.0)
    if args.load_checkpoint:
        ckpt = torch.load(args.load_checkpoint, map_location="cuda", weights_only=True)
        model.load_state_dict(ckpt["state_dict"])
        print(f"Loaded checkpoint {args.load_checkpoint} "
              f"(reported val_acc {ckpt.get('val_acc', '?')})")
        print(f"Skipping model reset; effective LR = {effective_lr}")

    teacher = None
    if args.distill_teacher:
        teacher = ImagenetteNet(num_classes=10).cuda().to(memory_format=torch.channels_last)
        tckpt = torch.load(args.distill_teacher, map_location="cuda", weights_only=True)
        teacher.load_state_dict(tckpt["state_dict"])
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        print(f"Loaded distillation teacher from {args.distill_teacher} "
              f"(reported val_acc {tckpt.get('val_acc', '?')})")

    all_results = []
    if args.eval_only:
        if not args.load_checkpoint:
            raise SystemExit("--eval-only requires --load-checkpoint")
        # Single eval run; emit a record with the same shape as a training one.
        val_loader = ImagenetteLoader("val", batch_size=256,
                                      resolution=args.resolution, pad=args.pad)
        val_acc = evaluate(model, val_loader, filter_fn=None)
        val_acc_residual = evaluate(model, val_loader, filter_fn=residual_filter)
        val_acc_blur = evaluate(model, val_loader, filter_fn=blur_filter)
        result = {
            "variant": "eval_only",
            "checkpoint": args.load_checkpoint,
            "val_acc": val_acc,
            "val_acc_residual_at_test": val_acc_residual,
            "val_acc_blur_at_test": val_acc_blur,
            "time_seconds": 0.0,
            "total_epochs": 0,
        }
        if corruption_loader is not None:
            if args.corruption_eval == "fast":
                severities, sub = (3,), 1000
            else:
                severities, sub = (1, 3, 5), None
            corr = evaluate_corruption_suite(model, corruption_loader,
                                              severities=severities, subsample=sub)
            result["corruption"] = corr
        all_results.append(result)
        print(f"eval_only checkpoint={args.load_checkpoint}  "
              f"val_acc={result['val_acc']:.4f}  "
              f"res={result['val_acc_residual_at_test']:.4f}  "
              f"blur={result['val_acc_blur_at_test']:.4f}  "
              f"corr={result.get('corruption', {}).get('_mean', float('nan')):.4f}")
        with open(args.out, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Wrote raw results to {args.out}")
        return

    for variant in args.variants:
        if variant not in SCHEDULES:
            print(f"unknown variant: {variant}, skipping")
            continue
        for r in range(args.runs):
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            if args.engine == "plateau":
                # When loading a checkpoint, only the first run uses the loaded
                # weights — subsequent runs would re-init via skip_reset=False,
                # which defeats the purpose. Keep skip_reset=True for all runs
                # but reload checkpoint between runs to ensure consistency.
                if args.load_checkpoint and r > 0:
                    ckpt = torch.load(args.load_checkpoint, map_location="cuda",
                                       weights_only=True)
                    model.load_state_dict(ckpt["state_dict"])
                result = run_plateau(variant, model,
                                     batch_size=args.batch_size,
                                     base_lr=effective_lr,
                                     max_final_phase=args.max_final_phase,
                                     skip_reset=bool(args.load_checkpoint),
                                     teacher=teacher,
                                     distill_temperature=args.distill_temperature,
                                     distill_alpha=args.distill_alpha)
                ep_count = result["total_epochs"]
            else:
                result = run_one(variant, SCHEDULES[variant], model,
                                 epochs=args.epochs, batch_size=args.batch_size,
                                 base_lr=args.lr, log_per_epoch=args.log_per_epoch)
                ep_count = args.epochs
            result["run"] = r

            corr_mean = float("nan")
            if corruption_loader is not None:
                if args.corruption_eval == "fast":
                    severities = (3,)
                    sub = 1000
                else:
                    severities = (1, 3, 5)
                    sub = None
                corr = evaluate_corruption_suite(model, corruption_loader,
                                                  severities=severities,
                                                  subsample=sub)
                result["corruption"] = corr
                corr_mean = corr["_mean"]

            if args.save_checkpoints:
                ckpt_path = os.path.join("checkpoints", f"{variant}_r{r}.pt")
                torch.save({"state_dict": model.state_dict(),
                            "variant": variant, "run": r,
                            "val_acc": result["val_acc"]}, ckpt_path)
                result["checkpoint"] = ckpt_path

            all_results.append(result)
            print(f"{variant:<22} {r:>3} {result['val_acc']:>9.4f} "
                  f"{result['val_acc_residual_at_test']:>9.4f} "
                  f"{result['val_acc_blur_at_test']:>9.4f} "
                  f"{corr_mean:>9.4f} "
                  f"{ep_count:>5} {result['time_seconds']:>8.2f}")

    # Summary.
    print("\n" + "=" * 90)
    print(f"{'variant':<22} {'val_acc':>20} {'res_test':>20} "
          f"{'blur_test':>20}  {'shape_gap':>10}")
    print("-" * 95)
    for variant in args.variants:
        if variant not in SCHEDULES:
            continue
        rs = [r for r in all_results if r["variant"] == variant]
        if not rs:
            continue
        va = torch.tensor([r["val_acc"] for r in rs])
        vr = torch.tensor([r["val_acc_residual_at_test"] for r in rs])
        vb = torch.tensor([r["val_acc_blur_at_test"] for r in rs])
        gap = va.mean() - vr.mean()
        print(f"{variant:<22} {va.mean():>10.4f} ± {va.std():>6.4f}  "
              f"{vr.mean():>10.4f} ± {vr.std():>6.4f}  "
              f"{vb.mean():>10.4f} ± {vb.std():>6.4f}  {gap:>10.4f}")

    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nWrote raw results to {args.out}")


if __name__ == "__main__":
    main()
