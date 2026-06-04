"""Subtractive augmentations on axes orthogonal to Laplacian band dropout.

Companion transforms to `BandDropAll` (band_dropout_transform.py). Each removes
information along an axis non-overlapping with spatial frequency, intended to
*stack* with band dropout for an additive effect:

  - BitDepthReduction : intensity quantization     (per-channel bit planes)
  - PCAColorDropout   : color-statistics eigenbasis (per-pixel channel covariance)

This module ships *both* shapes:

  - **Per-image classes** (`BitDepthReduction`, `PCAColorDropout`) operating on
    a single (C, H, W) tensor. Drop in after `transforms.ToTensor()` +
    `transforms.Normalize(mean, std)` inside torchvision DataLoader pipelines
    (e.g. the augmix harness's preprocess chain).

  - **Batched functional siblings** (`_bit_depth_drop`, `_pca_color_drop`)
    operating on (B, C, H, W). Mirrors the signature pattern of
    `imagenette_curriculum._band_dropout(inputs, p, drop_residual)` so the
    existing `compare_methods.apply_method_aug` dispatcher can stack them
    inline without per-sample Python loops.

Domain: NORMALIZED CHW / BCHW float tensor (post Normalize). These transforms
do NOT clip to [0, 1]. BitDepthReduction is the one exception: it must quantize
in pixel space, so it temporarily de-normalizes, quantizes, and re-normalizes
(taking the same mean/std as the preceding Normalize). PCAColorDropout centers
the data itself, so the Normalize offset is absorbed.

PCAColorDropout deliberately mirrors the BandDropAll API: per-component
Bernoulli keep-mask, `min_kept` with rejection resampling, and `component_mask`
analogous to BandDropAll's `band_mask`. C=3 components for RGB → all-dropped
event ~12.5% at p=0.5 (vs ~1.6% for 6 bands), so `min_kept=1` is materially
more important here than for BandDropAll.

Self-test: `python subtractive_transforms.py`.
"""

import torch

# Default normalization stats (IMAGENET) — used by BitDepthReduction's
# de-normalize round-trip. Imported lazily inside the batched function to
# avoid circular imports if anyone uses this module standalone.
_DEFAULT_MEAN = (0.485, 0.456, 0.406)
_DEFAULT_STD = (0.229, 0.224, 0.225)


def _as_chw(x: torch.Tensor):
    if x.dim() != 3:
        raise ValueError(f"expected (C, H, W) tensor, got shape {tuple(x.shape)}")
    return x


def _mean_std_tensors(mean, std, x):
    m = torch.as_tensor(mean, device=x.device, dtype=x.dtype).view(-1, 1, 1)
    s = torch.as_tensor(std, device=x.device, dtype=x.dtype).view(-1, 1, 1)
    return m, s


# --------------------------------------------------------------------------- #
# Per-image classes (use in torchvision DataLoader pipelines)
# --------------------------------------------------------------------------- #

class BitDepthReduction:
    """Stochastically reduce per-channel bit depth (posterization).

    Quantizes intensities to ``2**bits`` levels in pixel space, removing
    low-order intensity bits. Subtracts information along the
    intensity-quantization axis, independent of both spatial frequency and
    spatial location.

    De-normalizes to [0, 1] before quantizing and re-normalizes after, so bin
    spacing is in true intensity units rather than std-scaled normalized units.

    Parameters
    ----------
    bits_range : tuple of int, default=(2, 5)
        Inclusive range from which retained bit depth is sampled uniformly.
    mean, std : sequence of float
        Per-channel normalization stats used by the preceding Normalize.
    p : float, default=1.0
        Probability of applying the transform at all (else identity).
    """

    def __init__(self, bits_range=(2, 5),
                 mean=_DEFAULT_MEAN, std=_DEFAULT_STD,
                 p: float = 1.0):
        if bits_range[0] < 1:
            raise ValueError("bits must be >= 1")
        self.bits_range = bits_range
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = _as_chw(x)
        if torch.rand((), device=x.device) >= self.p:
            return x
        m, s = _mean_std_tensors(self.mean, self.std, x)
        img = (x * s + m).clamp_(0.0, 1.0)  # clamp guards normalize round-trip overshoot

        bits = int(torch.randint(self.bits_range[0],
                                  self.bits_range[1] + 1, ()).item())
        levels = 2 ** bits
        img = torch.round(img * (levels - 1)) / (levels - 1)
        return (img - m) / s

    def __repr__(self):
        return f"BitDepthReduction(bits_range={self.bits_range}, p={self.p})"


class PCAColorDropout:
    """Independently drop principal components of the per-pixel channel
    covariance.

    Pixels are treated as samples in C-dimensional channel space; PCA is fit
    per-image on their covariance and a per-component Bernoulli keep-mask
    zeroes selected components before reconstruction. Subtracts information
    along the color-statistics axis: dropping the leading component removes
    the dominant chromatic/illumination mode across the whole image while
    leaving all spatial detail intact.

    Operates directly on the normalized tensor (it centers internally).
    Grayscale (C<2) input is returned unchanged.

    Parameters
    ----------
    p : float, default=0.5
        Per-component drop probability (component KEPT with prob 1-p).
    min_kept : int, default=1
        Minimum components that must survive (rejection-resample, then fall
        back to forced single-keep).
    component_mask : {"all","leading","trailing"}, default="all"
        Which components are droppable; the rest always kept.
    apply_p : float, default=1.0
        Probability of applying the transform at all (else identity).
    """

    def __init__(self, p: float = 0.5, min_kept: int = 1,
                 component_mask: str = "all", apply_p: float = 1.0):
        if component_mask not in ("all", "leading", "trailing"):
            raise ValueError(f"unknown component_mask {component_mask!r}")
        self.p = p
        self.min_kept = min_kept
        self.component_mask = component_mask
        self.apply_p = apply_p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = _as_chw(x)
        c = x.shape[0]
        if c < 2 or torch.rand((), device=x.device) >= self.apply_p:
            return x

        flat = x.reshape(c, -1).t()  # (N_pixels, C)
        mean = flat.mean(dim=0, keepdim=True)
        centered = flat - mean
        cov = (centered.t() @ centered) / (centered.shape[0] - 1)
        evals, evecs = torch.linalg.eigh(cov)
        order = torch.argsort(evals, descending=True)
        evecs = evecs[:, order]

        half = c // 2
        if self.component_mask == "all":
            drop_idx = list(range(c))
        elif self.component_mask == "leading":
            drop_idx = list(range(half if half > 0 else 1))
        else:  # "trailing"
            drop_idx = list(range(half, c))
        n_drop = len(drop_idx)
        always_kept = c - n_drop
        target = max(0, self.min_kept - always_kept)

        keep = torch.rand(n_drop, device=x.device) > self.p
        if target > 0:
            for _ in range(8):
                if int(keep.sum()) >= target:
                    break
                keep = torch.rand(n_drop, device=x.device) > self.p
            if int(keep.sum()) < target:
                idx = int(torch.randint(0, n_drop, ()).item())
                keep = torch.zeros(n_drop, dtype=torch.bool, device=x.device)
                keep[idx] = True

        proj = centered @ evecs
        droppable_set = set(drop_idx)
        j = 0
        for comp in range(c):
            if comp in droppable_set:
                if not bool(keep[j]):
                    proj[:, comp] = 0.0
                j += 1

        out = (proj @ evecs.t() + mean).t().reshape(x.shape)
        return out

    def __repr__(self):
        return (f"PCAColorDropout(p={self.p}, min_kept={self.min_kept}, "
                f"component_mask={self.component_mask!r}, apply_p={self.apply_p})")


# --------------------------------------------------------------------------- #
# Batched functional siblings (for compare_methods.apply_method_aug)
# --------------------------------------------------------------------------- #
# Both take a normalized (B, C, H, W) tensor and return one of the same
# shape. Each sample has its own random apply-mask, drop-mask, and per-image
# eigenbasis (for PCA) — matching the per-sample independence semantics of
# `ic._band_dropout`.

def _bit_depth_drop(inputs: torch.Tensor, bits_range=(2, 5), apply_p: float = 1.0,
                    mean=_DEFAULT_MEAN, std=_DEFAULT_STD) -> torch.Tensor:
    """Batched BitDepthReduction. Per-sample bits and per-sample apply mask."""
    if inputs.dim() != 4:
        raise ValueError(f"expected (B,C,H,W), got {tuple(inputs.shape)}")
    B = inputs.shape[0]
    device, dtype = inputs.device, inputs.dtype
    m = torch.as_tensor(mean, device=device, dtype=dtype).view(1, -1, 1, 1)
    s = torch.as_tensor(std, device=device, dtype=dtype).view(1, -1, 1, 1)

    # Per-image bits ∈ [lo, hi]
    lo, hi = bits_range
    bits = torch.randint(lo, hi + 1, (B,), device=device)  # (B,)
    levels = (2 ** bits.float()).view(B, 1, 1, 1)          # (B,1,1,1)

    img = (inputs * s + m).clamp(0.0, 1.0)                  # (B,C,H,W)
    img_q = torch.round(img * (levels - 1)) / (levels - 1)
    out_q = (img_q - m) / s

    # Per-image apply mask
    apply = (torch.rand(B, device=device) < apply_p).view(B, 1, 1, 1).to(dtype)
    return apply * out_q + (1.0 - apply) * inputs


def _pca_color_drop(inputs: torch.Tensor, p: float = 0.5, min_kept: int = 1,
                    component_mask: str = "all",
                    apply_p: float = 1.0) -> torch.Tensor:
    """Batched PCAColorDropout. Per-image eigenbasis + per-image keep mask.

    Math:
      flat: (B,N,C)   N = H*W
      mean: (B,1,C)
      cov:  (B,C,C)  via batched matmul of centered ⊤ × centered / (N-1)
      eigh: (B,C) evals ascending, (B,C,C) evecs (cols = eigenvectors)
      flip last dim of evecs → descending-variance ordering
      proj = centered @ evecs   (B,N,C) — coords in eigenbasis
      zero dropped components, unproject, re-add mean, reshape back
    """
    if inputs.dim() != 4:
        raise ValueError(f"expected (B,C,H,W), got {tuple(inputs.shape)}")
    if component_mask not in ("all", "leading", "trailing"):
        raise ValueError(f"unknown component_mask {component_mask!r}")
    B, C, H, W = inputs.shape
    if C < 2:
        return inputs
    device, dtype = inputs.device, inputs.dtype
    N = H * W

    flat = inputs.reshape(B, C, N).transpose(1, 2)        # (B, N, C)
    mean = flat.mean(dim=1, keepdim=True)                  # (B, 1, C)
    centered = flat - mean                                 # (B, N, C)
    # eigh has no fp16 CUDA kernel. Disable autocast for the *entire* cov+eigh
    # block — otherwise autocast re-casts the matmul output back to fp16
    # despite our explicit .float() inputs.
    with torch.amp.autocast("cuda", enabled=False):
        cf = centered.transpose(1, 2).float() @ centered.float()
        cov32 = cf / max(N - 1, 1)
        evals32, evecs32 = torch.linalg.eigh(cov32)        # (B, C), (B, C, C)
    # eigh returns ascending eigenvalues; flip column ordering for descending.
    evecs = evecs32.flip(dims=(-1,)).to(dtype)             # (B, C, C)

    half = C // 2
    if component_mask == "all":
        drop_idx = list(range(C))
    elif component_mask == "leading":
        drop_idx = list(range(half if half > 0 else 1))
    else:  # trailing
        drop_idx = list(range(half, C))
    n_drop = len(drop_idx)
    always_kept = C - n_drop
    target = max(0, min_kept - always_kept)

    # Per-image per-component keep mask (over the droppable subset)
    keep = torch.rand(B, n_drop, device=device) > p        # (B, n_drop) bool
    if target > 0:
        bad = keep.sum(dim=1) < target
        for _ in range(8):
            if not bad.any():
                break
            keep[bad] = torch.rand(int(bad.sum()), n_drop, device=device) > p
            bad = keep.sum(dim=1) < target
        if bad.any():
            bi = bad.nonzero().view(-1)
            forced = torch.randint(0, n_drop, (bi.numel(),), device=device)
            keep[bi] = False
            keep[bi, forced] = True

    # Build full (B, C) keep mask: always-kept components stay True
    full_keep = torch.ones(B, C, dtype=torch.bool, device=device)
    for j, comp in enumerate(drop_idx):
        full_keep[:, comp] = keep[:, j]

    proj = centered @ evecs                                 # (B, N, C)
    proj = proj * full_keep.unsqueeze(1).to(dtype)          # zero dropped components
    out = proj @ evecs.transpose(1, 2) + mean               # (B, N, C)
    out = out.transpose(1, 2).reshape(B, C, H, W)

    # Per-image apply-or-not
    apply = (torch.rand(B, device=device) < apply_p).view(B, 1, 1, 1).to(dtype)
    return apply * out + (1.0 - apply) * inputs


# --------------------------------------------------------------------------- #
# Self-test. Run:  python subtractive_transforms.py
# --------------------------------------------------------------------------- #
def _selftest():
    torch.manual_seed(0)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    m = torch.tensor(mean).view(-1, 1, 1)
    s = torch.tensor(std).view(-1, 1, 1)

    img01 = torch.rand(3, 64, 64)
    x = (img01 - m) / s

    # === per-image checks (from user's spec) ===
    err = (BitDepthReduction(p=0.0)(x) - x).abs().max().item()
    print(f"[1a] BitDepthReduction p=0 identity: max abs err = {err:.2e}")
    assert err < 1e-6

    err = (PCAColorDropout(apply_p=0.0)(x) - x).abs().max().item()
    print(f"[1b] PCAColorDropout apply_p=0 identity: max abs err = {err:.2e}")
    assert err < 1e-6

    bd = BitDepthReduction(bits_range=(2, 2))(x)
    n_levels = ((bd * s + m).clamp(0, 1))[0].unique().numel()
    print(f"[2] bit_depth=2 distinct levels: {n_levels} (<= 4)")
    assert n_levels <= 4

    err = (PCAColorDropout(p=0.0, min_kept=0)(x) - x).abs().max().item()
    print(f"[3] PCA p=0 round-trip: max abs err = {err:.2e}")
    assert err < 1e-4

    flat0 = PCAColorDropout(p=1.0, min_kept=0)(x).reshape(3, -1)
    per_ch_var = flat0.var(dim=1).max().item()
    print(f"[4] PCA p=1 min_kept=0 -> mean image: max channel var = {per_ch_var:.2e}")
    assert per_ch_var < 1e-8

    out = PCAColorDropout(p=1.0, min_kept=1)(x)
    cen = (out - out.mean(dim=(1, 2), keepdim=True)).reshape(3, -1)
    cr = torch.linalg.matrix_rank(cen, tol=1e-4).item()
    print(f"[5] PCA p=1 min_kept=1 -> channel rank: {cr} (== 1)")
    assert cr == 1

    out = PCAColorDropout(p=1.0, min_kept=0, component_mask="trailing")(x)
    cen = (out - out.mean(dim=(1, 2), keepdim=True)).reshape(3, -1)
    cr = torch.linalg.matrix_rank(cen, tol=1e-4).item()
    print(f"[6] PCA trailing-drop p=1 -> leading kept, channel rank: {cr} (== 1)")
    assert cr == 1

    # === batched-sibling checks (the new bit) ===
    torch.manual_seed(11)
    B = 8
    img01_b = torch.rand(B, 3, 64, 64)
    xb = (img01_b - m.view(1, -1, 1, 1)) / s.view(1, -1, 1, 1)

    # [7] _bit_depth_drop apply_p=0 → identity.
    err = (_bit_depth_drop(xb, apply_p=0.0) - xb).abs().max().item()
    print(f"[7] _bit_depth_drop apply_p=0 identity: max abs err = {err:.2e}")
    assert err < 1e-6

    # [8] _bit_depth_drop bits=(2,2): each sample has <= 4 distinct levels per channel.
    out = _bit_depth_drop(xb, bits_range=(2, 2), apply_p=1.0)
    out_pix = (out * s.view(1, -1, 1, 1) + m.view(1, -1, 1, 1)).clamp(0, 1)
    max_lvls = max(out_pix[b, 0].unique().numel() for b in range(B))
    print(f"[8] batched bit_depth=2: max distinct levels (ch 0) across batch = {max_lvls} (<= 4)")
    assert max_lvls <= 4

    # [9] _pca_color_drop apply_p=0 → identity.
    err = (_pca_color_drop(xb, apply_p=0.0) - xb).abs().max().item()
    print(f"[9] _pca_color_drop apply_p=0 identity: max abs err = {err:.2e}")
    assert err < 1e-6

    # [10] _pca_color_drop p=0, min_kept=0 → identity (no components dropped).
    err = (_pca_color_drop(xb, p=0.0, min_kept=0) - xb).abs().max().item()
    print(f"[10] _pca_color_drop p=0 round-trip: max abs err = {err:.2e}")
    assert err < 5e-4

    # [11] _pca_color_drop p=1, min_kept=0 → mean image per sample.
    out = _pca_color_drop(xb, p=1.0, min_kept=0, apply_p=1.0)
    per_ch_var = out.var(dim=(2, 3)).max().item()
    print(f"[11] batched PCA p=1 min_kept=0 -> per-image mean: max channel var = {per_ch_var:.2e}")
    assert per_ch_var < 1e-5

    # [12] _pca_color_drop p=1, min_kept=1 → each sample's residual is rank-1 in channels.
    out = _pca_color_drop(xb, p=1.0, min_kept=1, apply_p=1.0)
    ranks = []
    for b in range(B):
        cen = (out[b] - out[b].mean(dim=(1, 2), keepdim=True)).reshape(3, -1)
        ranks.append(torch.linalg.matrix_rank(cen, tol=1e-3).item())
    print(f"[12] batched PCA p=1 min_kept=1 -> per-sample ranks: {ranks}")
    assert all(r == 1 for r in ranks)

    # [13] empirical per-component keep rate ~= 1-p (batched).
    torch.manual_seed(2)
    B2, p_test = 4000, 0.5
    rb = torch.rand(B2, 3) > p_test
    rates = rb.float().mean(dim=0).tolist()
    print("[13] per-component keep rate (target 0.50): "
          + ", ".join(f"{r:.3f}" for r in rates))
    assert all(abs(r - (1 - p_test)) < 0.03 for r in rates)

    print("\nAll checks passed.")


if __name__ == "__main__":
    _selftest()
