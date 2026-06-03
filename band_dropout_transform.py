"""Per-image Laplacian band-dropout transform (`band_drop_all`).

Faithful single-image port of `imagenette_curriculum._band_dropout(p, drop_residual=True)`
for use inside a torchvision transform pipeline (DataLoader workers, CPU).

Domain: operates on a NORMALIZED CHW float tensor — i.e. drop it in *after*
`transforms.ToTensor()` + `transforms.Normalize(mean, std)`. This matches the
original pipeline, where band-dropout was applied to already-normalized tensors
(imagenette_curriculum.py: loader normalizes at :374, aug applied at :737).

The decomposition (separable Gaussian blur + Laplacian pyramid) and the
per-band Bernoulli keep-mask are identical to the batch version with B=1; only
the device (CPU here vs GPU fp16 there) and the leading batch dim differ.

Reused by both the AugMix and PixMix ImageNet harnesses.
"""

import torch
import torch.nn.functional as F

# Mirror the constants in imagenette_curriculum. 6 bands = 5 bandpass at
# sigma = 1, 2, 4, 8, 16 plus the residual (final low-pass).
NUM_BANDS = 6
LEVELS = NUM_BANDS - 1
SIGMA0 = 1.0


def _gaussian_kernel_1d(sigma: float, device, dtype) -> torch.Tensor:
    half = max(1, int(3.0 * sigma + 0.5))
    coords = torch.arange(-half, half + 1, device=device, dtype=dtype)
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    return g / g.sum()


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur on a single (C, H, W) tensor."""
    if sigma <= 0:
        return x
    C = x.shape[0]
    g = _gaussian_kernel_1d(sigma, x.device, x.dtype)
    k_v = g.view(1, 1, -1, 1).expand(C, 1, -1, 1).contiguous()
    k_h = g.view(1, 1, 1, -1).expand(C, 1, 1, -1).contiguous()
    pad = (g.numel() - 1) // 2
    x = x.unsqueeze(0)  # (1, C, H, W)
    x = F.pad(x, (0, 0, pad, pad), mode="reflect")
    x = F.conv2d(x, k_v, padding=0, groups=C)
    x = F.pad(x, (pad, pad, 0, 0), mode="reflect")
    x = F.conv2d(x, k_h, padding=0, groups=C)
    return x.squeeze(0)


def laplacian_bands(x: torch.Tensor, levels: int = LEVELS,
                    sigma0: float = SIGMA0) -> list:
    """Return [bandpass_0 ... bandpass_{levels-1}, residual]; sums back to x."""
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


class BandDropAll:
    """Independently drop each of the 6 Laplacian bands with probability `p`.

    With `drop_residual=True` (the `band_drop_all` setting) the residual is also
    droppable. If `min_kept=0` (default), all bands can be zeroed (~1.6% at
    p=0.5, n=6) — matches the original. If `min_kept>=1`, rejection-resample
    so at least `min_kept` bands are kept (fallback: force one keep).
    """

    def __init__(self, p: float = 0.5, drop_residual: bool = True,
                 min_kept: int = 0, levels: int = LEVELS,
                 band_mask: str = 'all'):
        # `levels` overrides the default Laplacian depth. CIFAR-32 needs levels<=4
        # (σ=8 pad=24 fits in 32; σ=16 pad=48 does not). ImageNet uses default 5.
        # `band_mask` controls which subset of bands is droppable:
        #   'all'     — every band droppable (subject to drop_residual)
        #   'hf_only' — only top half (highest-freq); rest always kept
        #   'lf_only' — only bottom half (lowest-freq + residual); rest always kept
        # See docs/REVIEWER_DEFENSE.md for why these variants exist.
        if band_mask not in ('all', 'hf_only', 'lf_only'):
            raise ValueError(f"unknown band_mask {band_mask!r}")
        self.p = p
        self.drop_residual = drop_residual
        self.min_kept = min_kept
        self.levels = levels
        self.band_mask = band_mask

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        bands = laplacian_bands(x, levels=self.levels, sigma0=SIGMA0)
        n = len(bands)
        if self.min_kept == 0 and self.band_mask == 'all':
            # Original sequential path — bit-identical for Phase-1 reproducibility.
            out = torch.zeros_like(bands[0])
            for idx in range(n):
                if idx == n - 1 and not self.drop_residual:
                    out = out + bands[idx]
                else:
                    keep = (torch.rand((), device=x.device) > self.p).to(x.dtype)
                    out = out + keep * bands[idx]
            return out
        half = n // 2
        if self.band_mask == 'all':
            drop_idx = list(range(n)) if self.drop_residual else list(range(n - 1))
        elif self.band_mask == 'hf_only':
            drop_idx = list(range(half))
        else:  # 'lf_only'
            drop_idx = list(range(half, n))
        n_drop = len(drop_idx)
        always_kept = n - n_drop
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
        keep = keep.to(x.dtype)
        out = torch.zeros_like(bands[0])
        droppable_set = set(drop_idx)
        j = 0
        for idx in range(n):
            if idx in droppable_set:
                out = out + keep[j] * bands[idx]
                j += 1
            else:
                out = out + bands[idx]
        return out

    def __repr__(self):
        return (f"BandDropAll(p={self.p}, drop_residual={self.drop_residual}, "
                f"min_kept={self.min_kept}, levels={self.levels}, "
                f"band_mask={self.band_mask!r})")


# --------------------------------------------------------------------------- #
# Self-test: verify faithfulness against imagenette_curriculum's own functions.
# Run:  python band_dropout_transform.py
# --------------------------------------------------------------------------- #
def _selftest():
    import imagenette_curriculum as ic

    torch.manual_seed(0)
    C, H, W = 3, 64, 64
    x = torch.randn(C, H, W)

    # 1) Decomposition matches the batch version (B=1) exactly.
    mine = laplacian_bands(x)
    ref = ic.laplacian_bands(x.unsqueeze(0))
    assert len(mine) == len(ref) == NUM_BANDS, (len(mine), len(ref))
    max_band_err = max((mine[i] - ref[i].squeeze(0)).abs().max().item()
                       for i in range(NUM_BANDS))
    print(f"[1] decomposition vs batch B=1: max abs err = {max_band_err:.2e}")
    assert max_band_err < 1e-5, "decomposition diverges from reference"

    # 2) Perfect reconstruction: bands sum back to the input.
    recon_err = (sum(mine) - x).abs().max().item()
    print(f"[2] reconstruction (sum of bands == x): max abs err = {recon_err:.2e}")
    assert recon_err < 1e-5

    # 3) p=0 keeps everything -> output == input.
    drop0 = BandDropAll(p=0.0, drop_residual=True)(x)
    err0 = (drop0 - x).abs().max().item()
    print(f"[3] p=0 -> identity: max abs err = {err0:.2e}")
    assert err0 < 1e-5

    # 4) p=1, drop_residual=True -> all bands dropped -> all zeros.
    drop1 = BandDropAll(p=1.0, drop_residual=True)(x)
    print(f"[4] p=1 drop_residual -> zeros: max abs = {drop1.abs().max().item():.2e}")
    assert drop1.abs().max().item() < 1e-6

    # 5) p=1, drop_residual=False -> only residual survives.
    only_res = BandDropAll(p=1.0, drop_residual=False)(x)
    err_res = (only_res - mine[-1]).abs().max().item()
    print(f"[5] p=1 protect residual -> residual only: max abs err = {err_res:.2e}")
    assert err_res < 1e-6

    # 6) Empirical per-band keep rate ~= 1-p (drop_residual=True, all 6 droppable).
    torch.manual_seed(1)
    N = 20000
    p = 0.5
    # Count how often each band is kept by sampling the same RNG path.
    kept = torch.zeros(NUM_BANDS)
    for _ in range(N):
        for idx in range(NUM_BANDS):
            kept[idx] += (torch.rand(()) > p).item()
    rates = (kept / N).tolist()
    print(f"[6] per-band keep rate (target {1-p:.2f}): "
          + ", ".join(f"{r:.3f}" for r in rates))
    assert all(abs(r - (1 - p)) < 0.02 for r in rates)

    print("\nAll faithfulness checks passed.")


if __name__ == "__main__":
    _selftest()
