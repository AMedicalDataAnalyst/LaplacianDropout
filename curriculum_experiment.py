#############################################
#                  Setup                    #
#############################################

import os
import sys
import json
import argparse
from math import ceil
from dataclasses import dataclass, asdict
from typing import Callable

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

torch.backends.cudnn.benchmark = True

CIFAR_RAW_ROOT = r"C:\Users\JoyToy\Documents\Projects\data" if os.name == "nt" else "/mnt/c/Users/JoyToy/Documents/Projects/data"

#############################################
#         Laplacian decomposition           #
#############################################

def _gaussian_kernel_2d(sigma: float, device, dtype) -> torch.Tensor:
    """Truncated separable 2D Gaussian.

    Half-width 3σ retains >99.7% of mass; further truncation costs accuracy,
    further extension is wasted compute at 32×32 where σ=8 already touches
    a quarter of the image.
    """
    half = max(1, int(3.0 * sigma + 0.5))
    coords = torch.arange(-half, half + 1, device=device, dtype=dtype)
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g = g / g.sum()
    return torch.outer(g, g)


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Per-channel 2D Gaussian blur via grouped conv2d.

    Reflect padding matches the offline decomposition convention and avoids
    introducing dark borders that would themselves leak into low-frequency
    bands as boundary artefacts.
    """
    if sigma <= 0:
        return x
    C = x.shape[1]
    k = _gaussian_kernel_2d(sigma, x.device, x.dtype)
    k = k.expand(C, 1, *k.shape)
    pad = k.shape[-1] // 2
    x_padded = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    return F.conv2d(x_padded, k, padding=0, groups=C)


def laplacian_bands(x: torch.Tensor, levels: int = 4, sigma0: float = 1.0):
    """Decompose into per-pixel-resolution bandpass layers plus residual.

    Returns
    -------
    bands : list of Tensor, length ``levels + 1``
        ``bands[0]`` is the highest-frequency band (σ≈``sigma0``).
        ``bands[-1]`` is the low-pass residual.
        Summing reconstructs ``x`` exactly (machine precision).

    Examples
    --------
    >>> x = torch.randn(2, 3, 32, 32)
    >>> b = laplacian_bands(x, levels=4)
    >>> bool(torch.allclose(sum(b), x, atol=1e-3))
    True
    """
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


#############################################
#            Curriculum schedule            #
#############################################

NUM_BANDS = 5
RESIDUAL_IDX = NUM_BANDS - 1


@dataclass(frozen=True)
class StageSpec:
    """Active bands and dropout rate at a given training step.

    ``active_bands`` is high-frequency-first (band 0 = σ≈1, band 4 = residual).
    ``protect`` lists indices exempt from stochastic dropout.
    """
    active_bands: tuple
    protect: tuple
    dropout_p: float


def schedule_baseline(step: int, total_steps: int) -> StageSpec:
    return StageSpec(active_bands=tuple(range(NUM_BANDS)),
                     protect=tuple(range(NUM_BANDS)), dropout_p=0.0)


def schedule_residual_only(step: int, total_steps: int) -> StageSpec:
    return StageSpec(active_bands=(RESIDUAL_IDX,),
                     protect=(RESIDUAL_IDX,), dropout_p=0.0)


def _make_curriculum(dropout_p_final: float) -> Callable[[int, int], StageSpec]:
    """Gradual unlock at ~1.5 epochs per band, dropout only in the final phase.

    For 7.65 epochs total, the user requested ~1.5 epochs per band. With 4
    bandpass bands to introduce above the residual, that fills ~6 epochs
    and leaves ~1.65 epochs of "all-bands" final phase where dropout (if
    enabled) takes effect. Residual is always present from step 0.
    """
    def fn(step: int, total_steps: int) -> StageSpec:
        progress = step / max(1, total_steps)
        bands_to_unlock = min(NUM_BANDS - 1, int(progress / (1.5 / 7.65)))
        active = tuple(range(NUM_BANDS - 1 - bands_to_unlock, NUM_BANDS))
        all_unlocked = bands_to_unlock == NUM_BANDS - 1
        p = dropout_p_final if all_unlocked else 0.0
        return StageSpec(active_bands=active, protect=(RESIDUAL_IDX,), dropout_p=p)
    return fn


SCHEDULES = {
    "baseline": schedule_baseline,
    "decompose_identity": schedule_baseline,
    "curriculum": _make_curriculum(dropout_p_final=0.0),
    "curriculum_dropout": _make_curriculum(dropout_p_final=0.5),
    "residual_only": schedule_residual_only,
}


def assemble_input(bands: list, stage: StageSpec, use_decomposition: bool, original: torch.Tensor) -> torch.Tensor:
    """Reassemble active bands into a single image, applying per-sample dropout.

    ``use_decomposition=False`` returns the original input untouched, providing
    a clean baseline that bypasses any decomposition overhead. All other
    variants pass through the decompose-then-sum path so reconstruction
    quantization (negligible at fp16) is held constant.
    """
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


#############################################
#               Muon optimizer              #
#############################################

@torch.compile(fullgraph=True)
def _zeropower_via_newtonschulz5(gradients_4d, filter_meta_data, max_D, max_K,
                                  current_step, total_steps):
    a, b, c = (3.4576, -4.7391, 2.0843)
    eps_stable = 1e-05
    eps_gms = 1e-05
    progress_ratio = current_step / max(1, total_steps)
    initial_target_mag = 0.5012
    final_target_mag = 0.0786
    target_magnitude = (initial_target_mag * (1 - progress_ratio)
                        + final_target_mag * progress_ratio)
    if not filter_meta_data:
        return gradients_4d
    grad_list = []
    for meta in filter_meta_data:
        original_shape, reshaped_D, reshaped_K, list_idx = meta
        g_reshaped = gradients_4d[list_idx].reshape(reshaped_D, reshaped_K)
        g_padded = F.pad(g_reshaped, (0, max_K - reshaped_K, 0, max_D - reshaped_D),
                         "constant", 0)
        grad_list.append(g_padded)
    if not grad_list:
        return gradients_4d
    X = torch.stack(grad_list)
    current_batch_mags = X.norm(dim=(1, 2), keepdim=True)
    X = X * (target_magnitude / (current_batch_mags + eps_gms))
    X_norm = X.norm(dim=(1, 2), keepdim=True)
    X = X / (X_norm + eps_stable)
    transposed = False
    if X.size(1) > X.size(2):
        X = X.transpose(1, 2)
        transposed = True
    for _ in range(3):
        A = X @ X.transpose(1, 2)
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.transpose(1, 2)
    out = [None] * len(gradients_4d)
    for i, meta in enumerate(filter_meta_data):
        original_shape, reshaped_D, reshaped_K, list_idx = meta
        out[list_idx] = X[i][:reshaped_D, :reshaped_K].view(original_shape)
    return out


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.08, momentum=0.88, nesterov=True, norm_freq=1,
                 total_train_steps=None, weight_decay=0.0,
                 momentum_buffer_dtype=torch.half):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, norm_freq=norm_freq,
                        total_train_steps=total_train_steps, weight_decay=weight_decay,
                        momentum_buffer_dtype=momentum_buffer_dtype)
        super().__init__(params, defaults)
        self.step_count = 0
        self.last_norm_step = 0
        self.total_train_steps = total_train_steps
        self.filter_params_meta = []
        self.max_D, self.max_K = 0, 0
        for group in self.param_groups:
            for p in group["params"]:
                if len(p.shape) == 4 and p.requires_grad:
                    reshaped_D = p.shape[0]
                    reshaped_K = p.data.numel() // p.shape[0]
                    self.filter_params_meta.append({
                        "param": p, "original_shape": p.data.shape,
                        "reshaped_dims": (reshaped_D, reshaped_K)})
                    self.max_D = max(self.max_D, reshaped_D)
                    self.max_K = max(self.max_K, reshaped_K)
        self.max_D = max(1, self.max_D)
        self.max_K = (max(1, self.max_K) + 15) // 16 * 16

    @torch.no_grad()
    def step(self):
        self.step_count += 1
        group = self.param_groups[0]
        group["norm_freq"] = 2 + int(15 * self.step_count / self.total_train_steps)
        filter_params_with_grad = []
        filter_meta_for_current_step = []
        momentum_buffers = [] if group["momentum_buffer_dtype"] == torch.half else None
        for p_meta in self.filter_params_meta:
            p = p_meta["param"]
            if p.grad is not None:
                filter_params_with_grad.append(p)
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(
                        p.grad, dtype=group["momentum_buffer_dtype"],
                        memory_format=torch.preserve_format)
                if momentum_buffers is not None:
                    momentum_buffers.append(state["momentum_buffer"])
                filter_meta_for_current_step.append((
                    p_meta["original_shape"], p_meta["reshaped_dims"][0],
                    p_meta["reshaped_dims"][1], len(filter_params_with_grad) - 1))
        if not filter_params_with_grad:
            return
        if momentum_buffers is not None:
            torch._foreach_mul_(momentum_buffers, group["momentum"])
            grad_casts = [g.to(mb.dtype) for g, mb in zip(
                [p.grad for p in filter_params_with_grad], momentum_buffers)]
            torch._foreach_add_(momentum_buffers, grad_casts)
        else:
            momentum_buffers = [p.grad for p in filter_params_with_grad]
        if group["nesterov"]:
            nesterov_grads = torch._foreach_add(
                [p.grad for p in filter_params_with_grad], momentum_buffers,
                alpha=group["momentum"])
        else:
            nesterov_grads = momentum_buffers
        do_norm_scaling = (self.step_count - self.last_norm_step >= group["norm_freq"])
        if do_norm_scaling:
            self.last_norm_step = self.step_count
            norms = torch._foreach_norm(filter_params_with_grad)
            scale_factors = [(len(p.data) ** 0.5 / (n + 1e-07)).to(p.data.dtype)
                             for p, n in zip(filter_params_with_grad, norms)]
        final = _zeropower_via_newtonschulz5(
            nesterov_grads, filter_meta_for_current_step, self.max_D, self.max_K,
            self.step_count, self.total_train_steps)
        if do_norm_scaling:
            torch._foreach_mul_(filter_params_with_grad, scale_factors)
        torch._foreach_add_(filter_params_with_grad, final, alpha=-group["lr"])
        wd_factor = 1 - group["lr"] * group["weight_decay"]
        if wd_factor != 1.0:
            torch._foreach_mul_(filter_params_with_grad, wd_factor)

    def zero_grad(self, set_to_none: bool = True):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        p.grad.detach_(); p.grad.zero_()


#############################################
#                DataLoader                 #
#############################################

CIFAR_MEAN = torch.tensor((0.4914, 0.4822, 0.4465), dtype=torch.half)
CIFAR_STD = torch.tensor((0.247, 0.2435, 0.2616), dtype=torch.half)


@torch.compile()
def batch_color_jitter(inputs, brightness_range: float, contrast_range: float):
    B = inputs.shape[0]
    device, dtype = inputs.device, inputs.dtype
    bs = (torch.rand(B, 1, 1, 1, device=device, dtype=dtype) * 2 - 1) * brightness_range
    cs = (torch.rand(B, 1, 1, 1, device=device, dtype=dtype) * 2 - 1) * contrast_range + 1
    return (inputs + bs) * cs


@torch.compile()
def batch_flip_lr(inputs):
    flip_mask = (torch.rand(len(inputs), device=inputs.device) < 0.5).view(-1, 1, 1, 1)
    return torch.where(flip_mask, inputs.flip(-1), inputs)


@torch.compile()
def batch_crop(images, crop_size):
    B, C, H_padded, W_padded = images.shape
    r = (H_padded - crop_size) // 2
    y_offsets = (torch.rand(B, device=images.device) * (2 * r + 1)).long()
    x_offsets = (torch.rand(B, device=images.device) * (2 * r + 1)).long()
    by = torch.arange(crop_size, device=images.device).view(1, 1, crop_size, 1)
    bx = torch.arange(crop_size, device=images.device).view(1, 1, 1, crop_size)
    y = (y_offsets.view(B, 1, 1, 1) + by).expand(B, C, crop_size, crop_size)
    x = (x_offsets.view(B, 1, 1, 1) + bx).expand(B, C, crop_size, crop_size)
    bi = torch.arange(B, device=images.device).view(B, 1, 1, 1).expand_as(y)
    ci = torch.arange(C, device=images.device).view(1, C, 1, 1).expand_as(y)
    return images[bi, ci, y, x]


class CifarLoader:
    def __init__(self, path, train=True, batch_size=500, aug=None):
        data_path = os.path.join(path, "train.pt" if train else "test.pt")
        if not os.path.exists(data_path):
            dset = torchvision.datasets.CIFAR10(root=CIFAR_RAW_ROOT, download=False, train=train)
            images = torch.tensor(dset.data)
            labels = torch.tensor(dset.targets)
            os.makedirs(path, exist_ok=True)
            torch.save({"images": images, "labels": labels, "classes": dset.classes}, data_path)
        data = torch.load(data_path, map_location=torch.device("cuda"), weights_only=True)
        self.images, self.labels, self.classes = data["images"], data["labels"], data["classes"]
        self.images = ((self.images.half() / 255).permute(0, 3, 1, 2)
                       .to(memory_format=torch.channels_last))
        self.normalize = T.Normalize(CIFAR_MEAN, CIFAR_STD)
        self.proc_images = {}
        self.epoch = 0
        self.aug = aug or {}
        self.batch_size = batch_size
        self.drop_last = train
        self.shuffle = train
        self._indices = torch.empty(len(self.images), dtype=torch.long, device="cuda")

    def __len__(self):
        return (len(self.images) // self.batch_size if self.drop_last
                else ceil(len(self.images) / self.batch_size))

    def __iter__(self):
        if self.epoch == 0:
            images = self.proc_images["norm"] = self.normalize(self.images)
            if self.aug.get("flip", False):
                images = self.proc_images["flip"] = batch_flip_lr(images)
            pad = self.aug.get("translate", 0)
            if pad > 0:
                self.proc_images["pad"] = F.pad(images, (pad,) * 4, "reflect")
        if self.aug.get("translate", 0) > 0:
            images = batch_crop(self.proc_images["pad"], self.images.shape[-2])
        elif self.aug.get("flip", False):
            images = self.proc_images["flip"]
        else:
            images = self.proc_images["norm"]
        if self.aug.get("flip", False) and self.epoch % 2 == 1:
            images = images.flip(-1)
        cj = self.aug.get("color_jitter", {"enabled": False})
        if cj.get("enabled", False):
            images = batch_color_jitter(images, cj.get("brightness_range", 0.1),
                                         cj.get("contrast_range", 0.1))
        self.epoch += 1
        if self.shuffle:
            torch.randperm(len(self._indices), out=self._indices)
            indices = self._indices
        else:
            indices = torch.arange(len(self.images), device=self.images.device)
        for i in range(len(self)):
            idxs = indices[i * self.batch_size:(i + 1) * self.batch_size]
            yield (images[idxs], self.labels[idxs])


#############################################
#            Network Definition             #
#############################################

class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, momentum=0.5566, eps=1e-12):
        super().__init__(num_features, eps=eps, momentum=1 - momentum)
        self.weight.requires_grad = False


class Conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels, kernel_size=3, padding="same", bias=False)

    def reset_parameters(self):
        super().reset_parameters()
        w = self.weight.data
        torch.nn.init.dirac_(w[:w.size(1)])


class ConvGroup(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.conv1 = Conv(channels_in, channels_out)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = BatchNorm(channels_out)
        self.conv2 = Conv(channels_out, channels_out)
        self.norm2 = BatchNorm(channels_out)
        self.activ = nn.SiLU()

    def forward(self, x):
        x = self.activ(self.norm1(self.pool(self.conv1(x))))
        x = self.activ(self.norm2(self.conv2(x)))
        return x


class CifarNet(nn.Module):
    def __init__(self):
        super().__init__()
        widths = dict(block1=64, block2=256, block3=256)
        whiten_kernel_size = 2
        whiten_width = 2 * 3 * whiten_kernel_size ** 2
        self.whiten = nn.Conv2d(3, whiten_width, whiten_kernel_size, padding=0, bias=True)
        self.whiten.weight.requires_grad = False
        self.layers = nn.Sequential(
            nn.GELU(),
            ConvGroup(whiten_width, widths["block1"]),
            ConvGroup(widths["block1"], widths["block2"]),
            ConvGroup(widths["block2"], widths["block3"]),
            nn.MaxPool2d(3),
        )
        self.head = nn.Linear(widths["block3"], 10, bias=False)
        for mod in self.modules():
            mod.half()
        self.to(memory_format=torch.channels_last)

    def reset(self):
        for m in self.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
        w = self.head.weight.data
        w.mul_(1.0 / w.std())

    def init_whiten(self, train_images, eps=0.0005):
        c, (h, w) = train_images.shape[1], self.whiten.weight.shape[2:]
        patches = (train_images.unfold(2, h, 1).unfold(3, w, 1).transpose(1, 3)
                   .reshape(-1, c, h, w).float())
        pf = patches.view(len(patches), -1)
        cov = torch.mm(pf.t(), pf) / len(pf)
        U, S, V = torch.svd(cov)
        scaled = (U * torch.rsqrt(S + eps).unsqueeze(0)).T.reshape(-1, c, h, w)
        self.whiten.weight.data[:] = torch.cat((scaled, -scaled))

    def forward(self, x, whiten_bias_grad=True):
        x = x.to(memory_format=torch.channels_last)
        b = self.whiten.bias
        x = F.conv2d(x, self.whiten.weight, b if whiten_bias_grad else b.detach())
        x = self.layers(x)
        x = x.view(len(x), -1).contiguous()
        return self.head(x) / x.size(-1)


############################################
#                Training                  #
############################################

def evaluate_with_filter(model, loader, filter_fn=None):
    """Eval at fp16, optionally with input filter (e.g. residual-only)."""
    model.eval()
    test_images = loader.normalize(loader.images)
    if filter_fn is not None:
        test_images = filter_fn(test_images)
    with torch.no_grad():
        logits = torch.cat([model(b.contiguous(memory_format=torch.channels_last))
                            for b in test_images.split(2000)])
    return (logits.argmax(1) == loader.labels).float().mean().item()


def residual_filter(x: torch.Tensor) -> torch.Tensor:
    """Replace input with its residual band (shape-only view)."""
    bands = laplacian_bands(x, levels=4, sigma0=1.0)
    return bands[-1]


def run_one(variant_name: str, schedule_fn: Callable, model: CifarNet) -> dict:
    training_batch_size = 1536
    bias_lr = 0.0573
    head_lr = 0.5415
    wd = 1.0418e-06 * training_batch_size
    test_loader = CifarLoader("cifar10", train=False, batch_size=2000)
    train_loader = CifarLoader(
        "cifar10", train=True, batch_size=training_batch_size,
        aug={"flip": True, "translate": 2,
             "color_jitter": {"enabled": True, "brightness_range": 0.1399,
                              "contrast_range": 0.1308}})

    total_train_steps = ceil(489.6 * len(train_loader))
    whiten_bias_train_steps = ceil(0.2 * len(train_loader))
    model.reset()
    use_decomp = variant_name != "baseline"

    filter_params = [p for p in model.parameters() if len(p.shape) == 4 and p.requires_grad]
    norm_biases = [p for n, p in model.named_parameters() if "norm" in n and p.requires_grad]
    param_configs = [
        dict(params=[model.whiten.bias], lr=bias_lr, weight_decay=wd / bias_lr),
        dict(params=norm_biases, lr=bias_lr, weight_decay=wd / bias_lr),
        dict(params=[model.head.weight], lr=head_lr, weight_decay=wd / head_lr),
    ]
    optimizer1 = torch.optim.SGD(param_configs, momentum=0.825, nesterov=True, fused=True)
    optimizer2 = Muon(filter_params, lr=0.205, momentum=0.655, nesterov=True, norm_freq=4,
                      total_train_steps=total_train_steps, weight_decay=wd,
                      momentum_buffer_dtype=torch.half)
    optimizers = [optimizer1, optimizer2]
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    time_seconds = 0.0
    def start_timer(): starter.record()
    def stop_timer():
        nonlocal time_seconds
        ender.record()
        torch.cuda.synchronize()
        time_seconds += 1e-3 * starter.elapsed_time(ender)

    step = 0
    start_timer()

    with torch.no_grad():
        train_images_raw = train_loader.normalize(train_loader.images[:960])
        if use_decomp:
            init_bands = laplacian_bands(train_images_raw, levels=4, sigma0=1.0)
            init_stage = schedule_fn(0, total_train_steps)
            train_images_init = assemble_input(init_bands, init_stage, True, train_images_raw)
        else:
            train_images_init = train_images_raw
        model.init_whiten(train_images_init)

    lr_factor1_base = 1.0 / max(1, whiten_bias_train_steps)
    lr_factor2_base = 1.0 / total_train_steps
    lr_factor1_initial = optimizer1.param_groups[0]["initial_lr"]
    lr_factors2_initial = [g["initial_lr"] for g in optimizer1.param_groups[1:] + optimizer2.param_groups]

    @torch.compile(mode="max-autotune-no-cudagraphs", fullgraph=True)
    def forward_step(inputs, labels, whiten_bias_grad):
        outputs = model(inputs, whiten_bias_grad=whiten_bias_grad)
        return F.cross_entropy(outputs, labels, label_smoothing=0.09, reduction="sum")

    for epoch in range(ceil(total_train_steps / len(train_loader))):
        model.train()
        for inputs, labels in train_loader:
            if use_decomp:
                bands = laplacian_bands(inputs, levels=4, sigma0=1.0)
                stage = schedule_fn(step, total_train_steps)
                inputs_use = assemble_input(bands, stage, True, inputs)
            else:
                inputs_use = inputs

            whiten_bias_grad = step < whiten_bias_train_steps
            loss = forward_step(inputs_use, labels, whiten_bias_grad)
            loss.backward()

            lr_factor1 = 1 - step * lr_factor1_base
            lr_factor2 = 1 - step * lr_factor2_base
            optimizer1.param_groups[0]["lr"] = lr_factor1_initial * lr_factor1
            for i, group in enumerate(optimizer1.param_groups[1:] + optimizer2.param_groups):
                group["lr"] = lr_factors2_initial[i] * lr_factor2

            for opt in optimizers:
                opt.step()
                opt.zero_grad(set_to_none=True)

            step += 1
            if step >= total_train_steps:
                break
        if step >= total_train_steps:
            break

    stop_timer()

    val_acc = evaluate_with_filter(model, test_loader, filter_fn=None)
    val_acc_residual = evaluate_with_filter(model, test_loader, filter_fn=residual_filter)
    return {
        "variant": variant_name,
        "val_acc": val_acc,
        "val_acc_residual_at_test": val_acc_residual,
        "time_seconds": time_seconds,
    }


def warmup(model: CifarNet):
    """Run one schedule end-to-end with random data to trigger torch.compile.

    The warmup uses the curriculum_dropout schedule because it exercises every
    code path that any variant would use (decomposition, dropout, all bands
    active). Compiling once here avoids per-variant recompile costs in the
    main loop.
    """
    test_loader = CifarLoader("cifar10", train=False, batch_size=2000)
    train_loader = CifarLoader("cifar10", train=True, batch_size=1536,
                               aug={"flip": True, "translate": 2})
    train_loader.labels = torch.randint(0, 10, size=(len(train_loader.labels),),
                                         device="cuda")
    train_loader.images = torch.randn_like(train_loader.images)
    test_loader.labels = torch.randint(0, 10, size=(len(test_loader.labels),),
                                        device="cuda")
    test_loader.images = torch.randn_like(test_loader.images)

    model.reset()
    with torch.no_grad():
        x = train_loader.normalize(train_loader.images[:960])
        bands = laplacian_bands(x, levels=4, sigma0=1.0)
        stage = _make_curriculum(0.5)(0, 100)
        x_init = assemble_input(bands, stage, True, x)
        model.init_whiten(x_init)

    @torch.compile(mode="max-autotune-no-cudagraphs", fullgraph=True)
    def forward_step(inputs, labels, whiten_bias_grad):
        outputs = model(inputs, whiten_bias_grad=whiten_bias_grad)
        return F.cross_entropy(outputs, labels, label_smoothing=0.09, reduction="sum")

    for inputs, labels in train_loader:
        bands = laplacian_bands(inputs, levels=4, sigma0=1.0)
        stage = _make_curriculum(0.5)(0, 100)
        inputs_use = assemble_input(bands, stage, True, inputs)
        loss = forward_step(inputs_use, labels, True)
        loss.backward()
        break

    evaluate_with_filter(model, test_loader, filter_fn=None)
    evaluate_with_filter(model, test_loader, filter_fn=residual_filter)


############################################
#               Experiment                 #
############################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3, help="runs per variant")
    parser.add_argument("--variants", nargs="+", default=list(SCHEDULES.keys()))
    parser.add_argument("--out", type=str, default="curriculum_results.json")
    args = parser.parse_args()

    model = CifarNet().cuda().to(memory_format=torch.channels_last)
    model.compile(mode="max-autotune-no-cudagraphs")

    print("Warming up torch.compile (one-time cost)...")
    warmup(model)
    print("Warmup complete.\n")

    header = f"{'variant':<22} {'run':>3} {'val_acc':>9} {'res_test':>9} {'time_s':>8}"
    print(header)
    print("-" * len(header))

    all_results = []
    for variant in args.variants:
        if variant not in SCHEDULES:
            print(f"unknown variant: {variant}, skipping")
            continue
        for r in range(args.runs):
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            result = run_one(variant, SCHEDULES[variant], model)
            result["run"] = r
            all_results.append(result)
            print(f"{variant:<22} {r:>3} {result['val_acc']:>9.4f} "
                  f"{result['val_acc_residual_at_test']:>9.4f} {result['time_seconds']:>8.2f}")

    print("\n" + "=" * len(header))
    print(f"{'variant':<22} {'val_acc (mean±std)':>22} {'res_test (mean±std)':>22}  {'shape_gap':>10}")
    print("-" * 80)
    for variant in args.variants:
        if variant not in SCHEDULES:
            continue
        rs = [r for r in all_results if r["variant"] == variant]
        if not rs:
            continue
        va = torch.tensor([r["val_acc"] for r in rs])
        vr = torch.tensor([r["val_acc_residual_at_test"] for r in rs])
        gap = va.mean() - vr.mean()
        print(f"{variant:<22} {va.mean():>10.4f} ± {va.std():>6.4f}  "
              f"{vr.mean():>10.4f} ± {vr.std():>6.4f}  {gap:>10.4f}")

    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nWrote raw results to {args.out}")


if __name__ == "__main__":
    main()
