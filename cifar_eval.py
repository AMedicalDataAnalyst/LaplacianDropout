"""Full CIFAR evaluation suite for a trained WideResNet checkpoint.

Computes:
  - Clean test accuracy
  - CIFAR-C (15 corruptions × 5 severities) generated on-the-fly via the
    imagecorruptions library; mCE normalized to a supplied vanilla-baseline
    JSON if available, otherwise raw mean accuracy.
  - PGD@ε=2/255, 20 steps, untargeted (CIFAR convention)
  - Calibration (RMS-CE / AURRA) on clean + on the most-severe corruption

All outputs to a single JSON keyed by the checkpoint path. Re-runnable; later
the aggregator (`cifar_aggregate.py`) folds them into a table.

Usage:
  python cifar_eval.py --checkpoint snapshots/augmix_c10_s0/best.pth.tar \\
    --dataset cifar10 --layers 28 --widen-factor 10 \\
    --data-path /mnt/c/.../data --out cifar_eval_augmix_c10_s0.json

CIFAR-32 corruption gen takes ~5 minutes per (corruption × severity) at
batch 256 on a single GPU; full -C eval ≈ 6 hours per checkpoint. To make
the sweep tractable, --c-subsample limits to N images per (corruption × sev)
combination. 1000 is enough for a stable signal; 10000 (full) for the
published number on the winning checkpoint.
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

# augmix/pixmix WRN lives in their third_party. Use the augmix copy.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 'augmix', 'third_party', 'WideResNet_pytorch'))
from wideresnet import WideResNet  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pixmix'))
from calibration_tools import get_measures  # noqa: E402

import torchattacks

CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate',
    'jpeg_compression',
]

# augmix/pixmix CIFAR harnesses use [0.5,0.5,0.5] mean/std on tensors in [0,1].
CIFAR_MEAN = (0.5, 0.5, 0.5)
CIFAR_STD = (0.5, 0.5, 0.5)


def _corrupt(args):
    img_hwc, name, severity = args
    from imagecorruptions import corrupt
    return corrupt(img_hwc, corruption_name=name, severity=severity)


def load_model(arch_kwargs, checkpoint_path):
    """Load a WideResNet from an augmix/pixmix-style checkpoint dict."""
    net = WideResNet(arch_kwargs['layers'],
                     arch_kwargs['num_classes'],
                     arch_kwargs['widen_factor'],
                     arch_kwargs.get('droprate', 0.0))
    blob = torch.load(checkpoint_path, map_location='cuda', weights_only=False)
    sd = blob.get('state_dict', blob)
    sd = {k.replace('module.', '', 1) if k.startswith('module.') else k: v
          for k, v in sd.items()}
    net.load_state_dict(sd)
    return net.cuda().eval()


@torch.no_grad()
def clean_eval(model, test_data, batch_size, workers):
    """Returns (accuracy, confidence-array, correctness-bool-array)."""
    norm = transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
    loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)
    correct = total = 0
    all_conf, all_corr = [], []
    for x, y in loader:
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
        all_conf.append(conf.cpu()); all_corr.append((pred == y).cpu())
    return (correct / total,
            torch.cat(all_conf).numpy(),
            torch.cat(all_corr).numpy())


def _eval_one_corruption_array(model, imgs_uint8, labels, batch_size,
                                 mean_t, std_t):
    """imgs_uint8: (N,32,32,3) uint8. labels: (N,) int. Returns accuracy."""
    correct = total = 0
    with torch.no_grad():
        for i in range(0, len(labels), batch_size):
            batch = torch.from_numpy(imgs_uint8[i:i + batch_size]).cuda()
            batch = batch.permute(0, 3, 1, 2).contiguous().float() / 255.0
            batch = (batch - mean_t) / std_t
            lbl = torch.from_numpy(labels[i:i + batch_size]).cuda()
            preds = model(batch).argmax(dim=1)
            correct += (preds == lbl).sum().item()
            total += lbl.numel()
    return correct / total


def corruption_eval_reference(model, c_dir, batch_size, subsample):
    """Read the canonical Hendrycks CIFAR-C numpy arrays.
    Layout: c_dir/<corruption>.npy = (50000,32,32,3) uint8, with the 50000
    images being 10000 × 5 severities concatenated in order. c_dir/labels.npy
    = (50000,) int. Returns nested dict matching the on-the-fly path."""
    mean_t = torch.tensor(CIFAR_MEAN).view(1, 3, 1, 1).cuda()
    std_t = torch.tensor(CIFAR_STD).view(1, 3, 1, 1).cuda()
    all_labels = np.load(os.path.join(c_dir, 'labels.npy'))
    PER_SEV = len(all_labels) // 5  # = 10000 for CIFAR-10/-100
    results = {}
    for c in CORRUPTIONS:
        path = os.path.join(c_dir, f'{c}.npy')
        if not os.path.exists(path):
            print(f"  [skip] no {path}"); continue
        arr = np.load(path)  # (50000,32,32,3) uint8
        results[c] = {}
        for s in range(1, 6):
            i0, i1 = (s - 1) * PER_SEV, s * PER_SEV
            sev_imgs = arr[i0:i1]; sev_lbl = all_labels[i0:i1]
            if subsample and subsample < PER_SEV:
                rng = np.random.default_rng(0)
                idx = rng.choice(PER_SEV, size=subsample, replace=False)
                idx.sort()
                sev_imgs = sev_imgs[idx]; sev_lbl = sev_lbl[idx]
            t0 = time.time()
            acc = _eval_one_corruption_array(model, sev_imgs, sev_lbl,
                                              batch_size, mean_t, std_t)
            results[c][str(s)] = acc
            print(f"  {c} sev{s}: acc={acc:.4f} ({time.time()-t0:.1f}s)", flush=True)
    all_acc = []
    for c in CORRUPTIONS:
        if c in results:
            all_acc.extend(results[c].values())
    results['_mean'] = float(np.mean(all_acc))
    results['_source'] = 'reference_npy'
    return results


def corruption_eval_onthefly(model, raw_data, executor, batch_size, subsample):
    """Fallback: generate corruptions via imagecorruptions library (used when
    no --c-dir is supplied or it doesn't exist). NOT comparable to published
    numbers — the library applies ImageNet-tuned severity parameters."""
    mean_t = torch.tensor(CIFAR_MEAN).view(1, 3, 1, 1).cuda()
    std_t = torch.tensor(CIFAR_STD).view(1, 3, 1, 1).cuda()
    imgs = raw_data.data  # (N, 32, 32, 3) uint8
    labels = np.array(raw_data.targets)
    if subsample and subsample < len(labels):
        rng = np.random.default_rng(0)
        idx = rng.choice(len(labels), size=subsample, replace=False)
        idx.sort()
        imgs = imgs[idx]; labels = labels[idx]
    n = len(labels)
    results = {}
    for c in CORRUPTIONS:
        results[c] = {}
        for s in range(1, 6):
            t0 = time.time()
            corrupted = list(executor.map(
                _corrupt, ((imgs[i], c, s) for i in range(n)),
                chunksize=32))
            corrupted = np.stack(corrupted)
            acc = _eval_one_corruption_array(model, corrupted, labels,
                                              batch_size, mean_t, std_t)
            results[c][str(s)] = acc
            print(f"  {c} sev{s}: acc={acc:.4f} ({time.time()-t0:.0f}s)", flush=True)
    all_acc = []
    for c in CORRUPTIONS:
        all_acc.extend(results[c].values())
    results['_mean'] = float(np.mean(all_acc))
    results['_source'] = 'imagecorruptions_onthefly'
    return results


def pgd_eval(model, test_data, eps=2/255, steps=20, batch_size=128, workers=2,
             subsample=None):
    mean_t = torch.tensor(CIFAR_MEAN).view(1, 3, 1, 1).cuda()
    std_t = torch.tensor(CIFAR_STD).view(1, 3, 1, 1).cuda()
    normalize = lambda x: (x - mean_t) / std_t  # noqa: E731
    # Build loader without Normalize so the attack works in [0,1] space.
    test_data_raw = type(test_data)(
        test_data.root, train=False, transform=transforms.ToTensor())
    if subsample and subsample < len(test_data_raw):
        test_data_raw = torch.utils.data.Subset(
            test_data_raw,
            torch.randperm(len(test_data_raw),
                           generator=torch.Generator().manual_seed(0))
            [:subsample].tolist())
    loader = torch.utils.data.DataLoader(
        test_data_raw, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)
    atk = torchattacks.PGD(model, eps=eps, alpha=eps / 4, steps=steps,
                            random_start=True)
    atk.set_normalization_used(mean=list(CIFAR_MEAN), std=list(CIFAR_STD))
    correct = total = 0
    for x, y in loader:
        x = x.cuda(non_blocking=True); y = y.cuda(non_blocking=True)
        x_adv = atk(x, y)
        with torch.no_grad():
            pred = model(normalize(x_adv)).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--dataset', choices=['cifar10', 'cifar100'], required=True)
    ap.add_argument('--layers', type=int, default=28)
    ap.add_argument('--widen-factor', type=int, default=10)
    ap.add_argument('--droprate', type=float, default=0.0)
    ap.add_argument('--data-path', default='/mnt/c/Users/JoyToy/Documents/Projects/data')
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--c-subsample', type=int, default=1000,
                    help='# images per (corruption × severity); 10000=full')
    ap.add_argument('--pgd-subsample', type=int, default=2000,
                    help='PGD subsample; full 10k takes ~hour at 20 steps')
    ap.add_argument('--c-dir', default=None,
                    help='path to reference CIFAR-10-C / CIFAR-100-C dir (with '
                    'per-corruption .npy + labels.npy). When set, use the '
                    'canonical Hendrycks data instead of on-the-fly generation.')
    ap.add_argument('--skip-pgd', action='store_true')
    ap.add_argument('--skip-corruption', action='store_true')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    num_classes = 10 if args.dataset == 'cifar10' else 100
    ds_class = datasets.CIFAR10 if args.dataset == 'cifar10' else datasets.CIFAR100

    test_norm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    test_data = ds_class(args.data_path, train=False,
                         transform=test_norm, download=False)
    raw_data = ds_class(args.data_path, train=False,
                        transform=transforms.ToTensor(), download=False)

    print(f"Loading {args.checkpoint}")
    model = load_model({'layers': args.layers, 'widen_factor': args.widen_factor,
                        'droprate': args.droprate, 'num_classes': num_classes},
                       args.checkpoint)

    out = {'checkpoint': args.checkpoint, 'dataset': args.dataset,
           'arch': f'WRN-{args.layers}-{args.widen_factor}',
           'c_subsample': args.c_subsample, 'pgd_subsample': args.pgd_subsample}

    print("Clean eval...")
    acc, conf, corr = clean_eval(model, test_data, args.batch_size, args.workers)
    rms, aurra_v, mad, sf1 = get_measures(conf, corr)
    out['clean'] = {'acc': acc, 'rms_ce': float(rms), 'aurra': float(aurra_v)}
    print(f"  clean acc={acc:.4f}  RMS-CE={100*rms:.2f}  AURRA={100*aurra_v:.2f}")

    if not args.skip_corruption:
        if args.c_dir and os.path.isdir(args.c_dir):
            print(f"Corruption eval (reference CIFAR-C at {args.c_dir})...")
            corr_results = corruption_eval_reference(
                model, args.c_dir, args.batch_size, args.c_subsample)
        else:
            print("Corruption eval (on-the-fly via imagecorruptions; "
                  "WARNING: severity params ImageNet-tuned, not CIFAR-canonical)")
            with ProcessPoolExecutor(max_workers=4) as ex:
                corr_results = corruption_eval_onthefly(
                    model, raw_data, ex, args.batch_size, args.c_subsample)
        out['corruption'] = corr_results
        print(f"  mean corruption acc = {corr_results['_mean']:.4f}")

    if not args.skip_pgd:
        print(f"PGD eval (ε=2/255, 20 steps, subsample={args.pgd_subsample})...")
        pgd_acc = pgd_eval(model, test_data, batch_size=args.batch_size // 2,
                           workers=args.workers, subsample=args.pgd_subsample)
        out['adversarial'] = {'pgd_acc': pgd_acc, 'eps_255': 2, 'steps': 20}
        print(f"  PGD acc = {pgd_acc:.4f}")

    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.out}")


if __name__ == '__main__':
    main()
