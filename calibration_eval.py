"""Calibration eval (RMS-CE, AURRA) for trained ImageNet models.

Thin adapter that re-uses `calibration_tools.get_measures` from the pixmix
repo (so the metric implementation is the standard one) and optionally runs
over the ImageNet-C corruption subdirs too — the PixMix paper reports
calibration both on clean val and under corruption.

Usage:
  python calibration_eval.py --checkpoint ckpt.pth --arch resnet50 \\
      --val-dir data/imagenet/val --c-dir data/imagenet_c/ \\
      --out calibration.json
"""

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F
from torchvision import datasets, models, transforms

# Re-use pixmix's calibration metric implementations (same as their paper).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pixmix'))
from calibration_tools import get_measures  # noqa: E402

# Canonical Hendrycks ImageNet-C corruptions (same 15 the augmix harness uses).
CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate',
    'jpeg_compression',
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def test_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


@torch.no_grad()
def collect_conf_correct(model, loader):
    """Return (top-1 confidence array, correctness bool array) for one loader."""
    model.eval()
    confs, corrects = [], []
    for x, y in loader:
        x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        confs.append(conf.cpu())
        corrects.append((pred == y).cpu())
    return torch.cat(confs).numpy(), torch.cat(corrects).numpy()


def measure(loader_root, model, batch_size, workers, label):
    ds = datasets.ImageFolder(loader_root, transform=test_transform())
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)
    conf, correct = collect_conf_correct(model, loader)
    rms, aurra_v, mad, sf1 = get_measures(conf, correct)
    acc = float(correct.mean())
    print(f"{label:30s}  acc={acc:.4f}  RMS-CE={100*rms:.2f}  AURRA={100*aurra_v:.2f}  n={len(ds)}")
    return {'acc': acc, 'rms_ce': float(rms), 'aurra': float(aurra_v),
            'mad': float(mad), 'soft_f1': float(sf1), 'n': len(ds)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--arch', default='resnet50')
    ap.add_argument('--val-dir', required=True,
                    help='clean ImageNet val (ImageFolder)')
    ap.add_argument('--c-dir', default=None,
                    help='ImageNet-C root (with per-corruption/severity subdirs); '
                    'if set, calibration is measured per corruption per severity')
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--out', default='calibration_results.json')
    args = ap.parse_args()

    net = models.__dict__[args.arch]()
    ckpt = torch.load(args.checkpoint, map_location='cuda', weights_only=False)
    sd = ckpt.get('state_dict', ckpt)
    sd = {k.replace('module.', '', 1) if k.startswith('module.') else k: v
          for k, v in sd.items()}
    net.load_state_dict(sd)
    net = net.cuda()

    out = {'checkpoint': args.checkpoint, 'arch': args.arch}
    out['clean'] = measure(args.val_dir, net, args.batch_size, args.workers, 'clean val')

    if args.c_dir:
        out['corruption'] = {}
        for c in CORRUPTIONS:
            out['corruption'][c] = {}
            for s in range(1, 6):
                sub = os.path.join(args.c_dir, c, str(s))
                if not os.path.isdir(sub):
                    continue
                out['corruption'][c][str(s)] = measure(
                    sub, net, args.batch_size, args.workers, f'{c} sev{s}')

    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.out}")


if __name__ == '__main__':
    main()
