"""Out-of-distribution evaluation: ImageNet-R / ImageNet-A / ImageNet-Sketch.

Standalone eval over a trained torchvision ImageNet-1k model checkpoint. Reuses
the ImageNet-R/-A wnid masks already defined in pixmix/imagenet.py (so we
don't duplicate the 200-class subset lists). -Sketch uses all 1000 classes.

Usage:
  python ood_eval.py --checkpoint path/to/ckpt.pth --arch resnet50 \\
      --r-dir data/imagenet_r/ --a-dir data/imagenet_a/ \\
      --sketch-dir data/imagenet_sketch/ --out ood_results.json
"""

import argparse
import json
import os
import sys

import torch
from torchvision import datasets, models, transforms

# Import the wnid masks from the pixmix repo (they computed them once;
# don't duplicate 200-element lists in two places). pixmix's imagenet.py
# parses argv at module load and has required args, so fake argv around
# the import.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pixmix'))
_saved_argv = sys.argv
sys.argv = ['pixmix_imagenet_for_wnid_masks',
            '--mixing-set', '/tmp/_unused', '--num-classes', '1000']
from imagenet import imagenet_r_mask, imagenet_a_mask  # noqa: E402
sys.argv = _saved_argv

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_loader(root, batch_size, workers):
    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    ds = datasets.ImageFolder(root, transform=tf)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)


@torch.no_grad()
def evaluate(model, loader, mask=None):
    """Top-1 / top-5 accuracy. If `mask` is a bool list of length 1000,
    restrict argmax to those classes (the standard ImageNet-{R,A} protocol)."""
    model.eval()
    if mask is not None:
        keep = torch.tensor(mask, dtype=torch.bool, device='cuda')
    correct1 = correct5 = total = 0
    for x, y in loader:
        x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
        logits = model(x)
        if mask is not None:
            logits = logits[:, keep]
        _, top5 = logits.topk(5, dim=1)
        correct1 += (top5[:, 0] == y).sum().item()
        correct5 += (top5 == y.unsqueeze(1)).any(dim=1).sum().item()
        total += y.numel()
    return correct1 / total, correct5 / total


def load_model(arch, checkpoint):
    """Instantiate a torchvision model and load weights. Strips a 'module.'
    prefix if the checkpoint was saved under DataParallel."""
    net = models.__dict__[arch]()
    ckpt = torch.load(checkpoint, map_location='cuda', weights_only=False)
    sd = ckpt.get('state_dict', ckpt)
    sd = {k.replace('module.', '', 1) if k.startswith('module.') else k: v
          for k, v in sd.items()}
    net.load_state_dict(sd)
    return net.cuda()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--arch', default='resnet50')
    ap.add_argument('--r-dir', default=None)
    ap.add_argument('--a-dir', default=None)
    ap.add_argument('--sketch-dir', default=None)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--out', default='ood_results.json')
    args = ap.parse_args()

    net = load_model(args.arch, args.checkpoint)
    results = {'checkpoint': args.checkpoint, 'arch': args.arch}
    jobs = [
        ('imagenet_r', args.r_dir, imagenet_r_mask),
        ('imagenet_a', args.a_dir, imagenet_a_mask),
        ('imagenet_sketch', args.sketch_dir, None),
    ]
    for name, root, mask in jobs:
        if root is None:
            continue
        loader = build_loader(root, args.batch_size, args.workers)
        top1, top5 = evaluate(net, loader, mask=mask)
        results[name] = {'top1': top1, 'top5': top5, 'n': len(loader.dataset)}
        print(f"{name:18s}  top1={top1:.4f}  top5={top5:.4f}  n={len(loader.dataset)}")

    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {args.out}")


if __name__ == '__main__':
    main()
