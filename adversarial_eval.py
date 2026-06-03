"""Adversarial robustness eval (PGD / AutoAttack) for trained ImageNet models.

Thin wrapper around the torchattacks library so we don't implement attacks by
hand. ε is in [0,1] pixel space; we use torchattacks' set_normalization_used()
to handle the IMAGENET mean/std internally — pass the model that *expects*
normalized inputs (the standard trained one).

Usage:
  python adversarial_eval.py --checkpoint ckpt.pth --arch resnet50 \\
      --val-dir data/imagenet/val --attack pgd --eps 4 --subsample 5000
"""

import argparse
import json
import os

import torch
from torchvision import datasets, models, transforms
import torchattacks

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_loader(root, batch_size, workers, subsample):
    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),  # NOTE: no Normalize — the attack normalizes internally.
    ])
    ds = datasets.ImageFolder(root, transform=tf)
    if subsample and subsample < len(ds):
        g = torch.Generator().manual_seed(0)
        idx = torch.randperm(len(ds), generator=g)[:subsample].tolist()
        ds = torch.utils.data.Subset(ds, idx)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)


def make_attack(name, model, eps, steps):
    """Return a torchattacks attack with normalization wired up. ε is in [0,1] units."""
    if name == 'pgd':
        atk = torchattacks.PGD(model, eps=eps, alpha=eps / 4, steps=steps,
                               random_start=True)
    elif name == 'apgd':
        atk = torchattacks.APGD(model, eps=eps, steps=steps, norm='Linf')
    elif name == 'autoattack':
        atk = torchattacks.AutoAttack(model, norm='Linf', eps=eps,
                                      version='standard', n_classes=1000)
    else:
        raise ValueError(f"unknown attack {name!r}")
    atk.set_normalization_used(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD))
    return atk


@torch.no_grad()
def clean_acc(model, loader, normalize):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
        pred = model(normalize(x)).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / total


def adversarial_acc(model, loader, attack, normalize):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
        x_adv = attack(x, y)
        with torch.no_grad():
            pred = model(normalize(x_adv)).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / total


def load_model(arch, checkpoint):
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
    ap.add_argument('--val-dir', required=True)
    ap.add_argument('--attack', choices=['pgd', 'apgd', 'autoattack'], default='pgd')
    ap.add_argument('--eps', type=float, default=4.0,
                    help='Linf ε in 0-255 units (will be divided by 255 internally)')
    ap.add_argument('--steps', type=int, default=20,
                    help='attack iterations (pgd/apgd only; autoattack ignores)')
    ap.add_argument('--subsample', type=int, default=5000,
                    help='subsample val set (AutoAttack on full 50k is hours/checkpoint)')
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--out', default='adversarial_results.json')
    args = ap.parse_args()

    eps_unit = args.eps / 255.0
    print(f"attack={args.attack}  eps={args.eps}/255 ({eps_unit:.4f})  steps={args.steps}  subsample={args.subsample}")

    net = load_model(args.arch, args.checkpoint)
    loader = build_loader(args.val_dir, args.batch_size, args.workers, args.subsample)
    mean_t = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1).cuda()
    std_t = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1).cuda()
    normalize = lambda x: (x - mean_t) / std_t  # noqa: E731

    print("Clean eval...")
    c_acc = clean_acc(net, loader, normalize)
    print(f"  clean acc = {c_acc:.4f}")

    atk = make_attack(args.attack, net, eps_unit, args.steps)
    print(f"Running {args.attack}...")
    adv = adversarial_acc(net, loader, atk, normalize)
    print(f"  adv acc   = {adv:.4f}  (Δ = -{100*(c_acc-adv):.1f} pp)")

    out = {'checkpoint': args.checkpoint, 'arch': args.arch,
           'attack': args.attack, 'eps_255': args.eps, 'steps': args.steps,
           'subsample': args.subsample,
           'clean_acc': c_acc, 'adversarial_acc': adv}
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.out}")


if __name__ == '__main__':
    main()
