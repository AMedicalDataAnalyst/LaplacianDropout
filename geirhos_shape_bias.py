"""Geirhos cue-conflict shape-bias evaluation for ImageNet-1k models.

What this measures: present the model with images where SHAPE is from class A
but TEXTURE is from class B (the rgeirhos/texture-vs-shape stimuli). For each
prediction:
  - if pred class == shape label  -> shape-correct
  - if pred class == texture label-> texture-correct
  - else                          -> neither

  shape_bias = shape_correct / (shape_correct + texture_correct)

External data needed (download once, no auth):
  - Stimuli:       git clone https://github.com/rgeirhos/texture-vs-shape
                   The cue-conflict images live in
                   `data-cue-conflict/` (1280 PNGs, filenames encode
                   `<shape-class>/<image>-<texture-class>.png`).
  - 16-class map:  the texture-vs-shape repo also provides
                   `data/imagenet_mapping.txt` or similar — a file that maps
                   each of the 16 Geirhos classes to the list of ImageNet-1k
                   class indices that belong to it. Pass via --class-map-json
                   (we accept a JSON {geirhos_class: [imagenet_idx, ...]}).

Usage:
  python geirhos_shape_bias.py --checkpoint ckpt.pth --arch resnet50 \\
      --stimuli-dir /path/to/texture-vs-shape/data-cue-conflict \\
      --class-map-json geirhos_classes_to_imagenet.json \\
      --out shape_bias.json
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Geirhos's filename convention: '<shape>/<shape>10-<texture>3.png' under
# data-cue-conflict/, where <shape> and <texture> are 16-class names.
FILENAME_RE = re.compile(r'^([a-zA-Z_]+)\d+-([a-zA-Z_]+)\d+\.png$')


def parse_stimulus(path):
    """Return (shape_class, texture_class) parsed from a Geirhos filename."""
    m = FILENAME_RE.match(os.path.basename(path))
    if not m:
        return None
    return m.group(1), m.group(2)


def load_class_map(path):
    """Load {geirhos_class: [imagenet_idx, ...]} JSON."""
    with open(path) as f:
        m = json.load(f)
    # Validate.
    for k, v in m.items():
        assert isinstance(v, list) and all(isinstance(i, int) for i in v), \
            f"class map entry {k!r} must be list[int]"
    return m


def load_model(arch, ckpt):
    net = models.__dict__[arch]()
    blob = torch.load(ckpt, map_location='cuda', weights_only=False)
    sd = blob.get('state_dict', blob)
    sd = {k.replace('module.', '', 1) if k.startswith('module.') else k: v
          for k, v in sd.items()}
    net.load_state_dict(sd)
    return net.cuda().eval()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--arch', default='resnet50')
    ap.add_argument('--stimuli-dir', required=True,
                    help='root of Geirhos cue-conflict stimuli (per-class subdirs)')
    ap.add_argument('--class-map-json', required=True,
                    help='{geirhos_class: [imagenet_idx, ...]} JSON')
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--out', default='shape_bias.json')
    args = ap.parse_args()

    class_map = load_class_map(args.class_map_json)
    geirhos_classes = sorted(class_map.keys())
    # For each Geirhos class, indicator vector over the 1000 ImageNet classes.
    masks = torch.zeros(len(geirhos_classes), 1000, device='cuda')
    for gi, gc in enumerate(geirhos_classes):
        for idx in class_map[gc]:
            masks[gi, idx] = 1.0

    net = load_model(args.arch, args.checkpoint)
    tf = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    paths = []
    labels = []  # (shape_idx, texture_idx) pairs over geirhos_classes order
    for sub in sorted(os.listdir(args.stimuli_dir)):
        if sub not in geirhos_classes:
            continue  # ignore stray dirs
        for f in glob.glob(os.path.join(args.stimuli_dir, sub, '*.png')):
            parsed = parse_stimulus(f)
            if parsed is None:
                continue
            shape, texture = parsed
            if shape == texture:
                continue  # only cue-conflict images count
            if shape not in geirhos_classes or texture not in geirhos_classes:
                continue
            paths.append(f)
            labels.append((geirhos_classes.index(shape),
                           geirhos_classes.index(texture)))
    print(f"Loaded {len(paths)} cue-conflict stimuli over {len(geirhos_classes)} classes.")

    shape_correct = texture_correct = neither = 0
    per_class = defaultdict(lambda: {'shape': 0, 'texture': 0, 'neither': 0})
    for i in range(0, len(paths), args.batch_size):
        chunk = paths[i:i + args.batch_size]
        imgs = torch.stack([tf(Image.open(p).convert('RGB')) for p in chunk]).cuda()
        logits = net(imgs)  # (B, 1000)
        # For each image, pick the highest *Geirhos-class* score by max-pooling
        # the 1000 logits down to 16 via each class's ImageNet-index mask.
        probs = F.softmax(logits, dim=1)             # (B, 1000)
        scores = probs @ masks.t()                   # (B, 16) — sum of probs in each Geirhos bucket
        preds = scores.argmax(dim=1).tolist()        # predicted Geirhos class
        for j, p in enumerate(preds):
            sh, tx = labels[i + j]
            if p == sh:
                shape_correct += 1
                per_class[geirhos_classes[sh]]['shape'] += 1
            elif p == tx:
                texture_correct += 1
                per_class[geirhos_classes[sh]]['texture'] += 1
            else:
                neither += 1
                per_class[geirhos_classes[sh]]['neither'] += 1

    total_decided = shape_correct + texture_correct
    shape_bias = shape_correct / total_decided if total_decided else float('nan')
    print(f"shape_correct={shape_correct}  texture_correct={texture_correct}  "
          f"neither={neither}")
    print(f"shape_bias = {shape_bias:.4f}  ({100*shape_bias:.1f}%)")

    out = {
        'checkpoint': args.checkpoint, 'arch': args.arch,
        'n_stimuli': len(paths), 'shape_correct': shape_correct,
        'texture_correct': texture_correct, 'neither': neither,
        'shape_bias': shape_bias, 'per_class': dict(per_class),
    }
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.out}")


if __name__ == '__main__':
    main()
