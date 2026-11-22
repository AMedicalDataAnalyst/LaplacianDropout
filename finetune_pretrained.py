"""Does band dropout still help when the backbone is ImageNet-pretrained?

The cheap version of the ImageNet gate (docs/PHASE2_KICKOFF.md): fine-tune a
torchvision ResNet-18 (IMAGENET1K_V1) on a small dataset with and without
band dropout, reusing compare_methods.train_one unchanged (same in-GPU
loader, AdamW, JSD, EMA shadows, fast corruption suite).

One invocation = one (dataset, init, lr, seed) cell and writes one JSON with
four arms, forming two clean pairs:

    baseline              vs  band_drop_all            (neither uses JSD)
    augmix + JSD          vs  band_drop_all+augmix + JSD

Schedule is warm-up then constant LR, evaluated with weight-EMA shadows.

    python finetune_pretrained.py --dataset dtd --seed 0 --out results/x.json
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tvm

import imagenette_curriculum as ic
import compare_methods as cm

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("FT_DATA_ROOT", "/home/joytoy/projects/data")

# name -> (ImageFolder root with train/ and val/, cache dir)
DATASETS = {
    # the existing imagenette_cache/ was built from this root
    "imagenette": (os.path.join(DATA, "imagenette2-320"),
                   os.path.join(HERE, "imagenette_cache")),
    "imagewoof": (os.path.join(DATA, "imagewoof2-320"),
                  os.path.join(HERE, "imagenette_cache", "imagewoof")),
    # built by make_dtd_split(): train = train1 + val1, val = test1
    "dtd": (os.path.join(HERE, "imagenette_cache", "dtd_split1"),
            os.path.join(HERE, "imagenette_cache", "dtd")),
}

# (method, use_jsd)
ARMS = (("baseline", False), ("band_drop_all", False),
        ("augmix", True), ("band_drop_all+augmix", True))


def make_dtd_split(root: str):
    """Symlink DTD split 1 into an ImageFolder layout (idempotent)."""
    src = os.path.join(DATA, "dtd", "dtd")
    for split, files in (("train", ("train1.txt", "val1.txt")),
                         ("val", ("test1.txt",))):
        for fn in files:
            with open(os.path.join(src, "labels", fn)) as f:
                for line in f:
                    rel = line.strip()
                    if not rel:
                        continue
                    cls, name = rel.split("/")
                    d = os.path.join(root, split, cls)
                    os.makedirs(d, exist_ok=True)
                    dst = os.path.join(d, name)
                    if not os.path.lexists(dst):
                        os.symlink(os.path.join(src, "images", rel), dst)


class FinetuneNet(ic.ImagenetteNet):
    """ResNet-18 whose reset() restores the starting point of the run.

    init='pretrained': IMAGENET1K_V1 trunk, freshly initialised head.
    init='scratch': everything re-initialised (the repo's usual behaviour).
    """

    def __init__(self, num_classes: int, init: str):
        nn.Module.__init__(self)
        self.init = init
        if init == "pretrained":
            self.net = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
            self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)
            self._start = {k: v.detach().clone()
                           for k, v in self.net.state_dict().items()
                           if not k.startswith("fc.")}
        else:
            self.net = tvm.resnet18(weights=None, num_classes=num_classes)
        self.to(memory_format=torch.channels_last)

    def reset(self):
        if self.init == "scratch":
            return super().reset()
        self.net.load_state_dict(self._start, strict=False)
        self.net.fc.reset_parameters()  # head differs per seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    ap.add_argument("--init", choices=("pretrained", "scratch"), default="pretrained")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--resolution", type=int, default=224)
    ap.add_argument("--pad", type=int, default=16)
    ap.add_argument("--ema-decays", type=float, nargs="*", default=[0.99, 0.999])
    ap.add_argument("--arms", nargs="*", default=None,
                    help="subset of methods (default: all four)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root, cache = DATASETS[args.dataset]
    if args.dataset == "dtd":
        make_dtd_split(root)
    ic.IMAGENETTE_ROOT, ic.CACHE_DIR = root, cache
    cm.BAND_MASK_OVERRIDE = "all"

    corruption_loader = ic._build_corruption_eval_loader(
        resolution=args.resolution, pad=args.pad)
    n_cls = len(corruption_loader.classes)
    n_train = len(ic.get_or_build_cache("train", args.resolution, args.pad)["labels"])
    print(f"finetune_pretrained | {args.dataset} classes={n_cls} train={n_train} "
          f"val={corruption_loader.n} init={args.init} lr={args.lr:g} "
          f"epochs={args.epochs} seed={args.seed}", flush=True)

    model = FinetuneNet(n_cls, args.init).cuda().to(memory_format=torch.channels_last)

    results = []
    for method, use_jsd in ARMS:
        if args.arms and method not in args.arms:
            continue
        t0 = time.time()
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        torch.cuda.empty_cache()
        r = cm.train_one(method, model, max_epochs=args.epochs, base_lr=args.lr,
                         use_jsd=use_jsd, batch_size=args.batch_size,
                         resolution=args.resolution, pad=args.pad,
                         skip_reset=False, ema_decays=tuple(args.ema_decays),
                         schedule="constant")
        emas = r.pop("_ema_modules", {})
        suites = cm.evaluate_corruption_suite_multi(
            {"raw": model, **emas}, corruption_loader, severities=(3,), subsample=1000)
        r["corruption"] = suites["raw"]
        for k in emas:
            r["ema"][k]["corruption"] = suites[k]
        del emas
        r.update(dataset=args.dataset, init=args.init, lr=args.lr, seed=args.seed,
                 resolution=args.resolution, n_train=n_train, n_classes=n_cls,
                 seconds=round(time.time() - t0, 1))
        results.append(r)
        row = " ".join(f"ema{k}: {v['val_acc']:.4f}/{v['corruption']['_mean']:.4f}"
                       for k, v in r["ema"].items())
        print(f"{method:<22} raw clean/corr {r['val_acc']:.4f}/"
              f"{r['corruption']['_mean']:.4f}  {row}  {r['seconds']:.0f}s", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(results, f, indent=1)
    os.replace(tmp, args.out)
    print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
