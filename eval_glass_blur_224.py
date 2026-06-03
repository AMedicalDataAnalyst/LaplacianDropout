"""Add the 15th corruption (glass_blur) to the existing 224 FULL eval JSONs.

glass_blur is ~150x slower than the other corruptions in the imagecorruptions
library, so we run it separately, parallelizing the per-image corruption across
CPU cores, then merge the result into the existing 14-corruption JSON and
recompute the mean over all 15.
"""

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

import imagenette_curriculum as ic

RES, PAD = 224, 16
SEVERITIES = (1, 3, 5)
JOBS = [
    ("checkpoints/baseline_r0.pt", "results/phase1_imagenette_full/imagenette_results_FULL_224_baseline_r0.json"),
    ("checkpoints/band_drop_all_augmix_r0.pt", "results/phase1_imagenette_full/imagenette_results_FULL_224_bd_augmix_r0.json"),
]


def _corrupt_one(args):
    img_hwc, severity = args
    from imagecorruptions import corrupt
    return corrupt(img_hwc, corruption_name="glass_blur", severity=severity)


def eval_glass_blur(model, loader, executor) -> dict:
    pad, res = loader.pad, loader.resolution
    imgs = loader.images_uint8_cpu[:, :, pad:pad + res, pad:pad + res]
    cropped = imgs.permute(0, 2, 3, 1).contiguous().numpy()  # (N,H,W,3) uint8
    labels = loader.labels.cpu().numpy()
    mean_g = ic.IMAGENET_MEAN.view(1, 3, 1, 1).cuda().half()
    std_g = ic.IMAGENET_STD.view(1, 3, 1, 1).cuda().half()
    out = {}
    for sev in SEVERITIES:
        t0 = time.time()
        corrupted = list(executor.map(
            _corrupt_one, ((cropped[i], sev) for i in range(len(cropped))),
            chunksize=16))
        corrupted = np.stack(corrupted)  # (N,H,W,3) uint8
        correct = total = 0
        bs = 128
        for i in range(0, len(labels), bs):
            t = torch.from_numpy(corrupted[i:i + bs]).cuda()
            t = t.permute(0, 3, 1, 2).contiguous().half() / 255.0
            t = (t - mean_g) / std_g
            t = t.contiguous(memory_format=torch.channels_last)
            lbl = torch.from_numpy(labels[i:i + bs]).cuda()
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
                preds = model(t).argmax(1)
            correct += (preds == lbl).sum().item()
            total += lbl.numel()
        acc = correct / total
        out[str(sev)] = acc
        print(f"    glass_blur sev{sev}: acc={acc:.4f}  ({time.time()-t0:.0f}s)",
              flush=True)
    return out


def main():
    model = ic.ImagenetteNet(num_classes=10).cuda().to(
        memory_format=torch.channels_last)
    ic.warmup(model)
    loader = ic._build_corruption_eval_loader(resolution=RES, pad=PAD)

    with ProcessPoolExecutor(max_workers=4) as ex:
        for ckpt_path, json_path in JOBS:
            print(f"\n=== {ckpt_path} -> {json_path} ===", flush=True)
            ckpt = torch.load(ckpt_path, map_location="cuda", weights_only=True)
            model.load_state_dict(ckpt["state_dict"])
            model.eval()
            gb = eval_glass_blur(model, loader, ex)

            with open(json_path) as fh:
                data = json.load(fh)
            rec = data[0]
            corr = rec["corruption"]
            corr["glass_blur"] = {int(k): v for k, v in gb.items()}
            # Recompute mean over all per-corruption × severity accuracies (15 now).
            all_acc = []
            for c, sevs in corr.items():
                if c.startswith("_"):
                    continue
                all_acc.extend(sevs.values())
            corr["_mean"] = sum(all_acc) / len(all_acc)
            corr["_n_corruptions"] = len([c for c in corr if not c.startswith("_")])
            with open(json_path, "w") as fh:
                json.dump(data, fh, indent=2)
            print(f"  merged. new mean (15 corruptions) = {corr['_mean']:.4f}",
                  flush=True)


if __name__ == "__main__":
    main()
