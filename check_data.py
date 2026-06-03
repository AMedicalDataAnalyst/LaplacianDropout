"""Verify the tier-2 data root has everything in the right place.

Run:  python check_data.py [--root data]

For each expected path, prints OK / MISSING and a count of class-dirs or
files where applicable. Non-fatal: missing pieces just print MISSING so you
know what's still to stage.
"""

import argparse
import os


def status(label, path, expected=None, kind='dir'):
    if not os.path.exists(path):
        print(f"  [MISSING] {label:32s} -> {path}")
        return False
    if kind == 'dir':
        n = len([d for d in os.listdir(path)
                 if os.path.isdir(os.path.join(path, d))])
        extra = f" ({n} subdirs"
        if expected is not None:
            extra += f", expected {expected}"
        extra += ")"
        print(f"  [ OK    ] {label:32s} -> {path}{extra}")
    else:  # file
        print(f"  [ OK    ] {label:32s} -> {path}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/mnt/c/Users/JoyToy/Documents/Projects/data',
                    help='data root (default: /mnt/c/Users/JoyToy/Documents/Projects/data)')
    args = ap.parse_args()
    r = args.root

    print(f"Checking data root: {r}\n")
    print("NOTE: CIFAR-10/-100 and CIFAR-C tarballs were used in Phase-1.5\n"
          "      (transfer experiment, failed). They are NOT required for Phase 2.\n"
          "      See docs/PHASE2_STATUS.md for the decision record.\n")

    print("REQUIRED for Phase 2 headline:")
    status("imagenet train (1000 wnids)",  f"{r}/imagenet/train",          1000)
    status("imagenet val   (1000 wnids)",  f"{r}/imagenet/val",            1000)
    status("imagenet-c (15 corruptions)",  f"{r}/imagenet_c",              15)

    print("\nREQUIRED for Phase 2 OOD battery (R/A/Sketch):")
    status("imagenet-r (200 wnids)",       f"{r}/imagenet_r",              200)
    status("imagenet-a (200 wnids)",       f"{r}/imagenet_a",              200)
    status("imagenet-sketch (1000 wnids)", f"{r}/imagenet_sketch",         1000)

    print("\nREQUIRED for faithful PixMix (Phase 2 Phase-D):")
    status("fractals mixing set",          f"{r}/fractals_and_fvis/fractals/images", kind='dir')

    print("\nOPTIONAL — Tier-2 alt-corruption + perturbation (IPMix-grade only):")
    status("imagenet-c-bar",               f"{r}/imagenet_c_bar")
    status("imagenet-p",                   f"{r}/imagenet_p")
    status("imagenet-o",                   f"{r}/imagenet_o")

    print("\nOPTIONAL — Geirhos shape-bias (Phase 2 #12; needs ImageNet model):")
    status("texture-vs-shape stimuli",     f"{r}/texture-vs-shape/data-cue-conflict", 16)
    status("16-class -> ImageNet map",     "geirhos_classes_to_imagenet.json", kind='file')


if __name__ == '__main__':
    main()
