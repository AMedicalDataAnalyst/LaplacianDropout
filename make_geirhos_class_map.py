"""Generate geirhos_classes_to_imagenet.json for geirhos_shape_bias.py.

Extracts the canonical 16-class -> ImageNet-1k-index mapping from the
rgeirhos/texture-vs-shape repo (the `*_indices` class attributes in
code/helper/human_categories.py) so the mapping is provably Geirhos's own,
not a re-derivation.

Usage:
  python make_geirhos_class_map.py \
      --texture-vs-shape /path/to/texture-vs-shape \
      --out geirhos_classes_to_imagenet.json
"""

import argparse
import ast
import json
import os


def extract_index_lists(human_categories_path):
    """Parse `<category>_indices = [...]` assignments from the source file.

    Parsed with ast rather than imported, so we don't need the repo's
    Python-2-era imports to work.
    """
    with open(human_categories_path) as f:
        tree = ast.parse(f.read())

    mapping = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id.endswith('_indices'):
                category = target.id[:-len('_indices')]
                mapping[category] = ast.literal_eval(node.value)
    return mapping


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--texture-vs-shape', required=True,
                    help='root of a rgeirhos/texture-vs-shape clone')
    ap.add_argument('--out', default='geirhos_classes_to_imagenet.json')
    args = ap.parse_args()

    src = os.path.join(args.texture_vs_shape,
                       'code', 'helper', 'human_categories.py')
    mapping = extract_index_lists(src)

    assert len(mapping) == 16, f"expected 16 categories, got {len(mapping)}"
    all_indices = [i for v in mapping.values() for i in v]
    assert len(all_indices) == len(set(all_indices)), "categories overlap"
    assert all(0 <= i < 1000 for i in all_indices)

    with open(args.out, 'w') as f:
        json.dump({k: sorted(v) for k, v in sorted(mapping.items())},
                  f, indent=2)
    print(f"Wrote {args.out}: 16 categories, "
          f"{len(all_indices)} ImageNet indices total")


if __name__ == '__main__':
    main()
