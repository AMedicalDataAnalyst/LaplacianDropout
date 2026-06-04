# Band-targeted PGD vulnerability profile

Eval @ Imagenette-224, ε=4/255, 20-step PGD, 1000-image subsample.

Accuracy under the attack (higher = more robust).


| Model | none | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|---|
| baseline (no aug) | 0.031 | 0.441 | 0.796 | 0.886 | 0.899 | 0.911 | 0.903 |
| band_drop_all + AugMix (HEADLINE) | 0.011 | 0.648 | 0.757 | 0.826 | 0.882 | 0.904 | 0.908 |
| band_drop_all only | 0.002 | 0.598 | 0.706 | 0.734 | 0.866 | 0.906 | 0.908 |
| band_drop_all hf_only | 0.022 | 0.775 | 0.828 | 0.772 | 0.802 | 0.870 | 0.866 |
| band_drop_all lf_only | 0.001 | 0.050 | 0.621 | 0.861 | 0.899 | 0.904 | 0.905 |
