# CIFAR sweep results (auto-aggregated)

Aggregated from 2 (dataset, method) combinations, 2 total runs.

Mean ± std over seeds. mCE normalized to the per-dataset `baseline` runs (=100 by construction; lower is better).

| Dataset | Method | n | Clean | Corruption | mCE | PGD@2/255 | RMS-CE clean |
|---|---|---|---|---|---|---|---|
| cifar10 | `band_drop_all+augmix` | 1 | 0.9514 ± 0.0000 | 0.6673 ± 0.0000 | 163.3 | 0.1095 ± 0.0000 | 18.91 |
| cifar10 | `baseline` | 1 | 0.9575 ± 0.0000 | 0.7706 ± 0.0000 | 100.0 | — | 6.07 |
