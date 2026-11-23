# Band dropout on a pretrained backbone (night of 2026-11-22)

Corruption accuracy = mean over 14 corruptions at severity 3 (1000 validation images). All numbers in percentage points; differences are paired by seed, ± is the standard error.

## Paired differences (weight-EMA 0.99, the pre-registered read-out)

| dataset | init | lr | comparison | seeds | Δ corruption | Δ clean | verdict |
|---|---|---|---|---|---|---|---|
| dtd | pretrained | 0.0003 | primary: band drop on top of AugMix (JSD) | 5 | +7.93 ± 0.29 | -1.54 ± 0.26 | clean guard broken |
| dtd | pretrained | 0.0003 | secondary: band drop alone (no JSD) | 5 | +10.98 ± 0.31 | -3.39 ± 0.20 | clean guard broken |
| dtd | pretrained | 0.001 | primary: band drop on top of AugMix (JSD) | 3 | +7.73 ± 0.32 | -1.70 ± 0.48 | clean guard broken |
| dtd | pretrained | 0.001 | secondary: band drop alone (no JSD) | 3 | +12.74 ± 0.65 | -3.37 ± 1.13 | clean guard broken |
| dtd | scratch | 0.003 | primary: band drop on top of AugMix (JSD) | 2 | +12.15 ± 0.05 | +8.14 ± 0.43 | WIN |
| dtd | scratch | 0.003 | secondary: band drop alone (no JSD) | 2 | +6.57 ± 0.82 | -2.39 ± 1.65 | clean guard broken |
| imagenette | pretrained | 0.0003 | primary: band drop on top of AugMix (JSD) | 5 | +10.21 ± 0.22 | -1.27 ± 0.07 | clean guard broken |
| imagenette | pretrained | 0.0003 | secondary: band drop alone (no JSD) | 5 | +11.43 ± 0.68 | -0.69 ± 0.10 | WIN |
| imagenette | pretrained | 0.001 | primary: band drop on top of AugMix (JSD) | 3 | +9.97 ± 0.64 | -1.44 ± 0.09 | clean guard broken |
| imagenette | pretrained | 0.001 | secondary: band drop alone (no JSD) | 3 | +14.26 ± 0.83 | -0.65 ± 0.10 | WIN |
| imagewoof | pretrained | 0.0003 | primary: band drop on top of AugMix (JSD) | 5 | +14.07 ± 0.76 | -1.48 ± 0.12 | clean guard broken |
| imagewoof | pretrained | 0.0003 | secondary: band drop alone (no JSD) | 5 | +11.42 ± 0.80 | -1.05 ± 0.15 | clean guard broken |
| imagewoof | pretrained | 0.001 | primary: band drop on top of AugMix (JSD) | 3 | +11.06 ± 0.59 | -1.88 ± 0.28 | clean guard broken |
| imagewoof | pretrained | 0.001 | secondary: band drop alone (no JSD) | 3 | +15.02 ± 0.22 | -1.49 ± 0.27 | clean guard broken |
| imagewoof | scratch | 0.003 | primary: band drop on top of AugMix (JSD) | 2 | +10.73 ± 0.06 | -2.71 ± 0.17 | clean guard broken |
| imagewoof | scratch | 0.003 | secondary: band drop alone (no JSD) | 2 | +19.94 ± 0.41 | -1.86 ± 0.20 | clean guard broken |

## Absolute numbers (mean over seeds; corruption / clean)

### dtd, pretrained, lr 0.0003 (seeds [0, 1, 2, 3, 4])

| arm | EMA 0.99 | EMA 0.999 | raw weights |
|---|---|---|---|
| baseline | 35.7 / 71.5 | 38.3 / 72.3 | 34.3 / 69.4 |
| band_drop_all | 46.7 / 68.1 | 47.7 / 70.6 | 43.5 / 64.9 |
| augmix | 42.8 / 69.6 | 44.0 / 70.8 | 39.5 / 66.6 |
| band_drop_all+augmix | 50.7 / 68.1 | 50.5 / 70.0 | 46.9 / 64.3 |

### dtd, pretrained, lr 0.001 (seeds [0, 1, 2])

| arm | EMA 0.99 | EMA 0.999 | raw weights |
|---|---|---|---|
| baseline | 33.5 / 67.5 | 35.0 / 69.2 | 31.0 / 64.3 |
| band_drop_all | 46.2 / 64.1 | 46.8 / 66.7 | 39.9 / 58.1 |
| augmix | 41.6 / 67.1 | 41.9 / 67.8 | 35.8 / 61.0 |
| band_drop_all+augmix | 49.3 / 65.4 | 49.1 / 67.6 | 43.6 / 58.8 |

### dtd, scratch, lr 0.003 (seeds [0, 1])

| arm | EMA 0.99 | EMA 0.999 | raw weights |
|---|---|---|---|
| baseline | 21.3 / 43.5 | 14.9 / 29.8 | 18.4 / 37.8 |
| band_drop_all | 27.8 / 41.1 | 15.4 / 26.0 | 23.9 / 34.0 |
| augmix | 28.2 / 46.3 | 17.8 / 31.3 | 23.7 / 39.6 |
| band_drop_all+augmix | 40.3 / 54.4 | 28.6 / 40.6 | 36.0 / 47.8 |

### imagenette, pretrained, lr 0.0003 (seeds [0, 1, 2, 3, 4])

| arm | EMA 0.99 | EMA 0.999 | raw weights |
|---|---|---|---|
| baseline | 70.5 / 97.3 | 75.5 / 97.9 | 65.2 / 94.8 |
| band_drop_all | 82.0 / 96.6 | 84.7 / 97.4 | 78.7 / 95.1 |
| augmix | 77.8 / 96.7 | 82.1 / 97.4 | 71.2 / 95.2 |
| band_drop_all+augmix | 88.0 / 95.4 | 89.1 / 96.3 | 86.1 / 94.3 |

### imagenette, pretrained, lr 0.001 (seeds [0, 1, 2])

| arm | EMA 0.99 | EMA 0.999 | raw weights |
|---|---|---|---|
| baseline | 66.6 / 95.6 | 70.9 / 96.6 | 60.6 / 92.1 |
| band_drop_all | 80.9 / 94.9 | 82.4 / 96.0 | 76.8 / 91.3 |
| augmix | 76.7 / 95.3 | 77.9 / 96.1 | 68.1 / 92.5 |
| band_drop_all+augmix | 86.6 / 93.9 | 87.5 / 94.9 | 81.1 / 91.3 |

### imagewoof, pretrained, lr 0.0003 (seeds [0, 1, 2, 3, 4])

| arm | EMA 0.99 | EMA 0.999 | raw weights |
|---|---|---|---|
| baseline | 56.4 / 92.5 | 61.6 / 93.7 | 47.3 / 87.2 |
| band_drop_all | 67.8 / 91.4 | 74.1 / 93.0 | 63.6 / 88.6 |
| augmix | 64.8 / 92.0 | 68.9 / 93.2 | 59.6 / 89.9 |
| band_drop_all+augmix | 78.9 / 90.6 | 80.1 / 92.0 | 74.3 / 87.8 |

### imagewoof, pretrained, lr 0.001 (seeds [0, 1, 2])

| arm | EMA 0.99 | EMA 0.999 | raw weights |
|---|---|---|---|
| baseline | 51.5 / 90.5 | 55.8 / 91.7 | 42.9 / 85.7 |
| band_drop_all | 66.5 / 89.0 | 70.4 / 90.7 | 59.9 / 83.8 |
| augmix | 65.0 / 90.3 | 67.3 / 91.6 | 53.8 / 85.2 |
| band_drop_all+augmix | 76.1 / 88.4 | 77.9 / 90.2 | 69.1 / 83.4 |

### imagewoof, scratch, lr 0.003 (seeds [0, 1])

| arm | EMA 0.99 | EMA 0.999 | raw weights |
|---|---|---|---|
| baseline | 46.5 / 82.8 | 46.6 / 82.7 | 42.1 / 76.0 |
| band_drop_all | 66.5 / 81.0 | 66.4 / 80.8 | 58.9 / 74.9 |
| augmix | 61.8 / 83.8 | 62.7 / 84.0 | 51.1 / 76.7 |
| band_drop_all+augmix | 72.6 / 81.1 | 73.5 / 81.9 | 64.5 / 75.6 |

## Queue

- cells finished: 28
- failed or not started: 0
