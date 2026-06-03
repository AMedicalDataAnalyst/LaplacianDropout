# Imagenette tier-1 comparison: ours vs AugMix vs PixMix vs IPMix

Methodology: ResNet-18 on Imagenette-160, AMP+AdamW, plateau-based scheduler. The 'no JSD' rows are from the original baselines using the single-forward training loop. The '+ JSD' rows use the AugMix-style Jensen-Shannon consistency loss (3 forward passes per batch, λ=12). Corruption metric is mean accuracy across 14 Imagenette-C corruptions (glass_blur excluded), severity 3 (fast eval, 1000-image subset) or {1,3,5} (FULL eval, full 3925-image val set).

| Variant | n | Clean val_acc | Residual-only (shape proxy) | σ=2 blur eval | Mean corruption acc |
|---|---|---|---|---|---|
| baseline (no JSD) | 3 | 0.8745 ± 0.0021 | 0.1182 ± 0.0103 | 0.4802 ± 0.0104 | 0.5036 ± 0.0090 |
| baseline (FULL eval) | 2 | 0.8759 ± 0.0003 | 0.1110 ± 0.0006 | 0.4862 ± 0.0073 | 0.5429 ± 0.0064 |
| curriculum_dropout (orig) | 3 | 0.7604 ± 0.0020 | 0.6544 ± 0.0054 | 0.7530 ± 0.0010 | 0.6714 ± 0.0022 |
| band_drop_all (no JSD) | 3 | 0.8672 ± 0.0039 | 0.5469 ± 0.0103 | 0.8443 ± 0.0019 | 0.7126 ± 0.0034 |
| joint_blur (no JSD) | 3 | 0.8501 ± 0.0006 | 0.1639 ± 0.0066 | 0.8272 ± 0.0039 | 0.7089 ± 0.0025 |
| longA (no JSD, FULL eval) | 3 | 0.8864 ± 0.0019 | 0.3676 ± 0.0407 | 0.8510 ± 0.0027 | 0.7131 ± 0.0081 |
| extralongA (no JSD, FULL eval) | 3 | 0.8809 ± 0.0018 | 0.4119 ± 0.0373 | 0.8533 ± 0.0009 | 0.7360 ± 0.0057 |
| AugMix + JSD | 3 | 0.8923 ± 0.0011 | 0.1422 ± 0.0147 | 0.6520 ± 0.0120 | 0.6537 ± 0.0022 |
| PixMix + JSD | 3 | 0.8746 ± 0.0142 | 0.2272 ± 0.0101 | 0.6036 ± 0.0094 | 0.6677 ± 0.0175 |
| band_drop_all + JSD | 3 | 0.8618 ± 0.0103 | 0.6082 ± 0.0401 | 0.8497 ± 0.0089 | 0.7714 ± 0.0143 |
| band_drop_all + AugMix + JSD | 3 | 0.8796 ± 0.0022 | 0.6301 ± 0.0073 | 0.8630 ± 0.0021 | 0.8012 ± 0.0028 |
| band_drop_all + PixMix + JSD | 3 | 0.8861 ± 0.0021 | 0.5751 ± 0.0014 | 0.8482 ± 0.0037 | 0.7753 ± 0.0014 |
| IPMix + JSD | 3 | 0.8832 ± 0.0015 | 0.2172 ± 0.0224 | 0.6596 ± 0.0237 | 0.7001 ± 0.0064 |
| band_drop_all + IPMix + JSD | 3 | 0.8899 ± 0.0026 | 0.5470 ± 0.0023 | 0.8481 ± 0.0008 | 0.7181 ± 0.0065 |
| longA (bd_aug + AugMix → FT) + JSD | 3 | 0.8859 ± 0.0034 | 0.5726 ± 0.0102 | 0.8664 ± 0.0030 | 0.8004 ± 0.0034 |
