# Imagenette shape-bias experiments — summary

Methodology: ResNet-18 trained on Imagenette at 160² with AMP+AdamW. Plateau-based scheduler. `fast` corruption eval uses 1000-image subset × severity 3 across 14 corruptions (glass_blur excluded for runtime). `FULL` uses the full 3925-image val set × severities {1,3,5}.

| Variant | n | Clean val_acc | Residual-only acc (shape proxy) | σ=2 blur eval | Mean corruption acc |
|---|---|---|---|---|---|
| baseline (multi-seed fast eval) | 3 | 0.8745 ± 0.0021 | 0.1182 ± 0.0103 | 0.4802 ± 0.0104 | 0.5036 ± 0.0090 |
| baseline (FULL eval) | 2 | 0.8759 ± 0.0003 | 0.1110 ± 0.0006 | 0.4862 ± 0.0073 | 0.5429 ± 0.0064 |
| curriculum_dropout (orig, plateau) | 3 | 0.7604 ± 0.0020 | 0.6544 ± 0.0054 | 0.7530 ± 0.0010 | 0.6714 ± 0.0022 |
| band_drop (p=0.5, protect) | 3 | 0.8433 ± 0.0110 | 0.5727 ± 0.0276 | 0.8168 ± 0.0167 | 0.6838 ± 0.0183 |
| band_drop_all (p=0.5) | 3 | 0.8672 ± 0.0039 | 0.5469 ± 0.0103 | 0.8443 ± 0.0019 | 0.7126 ± 0.0034 |
| joint_blur (50/50) | 3 | 0.8501 ± 0.0006 | 0.1639 ± 0.0066 | 0.8272 ± 0.0039 | 0.7089 ± 0.0025 |
| twostage_A (lr/10 FT) | 3 | 0.8802 ± 0.0034 | 0.3393 ± 0.0201 | 0.8290 ± 0.0030 | 0.6988 ± 0.0032 |
| twostage_B (lr/10 FT) | 3 | 0.8723 ± 0.0024 | 0.5432 ± 0.0164 | 0.8451 ± 0.0080 | 0.6914 ± 0.0150 |
| twostage_A (full LR FT) | 3 | 0.8849 ± 0.0058 | 0.1163 ± 0.0065 | 0.7330 ± 0.0133 | 0.6671 ± 0.0072 |
| longA s2 (long pretrain + FT, fast eval) | 3 | 0.8860 ± 0.0004 | 0.3954 ± 0.0300 | 0.8499 ± 0.0008 | 0.7296 ± 0.0171 |
| longA s2 (FULL eval) | 3 | 0.8864 ± 0.0019 | 0.3676 ± 0.0407 | 0.8510 ± 0.0027 | 0.7131 ± 0.0081 |
| distill: band_drop_all + KL←baseline | 3 | 0.8695 ± 0.0053 | 0.5361 ± 0.0207 | 0.8476 ± 0.0068 | 0.7166 ± 0.0019 |
| distill: baseline + KL←longA (no aug) | 3 | 0.8763 ± 0.0032 | 0.1210 ± 0.0060 | 0.4876 ± 0.0094 | 0.5159 ± 0.0087 |
| extralongA s2 (max-pretrain 120, fast eval) | 3 | 0.8790 ± 0.0022 | 0.4264 ± 0.0310 | 0.8519 ± 0.0046 | 0.7532 ± 0.0089 |
| extralongA s2 (FULL eval) | 3 | 0.8809 ± 0.0018 | 0.4119 ± 0.0373 | 0.8533 ± 0.0009 | 0.7360 ± 0.0057 |
