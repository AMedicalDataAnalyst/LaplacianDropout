# Morning report, 2026-11-23

## What ran
Band dropout on an ImageNet-pretrained ResNet-18 (warm-up, constant learning rate, weight EMA).
The queue ran 21:03 to 05:40 and finished all 28 planned cells with no failures and nothing cut by the time guard.
(I said "up to 31 cells" last night; the plan was 28.)

## Pre-registered verdict: not a win, because the clean-accuracy guard broke
Primary test: band dropout on top of AugMix, pretrained, lr 3e-4, 5 seeds, EMA 0.99, paired by seed.

| dataset | corruption accuracy | clean accuracy | verdict |
|---|---|---|---|
| DTD (the real transfer test) | 42.8 → 50.7, +7.9 ± 0.3 pp | 69.6 → 68.1, −1.5 ± 0.3 pp | clean guard broken |
| Imagewoof | 64.8 → 78.9, +14.1 ± 0.8 pp | 92.0 → 90.6, −1.5 ± 0.1 pp | clean guard broken |
| Imagenette | 77.8 → 88.0, +10.2 ± 0.2 pp | 96.7 → 95.4, −1.3 ± 0.1 pp | clean guard broken |

The guard was "clean accuracy must not drop by more than 1 pp". It dropped 1.3 to 1.5 pp on all three, so the
rule I wrote down beforehand says no win, even though the robustness gain is 13 to 46 standard errors.

## What it means in plain terms
- The robustness effect survives pretraining. It is large on every dataset, both learning rates, and both with and without AugMix.
- It is not free. On a pretrained backbone it costs 1.3 to 1.5 pp of clean accuracy on top of AugMix, and 3.4 pp on DTD when used alone.
- Secondary test (band dropout alone against plain baseline): +11.0 (DTD), +11.4 (Imagewoof), +11.4 (Imagenette) pp corruption.
  Only Imagenette stays inside the guard (clean −0.7); DTD loses 3.4 and Imagewoof 1.1.
- lr 1e-3 (3 seeds) gives the same picture: +7.7 / +11.1 / +10.0 pp, clean −1.7 / −1.9 / −1.4.

## Caveats
- Imagenette and Imagewoof are ImageNet subsets, so the pretrained network already knows those classes. DTD is the real transfer test, and it has the smallest gain and the largest clean cost.
- Where the gain comes from (looked at after the fact): mostly noise, defocus blur, pixelate, frost and fog. Brightness, JPEG and zoom blur barely move, and zoom blur on DTD gets worse (−5.4 pp). So this is robustness to frequency-shaped damage, not general robustness.
- The slower EMA (0.999) gives higher accuracy for every arm and a slightly smaller clean cost (−0.8 / −1.2 / −1.1). That is a post hoc observation, not the registered read-out.
- From-scratch control: 2 seeds only. On DTD the from-scratch model is weak (43% clean), so its "WIN" row (+12.2 corruption, +8.1 clean) says little.
- One architecture (ResNet-18), short fine-tunes (about 2000 steps), 1000-image corruption subsample.

## Also finished last night: "uncertain" label for cell typing (ihc_stain_aug)
Mostly negative. Low model confidence detects reader disagreement with AUROC 0.63 to 0.65. With "uncertain" allowed the model
types 22 to 41% of cells at 0.83 to 0.90 accuracy for two readers (reader ceiling 0.92) and 7% for the third. See `ihc_stain_aug/docs/selective_typing.md`.

## Tonight
- 21:30: diffusion corrector on ddn-medseg samples. Code is committed but not armed; it gets sized on the free GPU this morning and armed only if the smoke test passes.
- Looped DDN trunk and pathology foundation features wait for your decisions.
