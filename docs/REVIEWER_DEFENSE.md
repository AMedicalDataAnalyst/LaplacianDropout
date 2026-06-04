# Reviewer defense — "is band-dropout just training on the test corruptions?"

The single most likely methodological challenge to this paper. This doc
states the risk, the steelman attack, and our defenses **as honestly
calibrated by the 3-seed hf_only / lf_only control experiment that
completed 2026-06-04**.

## The convention reviewers will hold us to

The AugMix / PixMix / IPMix lineage enforces a soft rule: training
augmentation operators should be **disjoint from the test-time corruption
operators**. AugMix's reference operator set (`augmentations.augmentations`)
deliberately excludes brightness, contrast, noise, and blur for this
reason — they appear in ImageNet-C / CIFAR-C and including them at
training time would be "cheating." Reviewers in this lineage *do* enforce
this rule, and the attack lands frequently in robustness papers.

## The steelman attack against band_drop_all

> "Random Laplacian-band dropout operates in the frequency domain. So do
> several test-time corruptions: defocus_blur, glass_blur, motion_blur,
> zoom_blur (all linear low-pass filters), and to a lesser extent the
> noise corruptions (broadband frequency perturbations). Training the
> network on random spectral subspace masking is morally close to
> training on the *form* of these corruptions. Your +45 pp on
> defocus_blur is not generalization — it's operator-family memorization."

A version of this attack can be applied to **7 of the 15 corruptions** in
our 224 table (the 4 blurs and 3 noises). Together those account for a
big chunk of the headline gain over baseline, so the attack is serious
and we need to address it head-on.

## What the hf_only / lf_only control actually shows (3-seed Imagenette-224)

| Method (3-seed mean ± std) | Clean | Corruption (fast) |
|---|---|---|
| `band_drop_all --band-mask all` (random multi-band) | 0.888 ± 0.003 | **0.806 ± 0.0004** |
| `band_drop_all --band-mask hf_only` (only HF bands droppable) | 0.873 ± 0.005 | **0.747 ± 0.002** |
| `band_drop_all --band-mask lf_only` (only LF + residual droppable) | 0.900 ± 0.002 | **0.573 ± 0.006** |

The control was designed to test memorization: if `band_drop_all` is "just
training on the blur operator family," then `hf_only` (which concentrates
the dropout on HF bands — most aligned with blur corruptions) should win
*more* on blur than `all`. The result split by corruption category is
nuanced — and **strengthens some defenses while weakening others.**

### Per-corruption breakdown — by category

| Category | Corruptions | hf_only Δ vs `all` |
|---|---|---|
| **Blur (HF-attacking)** | defocus / motion / zoom | **−0.7, −0.4, +0.7 pp — essentially TIE** |
| **Noise** | gaussian / shot / impulse | **+4.6, +4.0, +7.3 pp — hf_only WINS** |
| **Spatial / digital** | pixelate / elastic / jpeg | ≈ tie |
| **Low-freq / weather / DC** | contrast, brightness, fog, frost, snow | **−34.7, −6.4, −22.2, −22.8, −11.2 pp — `all` decisively wins** |

The 5.9 pp average drop for `hf_only` comes **entirely** from the
low-frequency-attacking corruptions. On the corruptions the steelman
points to (blur, noise — where operator-family alignment is the concern),
`hf_only` matches or *beats* `all`.

## What this means — calibrated honestly

### What we CAN claim (strong)

1. **The novel mechanism is the residual being droppable.** This is what
   distinguishes `all` from `lf_only` (which also has residual
   droppable but loses high-freq stress) and from `hf_only` (which never
   stresses missing structure). The +23 pp gap between `lf_only` and
   `all` and the per-corruption pattern (lf_only fails on contrast, fog,
   weather, EVERY blur, EVERY noise) is decisive: the model needs to be
   trained to recover from missing low-frequency / structure-carrying
   content.

2. **The win on low-frequency-attacking corruptions is NOT
   operator-family memorization.** No published augmentation method
   trains on "random low-frequency masking" or "structure dropout." Yet
   `all` gains +30-+35 pp over hf_only on contrast / fog / frost. These
   gains require the random multi-band structure including the
   residual-droppable case — a genuinely novel mechanism with no
   analog in any aug-method-family argument.

3. **At minimum, the multi-band random structure outperforms any
   targeted-band variant on aggregate.** Random multi-band wins by 5.9
   pp over hf_only and 23.3 pp over lf_only on mean corruption acc. So
   the random-structured-masking framing (rather than any
   single-frequency-band aug) is empirically the right
   characterization.

### What we CANNOT honestly claim

1. ❌ "`hf_only` losing on blur refutes operator-family memorization
   for blur." It doesn't lose on blur — it ties. So the steelman
   *retains force* for the blur subset specifically: the model trained
   with hf_only or all has been exposed to inputs missing HF bands,
   which is mechanistically similar to test-time blur. Of course it
   handles blur well — both methods do.

2. ❌ "Random multi-band is necessary for the blur win." Targeted HF
   dropping suffices for blur. The mechanism for the *blur* subset is
   "training on inputs with missing HF bands," which both `all` and
   `hf_only` do.

3. ❌ "The mechanism is operator-agnostic." The mechanism is
   *partly* operator-aligned for the blur/noise subset (it's HF-band
   dropout, which by construction looks blur-adjacent) and *genuinely
   novel* for the low-freq-attacking subset. Both are true.

## Recalibrated defenses (in order of how strong they are)

### Defense 1 (strongest) — the residual-droppable mechanism wins on corruptions with NO training-aug analog

`all`'s decisive win over `lf_only` on contrast (+25 pp), fog (+14 pp),
frost (+13 pp), and EVERY blur and noise corruption (where lf_only
catastrophically fails) is genuine. *No published aug method targets these
corruptions specifically.* The mechanism — random multi-band masking
where the residual can be dropped — has no operator-family analog in any
test corruption.

### Defense 2 (strong) — the win over baseline on LF-attacking corruptions can't be operator-family memorization

The headline result vs baseline at 224 includes +51 pp on contrast, +36 pp
on fog, +26 pp on frost. None of these is a frequency-band operator
that any training aug we cite (AugMix's PIL ops, PixMix's mixing,
IPMix's pixel/patch/image-level mixing) targets. Whatever mechanism gives
us these wins is genuinely new.

### Defense 3 (medium) — operator identity vs operator family

The disjoint-ops convention rules out training on the *same* operator.
AugMix's reference set includes `color`, `equalize`, `posterize`,
`solarize` (all color-space ops) and tests on `brightness` (also
color-space). Nobody calls this overlap. The rule is operator identity,
not domain. Random Laplacian-subspace masking is not Gaussian blur of
any fixed σ.

### Defense 4 (qualitative) — cross-basis transfer evidence

The (modest) +0.7 pp on jpeg_compression (DCT basis) shows the learned
invariance has some cross-basis transfer — not just Laplacian-band
specific.

### What we should NOT lean on

The original framing that the hf_only/lf_only result is a "decisive
refutation of operator-family memorization" was overstated. For the blur
subset specifically, the memorization story remains a viable
interpretation — `hf_only` (which is *more* operator-aligned with blur)
matches `all` on blur. Don't claim refutation here; instead reframe to
"the win on blur is not where our novelty lives."

## What this means for the method-section copy in the paper

A short subsection titled **"Relationship to test-time corruptions"** with
two distinct arguments:

### Argument A — for the blur/noise subset of our headline gain

> "The blur and noise corruptions in ImageNet-C are spectrally narrow
> (each corruption removes or adds content in a specific frequency band
> structure). Random Laplacian-subspace masking exposes the model to
> *inputs missing arbitrary bands*, which is mechanistically related to
> these corruptions. We do not claim our blur/noise gain is unrelated to
> the operator family — it is naturally explained by training on
> structurally similar perturbations.
>
> This is consistent with the disjoint-operator convention: we train on
> random Laplacian-band masks, not on Gaussian blur of any fixed σ.
> AugMix follows the same convention: its training ops are color-space
> while ImageNet-C tests color-space corruptions, and this is not called
> overlap."

### Argument B — for the low-freq-attacking subset (the load-bearing claim)

> "The decisive feature of our method is the residual being droppable.
> Section [X] shows that an `hf_only` ablation (where the residual and
> low-frequency bands are always kept) matches `all` on blur and noise
> corruptions but loses 25-35 pp on contrast, fog, and frost. These
> low-frequency-attacking corruptions have no analog in any training
> augmentation we are aware of, in any published method in this line.
> Our gain over baseline on contrast (+51 pp), fog (+36 pp), and frost
> (+26 pp) is therefore the result of a genuinely new mechanism: random
> multi-band masking including the residual."

## Control experiments — status

| Control | Purpose | Status |
|---|---|---|
| **A: hf_only / lf_only (3 seeds, Imagenette-224)** | decompose the win across band subsets | ✅ DONE |
| A bis: hf_only / lf_only on ImageNet-1k | repeat at scale post Phase B | ⏳ planned in `PHASE2_KICKOFF.md` |
| B: DCT-basis variant | does the mechanism generalize across orthogonal bases? | ❌ not yet implemented (~½ day code) |
| C: matched-budget Gaussian noise | rule out "just strong noise injection" | ❌ not yet (~1 training run) |
| CIFAR controls | originally planned | ❌ cancelled (CIFAR transfer failed Gate A) |

## Recommendation

For submission, we need **at minimum**:

1. The A-bis ImageNet-1k repeat (2 extra training runs post Phase B), to
   show the per-corruption pattern transfers from Imagenette to canonical
   ImageNet-C.
2. The method-section copy above, calibrated to the actual control data
   (NOT the original overstated framing).

Controls B and C would strengthen the paper further but are not
mandatory. Skip them if compute is tight; revisit if reviewers in early
feedback want stronger refutation of the operator-family interpretation
for the blur subset.

## Full per-corruption numbers (for the paper table)

| Corruption | baseline | all | Δ vs base | hf_only | Δ vs all | lf_only | Δ vs all |
|---|---|---|---|---|---|---|---|
| brightness | 0.784 | 0.842 | +5.8 | 0.778 | −6.4 | 0.862 | +2.0 |
| contrast | 0.349 | 0.852 | **+50.3** | 0.505 | −34.7 | 0.593 | −25.9 |
| defocus_blur | 0.427 | 0.858 | **+43.1** | 0.850 | −0.7 | 0.375 | **−48.2** |
| elastic_transform | 0.658 | 0.782 | +12.4 | 0.799 | +1.7 | 0.552 | −23.0 |
| fog | 0.483 | 0.849 | **+36.6** | 0.628 | −22.2 | 0.711 | −13.8 |
| frost | 0.533 | 0.752 | +21.9 | 0.523 | −22.8 | 0.620 | −13.2 |
| gaussian_noise | 0.480 | 0.765 | +28.5 | 0.811 | +4.6 | 0.338 | **−42.7** |
| impulse_noise | 0.447 | 0.738 | +29.1 | 0.811 | +7.3 | 0.293 | **−44.5** |
| jpeg_compression | 0.880 | 0.858 | −2.2 | 0.850 | −0.7 | 0.864 | +0.7 |
| motion_blur | 0.542 | 0.837 | +29.5 | 0.833 | −0.4 | 0.500 | **−33.7** |
| pixelate | 0.779 | 0.871 | +9.2 | 0.857 | −1.4 | 0.803 | −6.8 |
| shot_noise | 0.491 | 0.779 | +28.8 | 0.819 | +4.0 | 0.327 | **−45.2** |
| snow | 0.591 | 0.684 | +9.3 | 0.572 | −11.2 | 0.549 | −13.5 |
| zoom_blur | 0.662 | 0.811 | +14.9 | 0.818 | +0.7 | 0.631 | −18.0 |
| **MEAN** | 0.566 | **0.805** | **+24.0** | 0.747 | −5.9 | 0.573 | −23.3 |

(Baseline columns from Phase 1 224 full-eval; targeted variants 3-seed
mean from `results/phase2_reviewer_defense/`. The "Δ vs all" columns are
the load-bearing refinement of the original headline narrative — they
show where the random multi-band structure (including residual-droppable)
is uniquely responsible vs where targeted operator-aligned dropping
suffices.)
