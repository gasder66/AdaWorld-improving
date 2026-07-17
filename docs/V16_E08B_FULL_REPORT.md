# V16 E08B Boxing full experiment report

## Scope

E08B keeps the E07C high-resolution stride-2 CNN, convolutional IDM, independent
FDM, learned upsampling, dynamic-mask loss, and edge loss. It adds a causal
three-frame temporal Transformer to object-state feature extraction:

- current state: `[t-2, t-1, t]`
- next state: `[t-1, t, t+1]`
- latent action: `IDM(h_t, h_t+1)`

The FDM only receives `h_t` and `z`, so the target frame cannot leak into the
forward prediction path. This experiment evaluates temporal feature extraction;
it does not add an inter-object Transformer to the FDM.

## Training

Two E08B seeds were initialized from E07C and trained for 3,000 phase-balanced
steps. Both runs completed normally.

| Metric | Seed 0 | Seed 1 | Mean |
|---|---:|---:|---:|
| normal-z mask IoU | 0.9870 | 0.9872 | 0.9871 |
| zero-z mask IoU | 0.7604 | 0.7604 | 0.7604 |
| shuffle-z mask IoU | 0.8287 | 0.8285 | 0.8286 |

The large normal/zero and normal/shuffle gaps show that the learned object
latent is necessary and transition-specific.

## Phase-balanced contour evaluation

The evaluation uses 500 held-out transitions for each of six phases (3,000
total). E08B values below are the mean of two seeds.

| Metric | E07C | E08B mean | Delta |
|---|---:|---:|---:|
| predicted mask IoU | 0.9094 | 0.9169 | +0.0074 |
| boundary F1 | 0.9863 | 0.9893 | +0.0030 |
| dynamic-region IoU | 0.7748 | 0.7798 | +0.0050 |
| dynamic-region L1 | 0.0449 | 0.0398 | -0.0051 |

Per-phase predicted mask IoU:

| Phase | E07C | E08B mean |
|---|---:|---:|
| movement | 0.9915 | 0.9911 |
| onset | 0.8655 | 0.8697 |
| extend | 0.8648 | 0.8791 |
| hold | 0.9127 | 0.9286 |
| retract | 0.9490 | 0.9515 |
| switch | 0.8730 | 0.8811 |

Temporal features improve all punch phases, especially extend and hold, while
ordinary movement is unchanged.

## Six-way phase probe

The probe uses 1,500 training and 500 validation examples per phase (9,000 /
3,000 total). Chance balanced accuracy is 0.1667.

| Model | Linear | MLP |
|---|---:|---:|
| E07C | 0.4600 | 0.5583 |
| E08B seed 0 | 0.4687 | 0.5760 |
| E08B seed 1 | 0.4660 | 0.5610 |
| E08B mean | 0.4673 | 0.5685 |

The gain is positive but modest. Mean E08B MLP recall is strong for movement
(0.809), onset (0.850), and hold (0.811), but extend (0.275) and switch (0.207)
remain difficult. The temporal window therefore helps selected phases without
producing a clean, globally ordered phase representation.

## Latent semantic probes

Each semantic visualization uses all available rare interactions and up to 500
examples per event (3,323 points for each E08B seed). Values below are held-out
linear-probe scores.

| Probe | E07C | E08B mean |
|---|---:|---:|
| movement dx R2 | 0.6309 | 0.5608 |
| movement dy R2 | 0.8763 | 0.8664 |
| movement direction balanced accuracy | 0.9282 | 0.9374 |
| punch balanced accuracy | 0.7163 | 0.7479 |
| interaction balanced accuracy | 0.4443 | 0.4266 |
| fighter identity balanced accuracy | 0.8473 | 0.8861 |

The latent reliably represents movement direction and improves punch-active
readability. Horizontal displacement is seed-sensitive (`0.4907` versus
`0.6308`), so the temporal branch is not uniformly better for continuous
motion. Interaction decoding does not improve. Fighter identity is more
readable; in Boxing this is partly expected because the two mirrored sprites
have different object-relative motion geometry, but it confirms that the
latent is not a pure identity-free action code.

## Qualitative evaluation

- Thresholded masks preserve clear fighter contours and recognizable
  onset/extend/hold/retract shapes.
- Probability maps remain soft at changing arm boundaries.
- Full RGB reconstructions still show ghosting in punch-miss, occlusion, and
  recovery examples.
- Normal-z predictions are substantially more stable than shuffled-z
  predictions, but individual easy transitions can be reconstructed from the
  current state with zero z.
- UMAP and t-SNE separate movement directions well, but the representation is a
  collection of pose/identity-dependent subclusters rather than one smooth
  action manifold.
- Interaction labels overlap heavily in UMAP, consistent with the weak
  interaction probe and the limited causal interaction in Boxing.

## Verdict

E08B is a successful temporal-feature experiment, but not a solution to pure
action disentanglement.

1. The object-wise IDM/FDM data flow is stable and the latent is necessary.
2. Three-frame causal temporal features modestly improve punch contours and
   phase decoding.
3. The latent contains useful movement and punch semantics, but also
   pose/identity structure.
4. A longer temporal window alone does not organize all punch phases or improve
   interaction semantics.
5. The remaining visual failure is concentrated in dynamic RGB rendering more
   than binary contour recovery.

## Recommended next experiment

Keep E08B as the temporal front end, then add a phase-sensitive predictive
objective instead of increasing Transformer depth:

1. predict a short sequence of future object masks/features, not only `t+1`;
2. add temporal-order or arm-change contrastive supervision using labels only
   for evaluation/auxiliary ablation;
3. retain normal/zero/shuffle diagnostics;
4. evaluate the independent FDM before adding an interaction Transformer;
5. treat a different Atari environment with actual state-changing interactions
   as a separate experiment, because Boxing cannot strongly validate causal
   multi-agent interaction.

## Engineering note

Evaluation originally reloaded the same episode file for every randomly ordered
transition. Grouping indices by episode and caching the most recently loaded
sample reduced a 60-sample contour check from approximately 95 seconds
(extrapolated from the old six-sample smoke) to 4.6 seconds, without changing
the selected samples or metrics.
