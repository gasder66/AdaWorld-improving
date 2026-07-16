# V16 Boxing experiment ledger

This file records milestone experiments for the V16 object-wise Boxing LAM. Individual hyperparameter runs remain under ignored `result/` and `reports/` directories and do not receive Git commits.

## Git policy

- Keep one branch for the current architecture line: `codex/v16-boxing-object-lam`.
- Commit changes to model behavior, data protocol, evaluation definitions, or reproducibility documentation.
- Do not commit checkpoints, generated datasets, plots, logs, or parameter-only reruns.
- Tag only reproducible milestones that change the conclusion of the project.
- Name experiment directories `v16_eNN_<short-purpose>` and save the exact command, source commit, dataset roots, seed, metrics, and artifact paths in `experiment.json`.

## Milestones

### E00 — movement smoke test

- Purpose: verify OCAtari RGB/mask data flow and independent object reconstruction.
- Data: `data/v16_boxing/stage1_movement_large`.
- Status: complete; infrastructure baseline only.

### E01 — adjacent mixed movement and isolated punch

- Purpose: establish whether object-wise IDM latents contain movement and punch information.
- Data: movement plus `data/v16_boxing/isolated_punch_large`.
- Status: complete; movement readable, punch nonlinear but phase coverage imbalanced.

### E02 — transition-balanced predictive baseline

- Source commit: `ae8dc3b`.
- Output: `result/v16/boxing_stage2_transition_balanced`.
- State MSE: normal `0.009117`, zero `0.025771`, genuine shuffle `0.023623`.
- Punch probe: linear `0.7302`, MLP `0.8013`, RBF-SVM `0.8038` balanced accuracy.
- Motion probe: dx `0.8612`, dy `0.9739` R2.
- Conclusion: the matching latent is necessary for every evaluated transition class, but punch phase and arm extension are not yet organized cleanly.

### E03 — targeted rare punch phases

- Purpose: increase genuinely distinct Player/Enemy onset, extend, hold, and switch transitions before adding interaction modeling.
- Data selection: OCAtari labels filter generated clips; labels remain excluded from model inputs.
- Planned outputs: `data/v16_boxing/targeted_punch_*`, `data/v16_boxing/transition_index_v2`, and `result/v16/v16_e03_targeted_punch`.
- Acceptance targets: preserve normal < zero/shuffle for all phases; punch linear balanced accuracy >= 0.80; punch MLP >= 0.85; arm-delta R2 > 0.10 without materially reducing dx/dy R2.

### E04 — interaction Transformer

- Start only after E03 produces a stable independent-FDM baseline.
- Compare independent FDM with a two-fighter Transformer on contact, hit, and occlusion subsets.
