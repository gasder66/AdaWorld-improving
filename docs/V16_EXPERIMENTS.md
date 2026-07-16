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
- Source commit: `640aab4`.
- Data: `targeted_enemy_phases_v1`, `targeted_enemy_switch_v1`, and `transition_index_v2`.
- Output: `result/v16/v16_e03_targeted_punch`.
- Overall state MSE: normal `0.009029`, zero `0.025349`, genuine shuffle `0.024556`.
- Original validation probe: linear `0.7202`, MLP `0.8192`, RBF-SVM `0.8048`; dx `0.8568`, dy `0.9766` R2.
- Targeted Enemy probe: linear `0.8468`, MLP `0.8840`, RBF-SVM `0.9090`; arm-delta left `0.0294`, right `0.0685` R2.
- Event ablation: normal beats zero and genuine shuffle for all six events, with 161 onset, 1,131 extend, 536 hold, 4,696 retract, and 289 switch validation transitions in the bounded evaluation.
- Conclusion: rare-stage coverage and punch readability improved substantially without reducing the predictive need for matching latents. Fine-grained arm-extension geometry remains below the `0.10` R2 target, so E03 is a successful punch-detection baseline but not yet a complete phase-regression result.

### E04 — interaction Transformer

- Data protocol: `interaction_general_v1`, balanced `interaction_hit_player_v1` / `interaction_hit_enemy_v1`, `interaction_occlusion_v1`, and `transition_index_v3`.
- Validation coverage: 314 hit, 314 received-hit, 299 occlusion, 600 contact, and 600 recovery target transitions in the bounded comparison.
- `v16_e04b_interaction_independent`: overall state MSE `0.009747`.
- Replacement Transformer (`v16_e04b_interaction_transformer`) failed: overall state MSE `0.012588` and worse results in every event class. It is retained only as a negative experiment.
- Residual Transformer (`v16_e04c_residual_transformer`) preserves the pretrained independent FDM and learns a two-token interaction correction. Overall state MSE is `0.009534`, with normal/zero/shuffle at `0.009534` / `0.025162` / `0.025321`.
- Relative to the same-data independent baseline, the residual Transformer improves non-interaction `2.0%`, contact `1.9%`, hit `4.5%`, received-hit `2.5%`, occlusion `3.4%`, and recovery `2.0%`.
- Original punch probe remains stable: linear `0.7550`, MLP `0.8096`, RBF-SVM `0.8093`; dx `0.8836`, dy `0.9785` R2.
- Opponent-state masking raises target error most for hit and occlusion, but opponent-state shuffling is nearly neutral. Current evidence supports a small generic cross-object context benefit, not yet strong matching-specific interaction.
- A position-aware visual-token variant is implemented as `spatial_interaction` but remains untrained. A batch-size guard now prevents transition training with batch `< 2`, because genuine same-object shuffle is impossible with one transition per object.
