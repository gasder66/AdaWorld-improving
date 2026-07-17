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

### E05 — oracle-mask ObjectLAM

- Purpose: test whether OCAtari masks can serve directly as a supervised object factorizer, removing RGB appearance from the IDM path while preserving object motion and punch information.
- Model interface: `object_input_mode` supports `masked_rgb_mask` (the E04 default), `mask_only`, and `masked_rgb`. Old checkpoints default to `masked_rgb_mask` and remain loadable.
- Mask-only invariant: object states and latents depend only on each fighter's one-channel oracle mask. RGB remains available only to the separate background encoder and auxiliary full-frame decoder.
- `v16_e05a_mask_only_independent`: overall normal/zero/shuffle state MSE is `0.010812` / `0.017994` / `0.019088`; mask IoU is `0.9054` / `0.7609` / `0.7724`.
- E05A event state MSE normal versus zero/shuffle remains favorable for all eight interaction classes. On occlusion it is `0.03201` versus `0.04220` / `0.05300`; on received-hit it is `0.01561` versus `0.02495` / `0.02695`.
- `v16_e05b_mask_only_residual_transformer`: overall normal/zero/shuffle state MSE is `0.009725` / `0.017879` / `0.018761`; mask IoU is `0.9113` / `0.7615` / `0.7761`.
- Relative to E05A, the residual Transformer lowers state MSE by approximately `10.1%` overall, `9.5%` on non-interaction, `8.2%` on contact, `5.5%` on hit, `9.4%` on received-hit, `9.0%` on occlusion, and `9.8%` on recovery.
- E05B latent probes: punch-change linear `0.7158`, punch-active MLP `0.7898`, movement dx/dy `0.8892` / `0.9773` R2. On the balanced interaction manifold, direction balanced accuracy is `0.9263` and identity balanced accuracy is `0.8400`, down from E04's `0.9862` identity result.
- Mask contours are substantially sharper and more directly interpretable than RGB reconstruction. The auxiliary RGB decoder still has rare catastrophic samples and is not the primary E05 quality criterion.
- Opponent-state masking raises error slightly, but matched versus shuffled opponent state remains effectively identical. E05 therefore strengthens the generic cross-object context result but still does not establish matching-specific one-step interaction.
- Conclusion: oracle masks are a viable and cleaner object-state interface for Atari. They preserve movement and punch semantics, make the matching latent necessary for future contour prediction, and reduce—but do not eliminate—fighter identity information.

### E06 — bounded structure-content conditioning

- Purpose: retain the E05 mask-only structure/IDM/FDM path while adding the smallest possible RGB content condition for object appearance. This is a bounded test, not a claim that action can be made completely structure-independent.
- Information path: oracle masks alone produce object structure states and the two object latents. RGB content is pooled separately under each mask and enters only the object decoder through FiLM. The decoder never receives `z` directly, and content never enters IDM/FDM.
- Training intervention: with probability `0.8`, each fighter is assigned a random color that remains fixed across the two transition frames. This prevents the decoder from recovering fighter appearance from slot identity alone.
- Output: `result/v16/v16_e06a_structure_content`; independent FDM; 1,200 visual pretraining and 4,000 dynamics steps on `transition_index_v3`.
- Overall normal/zero/shuffle state MSE is `0.010075` / `0.017994` / `0.018912`; mask IoU is `0.9143` / `0.7614` / `0.7757`. Relative to E05A, normal state MSE improves about `6.8%`, mask IoU improves from `0.9054` to `0.9143`, and object RGB L1 improves from `0.00777` to `0.00590`.
- Content intervention on 3,427 balanced validation transitions leaves predicted structure exactly unchanged within numerical precision. On original RGB, object RGB L1 is `0.00792` with normal content, `0.41770` with zero content, and `0.78492` after swapping fighter content.
- Rebinding succeeds quantitatively: after swapping content, predicted fighter color is much closer to the donor fighter than the original target (`0.06174` versus `0.78492` L1). With controlled random recoloring, the same donor preference remains (`0.02165` versus `0.26100`).
- Frozen-latent probes remain useful: movement dx/dy R2 is `0.7951` / `0.9519`, direction balanced accuracy is `0.9288`, and punch balanced accuracy is `0.7163`. Identity balanced accuracy is still `0.8592`, so this experiment does not produce a pure identity-free action code.
- Visual inspection shows that normal content restores fighter colors, zero content produces dark generic fighters, and swapping content exchanges appearance while preserving the predicted mask geometry. Rare catastrophic reconstruction remains, especially in one received-hit example, so reconstruction quality is improved but not solved.
- `v16_e06b_structure_content_transformer` adds the E04/E05 residual two-token Transformer on top of E06A. Overall normal/zero/shuffle state MSE is `0.009439` / `0.017786` / `0.018473`, a `6.3%` state-MSE improvement over E06A; mask IoU is `0.9157`.
- E06B probes remain similar: movement dx/dy R2 `0.7888` / `0.9484`, direction balanced accuracy `0.9316`, punch balanced accuracy `0.7087`, and identity balanced accuracy `0.8507`.
- As in E04/E05, opponent masking has only tiny effects and matched opponent shuffling is neutral. Even in occlusion, opponent-state masking changes MSE by only `0.000066`, while shuffling changes it by `-0.00000025`. The Transformer is a useful learned residual in aggregate but still does not establish matching-specific one-step interaction.
- Conclusion: the minimal conditional-content path is viable and does not contaminate the mask-only transition path. It improves appearance reconstruction, supports controlled rebinding, and remains compatible with the residual interaction FDM, but the latent should still be described as an object-specific visual transition rather than a pure action representation.

### E07 — punch-contour resolution and dynamic-boundary loss

- Purpose: test the hypothesis that stride-4 visual states, global IDM pooling, and bilinear mask decoding blur the thin arm contour and collapse distinct punch stages.
- E07A first separates decoder capacity from transition prediction by decoding the oracle next state with the same frozen decoder. On 3,000 phase-balanced transitions, E06B oracle/predicted mask IoU is `0.9355` / `0.8670`, and dynamic-region IoU is `0.8077` / `0.7341`. The large oracle-versus-predicted gap shows that the FDM/latent path is a larger limitation than the old decoder alone.
- E07B (`v16_e07b_highres_spatial_idm`) changes the object-state map from stride 4 to stride 2, replaces global IDM pooling with a `4x4` spatial grid, and uses learned PixelShuffle upsampling. Oracle mask IoU rises to `0.9904`, confirming that the old low-resolution decoder discarded recoverable contour detail. Predicted mask IoU rises to `0.8847`.
- E07C (`v16_e07c_dynamic_edge`) fine-tunes E07B with a dynamic-region mask loss (`2.0`) and an edge-gradient loss (`0.5`). Overall predicted mask IoU rises to `0.9094`, dynamic-region IoU rises to `0.7748`, and dynamic-region L1 falls from E06B's `0.0631` to `0.0449`.
- E06B to E07C predicted mask IoU improves for every transition phase: movement `0.9201 -> 0.9915`, onset `0.8397 -> 0.8655`, extend `0.8376 -> 0.8648`, hold `0.8616 -> 0.9127`, retract `0.8972 -> 0.9490`, and switch `0.8460 -> 0.8730`.
- Matching latents remain necessary: main validation normal/zero/shuffle mask IoU is `0.9969` / `0.7595` / `0.8475`. The very high global IoU is dominated by full-mask overlap, so phase-balanced target-slot contour metrics above remain the stricter result.
- A new balanced six-way probe uses 1,500 training and 500 validation examples per phase. E06B linear/MLP balanced accuracy is `0.4650` / `0.5763`; E07C is `0.4617` / `0.5370`, versus `0.1667` chance. Onset and hold remain readable, while extend, retract, and switch are heavily confused.
- The existing continuous probes agree with this result: E07C punch-active MLP accuracy is `0.8001` and movement dx/dy R2 is `0.8596` / `0.9783`, but left/right arm-delta R2 remains approximately `-0.0014` / `0.0081`.
- Visual inspection confirms that thresholded masks now preserve much sharper punch contours and stage-dependent arm shapes. Probability maps remain softer around changing boundaries, and rare catastrophic full-frame reconstructions still occur, especially for selected received-hit and recovery examples.
- Conclusion: the pooling/decoder hypothesis was partly correct for visual sharpness but incorrect as a complete explanation of latent semantics. E07 materially improves contour reconstruction, including every punch phase, but does not create a better phase-organized action space. Future work should add an explicit short temporal window or phase-sensitive predictive objective before further CNN kernel tuning.

### E08A — original AdaWorld-style two-frame ST-IDM

- Purpose: isolate whether AdaWorld's action-prompt spatiotemporal attention extracts a more phase-readable two-frame transition than E07C's convolutional difference IDM.
- Source check: the unmodified AdaWorld code at repository commit `fe3701e` trains its latent-action autoencoder with `num_frames: 2`. It applies spatial attention within each frame, temporal attention across the two frames, and reads the second-frame action-prompt token. E08A follows this two-frame information path per fighter; it is not a longer temporal window.
- Replacement ST-IDM (`v16_e08a_adaworld_st_idm`) uses an `8x8` object token grid, two factorized spatial/temporal blocks, four heads, and one action prompt per frame. It is initialized from E07C for all compatible encoder, FDM, and decoder parameters.
- The replacement fails to integrate with the existing object-state/FDM interface. Overall normal state MSE is `0.01488`, worse than both E07C (`0.00705`) and the identity state (`0.01175`). Main mask IoU falls to `0.8060`; phase-balanced target-slot contour IoU falls from `0.9094` to `0.7622`.
- Six-way phase balanced accuracy also falls: linear/MLP is `0.3713` / `0.4027`, compared with E07C's `0.4617` / `0.5370`. This rules out the interpretation that reconstruction alone failed while ST latents became more semantic.
- Residual ST-IDM (`v16_e08a2_residual_st_idm`) retains the trained E07C convolutional IDM and adds a gated ST correction. The base IDM and FDM are frozen, so the experiment tests only incremental information supplied by two-frame ST attention.
- The residual design preserves the baseline: normal/zero/shuffle mask IoU is `0.9973` / `0.7595` / `0.8475`, state MSE is `0.00708`, phase-balanced target-slot contour IoU is `0.9088`, and dynamic-region IoU is `0.7751`.
- It supplies no measurable phase benefit. Six-way linear/MLP balanced accuracy is `0.4613` / `0.5363`, effectively identical to E07C's `0.4617` / `0.5370`.
- Conclusion: two-frame ST attention is neither a drop-in replacement for the current IDM nor an incremental source of phase semantics. A longer window must be paired with a phase-sensitive self-supervised objective; merely allowing attention over more frames is unlikely to be used when the ordinary one-step reconstruction target is already determined by the final frame pair.
