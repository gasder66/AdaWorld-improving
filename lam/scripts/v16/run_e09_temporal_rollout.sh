#!/usr/bin/env bash
set -euo pipefail

GPU="${E09_GPU:-0}"
SEED="${E09_SEED:-0}"
STEPS="${E09_STEPS:-3000}"
MAX_EVAL_BATCHES="${E09_MAX_EVAL_BATCHES:-500}"
CONDA_ENV="${E09_CONDA_ENV:-atari-v2}"
OUTPUT="${E09_OUTPUT:-result/v16/v16_e09_temporal_rollout_seed${SEED}_m2}"
INIT_CHECKPOINT="${E09_INIT_CHECKPOINT:-result/v16/v16_e08b_temporal_features_seed${SEED}_m2/model.pt}"

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

PYTHONPATH=lam python lam/scripts/v16/run_boxing_stage1.py \
  --init_checkpoint "${INIT_CHECKPOINT}" \
  --transition_index_dir data/v16_boxing/transition_index_v3 \
  --balanced_samples 50000 \
  --balance_mode phase \
  --max_eval_batches "${MAX_EVAL_BATCHES}" \
  --eval_batch_size 8 \
  --output "${OUTPUT}" \
  --steps "${STEPS}" \
  --pretrain_steps 0 \
  --batch_size 2 \
  --state_dim 64 \
  --latent_dim 16 \
  --fdm_type independent \
  --object_input_mode mask_structure_content \
  --lr 0.0002 \
  --gpu "${GPU}" \
  --seed "${SEED}" \
  --transition_gap 1 \
  --content_recolor_probability 0.8 \
  --structure_scale 2 \
  --idm_grid_size 4 \
  --idm_type conv \
  --temporal_context 3 \
  --prediction_horizon 2 \
  --rollout_weight 1.0 \
  --temporal_token_grid 8 \
  --temporal_layers 2 \
  --temporal_heads 4 \
  --learned_upsampling \
  --dynamic_mask_weight 2.0 \
  --edge_weight 0.5
