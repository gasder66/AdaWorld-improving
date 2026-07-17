#!/usr/bin/env bash
set -euo pipefail

GPU="${E08B_GPU:-0}"
MIN_FREE_MIB="${E08B_MIN_FREE_MIB:-8000}"
OUTPUT="${E08B_OUTPUT:-result/v16/v16_e08b_temporal_features}"

while true; do
  free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${GPU}")"
  if (( free_mib >= MIN_FREE_MIB )); then
    break
  fi
  echo "waiting for GPU ${GPU}: ${free_mib} MiB free, need ${MIN_FREE_MIB} MiB"
  sleep 30
done

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate adaworld

PYTHONPATH=lam python lam/scripts/v16/run_boxing_stage1.py \
  --init_checkpoint result/v16/v16_e07c_dynamic_edge/model.pt \
  --transition_index_dir data/v16_boxing/transition_index_v3 \
  --balanced_samples 50000 \
  --balance_mode phase \
  --max_eval_batches 500 \
  --eval_batch_size 8 \
  --output "${OUTPUT}" \
  --steps 3000 \
  --pretrain_steps 0 \
  --batch_size 2 \
  --state_dim 64 \
  --latent_dim 16 \
  --fdm_type independent \
  --object_input_mode mask_structure_content \
  --lr 0.0002 \
  --gpu "${GPU}" \
  --transition_gap 1 \
  --content_recolor_probability 0.8 \
  --structure_scale 2 \
  --idm_grid_size 4 \
  --idm_type conv \
  --temporal_context 3 \
  --temporal_token_grid 8 \
  --temporal_layers 2 \
  --temporal_heads 4 \
  --learned_upsampling \
  --dynamic_mask_weight 2.0 \
  --edge_weight 0.5
