#!/bin/bash
source /home/xiaojy/miniconda3/etc/profile.d/conda.sh
conda activate adaworld

# Use system CUDA 12.1 NCCL (compatible with driver) instead of conda's cu11 NCCL
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}

export CUDA_VISIBLE_DEVICES=0,1,2
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}
export NCCL_ASYNC_ERROR_HANDLING=1
export LAM_RUN_ID=${LAM_RUN_ID:-resume_3gpu_opt}

python resume_a2d_weights_only.py 2>&1 | tee output_train_a2d_3gpu.log
