#!/bin/bash
# Train V12 on gridworld_maze_multiobject — all 3 phases sequentially.
set -e
cd /home/xiaojy/projects/AdaWorld-improving

export PYTHONPATH=lam
export PYTHONUNBUFFERED=1
PY=/home/xiaojy/miniconda3/envs/adaworld/bin/python
GPU=7
DATASET=gridworld_maze_multiobject
BS=16
WORKERS=4

echo "=========================================="
echo "Phase A: decoder verification (3000 steps)"
echo "=========================================="
$PY lam/scripts/v12/run.py \
    --config phaseA --dataset $DATASET --phase A \
    --batch_size $BS --steps 3000 --gpu $GPU --num_workers $WORKERS \
    --checkpoint_every 1000

echo "=========================================="
echo "Phase B: structure LAM (5000 steps)"
echo "=========================================="
$PY lam/scripts/v12/run.py \
    --config phaseB --dataset $DATASET --phase B \
    --batch_size $BS --steps 5000 --gpu $GPU --num_workers $WORKERS \
    --checkpoint_every 1000

echo "=========================================="
echo "Phase C: joint training (5000 steps, from Phase B)"
echo "=========================================="
$PY lam/scripts/v12/run.py \
    --config phaseC --dataset $DATASET --phase C \
    --batch_size $BS --steps 5000 --gpu $GPU --num_workers $WORKERS \
    --checkpoint_every 1000 \
    --checkpoint result/v12/$DATASET/phaseB/model.pt

echo "=========================================="
echo "All phases complete!"
echo "=========================================="
