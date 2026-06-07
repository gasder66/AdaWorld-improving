#!/bin/bash
# Wrapper script to run experiments reliably with output logging
USAGE="Usage: $0 <gpu_id> <name> <num_slots> <batch_size> <steps> [--aux_loss <val>] [--competition] [--no_checkpoint]"

if [ $# -lt 5 ]; then
    echo "$USAGE"
    exit 1
fi

GPU=$1
NAME=$2
NUM_SLOTS=$3
BATCH_SIZE=$4
STEPS=$5
shift 5

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/results/slot_attention_exp_v2/log_${NAME}.txt"
RESULTS_DIR="${SCRIPT_DIR}/results/slot_attention_exp_v2"

# Build command
CMD="CUDA_VISIBLE_DEVICES=${GPU} ${HOME}/miniconda3/envs/adaworld/bin/python ${SCRIPT_DIR}/run_single.py --gpu 0 --name ${NAME} --num_slots ${NUM_SLOTS} --batch_size ${BATCH_SIZE} --steps ${STEPS} $@"

# Run with unbuffered output  
export CUDA_VISIBLE_DEVICES="${GPU}"
echo "Starting on GPU ${GPU}: ${CMD}" > "${LOG_FILE}"
echo "Logging to: ${LOG_FILE}"
stdbuf -oL -eL ${HOME}/miniconda3/envs/adaworld/bin/python ${SCRIPT_DIR}/run_single.py --gpu 0 --name ${NAME} --num_slots ${NUM_SLOTS} --batch_size ${BATCH_SIZE} --steps ${STEPS} $@ >> "${LOG_FILE}" 2>&1 &
PID=$!
echo "PID: ${PID}" >> "${LOG_FILE}"
echo "Process started with PID: ${PID}" 
