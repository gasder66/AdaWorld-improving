#!/bin/bash
# V12.1: 4 Diagnostic Experiments
#
# A: No-z FDM baseline — FDM(s_t, z=0) -> s_{t+1} on synthetic
# B: No-velocity structure — drop velocity from raw structure
# C: Causal/per-frame StructureEncoder — no future leakage
# D: Branching synthetic dataset — multimodal p(s_{t+1}|s_t)
#
# Success criterion: BoxIoU(normal) - BoxIoU(z=0) > 0.1
#
# Usage: bash lam/scripts/v12/run_diagnostics.sh [first_gpu=4]

set -e
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PYTHON="$(conda info --base 2>/dev/null || echo /home/xiaojy/miniconda3)/envs/adaworld/bin/python"
export PYTHONPATH="${ROOT}/lam"

GPU=${1:-4}
BATCH=16
STEPS=5000

run_train() {
    local cfg="$1" dataset="$2" phase="$3" gpu="$4" ckpt="$5" extra_flags="$6"
    echo "===== $cfg / $dataset / Phase $phase (GPU $gpu) ====="
    local cmd="$PYTHON lam/scripts/v12/run.py \
        --config $cfg --dataset $dataset --phase $phase \
        --batch_size $BATCH --steps $STEPS --gpu $gpu --seed 42 $extra_flags"
    if [ -n "$ckpt" ]; then
        cmd="$cmd --checkpoint $ckpt"
    fi
    echo "  $cmd"
    $cmd
}

eval_full() {
    local ckpt="$1" dataset="$2" gpu="$3" extra_flags="$4"
    echo "===== Eval: $ckpt (GPU $gpu) ====="
    $PYTHON lam/scripts/v12/eval_v12.py \
        --checkpoint "$ckpt" --dataset "$dataset" --gpu $gpu \
        --batch_size 8 --max_latent 2000 --seed 42 $extra_flags
}

# ============================================================
# Experiment A: No-z FDM baseline (Phase B only)
# ============================================================
echo "######## Experiment A: No-z FDM baseline ########"
run_train "expA_noz" "synthetic_minimal_nooverlap" "B" "$GPU" "" "--no_z"
eval_full "${ROOT}/result/v12/synthetic_minimal_nooverlap/expA_noz/model.pt" \
    "synthetic_minimal_nooverlap" "$GPU" "--no_z"

# ============================================================
# Experiment B: No-velocity (Phase B -> Phase C)
# ============================================================
echo ""
echo "######## Experiment B: No-velocity structure ########"
GPU_B=$((GPU + 1))
run_train "expB_novel_B" "synthetic_minimal_nooverlap" "B" "$GPU_B" "" "--no_velocity"
run_train "expB_novel_C" "synthetic_minimal_nooverlap" "C" "$GPU_B" \
    "${ROOT}/result/v12/synthetic_minimal_nooverlap/expB_novel_B/model.pt" "--no_velocity"
eval_full "${ROOT}/result/v12/synthetic_minimal_nooverlap/expB_novel_C/model.pt" \
    "synthetic_minimal_nooverlap" "$GPU_B" "--no_velocity"

# ============================================================
# Experiment C1: Causal StructureEncoder (Phase B -> Phase C)
# ============================================================
echo ""
echo "######## Experiment C1: Causal StructureEncoder ########"
GPU_C1=$((GPU + 2))
run_train "expC_causal_B" "synthetic_minimal_nooverlap" "B" "$GPU_C1" "" "--encoder_mode causal"
run_train "expC_causal_C" "synthetic_minimal_nooverlap" "C" "$GPU_C1" \
    "${ROOT}/result/v12/synthetic_minimal_nooverlap/expC_causal_B/model.pt" "--encoder_mode causal"
eval_full "${ROOT}/result/v12/synthetic_minimal_nooverlap/expC_causal_C/model.pt" \
    "synthetic_minimal_nooverlap" "$GPU_C1" "--encoder_mode causal"

# ============================================================
# Experiment C2: Per-frame StructureEncoder (Phase B -> Phase C)
# ============================================================
echo ""
echo "######## Experiment C2: Per-frame StructureEncoder ########"
GPU_C2=$((GPU + 3))
run_train "expC_perfrm_B" "synthetic_minimal_nooverlap" "B" "$GPU_C2" "" "--encoder_mode per_frame"
run_train "expC_perfrm_C" "synthetic_minimal_nooverlap" "C" "$GPU_C2" \
    "${ROOT}/result/v12/synthetic_minimal_nooverlap/expC_perfrm_B/model.pt" "--encoder_mode per_frame"
eval_full "${ROOT}/result/v12/synthetic_minimal_nooverlap/expC_perfrm_C/model.pt" \
    "synthetic_minimal_nooverlap" "$GPU_C2" "--encoder_mode per_frame"

# ============================================================
# Experiment D: Branching synthetic dataset (Phase B only)
# ============================================================
echo ""
echo "######## Experiment D: Branching synthetic dataset ########"
GPU_D=$((GPU))

echo "  Generating branching synthetic dataset..."
$PYTHON lam/scripts/v12/gen_branching_synthetic.py \
    --n_train 5000 --n_val 500 \
    --out "${ROOT}/data/v12/branching_synthetic"

run_train "expD_branch" "branching_synthetic" "B" "$GPU_D" "" ""
eval_full "${ROOT}/result/v12/branching_synthetic/expD_branch/model.pt" \
    "branching_synthetic" "$GPU_D" ""

echo ""
echo "========== ALL EXPERIMENTS DONE =========="
echo ""
echo "Summary targets:"
echo "  A: result/v12/synthetic_minimal_nooverlap/expA_noz/eval/eval.json"
echo "  B: result/v12/synthetic_minimal_nooverlap/expB_novel_C/eval/eval.json"
echo "  C1: result/v12/synthetic_minimal_nooverlap/expC_causal_C/eval/eval.json"
echo "  C2: result/v12/synthetic_minimal_nooverlap/expC_perfrm_C/eval/eval.json"
echo "  D: result/v12/branching_synthetic/expD_branch/eval/eval.json"
