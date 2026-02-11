#!/bin/bash
#SBATCH --job-name=olmo2_1b_eval
#SBATCH --mail-user=hsun74@jhu.edu
#SBATCH --mail-type=FAIL,END
#SBATCH --partition=a100
#SBATCH -A mdredze1_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=50G
#SBATCH --gpus=1
#SBATCH --time=1-15:00:00
#SBATCH --chdir=/scratch4/mdredze1/hsun74/ElementalTask
#SBATCH --export=all
#SBATCH --output=/scratch4/mdredze1/hsun74/ElementalTask/logs/output_olmo2_1b_eval.log
#SBATCH --error=/scratch4/mdredze1/hsun74/ElementalTask/logs/error_olmo2_1b_eval.log

# ============================================================================
# SLURM Script: OLMo-2 1B Evaluation + Trajectory Analysis
# ============================================================================

BASE_DIR="/scratch4/mdredze1/hsun74/ElementalTask"
RESULTS_DIR="${BASE_DIR}/results/olmo2_1b"
PLOTS_DIR="${BASE_DIR}/plots/olmo2_1b"

# Create output directories
mkdir -p "${RESULTS_DIR}"
mkdir -p "${PLOTS_DIR}"
mkdir -p "${BASE_DIR}/logs"

# Activate conda environment
source activate elementaltask

export PYTHONPATH="${BASE_DIR}:${PYTHONPATH}"

# ============================================================================
# Step 1: Evaluate OLMo-2 1B across all checkpoints
# ============================================================================
echo "============================================"
echo "Step 1: Running evaluation across checkpoints"
echo "============================================"

# python scripts/eval_across_checkpoints.py \
#     --model_configs eval_configs/olmo2_checkpoints_1b.json \
#     --output_path "${RESULTS_DIR}" \
#     --load_vllm \
#     --max_new_tokens 100

# python scripts/eval_across_checkpoints.py \
#     --model_configs eval_configs/sanity_check.json \
#     --tasks simple_icl \
#     --output_path results/test_run \
#     --load_vllm \
#     --max_new_tokens 100

echo "Evaluation complete."

# # ============================================================================
# # Step 2: Trajectory Analysis - Emergence Points
# # ============================================================================
# echo "============================================"
# echo "Step 2: Finding emergence points"
# echo "============================================"

python scripts/trajectory_analysis/get_emergence_point.py \
    --results_dir results/test_run \
    --method relative \
    --threshold 0.5 \
    --plot \
    --plot_dir "${PLOTS_DIR}/emergence" \
    --output "${RESULTS_DIR}/emergence_points.csv"

# # ============================================================================
# # Step 3: Trajectory Analysis - Predict Compositional from Components
# # ============================================================================
# echo "============================================"
# echo "Step 3: Predicting compositional from components"
# echo "============================================"

python scripts/trajectory_analysis/predict_compositional_from_components.py \
    --results_dir results/test_run \
    --output_dir "${PLOTS_DIR}/compositional_prediction" \
    --method all

# # ============================================================================
# # Step 4: Trajectory Analysis - Trajectory Chaos
# # ============================================================================
# echo "============================================"
# echo "Step 4: Analyzing trajectory chaos"
# echo "============================================"

python scripts/trajectory_analysis/get_trajectory_chaos.py \
    --results_dir results/test_run \
    --output "${RESULTS_DIR}/chaos_metrics.csv"

# echo "============================================"
# echo "All steps complete!"
# echo "Results: ${RESULTS_DIR}"
# echo "Plots:   ${PLOTS_DIR}"
# echo "============================================"
