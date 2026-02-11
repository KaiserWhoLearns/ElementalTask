#!/bin/bash
#SBATCH --job-name=k2v2_eval
#SBATCH --mail-user=hsun74@jhu.edu
#SBATCH --mail-type=FAIL,END
#SBATCH -A mdredze80_gpu
#SBATCH --partition=ica100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=180G
#SBATCH --gpus=2
#SBATCH --time=3-00:00:00
#SBATCH --chdir=/scratch4/mdredze1/hsun74/ElementalTask
#SBATCH --export=all
#SBATCH --output=/scratch4/mdredze1/hsun74/ElementalTask/logs/output_k2v2_eval2.log
#SBATCH --error=/scratch4/mdredze1/hsun74/ElementalTask/logs/error_k2v2_eval2.log

module load gcc/11.4.0
module load anaconda
conda activate elementaltask

# Force newer libstdc++ for FlashInfer/vLLM worker subprocesses
# (system /lib64/libstdc++.so.6 only has GLIBCXX up to 3.4.25, FlashInfer needs 3.4.26)
export LD_PRELOAD=/data/apps/extern/spack_on/gcc/9.3.0/gcc/11.4.0-hzz5maaw347vs5ygsiqkl77ua35qa2d7/lib64/libstdc++.so.6

# Clear stale FlashInfer cache (was built against old libstdc++)
rm -rf /home/hsun74/.cache/flashinfer

# ============================================================================
# SLURM Script: LLM360 K2-V2 (70B) Evaluation + Trajectory Analysis
# ============================================================================
# Kill any lingering vLLM/Python processes for your user
# KILL THE ZOOMBIE PROCESSES
pkill -u hsun74 -f python


BASE_DIR="/scratch4/mdredze1/hsun74/ElementalTask"
RESULTS_DIR="${BASE_DIR}/results/k2v2"
PLOTS_DIR="${BASE_DIR}/plots/k2v2"

# Create output directories
mkdir -p "${RESULTS_DIR}"
mkdir -p "${PLOTS_DIR}"
mkdir -p "${BASE_DIR}/logs"

# Activate conda environment
source activate elementaltask

export PYTHONPATH="${BASE_DIR}:${PYTHONPATH}"

# Force IPv4 for gloo/nccl to avoid IPv4/IPv6 address family mismatch
# (RuntimeError: ss1.ss_family == ss2.ss_family)
# Dynamically detect the first non-loopback interface (bond0 doesn't exist on compute nodes)
export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')
IFACE=$(ip -o -4 addr show | grep -v ' lo ' | head -1 | awk '{print $2}')
echo "Detected network interface: ${IFACE}"
export GLOO_SOCKET_IFNAME=${IFACE}
export NCCL_SOCKET_IFNAME=${IFACE}
export NCCL_SOCKET_FAMILY=AF_INET

# ============================================================================
# Step 1: Evaluate K2-V2 across all checkpoints
# ============================================================================
echo "============================================"
echo "Step 1: Running evaluation across checkpoints"
echo "============================================"

python scripts/eval_across_checkpoints.py \
    --model_configs eval_configs/k2v2_sanitycheck.json \
    --output_path "${RESULTS_DIR}" \
    --load_vllm \
    --max_new_tokens 20 \
    --quantization bitsandbytes

echo "Evaluation complete."

# # ============================================================================
# # Step 2: Trajectory Analysis - Emergence Points
# # ============================================================================
# echo "============================================"
# echo "Step 2: Finding emergence points"
# echo "============================================"

# python scripts/trajectory_analysis/get_emergence_point.py \
#     --results_dir "${RESULTS_DIR}" \
#     --method relative \
#     --threshold 0.5 \
#     --plot \
#     --plot_dir "${PLOTS_DIR}/emergence" \
#     --output "${RESULTS_DIR}/emergence_points.csv"

# # ============================================================================
# # Step 3: Trajectory Analysis - Predict Compositional from Components
# # ============================================================================
# echo "============================================"
# echo "Step 3: Predicting compositional from components"
# echo "============================================"

# python scripts/trajectory_analysis/predict_compositional_from_components.py \
#     --results_dir "${RESULTS_DIR}" \
#     --output_dir "${PLOTS_DIR}/compositional_prediction" \
#     --method all

# # ============================================================================
# # Step 4: Trajectory Analysis - Trajectory Chaos
# # ============================================================================
# echo "============================================"
# echo "Step 4: Analyzing trajectory chaos"
# echo "============================================"

# python scripts/trajectory_analysis/get_trajectory_chaos.py \
#     --results_dir "${RESULTS_DIR}" \
#     --output "${RESULTS_DIR}/chaos_metrics.csv"

# echo "============================================"
# echo "All steps complete!"
# echo "Results: ${RESULTS_DIR}"
# echo "Plots:   ${PLOTS_DIR}"
# echo "============================================"
