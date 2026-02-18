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

# python scripts/eval_across_checkpoints.py \
#     --model_configs eval_configs/k2v2_checkpoints.json \
#     --output_path "${RESULTS_DIR}" \
#     --load_vllm \
#     --max_new_tokens 20 \
#     --force_reeval

# echo "Evaluation complete."

# ============================================================================
# Step 1.5: Generate per-task pivot files from detailed_results.csv
# ============================================================================
echo "============================================"
echo "Step 1.5: Generating per-task pivot files"
echo "============================================"

python -c "
import pandas as pd
from pathlib import Path

results_dir = Path('${RESULTS_DIR}')
detailed_file = results_dir / 'detailed_results.csv'

if not detailed_file.exists():
    print(f'No detailed_results.csv found in {results_dir}')
    exit(0)

print(f'Loading {detailed_file}...')
df = pd.read_csv(detailed_file)

# Get all accuracy subtask columns
acc_cols = [c for c in df.columns if c.startswith('accuracy_') and c != 'accuracy']

# Task -> subtask mapping based on which rows have data
TASK_PREFIXES = {
    'compositional': 'compositional:',
    'simple_icl': 'simple_icl:',
}

created = 0

# Generate per-subtask pivot files
for task_category in df['task'].unique():
    task_df = df[df['task'] == task_category]

    # Find subtask columns with data for this task
    for acc_col in acc_cols:
        if task_df[acc_col].isna().all():
            continue

        subtask = acc_col.replace('accuracy_', '')

        # Determine full task name
        if task_category in TASK_PREFIXES:
            full_name = f'{TASK_PREFIXES[task_category]}{subtask}'
        else:
            full_name = subtask

        # Create pivot data
        pivot_data = task_df[['model', 'checkpoint', acc_col]].dropna(subset=[acc_col])
        if pivot_data.empty:
            continue

        pivot_data = pivot_data.rename(columns={acc_col: full_name})

        # Save with sanitized filename
        sanitized = full_name.replace(':', '_').replace('/', '_')
        output_file = results_dir / f'accuracy_pivot_{sanitized}.csv'
        pivot_data.to_csv(output_file, index=False)
        created += 1

# Also create aggregate task pivot files
for task in df['task'].unique():
    task_df = df[df['task'] == task][['model', 'checkpoint', 'accuracy']].dropna(subset=['accuracy'])
    if task_df.empty:
        continue
    task_df = task_df.rename(columns={'accuracy': task})
    sanitized = task.replace(':', '_').replace('/', '_')
    output_file = results_dir / f'accuracy_pivot_{sanitized}.csv'
    task_df.to_csv(output_file, index=False)
    created += 1

print(f'Created {created} per-task pivot files')
"

# ============================================================================
# Step 2: Trajectory Analysis - Emergence Points
# ============================================================================
echo "============================================"
echo "Step 2: Finding emergence points"
echo "============================================"

python scripts/trajectory_analysis/get_emergence_point.py \
    --results_dir "${RESULTS_DIR}" \
    --method relative \
    --threshold 0.5 \
    --plot \
    --plot_dir "${PLOTS_DIR}/emergence" \
    --output "${RESULTS_DIR}/emergence_points.csv"

# ============================================================================
# Step 3: Trajectory Analysis - Predict Compositional from Components
# ============================================================================
echo "============================================"
echo "Step 3: Predicting compositional from components"
echo "============================================"

python scripts/trajectory_analysis/predict_compositional_from_components.py \
    -d "${RESULTS_DIR}" \
    -o "${PLOTS_DIR}/compositional_prediction" \
    -m all

# ============================================================================
# Step 4: Trajectory Analysis - Trajectory Chaos
# ============================================================================
echo "============================================"
echo "Step 4: Analyzing trajectory chaos"
echo "============================================"

python scripts/trajectory_analysis/get_trajectory_chaos.py \
    -d "${RESULTS_DIR}" \
    -o "${RESULTS_DIR}/chaos_metrics.csv"

echo "============================================"
echo "All steps complete!"
echo "Results: ${RESULTS_DIR}"
echo "Plots:   ${PLOTS_DIR}"
echo "============================================"
