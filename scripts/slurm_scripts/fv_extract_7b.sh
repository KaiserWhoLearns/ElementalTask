#!/bin/bash
#SBATCH --job-name=fv_7b
#SBATCH --output=logs/fv_extract_7b_%j.out
#SBATCH --error=logs/fv_extract_7b_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuA100x4
#SBATCH --account=bfcu-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# =============================================================================
# FV EXTRACTION — OLMo-2-1124-7B
# Extracts function vectors and builds skill basis for 7B model
# =============================================================================

# Redirect all caches to nvme
export HF_HOME=/work/nvme/bfcu/mliu7/hf_cache
export HUGGINGFACE_HUB_CACHE=/work/nvme/bfcu/mliu7/hf_cache/hub
export TRANSFORMERS_CACHE=/work/nvme/bfcu/mliu7/hf_cache/hub
export XDG_CACHE_HOME=/work/nvme/bfcu/mliu7/cache
export XET_CACHE=/work/nvme/bfcu/mliu7/xet_cache
export TMPDIR=/work/nvme/bfcu/mliu7/tmp
# Disable xet chunk-cache downloader — it writes to ~/.cache/huggingface/xet/
# which symlinks to /work/hdd/bfcu (over quota). Regular HTTP download works fine.
export HF_HUB_DISABLE_XET=1
mkdir -p "$HF_HOME/hub" "$XDG_CACHE_HOME" "$XET_CACHE" "$TMPDIR"

# CRITICAL: ~/.cache/huggingface is a symlink to /work/hdd/bfcu which is over quota.
# Override it so nothing accidentally follows the symlink.
if [ -L "$HOME/.cache/huggingface" ]; then
    rm "$HOME/.cache/huggingface"
    ln -s /work/nvme/bfcu/mliu7/hf_cache "$HOME/.cache/huggingface"
    echo "Redirected ~/.cache/huggingface -> /work/nvme/bfcu/mliu7/hf_cache"
fi

source ~/.bashrc
conda activate elemental_tasks

mkdir -p logs

export PYTHONPATH=/projects/bfcu/ElementalTask:$PYTHONPATH

echo "========================================================================"
echo "FV Extraction: OLMo-2-1124-7B"
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "========================================================================"

python function_vecs/experiments/analyze_real_tasks.py \
    --model allenai/OLMo-2-1124-7B \
    --device cuda \
    --dtype bfloat16 \
    --layer -1 \
    --num-heads 10 \
    --num-samples 20 \
    --only-correct \
    --results-dir results/olmo2_continuous_7b_early_revised \
    --output-dir function_vecs/results/olmo2_7b_correct_only \
    --holdout-compositional \
    --basis-method svd \
    --epsilons 0.5 0.3 0.2 0.15 0.1 0.05 0.01

EXIT_CODE=$?

echo ""
echo "========================================================================"
echo "Job completed! Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "========================================================================"

exit $EXIT_CODE
