#!/bin/bash
#SBATCH --job-name=eval_final_crystal
#SBATCH --output=logs/eval_final_crystal_%A_%a.out
#SBATCH --error=logs/eval_final_crystal_%A_%a.err
#SBATCH --mail-user=hsun74@jhu.edu
#SBATCH --mail-type=FAIL,END
#SBATCH -A mdredze1_gpu
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --array=0-108

BASE_DIR="/scratch4/mdredze1/hsun74/ElementalTask"

source "${BASE_DIR}/scripts/slurm_scripts/rockfish_scripts/eval_tasks_final_lists.sh"

CONFIG="${BASE_DIR}/eval_configs/crystal_checkpoints_0b_1t_main.json"
OUTPUT_BASE="results/crystal_continuous_final_iteration"

NUM_TASKS=${#FINAL_TASKS[@]}
TASK_IDX=$((SLURM_ARRAY_TASK_ID % NUM_TASKS))
TASK=${FINAL_TASKS[$TASK_IDX]}

echo "Final CrystalCoder run | task_idx=$TASK_IDX"
echo "Task: $TASK"
echo "Config: $CONFIG"

cd "${BASE_DIR}" || exit 1

TASK_SANITIZED=$(echo "$TASK" | tr ':' '_')
EXISTING=$(find "$OUTPUT_BASE" -name "*_${TASK_SANITIZED}_metrics.json" 2>/dev/null | wc -l)
EXPECTED=$(python3 -c "import json; d=json.load(open('$CONFIG')); print(sum(len(v) for v in d.values()))")

if [ "$EXISTING" -ge "$EXPECTED" ]; then
    echo "Task already complete for all checkpoints, skipping."
    exit 0
fi

module load gcc/11.4.0
module load anaconda
conda activate elementaltask

# Force newer libstdc++ for FlashInfer/vLLM worker subprocesses
export LD_PRELOAD=/data/apps/extern/spack_on/gcc/9.3.0/gcc/11.4.0-hzz5maaw347vs5ygsiqkl77ua35qa2d7/lib64/libstdc++.so.6

mkdir -p logs
export PYTHONPATH="${BASE_DIR}:${PYTHONPATH}"

python scripts/eval_across_checkpoints.py \
    --model_configs "$CONFIG" \
    --output_path "$OUTPUT_BASE" \
    --tasks "$TASK" \
    --max_new_tokens 50 \
    --num_shots 5 \
    --eval_mode all \
    "$@"

exit $?
