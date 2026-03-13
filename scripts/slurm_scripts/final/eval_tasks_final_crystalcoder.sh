#!/bin/bash
#SBATCH --job-name=eval_final_crystal
#SBATCH --output=logs/eval_final_crystal_%A_%a.out
#SBATCH --error=logs/eval_final_crystal_%A_%a.err
#SBATCH --time=2-00:00:00
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# 109 final tasks x 1 config = 109 array jobs
#SBATCH --array=0-108
#SBATCH --mail-user=emmy@cmu.edu
#SBATCH --mail-type=ALL

source /data/user_data/mengyan3/tir3/ElementalTask/scripts/slurm_scripts/final/eval_tasks_final_lists.sh

CONFIG="/data/user_data/mengyan3/tir3/ElementalTask/eval_configs/crystal_checkpoints_0b_1t_main.json"
OUTPUT_BASE="results/crystal_continuous_final_iteration"

NUM_TASKS=${#FINAL_TASKS[@]}
TASK_IDX=$((SLURM_ARRAY_TASK_ID % NUM_TASKS))
TASK=${FINAL_TASKS[$TASK_IDX]}

echo "Final CrystalCoder run | task_idx=$TASK_IDX"
echo "Task: $TASK"
echo "Config: $CONFIG"

cd /data/user_data/mengyan3/tir3/ElementalTask || exit 1

TASK_SANITIZED=$(echo "$TASK" | tr ':' '_')
EXISTING=$(find "$OUTPUT_BASE" -name "*_${TASK_SANITIZED}_metrics.json" 2>/dev/null | wc -l)
EXPECTED=$(python3 -c "import json; d=json.load(open('$CONFIG')); print(sum(len(v) for v in d.values()))")

if [ "$EXISTING" -ge "$EXPECTED" ]; then
    echo "Task already complete for all checkpoints, skipping."
    exit 0
fi

source ~/.bashrc
conda activate elemental_tasks
mkdir -p logs
export PYTHONPATH=/data/user_data/mengyan3/tir3/ElementalTask:$PYTHONPATH

python scripts/eval_across_checkpoints.py \
    --model_configs "$CONFIG" \
    --output_path "$OUTPUT_BASE" \
    --tasks "$TASK" \
    --max_new_tokens 50 \
    --num_shots 5 \
    --eval_mode all \
    "$@"

exit $?