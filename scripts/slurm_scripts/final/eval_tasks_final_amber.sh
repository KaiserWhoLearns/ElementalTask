#!/bin/bash
#SBATCH --job-name=eval_final_amber
#SBATCH --output=logs/eval_final_amber_%A_%a.out
#SBATCH --error=logs/eval_final_amber_%A_%a.err
#SBATCH --time=2-00:00:00
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# 109 final tasks x 1 config = 109 array jobs
#SBATCH --array=90-100
#SBATCH --mail-user=emmy@cmu.edu
#SBATCH --mail-type=ALL

# currently on:  90-100 (out of 108)
source /data/user_data/mengyan3/tir3/ElementalTask/scripts/slurm_scripts/final/eval_tasks_final_lists.sh

CONFIG="/data/user_data/mengyan3/tir3/ElementalTask/eval_configs/amber_checkpoints_0b_1t_main.json"
OUTPUT_BASE="results/amber_continuous_final_iteration"

NUM_TASKS=${#FINAL_TASKS[@]}
TASK_IDX=$((SLURM_ARRAY_TASK_ID % NUM_TASKS))
TASK=${FINAL_TASKS[$TASK_IDX]}

echo "Final Amber run | task_idx=$TASK_IDX"
echo "Task: $TASK"
echo "Config: $CONFIG"

cd /data/user_data/mengyan3/tir3/ElementalTask || exit 1

TASK_SANITIZED=$(echo "$TASK" | tr ':' '_')
TASK_COMPLETE=$(python3 - <<PY
import glob
import json
import os
import re

config = "$CONFIG"
output_base = "$OUTPUT_BASE"
task_sanitized = "$TASK_SANITIZED"

cfg = json.load(open(config))
model_id = next(iter(cfg.keys()))
checkpoints = cfg[model_id]
model_safe = model_id.replace('/', '_')

pattern = os.path.join(output_base, "**", f"*_{task_sanitized}_metrics.json")
files = glob.glob(pattern, recursive=True)

done = set()
rx = re.compile(rf"^{re.escape(model_safe)}_(.+)_{re.escape(task_sanitized)}_metrics\\.json$")
for path in files:
    m = rx.match(os.path.basename(path))
    if m:
        done.add(m.group(1))

print("1" if set(checkpoints).issubset(done) else "0")
PY
)

if [ "$TASK_COMPLETE" = "1" ]; then
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