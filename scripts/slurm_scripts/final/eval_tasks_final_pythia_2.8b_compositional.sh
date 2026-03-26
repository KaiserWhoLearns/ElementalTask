#!/bin/bash
#SBATCH --account=bfcu-delta-gpu
#SBATCH --job-name=eval_final_pythia_2.8b_compositional
#SBATCH --output=logs/eval_final_pythia_2.8b_compositional_%A_%a.out
#SBATCH --error=logs/eval_final_pythia_2.8b_compositional_%A_%a.err
#SBATCH --time=2-00:00:00
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-43
#SBATCH --mail-user=isabelle_lee@seas.harvard.edu
#SBATCH --mail-type=ALL

source /u/glee4/ElementalTask/scripts/slurm_scripts/final/eval_tasks_final_lists.sh

CONFIG="/u/glee4/ElementalTask/eval_configs/pythia_2_8b_checkpoints_0b_1t_main.json"
OUTPUT_BASE="results/pythia_2.8b_continuous_final_iteration_compositional"

NUM_TASKS=${#COMPOSITIONAL_TASKS[@]}
TASK_IDX=$((SLURM_ARRAY_TASK_ID % NUM_TASKS))
TASK=${COMPOSITIONAL_TASKS[$TASK_IDX]}

cd /u/glee4/ElementalTask || exit 1
TASK_SANITIZED=$(echo "$TASK" | tr ':' '_')
EXISTING=$(find "$OUTPUT_BASE" -name "*_${TASK_SANITIZED}_metrics.json" 2>/dev/null | wc -l)
EXPECTED=$(python3 -c "import json; d=json.load(open('$CONFIG')); print(sum(len(v) for v in d.values()))")
[ "$EXISTING" -ge "$EXPECTED" ] && exit 0

source ~/.bashrc
conda activate base
mkdir -p logs
export PYTHONPATH=/u/glee4/ElementalTask:$PYTHONPATH

python scripts/eval_across_checkpoints.py \
  --model_configs "$CONFIG" \
  --output_path "$OUTPUT_BASE" \
  --tasks "$TASK" \
  --max_new_tokens 50 \
  --num_shots 5 \
  --eval_mode all \
  "$@"
