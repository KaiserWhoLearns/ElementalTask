#!/bin/bash
#SBATCH --job-name=eval_final_pythia_1.4b_compositional
#SBATCH --output=logs/eval_final_pythia_1.4b_compositional_%A_%a.out
#SBATCH --error=logs/eval_final_pythia_1.4b_compositional_%A_%a.err
#SBATCH --time=2-00:00:00
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-43
#SBATCH --mail-user=isabelle_lee@seas.harvard.edu
#SBATCH --mail-type=ALL

source /n/holylabs/dam_lab/Users/iglee/ElementalTask/scripts/slurm_scripts/final/eval_tasks_final_lists.sh

CONFIG="/n/holylabs/dam_lab/Users/iglee/ElementalTask/eval_configs/pythia_1.4b_checkpoints_main.json"
OUTPUT_BASE="results/pythia_1.4b_continuous_final_iteration"

NUM_TASKS=${#COMPOSITIONAL_TASKS[@]}
TASK_IDX=$((SLURM_ARRAY_TASK_ID % NUM_TASKS))
TASK=${COMPOSITIONAL_TASKS[$TASK_IDX]}

cd /n/holylabs/dam_lab/Users/iglee/ElementalTask || exit 1
TASK_SANITIZED=$(echo "$TASK" | tr ':' '_')
EXISTING=$(find "$OUTPUT_BASE" -name "*_${TASK_SANITIZED}_metrics.json" 2>/dev/null | wc -l)
EXPECTED=$(python3 -c "import json; d=json.load(open('$CONFIG')); print(sum(len(v) for v in d.values()))")
[ "$EXISTING" -ge "$EXPECTED" ] && exit 0

source ~/.bashrc
conda activate base
mkdir -p logs
export PYTHONPATH=/n/holylabs/dam_lab/Users/iglee/ElementalTask:$PYTHONPATH

python scripts/eval_across_checkpoints.py \
  --model_configs "$CONFIG" \
  --output_path "$OUTPUT_BASE" \
  --tasks "$TASK" \
  --max_new_tokens 50 \
  --num_shots 5 \
  --eval_mode all \
  "$@"
