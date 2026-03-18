#!/bin/bash
#SBATCH --job-name=eval_pythia14b
#SBATCH --output=logs/eval_pythia14b_%A_%a.out
#SBATCH --error=logs/eval_pythia14b_%A_%a.err
#SBATCH --time=8:00:00
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuA100x4
#SBATCH --account=bfcu-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-97%8
#SBATCH --mail-user=isabelle_lee@seas.harvard.edu
#SBATCH --mail-type=ALL

# =============================================================================
# PYTHIA-1.4B EVALUATION — ALL TASKS
# =============================================================================

# Regular tasks (22 tasks)
REGULAR_TASKS=(
    "basic_arithmetic"
    "ignoring_context"
    "simple_icl:uppercase"
    "simple_icl:lowercase"
    "simple_icl:first_letter"
    "simple_icl:last_letter"
    "simple_icl:translate_eng_fr"
    "simple_icl:translate_fr_eng"
    "simple_icl:translate_eng_sp"
    "simple_icl:translate_sp_eng"
    "simple_icl:present_to_gerund"
    "simple_icl:singular_to_plural"
    "simple_icl:country_to_capital"
    "simple_icl:country_to_currency"
    "copying"
    "simple"
    "token_reversal"
    "string_analogy"
    "textfrct"
    "part_of_speech"
    "math"
    "ioi_task"
)

# TextFRCT category tasks (22 tasks)
TEXTFRCT_TASKS=(
    "textfrct:CV1"
    "textfrct:CV2"
    "textfrct:CV3"
    "textfrct:FA3"
    "textfrct:FE1"
    "textfrct:I1"
    "textfrct:I2"
    "textfrct:MA2"
    "textfrct:MA3"
    "textfrct:RG1"
    "textfrct:RG2"
    "textfrct:RG3"
    "textfrct:RL1"
    "textfrct:RL3"
    "textfrct:RL4"
    "textfrct:V1"
    "textfrct:V2"
    "textfrct:V3"
    "textfrct:V4"
    "textfrct:V5"
    "textfrct:XU1"
    "textfrct:XU2"
)

# Original compositional tasks (26 tasks)
COMPOSITIONAL_TASKS=(
    "compositional:first_upper"
    "compositional:last_upper"
    "compositional:lower_first"
    "compositional:lower_last"
    "compositional:lower_reverse"
    "compositional:reverse_first"
    "compositional:reverse_last"
    "compositional:reverse_lower"
    "compositional:reverse_upper"
    "compositional:upper_first"
    "compositional:upper_last"
    "compositional:upper_reverse"
    "compositional:gerund_first"
    "compositional:gerund_last"
    "compositional:gerund_reverse"
    "compositional:gerund_upper"
    "compositional:plural_first"
    "compositional:plural_last"
    "compositional:plural_reverse"
    "compositional:plural_upper"
    "compositional:translate_eng_fr_reverse"
    "compositional:translate_eng_fr_upper"
    "compositional:translate_eng_sp_reverse"
    "compositional:translate_eng_sp_upper"
    "compositional:translate_fr_eng_upper"
    "compositional:translate_sp_eng_upper"
)

# New compositional tasks (28 tasks)
NEW_COMPOSITIONAL_TASKS=(
    "compositional:gerund_lower"
    "compositional:gerund_reverse_first"
    "compositional:gerund_upper_reverse"
    "compositional:lower_reverse_first"
    "compositional:lower_reverse_last"
    "compositional:plural_lower"
    "compositional:plural_reverse_first"
    "compositional:plural_upper_reverse"
    "compositional:reverse_lower_first"
    "compositional:reverse_upper_first"
    "compositional:translate_eng_fr_first"
    "compositional:translate_eng_fr_last"
    "compositional:translate_eng_fr_lower"
    "compositional:translate_eng_fr_upper_reverse"
    "compositional:translate_eng_sp_first"
    "compositional:translate_eng_sp_last"
    "compositional:translate_eng_sp_lower"
    "compositional:translate_eng_sp_upper_reverse"
    "compositional:translate_fr_eng_first"
    "compositional:translate_fr_eng_last"
    "compositional:translate_fr_eng_lower"
    "compositional:translate_fr_eng_reverse"
    "compositional:translate_sp_eng_first"
    "compositional:translate_sp_eng_last"
    "compositional:translate_sp_eng_lower"
    "compositional:translate_sp_eng_reverse"
    "compositional:upper_reverse_first"
    "compositional:upper_reverse_last"
)

# Combine all tasks
TASKS=("${REGULAR_TASKS[@]}" "${TEXTFRCT_TASKS[@]}" "${COMPOSITIONAL_TASKS[@]}" "${NEW_COMPOSITIONAL_TASKS[@]}")

# =============================================================================
# CONFIG
# =============================================================================

CONFIG="/projects/bfcu/ElementalTask/eval_configs/pythia_1.4b_checkpoints.json"
OUTPUT_BASE="results/pythia_1.4b"
PYTHIA_BASE="/work/nvme/bfcu/glee4/pythia_checkpoints"

# =============================================================================
# JOB LOGIC
# =============================================================================

NUM_TASKS=${#TASKS[@]}
TASK_IDX=$((SLURM_ARRAY_TASK_ID % NUM_TASKS))
TASK=${TASKS[$TASK_IDX]}
CONFIG_NAME=$(basename "$CONFIG" .json)

echo "========================================================================"
echo "SLURM ARRAY JOB: $SLURM_ARRAY_JOB_ID - Task ID: $SLURM_ARRAY_TASK_ID"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Config: $CONFIG_NAME"
echo "Task: $TASK (idx=$TASK_IDX)"
echo "Output: $OUTPUT_BASE"
echo "Pythia base: $PYTHIA_BASE"
echo ""

cd /projects/bfcu/ElementalTask || exit 1

# =============================================================================
# SKIP CHECK
# =============================================================================

TASK_SANITIZED=$(echo "$TASK" | tr ':' '_')
EXISTING=$(find "$OUTPUT_BASE" -name "*_${TASK_SANITIZED}_metrics.json" 2>/dev/null | wc -l)
EXPECTED=$(python3 -c "
import json
with open('$CONFIG') as f:
    config = json.load(f)
total = sum(len(ckpts) for ckpts in config.values())
print(total)
")

echo "Metrics files found: $EXISTING / $EXPECTED expected"

if [ "$EXISTING" -ge "$EXPECTED" ]; then
    echo "Task already complete! Skipping."
    exit 0
fi

echo "Proceeding with evaluation..."
echo ""

# =============================================================================
# RUN
# =============================================================================

# Use nvme storage for all caches
export HF_HOME=/work/nvme/bfcu/glee4/hf_cache
export HUGGINGFACE_HUB_CACHE=/work/nvme/bfcu/glee4/hf_cache/hub
export TRANSFORMERS_CACHE=/work/nvme/bfcu/glee4/hf_cache/hub
export XDG_CACHE_HOME=/work/nvme/bfcu/glee4/cache
export XET_CACHE=/work/nvme/bfcu/glee4/xet_cache
export TMPDIR=/work/nvme/bfcu/glee4/tmp
export HF_HUB_DISABLE_XET=1

mkdir -p "$HF_HOME/hub" "$XDG_CACHE_HOME" "$XET_CACHE" "$TMPDIR" logs "$OUTPUT_BASE"

source ~/.bashrc
conda activate elemental_tasks

export PYTHONPATH=/projects/bfcu/ElementalTask:$PYTHONPATH
export PYTHIA_CHECKPOINT_BASE="$PYTHIA_BASE"

echo "Command: python scripts/eval_across_checkpoints.py \\"
echo "    --model_configs $CONFIG \\"
echo "    --output_path $OUTPUT_BASE \\"
echo "    --tasks $TASK \\"
echo "    --max_new_tokens 50 \\"
echo "    --num_shots 5 \\"
echo "    --eval_mode all"
echo ""

python scripts/eval_across_checkpoints.py \
    --model_configs "$CONFIG" \
    --output_path "$OUTPUT_BASE" \
    --tasks "$TASK" \
    --max_new_tokens 50 \
    --num_shots 5 \
    --eval_mode all \
    "$@"

EXIT_CODE=$?

echo ""
echo "========================================================================"
echo "Job completed! Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "Task: $TASK"
echo "========================================================================"

exit $EXIT_CODE