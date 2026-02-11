#!/bin/bash
#SBATCH --job-name=eval_v2
#SBATCH --output=logs/eval_v2_%A_%a.out
#SBATCH --error=logs/eval_v2_%A_%a.err
#SBATCH --time=8:00:00
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuA100x4
#SBATCH --account=bfcu-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# Total tasks: 68 tasks (22 regular + 20 textfrct + 26 compositional) x 2 configs = 136 array jobs (0-135)
#SBATCH --array=0-135

# =============================================================================
# SPACED MODE CONFIGURATION
# Set SPACED=true to run spaced version of tasks (spaces between characters)
# Usage: SPACED=true sbatch scripts/eval_array_job_v2.sh
# =============================================================================
SPACED=${SPACED:-false}

# =============================================================================
# TASK DEFINITIONS
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

# TextFRCT category tasks (20 tasks with 10+ samples)
TEXTFRCT_TASKS=(
    "textfrct:CV1"   # 50 examples
    "textfrct:CV2"   # 40 examples
    "textfrct:CV3"   # 36 examples
    # "textfrct:FA1"   # 8 examples
    # "textfrct:FA2"   # 8 examples
    "textfrct:FA3"   # 10 examples
    "textfrct:FE1"   # 20 examples
    # "textfrct:FE2"   # 2 examples
    # "textfrct:FE3"   # 6 examples
    # "textfrct:FI1"   # 2 examples
    # "textfrct:FI2"   # 2 examples
    # "textfrct:FI3"   # 2 examples
    # "textfrct:FW1"   # 2 examples
    # "textfrct:FW2"   # 2 examples
    # "textfrct:FW3"   # 2 examples
    "textfrct:I1"    # 30 examples
    "textfrct:I2"    # 28 examples
    "textfrct:MA2"   # 30 examples
    "textfrct:MA3"   # 30 examples
    "textfrct:RG1"   # 30 examples
    "textfrct:RG2"   # 30 examples
    "textfrct:RG3"   # 30 examples
    "textfrct:RL1"   # 30 examples
    "textfrct:RL3"   # 20 examples
    "textfrct:RL4"   # 24 examples
    "textfrct:V1"    # 36 examples
    "textfrct:V2"    # 36 examples
    "textfrct:V3"    # 48 examples
    "textfrct:V4"    # 36 examples
    "textfrct:V5"    # 36 examples
    "textfrct:XU1"   # 20 examples
    "textfrct:XU2"   # 20 examples
    # "textfrct:XU3"   # 4 examples
    # "textfrct:XU4"   # 8 examples
)

# Compositional tasks - 2-way only (26 tasks)
COMPOSITIONAL_TASKS=(
    # String ops
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
    # Gerund + string ops
    "compositional:gerund_first"
    "compositional:gerund_last"
    "compositional:gerund_reverse"
    "compositional:gerund_upper"
    # Plural + string ops
    "compositional:plural_first"
    "compositional:plural_last"
    "compositional:plural_reverse"
    "compositional:plural_upper"
    # Translation + string ops
    "compositional:translate_eng_fr_reverse"
    "compositional:translate_eng_fr_upper"
    "compositional:translate_eng_sp_reverse"
    "compositional:translate_eng_sp_upper"
    "compositional:translate_fr_eng_upper"
    "compositional:translate_sp_eng_upper"
)

# Combine all tasks
TASKS=("${REGULAR_TASKS[@]}" "${TEXTFRCT_TASKS[@]}" "${COMPOSITIONAL_TASKS[@]}")

# =============================================================================
# CONFIG DEFINITIONS
# =============================================================================

CONFIGS=(
    "/projects/bfcu/ElementalTask/eval_configs/olmo2_checkpoints_1b_early.json"
    "/projects/bfcu/ElementalTask/eval_configs/olmo2_checkpoints_7b_early.json"
)

# Output directories - add _spaced suffix if running spaced mode
if [ "$SPACED" = "true" ]; then
    OUTPUT_DIRS=(
        "results/olmo2_continuous_1b_early_spaced"
        "results/olmo2_continuous_7b_early_spaced"
    )
else
    OUTPUT_DIRS=(
        "results/olmo2_continuous_1b_early_revised"
        "results/olmo2_continuous_7b_early_revised"
    )
fi

# =============================================================================
# JOB LOGIC
# =============================================================================

# Calculate which task and config to run
NUM_TASKS=${#TASKS[@]}
CONFIG_IDX=$((SLURM_ARRAY_TASK_ID / NUM_TASKS))
TASK_IDX=$((SLURM_ARRAY_TASK_ID % NUM_TASKS))

TASK=${TASKS[$TASK_IDX]}
CONFIG=${CONFIGS[$CONFIG_IDX]}
OUTPUT_BASE=${OUTPUT_DIRS[$CONFIG_IDX]}
CONFIG_NAME=$(basename $CONFIG .json)

# Print job information
echo "========================================================================"
echo "SLURM ARRAY JOB: $SLURM_ARRAY_JOB_ID - Task ID: $SLURM_ARRAY_TASK_ID"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Config: $CONFIG_NAME (idx=$CONFIG_IDX)"
echo "Task: $TASK (idx=$TASK_IDX)"
echo "Spaced mode: $SPACED"
echo "Output: $OUTPUT_BASE"
echo ""

cd /projects/bfcu/ElementalTask || exit 1

# =============================================================================
# SIMPLE CHECK: Count existing metrics files for this task
# =============================================================================

# Sanitize task name for file matching (: -> _)
TASK_SANITIZED=$(echo "$TASK" | tr ':' '_')

# Add _spaced suffix for spaced mode file checking
if [ "$SPACED" = "true" ]; then
    TASK_SANITIZED="${TASK_SANITIZED}_spaced"
fi

# Count how many metrics files exist for this task
EXISTING=$(find "$OUTPUT_BASE" -name "*_${TASK_SANITIZED}_metrics.json" 2>/dev/null | wc -l)

# Count expected checkpoints from config
EXPECTED=$(python3 -c "
import json
with open('$CONFIG') as f:
    config = json.load(f)
total = sum(len(ckpts) for ckpts in config.values())
print(total)
")

echo "Metrics files found: $EXISTING / $EXPECTED expected"

if [ "$EXISTING" -ge "$EXPECTED" ]; then
    echo ""
    echo "========================================================================"
    echo "Task already complete! Skipping."
    echo "========================================================================"
    exit 0
fi

echo "Proceeding with evaluation..."
echo ""

# =============================================================================
# RUN EVALUATION
# =============================================================================

# Set HuggingFace cache
export HF_HOME=/projects/bfcu/hf_cache
export HUGGINGFACE_HUB_CACHE=/projects/bfcu/hf_cache
echo "HF_HOME set to: $HF_HOME"
echo ""

# Activate conda environment
source ~/.bashrc
conda activate elemental_tasks
echo "Activated conda environment: elemental_tasks"
echo ""

mkdir -p logs

# Add project root to Python path
export PYTHONPATH=/projects/bfcu/ElementalTask:$PYTHONPATH
echo "PYTHONPATH set to: $PYTHONPATH"

# Build spaced flag if needed
SPACED_FLAG=""
if [ "$SPACED" = "true" ]; then
    SPACED_FLAG="--spaced"
fi

# Run the evaluation for this specific task and config
echo "Starting evaluation..."
echo "Command: python scripts/eval_across_checkpoints.py \\"
echo "    --model_configs $CONFIG \\"
echo "    --output_path $OUTPUT_BASE \\"
echo "    --tasks $TASK \\"
echo "    --max_new_tokens 50 \\"
echo "    --num_shots 5 \\"
echo "    --eval_mode all $SPACED_FLAG"
echo ""

python scripts/eval_across_checkpoints.py \
    --model_configs "$CONFIG" \
    --output_path "$OUTPUT_BASE" \
    --tasks "$TASK" \
    --max_new_tokens 50 \
    --num_shots 5 \
    --eval_mode all \
    $SPACED_FLAG \
    "$@"  # Pass through additional arguments like --force_reeval

EXIT_CODE=$?

# Print completion info
echo ""
echo "========================================================================"
echo "Job completed!"
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "Config: $CONFIG_NAME"
echo "Task: $TASK"
echo "Spaced: $SPACED"
echo "Output: $OUTPUT_BASE"
echo "========================================================================"

exit $EXIT_CODE
