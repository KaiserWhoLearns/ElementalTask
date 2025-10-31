#!/bin/bash
# Helper script to run tests with proper environment setup

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Project root is the parent directory of scripts/
PROJECT_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"

# Set PYTHONPATH to include project root
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Change to project directory
cd "${PROJECT_ROOT}"

# # Activate conda environment if specified
# # Usage: Set CONDA_ENV environment variable or it will look for 'elemental_tasks' by default
# CONDA_ENV_NAME="${CONDA_ENV:-elemental_tasks}"

# # Try to initialize conda if not already initialized
# if ! command -v conda &> /dev/null; then
#     # Common conda installation paths
#     CONDA_PATHS=(
#         "${HOME}/anaconda3"
#         "${HOME}/miniconda3"
#         "/opt/anaconda3"
#         "/opt/miniconda3"
#         "/usr/local/anaconda3"
#         "/usr/local/miniconda3"
#         "/sw/external/python/anaconda3"
#     )

#     for conda_path in "${CONDA_PATHS[@]}"; do
#         if [ -f "${conda_path}/etc/profile.d/conda.sh" ]; then
#             source "${conda_path}/etc/profile.d/conda.sh"
#             break
#         fi
#     done
# fi

# # Activate the conda environment if conda is available
# if command -v conda &> /dev/null; then
#     conda activate "${CONDA_ENV_NAME}" 2>/dev/null || echo "Warning: Could not activate conda environment '${CONDA_ENV_NAME}'"
# else
#     echo "Warning: conda not found. Proceeding with system Python."
# fi

# Run the test or script passed as argument
if [ $# -eq 0 ]; then
    echo "Usage: bash scripts/run_test.sh <test_file_or_script>"
    echo "Example: bash scripts/run_test.sh tests/test_simple_interface.py"
    exit 1
fi

python "$@"
