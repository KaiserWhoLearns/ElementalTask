#!/bin/bash
export base_dir=/scratch4/mdredze1/hsun74/ElementalTask
export exp_name=interp_perf_olmo

cd $base_dir
sbatch <<EOT
#!/bin/bash

#SBATCH --job-name=$exp_name
#SBATCH --mail-user=hsun74@jhu.edu
#SBATCH --mail-type=FAIL,END
#SBATCH -A mdredze80_gpu
#SBATCH --partition=ica100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=50G
#SBATCH --gpus=1
#SBATCH --time=1-00:00:00 # Max runtime in DD-HH:MM:SS format.
#SBATCH --chdir=${BASE_DIR}
#SBATCH --export=all
#SBATCH --output=${base_dir}/logs/output_${exp_name}.log
#SBATCH --error=${base_dir}/logs/error_${exp_name}.log

module load anaconda3
conda activate eval-pipeline
# source "/home/hsun74/.bashrc"
cd $base_dir

# python scripts/measure_ckpt_interp_perf.py --model_id LLM360/Crystal \
#     --data_path dataset/simple.csv \
#     --num_checkpoints 10 \
#     --output_path output/Crystal_ckpt_interp_results.csv

# python scripts/measure_ckpt_interp_perf.py --model_id LLM360/Crystal \
#     --data_path dataset/simple.csv \
#     --begin 1 \
#     --end 10 \
#     --output_path output/Crystal_ckpt_interp_results_firstckpts.csv

# python scripts/measure_ckpt_interp_perf.py --model_id allenai/OLMo-2-0425-1B \
#     --use_vllm \
#     --data_path dataset/simple.csv \
#     --num_checkpoints 10 \
#     --output_path output/olmo2_ckpt_interp_results.csv

python scripts/measure_ckpt_interp_perf.py --model_id allenai/OLMo-2-0425-1B \
    --use_vllm \
    --data_path dataset/simple.csv \
    --begin 1 \
    --end 10 \
    --output_path output/olmo2_ckpt_interp_results_firstckpts.csv

EOT