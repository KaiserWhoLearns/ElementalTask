#!/bin/bash
#SBATCH --job-name=predownload_crystal
#SBATCH --output=logs/predownload_crystal_%j.out
#SBATCH --error=logs/predownload_crystal_%j.err
#SBATCH --mail-user=hsun74@jhu.edu
#SBATCH --mail-type=FAIL,END
#SBATCH -A mdredze1_gpu
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=12:00:00

# Pre-download all CrystalCoder checkpoints sequentially to avoid
# concurrent download corruption when running array jobs.

BASE_DIR="/scratch4/mdredze1/hsun74/ElementalTask"
cd "${BASE_DIR}" || exit 1

module load gcc/11.4.0
module load anaconda
conda activate elementaltask

export LD_PRELOAD=/data/apps/extern/spack_on/gcc/9.3.0/gcc/11.4.0-hzz5maaw347vs5ygsiqkl77ua35qa2d7/lib64/libstdc++.so.6
export PYTHONPATH="${BASE_DIR}:${PYTHONPATH}"

CONFIG="${BASE_DIR}/eval_configs/crystal_checkpoints_0b_1t_main.json"

echo "Pre-downloading all CrystalCoder checkpoints..."
python3 -c "
import json
from huggingface_hub import snapshot_download

with open('${CONFIG}') as f:
    config = json.load(f)

for model_name, checkpoints in config.items():
    print(f'Model: {model_name}, {len(checkpoints)} checkpoints')
    for i, ckpt in enumerate(checkpoints):
        print(f'  [{i+1}/{len(checkpoints)}] Downloading {ckpt}...')
        try:
            snapshot_download(model_name, revision=ckpt)
            print(f'    OK')
        except Exception as e:
            print(f'    ERROR: {e}')

print('Pre-download complete.')
"

echo "Verifying downloaded files..."
python3 -c "
import json, os, struct
from huggingface_hub import snapshot_download

with open('${CONFIG}') as f:
    config = json.load(f)

errors = 0
for model_name, checkpoints in config.items():
    for ckpt in checkpoints:
        path = snapshot_download(model_name, revision=ckpt, local_files_only=True)
        for fname in os.listdir(path):
            if fname.endswith('.bin'):
                fpath = os.path.join(path, fname)
                with open(fpath, 'rb') as f:
                    header = f.read(2)
                if header != b'PK':
                    print(f'CORRUPTED: {ckpt}/{fname} (header: {header.hex()})')
                    errors += 1

if errors == 0:
    print('All checkpoint files verified OK.')
else:
    print(f'{errors} corrupted files found!')
    exit(1)
"

exit $?
