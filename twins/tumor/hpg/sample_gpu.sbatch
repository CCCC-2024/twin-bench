#!/bin/bash
# Optional GPU timing test (UNTESTED code path: --device cuda). Runs 4 simulations on one L4 GPU.
# For B200 add: #SBATCH --partition=hpg-b200   (see hpg/README.md before using B200)
#SBATCH --job-name=tt_gpu_test
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=00:30:00
#SBATCH --output=logs/gpu_test_%j.log
set -euo pipefail
ENV=/blue/${GROUP:?export GROUP=<your HiPerGator group>}/$USER/conda/envs/tumortwin
export PATH=$ENV/bin:$PATH TQDM_DISABLE=1 PYTHONWARNINGS=ignore
nvidia-smi
python scripts/sample_sims.py --n 4 --seed 99 --device cuda --threads 4
