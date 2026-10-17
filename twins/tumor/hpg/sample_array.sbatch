#!/bin/bash
# CPU job array: 32 tasks x 8 simulations = 256 runs. Each run ~30 s on 2 cores -> each task ~5 min.
#SBATCH --job-name=tt_sims
#SBATCH --cpus-per-task=2
#SBATCH --mem=8gb
#SBATCH --time=00:45:00
#SBATCH --array=0-31
#SBATCH --output=logs/sims_%A_%a.log
set -euo pipefail
ENV=/blue/${GROUP:?export GROUP=<your HiPerGator group>}/$USER/conda/envs/tumortwin
export PATH=$ENV/bin:$PATH TQDM_DISABLE=1 PYTHONWARNINGS=ignore
N=${N:-256}; PER=${PER:-8}; SEED=${SEED:-0}; PATIENT=${PATIENT:-HGG_demo_001}
START=$((SLURM_ARRAY_TASK_ID * PER)); END=$((START + PER))
python scripts/sample_sims.py --n $N --seed $SEED --start $START --end $END --threads 2 --patient $PATIENT
