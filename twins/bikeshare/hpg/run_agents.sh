#!/bin/bash
# Run an agent experiment on a GPU node (an interactive GPU session, or inside an sbatch job), from the repository
# root:  bash twins/bikeshare/hpg/run_agents.sh new_conditions.py qwen3.5:9b --reasoning none --check
# Starts Ollama on this node, downloads the model the first time, runs the script, stops Ollama.
set -euo pipefail
GROUP=${GROUP:?export GROUP=<your HiPerGator group> first}
BASE=/blue/$GROUP/$USER
SCRIPT=$1; shift
MODEL=$1                                     # the script's first argument is the model
export OLLAMA_MODELS=$BASE/ollama/models     # about 21 GB for qwen3.5:9b and gpt-oss:20b
export OLLAMA_HOST=127.0.0.1:11434
mkdir -p logs
nvidia-smi -L                                # fails on a node without a GPU: then do not run here
$BASE/ollama/bin/ollama serve > logs/ollama_$$.log 2>&1 &
trap 'kill $!' EXIT
until curl -s $OLLAMA_HOST/api/version > /dev/null; do sleep 1; done
$BASE/ollama/bin/ollama pull $MODEL
$BASE/conda/envs/twin-bench/bin/python twins/bikeshare/scripts/$SCRIPT "$@"
