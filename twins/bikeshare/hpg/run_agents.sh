#!/bin/bash
# Run an agent experiment on a GPU node (an interactive GPU session, or inside an sbatch job), from the repository
# root:  bash twins/bikeshare/hpg/run_agents.sh new_conditions.py laqmwe --reasoning none --check
# Starts Ollama on this node, uses an installed model (or downloads a registry model), runs the script, stops Ollama.
set -euo pipefail
GROUP=${GROUP:?export GROUP=<your HiPerGator group> first}
BASE=/blue/$GROUP/$USER
SCRIPT=$1; shift
MODEL=$1                                     # the script's first argument is the model
export OLLAMA_MODELS=$BASE/ollama/models     # shared model store, including local aliases
export OLLAMA_HOST=127.0.0.1:11434
mkdir -p logs
nvidia-smi -L                                # fails on a node without a GPU: then do not run here
$BASE/ollama/bin/ollama serve > logs/ollama_$$.log 2>&1 &
trap 'kill $!' EXIT
until curl -s $OLLAMA_HOST/api/version > /dev/null; do sleep 1; done
if ! "$BASE/ollama/bin/ollama" show "$MODEL" > /dev/null 2>&1; then
  case "$MODEL" in
    laqmwe|laqmwe:latest)
      echo "Local model '$MODEL' is not installed. Provision it before running this script." >&2
      exit 1
      ;;
    *) "$BASE/ollama/bin/ollama" pull "$MODEL" ;;
  esac
fi
$BASE/conda/envs/twin-bench/bin/python twins/bikeshare/scripts/$SCRIPT "$@"
