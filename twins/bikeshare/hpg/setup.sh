#!/bin/bash
# One-time setup on HiPerGator for the LLM agent experiments (run from the repository root, on a login node).
# Puts the Python environment, Ollama and the models under /blue (the home directory is limited to 40 GB).
set -euo pipefail
GROUP=${GROUP:?export GROUP=<your HiPerGator group> first}
BASE=/blue/$GROUP/$USER
ENV=$BASE/conda/envs/twin-bench
module load conda

mkdir -p $BASE/.cache $BASE/ollama
export PIP_CACHE_DIR=$BASE/.cache/pip
export CONDA_PKGS_DIRS=/tmp/$USER-conda-pkgs   # on /blue, conda crashed reading its package index (Bus error)
twins/bikeshare/fetch_upstream.sh

[ -d $ENV ] || conda create -y -p $ENV python=3.11
$ENV/bin/pip install -e ".[twins,dev]"     # the agent needs no more than this; the RNN tests skip without torch

# Reuse the installed runtime: local models may depend on a custom architecture.
if [ ! -x "$BASE/ollama/bin/ollama" ]; then
  curl -fsSL https://ollama.com/download/ollama-linux-amd64.tar.zst | zstd -d | tar x -C "$BASE/ollama"
fi
$BASE/ollama/bin/ollama --version || true  # "could not connect" is fine here: the server starts in run_agents.sh
$ENV/bin/python -m pytest -q tests/test_dtcore.py tests/test_bikeshare.py
echo "ready: env $ENV, ollama $BASE/ollama"
