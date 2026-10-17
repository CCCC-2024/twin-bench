#!/bin/bash
# One-time setup on HiPerGator (run on a login node, from the twins/tumor folder).
# Creates the conda env under /blue, downloads TumorTwin at the pinned commit, installs everything.
set -euo pipefail
GROUP=${GROUP:?export GROUP=<your HiPerGator group> first}
BASE=/blue/$GROUP/$USER
ENV=$BASE/conda/envs/tumortwin
module load conda

# keep caches off the 40 GB home directory
mkdir -p $BASE/.cache
export PIP_CACHE_DIR=$BASE/.cache/pip

./fetch_upstream.sh

if [ ! -d $ENV ]; then
  conda create -y -p $ENV python=3.11
fi
$ENV/bin/pip install --upgrade pip
$ENV/bin/pip install torch            # Linux wheels from PyPI include CUDA support
$ENV/bin/pip install -e upstream/TumorTwin scikit-learn
$ENV/bin/python -c "import torch, tumortwin, sklearn; print('ok', torch.__version__, 'cuda available on this node:', torch.cuda.is_available())"
echo "env ready: $ENV"
