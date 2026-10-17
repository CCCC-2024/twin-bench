#!/usr/bin/env bash
# TumorTwin (UT Austin Research License: academic / research use only).
set -euo pipefail
cd "$(dirname "$0")"
../../tools/fetch_github.sh OncologyModelingGroup/TumorTwin bedf90a6d47ba48cf5cdb25901967d84730061d1 \
  upstream/TumorTwin
