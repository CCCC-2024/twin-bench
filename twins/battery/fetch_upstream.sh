#!/usr/bin/env bash
# Thelen et al. battery digital twin (MIT): code and the 124 LFP cells (Severson et al. 2019 data).
set -euo pipefail
cd "$(dirname "$0")"
../../tools/fetch_github.sh acthelen/battery_digital_twin 869a1f61362983fd29b82f4414966db94e9b69f1 \
  upstream/battery_digital_twin
