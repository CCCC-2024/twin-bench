#!/usr/bin/env bash
# Kapteyn et al. UAV digital twin (BSD 3-Clause), pinned to commits.
set -euo pipefail
cd "$(dirname "$0")"
../../tools/fetch_github.sh michaelkapteyn/UAV-Digital-Twin b51e42dfaadd35fbf6e17327fa1588bd73c1f975 \
  upstream/UAV-Digital-Twin
# Calibration repository: only the README, license and Step 3_2 script are used.
../../tools/fetch_github.sh michaelkapteyn/UAV-Experimental-Calibration 1dab6340c0782d70dff0ef1dcfe17b4006751622 \
  upstream/UAV-Experimental-Calibration README.md license.txt src/calibrationStep3_2_optimize_point_masses.py
