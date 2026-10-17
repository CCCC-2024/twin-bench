#!/usr/bin/env bash
# Public experiments of the Aarhus University incubator (INTO-CPS example_digital-twin_incubator).
# The upstream repository has no license file, so only the data files used here are downloaded,
# and nothing derived from them that reproduces the data is committed or published.
set -euo pipefail
cd "$(dirname "$0")"
D=software/incubator/datasets
../../tools/fetch_github.sh INTO-CPS-Association/example_digital-twin_incubator 0a851c027587c96664ff319d82fff8e8d68cf80d \
  upstream/example_digital-twin_incubator \
  $D/20230501_calibration_empty_system/20230501_calibration_empty_system.csv \
  $D/20230501_calibration_empty_system/README.md \
  $D/20230501_tempeh_batch/rec_2023-05-01__08_09_06.csv \
  $D/20230501_tempeh_batch/events.csv \
  $D/20230501_tempeh_batch/README.md
