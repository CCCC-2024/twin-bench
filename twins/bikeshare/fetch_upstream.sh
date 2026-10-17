#!/usr/bin/env bash
# Gammelli et al. 2022 (MIT): the queueing decision code, one Citi Bike station's 2018 events and hourly
# counts, and the authors' saved demand forecasts. The events come from Citi Bike System Data
# (NYCBS Data Use Policy: no redistribution as a stand-alone dataset), so they are fetched, never committed.
set -euo pipefail
cd "$(dirname "$0")"
S=${1:-426}
P=saved_files/60min/predicted_demand
../../tools/fetch_github.sh DanieleGammelli/variational-poisson-rnn abf77f79fc64be75ae9102ec8d537f77ed9c5f8f \
  upstream/variational-poisson-rnn \
  LICENSE src/algos/inventory_decision_hourly.py \
  data/raw/station_information_citibike.csv data/raw/${S}_allbooking_2018.csv \
  data/demand_rate/60min/${S}_hourlyRatesByDay_2018.csv \
  $P/so_rnn_${S}_pickup_mean.npy $P/so_rnn_${S}_return_mean.npy \
  $P/mo_rnn_${S}_pickup_mean.npy $P/mo_rnn_${S}_return_mean.npy
