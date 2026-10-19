#!/usr/bin/env bash
# Gammelli et al. 2022 (MIT): the queueing decision code, one Citi Bike station's 2018 events and hourly
# counts, the authors' saved demand forecasts, and their trained VP-RNN for the station (weights, the hourly
# weather it reads, and the accuracy they recorded for it). The events come from Citi Bike System Data
# (NYCBS Data Use Policy: no redistribution as a stand-alone dataset), so they are fetched, never committed.
set -euo pipefail
cd "$(dirname "$0")"
S=${1:-426}
P=saved_files/60min/predicted_demand
M=saved_files/60min/trained_models/rnn
../../tools/fetch_github.sh DanieleGammelli/variational-poisson-rnn abf77f79fc64be75ae9102ec8d537f77ed9c5f8f \
  upstream/variational-poisson-rnn \
  LICENSE src/algos/inventory_decision_hourly.py \
  data/raw/station_information_citibike.csv data/raw/${S}_allbooking_2018.csv \
  data/demand_rate/60min/${S}_hourlyRatesByDay_2018.csv \
  $P/so_rnn_${S}_pickup_mean.npy $P/so_rnn_${S}_return_mean.npy \
  $P/mo_rnn_${S}_pickup_mean.npy $P/mo_rnn_${S}_return_mean.npy \
  $M/so_st${S}_pickup $M/so_st${S}_return data/raw/weather2018_60min.csv \
  saved_files/60min/results/prediction/output_st${S}_so_rnn.txt
