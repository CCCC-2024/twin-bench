"""Read one Citi Bike station: dock capacity, every recorded rental / return, hourly counts, the paper's
demand forecasts, and its trained VP-RNN with the weather it reads. The files come from fetch_upstream.sh
(Gammelli et al. 2022 repository).

Two benchmark rates in the hourly file are decoded here (checked to 1e-14 on station 426):
  historical_average_*          mean of ALL earlier days, same hour, same weekday/weekend type
  monthly_historical_average_*  the same, but only the last 30 days
"""
from pathlib import Path

import numpy as np
import pandas as pd

UPSTREAM = Path(__file__).resolve().parent / "upstream" / "variational-poisson-rnn"
DATA = UPSTREAM / "data"
FORECASTS = UPSTREAM / "saved_files" / "60min" / "predicted_demand"
RNN_MODELS = UPSTREAM / "saved_files" / "60min" / "trained_models" / "rnn"
RNN_ACCURACY = UPSTREAM / "saved_files" / "60min" / "results" / "prediction"


def capacity(station=426):
    info = pd.read_csv(DATA / "raw" / "station_information_citibike.csv")
    return int(info.loc[info["id"] == int(station), "capacity"].iloc[0])


def name(station=426):
    info = pd.read_csv(DATA / "raw" / "station_information_citibike.csv")
    return str(info.loc[info["id"] == int(station), "address"].iloc[0])


def events(station=426):
    """Every recorded event in time order: change = -1 rental (bike leaves), +1 return (bike arrives)."""
    e = pd.read_csv(DATA / "raw" / f"{station}_allbooking_2018.csv")
    return pd.DataFrame(dict(t=pd.to_datetime(e["event_time"], format="mixed"),
                             change=e["inventory_change"].astype(int),
                             date=pd.to_datetime(e["date"]).dt.date, hour=e["hour"].astype(int)))


def hourly(station=426):
    """One row per hour of 2018: counts and the two benchmark rates used in the paper."""
    d = pd.read_csv(DATA / "demand_rate" / "60min" / f"{station}_hourlyRatesByDay_2018.csv")
    return pd.DataFrame(dict(date=pd.to_datetime(d["date"]).dt.date, hour=d["hour"].astype(int),
                             pickups=d["count_pickup"].astype(int), returns=d["count_return"].astype(int),
                             weekday=d["day_of_week"] < 5,
                             ha_pickup=d["historical_average_pickup"], ha_return=d["historical_average_return"],
                             ha30_pickup=d["monthly_historical_average_pickup"],
                             ha30_return=d["monthly_historical_average_return"]))


def rnn_rates(model="so_rnn", station=426):
    """The paper's saved forecasts (so_rnn = VP-RNN, mo_rnn = MOVP-RNN) as functions of (date, hour).

    Same indexing as the upstream decision code: element k forecasts hour-of-year k+1; negative -> 0.
    """
    out = {}
    for k in ("pickup", "return"):
        f = np.load(FORECASTS / f"{model}_{station}_{k}_mean.npy").ravel()
        out[k] = np.where(f >= 0, f, 0.0)

    def rates(date, hours):
        doy = pd.Timestamp(date).dayofyear
        idx = [h + 24 * (doy - 1) - 1 for h in hours]
        return out["pickup"][idx], out["return"][idx]
    return rates


def precip():
    """The recorded precipitation probability of every hour of 2018 (New York), as precip(date, hour).

    Row k of the authors' weather file is the same date and hour as row k of the hourly counts; they align them by row.
    """
    w = pd.read_csv(DATA / "raw" / "weather2018_60min.csv")
    p = dict(zip(zip(pd.to_datetime(w["date"], format="%m/%d/%Y").dt.date, w["hour"].astype(int)),
                 w["precip_prob"].astype(float)))
    return lambda date, hour: p[(date, hour)]
