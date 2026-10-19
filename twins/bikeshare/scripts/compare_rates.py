"""Compare the twin's rate models on the paper's test days (Nov-Dec 2018); nothing else in the twin changes.

The same StationTwin replays 2018 on dtcore with each rate model and decides every night at 23:00; each decision is
scored by replaying the next day's recorded events, as in step1_model.py. Rate models in the twin:
  historical_all   HistoricalRates(): all earlier days (the paper's "historical average")
  historical_30    HistoricalRates(window_days=30): the last 30 days (its "monthly historical average")
  rnn              RNNRates(): the paper's trained VP-RNN, forecasting the day from what is known at 23:00
  rnn_no_weather   RNNRates with the precipitation probability always 0: the forecast without any weather
and, outside the twin, the paper's protocol with the same network:
  rnn_one_step     each hour's one-step forecast, which needs the count of the hour before, on the same day
Also each forecast's accuracy over the test days (RMSE against the recorded counts, rentals and returns).
Writes results/compare_rates.json. Needs the rnn extra (PyTorch); about 20 s.
"""
import datetime
import json
from pathlib import Path

import numpy as np

from dtcore import Recorder, run
from twins.bikeshare import HistoricalRates, ReplayStation, RNNRates, StationTwin, data, experiment, model

ROOT = Path(__file__).resolve().parents[1]
JAN1 = datetime.date(2018, 1, 1)
paper = experiment.load()
C, events = paper["C"], data.events()
first = (experiment.TEST_DAYS[0] - JAN1).days * 24                     # hour index of 1 Nov, 00:00
counts = data.hourly()[["pickups", "returns"]].to_numpy(float)[first:]  # the test days, rentals / returns


def in_the_twin(rates):
    """Replay 2018 with this rate model (the weather file ends with the year, so not its last hour). Returns the
    decision taken at 23:00 before each test day, the twin's forecast of each test day from that night, and the
    twin's records."""
    twin, rec, forecast = StationTwin(C, rates), Recorder("compare-rates"), {}

    def at_midnight(k, source, twin):
        day = JAN1 + datetime.timedelta(days=k // 24)
        if k % 24 == 0 and day in experiment.TEST_DAYS:
            forecast[day] = np.stack(twin.rates.ahead(24), axis=1)
    run(ReplayStation(C, events, JAN1, 365), twin, max_steps=365 * 24 - 1, recorder=rec, before_step=at_midnight)
    decided = {day: rec.rows[(day - JAN1).days * 24 - 1]["action"] for day in experiment.TEST_DAYS}
    return decided, forecast, rec.rows


def scored(day, start, expected):
    lost_rent, lost_return, _ = model.replay(C, start, paper["events"].get(day, np.array([], int)))
    return dict(start=start, expected=expected, lost_rent=lost_rent, lost_return=lost_return)


def rmse(forecast):
    f = np.concatenate([forecast[day] for day in experiment.TEST_DAYS])
    return [float(x) for x in np.sqrt(np.mean((f - counts) ** 2, axis=0))]


res = dict(forecast_rmse={}, days={str(day): {} for day in experiment.TEST_DAYS})
models = dict(historical_all=HistoricalRates(), historical_30=HistoricalRates(window_days=30), rnn=RNNRates(),
              rnn_no_weather=RNNRates(precip=lambda day, hour: 0.0))
for name, rates in models.items():
    decided, forecast, rows = in_the_twin(rates)
    res["forecast_rmse"][name] = rmse(forecast)
    for day, a in decided.items():
        res["days"][str(day)][name] = scored(day, a["set_bikes"], a["expected_lost"])
    if name == "rnn":                                                   # row t holds the forecast of hour t + 1
        one_step = {day: np.array([[r["est"]["next_rent_rate"], r["est"]["next_return_rate"]]
                                   for r in rows[(day - JAN1).days * 24 - 1:][:24]])
                    for day in experiment.TEST_DAYS}
        res["forecast_rmse"]["rnn_one_step"] = rmse(one_step)
        for day, f in one_step.items():
            res["days"][str(day)]["rnn_one_step"] = scored(day, *model.best_start(f[:, 0], f[:, 1], C))
json.dump(res, open(ROOT / "results" / "compare_rates.json", "w"), indent=0)

lost = {m: np.array([d[m]["lost_rent"] + d[m]["lost_return"] for d in res["days"].values()])
        for m in res["forecast_rmse"]}
print(f"{'rate model':<16}{'RMSE rent / return':>20}{'start (median)':>16}{'expects':>9}{'lost/day':>10}")
for m, (rp, rr) in res["forecast_rmse"].items():
    rows = [d[m] for d in res["days"].values()]
    print(f"{m:<16}{rp:>11.2f} / {rr:<6.2f}{np.median([r['start'] for r in rows]):>16.0f}"
          f"{np.mean([r['expected'] for r in rows]):>9.1f}{lost[m].mean():>10.2f}")
print("\npaired daily difference in lost rentals + returns (mean +/- standard error, 61 days):")
for a, b in (("historical_all", "rnn"), ("historical_30", "rnn"), ("rnn", "rnn_one_step"), ("rnn_no_weather", "rnn")):
    d = lost[a] - lost[b]
    print(f"  {a} - {b}: {d.mean():+.2f} +/- {d.std(ddof=1) / np.sqrt(len(d)):.2f}")
