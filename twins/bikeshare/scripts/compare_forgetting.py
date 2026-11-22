"""How much should the twin forget? Rate models that forget differently, in the same twin, over 2018.

The twin learned its rates from January on, and by November they were about twice the real demand: averaging all
earlier days adapts too slowly. The same StationTwin replays 2018 on dtcore with each rate model below and decides
every night at 23:00; each decision is scored by replaying the next day's recorded events, as in compare_rates.py.
  all            HistoricalRates(): every earlier day, nothing forgotten
  window_<n>     HistoricalRates(window_days=n): only the last n days
  half_life_<n>  DiscountedRates(n): every earlier day, weighted down by half every n days
  selected       SelectedRates over all of the above: each night the twin uses the one whose day-ahead forecasts had
                 the smallest error over the last 28 days, so it needs no window chosen for it
How much to forget is chosen on the days before the test days (CHOOSE_DAYS, May-Oct) and reported on the paper's test
days (Nov-Dec), so the choice never sees the days it is reported on. Writes results/compare_forgetting.json; about 1 min.
"""
import datetime
import json
from pathlib import Path

import numpy as np

from dtcore import Recorder, run
from twins.bikeshare import (DiscountedRates, HistoricalRates, ReplayStation, SelectedRates, StationTwin, data,
                             experiment, model)

ROOT = Path(__file__).resolve().parents[1]
JAN1 = datetime.date(2018, 1, 1)
CHOOSE_DAYS = [datetime.date(2018, 5, 1) + datetime.timedelta(days=i) for i in range(184)]   # May-Oct 2018
TEST_DAYS = experiment.TEST_DAYS
WINDOWS, HALF_LIVES = (7, 14, 21, 30, 45, 60, 90, 120), (5, 10, 20, 40, 80)
paper = experiment.load()
C, events = paper["C"], data.events()
counts = data.hourly()[["pickups", "returns"]].to_numpy(float)


def fixed():
    return dict(all=HistoricalRates(), **{f"window_{n}": HistoricalRates(window_days=n) for n in WINDOWS},
                **{f"half_life_{n}": DiscountedRates(n) for n in HALF_LIVES})


def in_the_twin(rates):
    """Replay 2018 with this rate model. For every scored day: the start decided at 23:00 the night before, the lost
    rentals + returns when that day's recorded events are replayed from it, the squared error of the twin's forecast
    of the day, and (SelectedRates only) which rate model was in use."""
    twin, rec, out = StationTwin(C, rates), Recorder("compare-forgetting"), {}

    def at_midnight(k, source, twin):
        day = JAN1 + datetime.timedelta(days=k // 24)
        if k % 24 == 0 and day in scored_days:
            f = np.stack(twin.rates.ahead(24), axis=1)
            out[day] = dict(sq_error=((f - counts[k:k + 24]) ** 2).sum(axis=0).tolist(),
                            chosen=getattr(twin.rates, "chosen", None))
    run(ReplayStation(C, events, JAN1, 365), twin, max_steps=365 * 24 - 1, recorder=rec, before_step=at_midnight)
    for day, d in out.items():
        start = rec.rows[(day - JAN1).days * 24 - 1]["action"]["set_bikes"]
        lost_rent, lost_return, _ = model.replay(C, start, paper["events"].get(day, np.array([], int)))
        d.update(start=start, lost=lost_rent + lost_return)
    return out


scored_days = set(CHOOSE_DAYS + TEST_DAYS)
models = dict(fixed(), selected=SelectedRates(fixed()))
days = {name: in_the_twin(rates) for name, rates in models.items()}
lost = {name: {p: np.array([d[day]["lost"] for day in period]) for p, period in (("choose", CHOOSE_DAYS), ("test", TEST_DAYS))}
        for name, d in days.items()}
rmse = {name: {p: np.sqrt(np.mean([d[day]["sq_error"] for day in period], axis=0) / 24).tolist()
               for p, period in (("choose", CHOOSE_DAYS), ("test", TEST_DAYS))} for name, d in days.items()}
chosen = min((m for m in models if m != "selected"), key=lambda m: lost[m]["choose"].mean())   # on May-Oct only

res = dict(chosen_on_choose_days=chosen, forecast_rmse=rmse,
           days={str(day): {name: days[name][day] for name in models} for day in sorted(scored_days)})
json.dump(res, open(ROOT / "results" / "compare_forgetting.json", "w"), indent=0)

print(f"{'rate model':<15}{'May-Oct lost/day':>17}{'RMSE rent / return':>21}{'Nov-Dec lost/day':>18}{'RMSE rent / return':>21}")
for m in models:
    print(f"{m:<15}{lost[m]['choose'].mean():>17.2f}{rmse[m]['choose'][0]:>12.2f} / {rmse[m]['choose'][1]:<6.2f}"
          f"{lost[m]['test'].mean():>18.2f}{rmse[m]['test'][0]:>12.2f} / {rmse[m]['test'][1]:<6.2f}")
print(f"\nchosen on May-Oct (fewest lost per day): {chosen}")
print("paired daily difference in lost rentals + returns on Nov-Dec (mean +/- standard error, 61 days):")
for a, b in (("all", chosen), ("all", "selected"), ("window_30", chosen), (chosen, "selected")):
    d = lost[a]["test"] - lost[b]["test"]
    print(f"  {a} - {b}: {d.mean():+.2f} +/- {d.std(ddof=1) / np.sqrt(len(d)):.2f}")
use = [days["selected"][day]["chosen"] for day in TEST_DAYS]
print("selected used on Nov-Dec:", {m: use.count(m) for m in sorted(set(use), key=use.count, reverse=True)})
