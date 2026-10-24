"""Export what the page needs for the bike-share twin, and a Python reference run for web/checks/check_bikeshare.js.

web/bikeshare_model.js (published): station capacity; the twin's learned totals per hour and day type after the
    real record up to 31 Oct 2018 (n days, total rentals, total returns: aggregates only); the Nov-Dec 2018 average
    demand per hour and day type, used as the simulated station's true rates; the twin's decision for 1 Nov.
web/bikeshare_replay.js (local only, git-ignored): the recorded rentals/returns of Nov-Dec 2018, hour by hour, and,
    with the rnn extra, what the twin with the VP-RNN as its rate model forecasts after each of those hours (the page
    reads it with PrecomputedRates). Citi Bike data may not be redistributed as a stand-alone dataset, so this file is
    never committed or published.
web/bikeshare_events.js (published): for each night of scripts/new_conditions.py, the twin's rental and return rates
    for the next day as it forecast them at midnight, its brief() then, and the recorded agent runs
    (results/new_conditions.json); the page draws, from these rates, what each start would lose with an event and with
    the inputs the agent extracted from it. Opened locally, the page can also ask a model served on the same computer
    to extract them from an event typed in words: it sends twin.EXTRACT_EVENT_PROMPT and the night's brief.
results/ref_web_bikeshare.json: a simulated run on dtcore (3 days from 1 Nov; demand x1.5 from hour 30; a truck sets
    5 bikes before hour 44) with every simulated event, the twin's estimates and decisions, forecasts and what-ifs at
    a few hours, and model-only cases. The check feeds the same events to the page's code and compares.

Usage: python twins/bikeshare/scripts/export_web.py [--replay-only]   (--replay-only writes only bikeshare_replay.js)
"""
import datetime
import json
import sys
from pathlib import Path

import numpy as np

from dtcore import Recorder, run
from twins.bikeshare import ReplayStation, SimStation, StationTwin, data, model
from twins.bikeshare.twin import EXTRACT_EVENT_PROMPT

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT.parents[1] / "web"

STATION, JAN1, NOV1 = 426, datetime.date(2018, 1, 1), datetime.date(2018, 11, 1)
C, ev, hourly = data.capacity(STATION), data.events(STATION), data.hourly(STATION)
DAY0 = (NOV1 - JAN1).days                                   # day index of 1 Nov (Jan 1 = 0, a Monday)


def rnn_replay(rates):
    """The twin with the VP-RNN as its rate model over the recorded days, as the page reads it: its decision for 1 Nov,
    and after each hour the rates of the next 24 hours and the sd of the next hour's. The weather the network reads
    ends with 2018, so the table stops at 30 Dec 23:00, the decision for 31 Dec."""
    first, n = DAY0 * 24, 60 * 24
    rec, ahead = Recorder("bikeshare-rnn-replay"), {}

    def keep(k, source, twin):                              # before hour k the twin has seen hour k - 1
        if first < k <= first + n:
            ahead[k - 1 - first] = [v.tolist() for v in twin.rates.ahead(24)]
    run(ReplayStation(C, ev, JAN1, 365), StationTwin(C, rates), max_steps=first + n + 1, recorder=rec, before_step=keep)
    sd = [[r["est"]["next_rent_sd"], r["est"]["next_return_sd"]] for r in rec.rows[first:first + n]]
    return dict(start_bikes=rec.rows[first - 1]["action"]["set_bikes"], table=[ahead[k] + [sd[k]] for k in range(n)])


# local-only replay of the recorded Nov-Dec days, one string per hour ("+" return, "-" rental)
by_hour = {(d, hr): g["change"].to_numpy() for (d, hr), g in ev[ev["date"] >= NOV1].groupby(["date", "hour"])}
hours = []
for i in range(61 * 24):
    d = NOV1 + datetime.timedelta(days=i // 24)
    hours.append("".join("+" if c > 0 else "-" for c in by_hour.get((d, i % 24), [])))
replay = dict(day0=DAY0, hours=hours)
try:
    from twins.bikeshare import RNNRates
except ImportError:                                         # the rnn extra (PyTorch) is not installed
    print("no PyTorch: bikeshare_replay.js is written without the VP-RNN rates (pip install -e \".[rnn]\")")
else:
    replay["rnn"] = rnn_replay(RNNRates())
(WEB / "bikeshare_replay.js").write_text("window.BIKESHARE_REPLAY = " + json.dumps(replay) + ";\n")
if "--replay-only" in sys.argv:
    print(f"wrote {WEB / 'bikeshare_replay.js'} (local only)")
    sys.exit()

# 1. the twin learns from the real record, 1 Jan - 31 Oct, deciding every night
st, tw = ReplayStation(C, ev, JAN1, DAY0), StationTwin(C)
rec = Recorder("bikeshare-learn")
run(st, tw, recorder=rec)
s0 = rec.rows[-1]["action"]["set_bikes"]                    # decided at 23:00 on 31 Oct
base = {}
for wk in (True, False):
    rows = []
    for hr in range(24):
        dates, cp, cr = tw.rates.hist[(wk, hr)]
        assert dates[-1] < NOV1
        rows.append([len(dates), cp[-1], cr[-1]])
    base["weekday" if wk else "weekend"] = rows

# 2. the simulated station's true demand: Nov-Dec 2018 average per hour and day type
h = hourly[hourly["date"] >= NOV1]
truth = {}
for wk in (True, False):
    g = h[h["weekday"] == wk].groupby("hour")[["pickups", "returns"]].mean()
    truth["weekday" if wk else "weekend"] = [[float(g.loc[hr, "pickups"]), float(g.loc[hr, "returns"])] for hr in range(24)]


def true_rates(day, hour):
    return truth["weekday" if day.weekday() < 5 else "weekend"][hour]


web = dict(station=STATION, name=data.name(STATION), C=C, day0=DAY0, start_bikes=s0, base=base, truth=truth,
           source="Gammelli et al. 2022 (variational-poisson-rnn @ abf77f7, MIT); Citi Bike System Data 2018 (aggregates)")
(WEB / "bikeshare_model.js").write_text("window.BIKESHARE_MODEL = " + json.dumps(web, separators=(",", ":")) + ";\n")

# 3. reference run on a simulated station, continuing the learned twin
N, SURGE_AT, TRUCK_AT, PRED_AT = 72, 30, 44, (5, 23, 44, 60)


class RecordingSim(SimStation):
    """The simulated station, keeping each hour's events so the page can be fed exactly the same ones."""
    changes = []

    def observe(self):
        o = super().observe()
        if o is not None:
            self.changes.append("".join("+" if c > 0 else "-" for c in self.last_changes))
        return o


sim, preds = RecordingSim(C, true_rates, NOV1, 3, bikes=s0, seed=5), []


def events(k, source, twin):
    if k == SURGE_AT:
        source.surge = 1.5
    if k == TRUCK_AT:
        source.act({"set_bikes": 5})
    if k in PRED_AT:
        for s in (None, 0, 12, 30):
            p = twin.predict(12, set_bikes=s)
            preds.append(dict(k=k, set_bikes=s, **{key: (v.tolist() if hasattr(v, "tolist") else v) for key, v in p.items()}))


rec = Recorder("bikeshare-web")
run(sim, tw, max_steps=N, recorder=rec, before_step=events)
changes, rows = sim.changes, []
for r in rec.rows:
    o, e, a = r["obs"], r["est"], r["action"]
    rows.append(dict(bikes=o["bikes"], lost_rent=o["lost_rent"], lost_return=o["lost_return"], pickups=o["pickups"],
                     returns=o["returns"], rent_rate=e["next_rent_rate"], return_rate=e["next_return_rate"],
                     rent_sd=e["next_rent_sd"], return_sd=e["next_return_sd"], days_seen=e["days_seen"],
                     set_bikes=None if a is None else a["set_bikes"], expected_lost=None if a is None else a["expected_lost"]))

rng = np.random.default_rng(1)
cases = []
for _ in range(6):
    lp, lr = rng.uniform(0, 40, 24), rng.uniform(0, 40, 24)
    start = rng.dirichlet(np.ones(C + 1))
    f = model.forecast(lp, lr, C, start)
    cases.append(dict(lp=lp.tolist(), lr=lr.tolist(), start=start.tolist(), udf=model.udf(lp, lr, C).tolist(),
                      best=model.best_start(lp, lr, C)[0], dist_end=f["dist"][-1, 0].tolist(),
                      lost_rent=f["lost_rent"][:, 0].tolist(), lost_return=f["lost_return"][:, 0].tolist()))
ref = dict(day0=DAY0, start_bikes=s0, surge_at=SURGE_AT, truck_at=TRUCK_AT, changes=changes, rows=rows, preds=preds, cases=cases)
json.dump(ref, open(ROOT / "results" / "ref_web_bikeshare.json", "w"))

# 4. the agent runs on events the data never showed: the same replay as scripts/new_conditions.py, keeping the rates
nc = json.loads((ROOT / "results" / "new_conditions.json").read_text())
EVENTS = {"A": (3, 1, 7, 10), "B": (2, 1, 6, 9), "C": (4, 1, 21, 23)}   # as in new_conditions.py: factors, from, to hour
ahead, brief = {}, {}


def at_midnight(k, source, twin):
    day = str(JAN1 + datetime.timedelta(days=k // 24))
    if k % 24 == 0 and day in nc["days"]:
        ahead[day], brief[day] = [v.tolist() for v in twin.rates.ahead(24)], twin.brief()
run(ReplayStation(C, ev, JAN1, 365), StationTwin(C), max_steps=365 * 24 - 1, before_step=at_midnight)
for day, (lp, lr) in ahead.items():                         # the rates give back every cost the experiment saved
    for key, (fp, fr, a, b) in EVENTS.items():
        w = (np.arange(24) >= a) & (np.arange(24) < b)
        cost, row = model.udf(np.where(w, np.array(lp) * fp, lp), np.where(w, np.array(lr) * fr, lr), C), nc["days"][day][key]
        assert int(np.argmin(cost)) == row["best"]["start"] and model.best_start(np.array(lp), np.array(lr), C)[0] == row["rule"]["start"]
        assert all(abs(cost[r["start"]] - r["expected"]) < 1e-9 for r in row.values() if r["start"] is not None)
events = dict(events={k: dict(text=nc["events"][k], truth=v) for k, v in EVENTS.items()}, agents=list(nc["models"]), extract_prompt=EXTRACT_EVENT_PROMPT,
              nights=[dict(day=day, lp=ahead[day][0], lr=ahead[day][1], brief=brief[day], **nc["days"][day]) for day in sorted(nc["days"])])
(WEB / "bikeshare_events.js").write_text("window.BIKESHARE_EVENTS = " + json.dumps(events, separators=(",", ":")) + ";\n")
lost = sum(r["lost_rent"] + r["lost_return"] for r in rows)
print(f"wrote {WEB / 'bikeshare_model.js'}, {WEB / 'bikeshare_events.js'}, {WEB / 'bikeshare_replay.js'} (local only)")
print(f"twin learned on {base['weekday'][8][0]} weekdays + {base['weekend'][8][0]} weekend days; decision for 1 Nov: {s0} bikes")
print(f"reference: {N} simulated hours, {lost} lost rentals + returns, nightly decisions "
      f"{[r['set_bikes'] for r in rows if r['set_bikes'] is not None]}")
