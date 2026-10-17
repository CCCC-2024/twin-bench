"""Export what the page needs for the bike-share twin, and a Python reference run for web/checks/check_bikeshare.js.

web/bikeshare_model.js (published): station capacity; the twin's learned totals per hour and day type after the
    real record up to 31 Oct 2018 (n days, total rentals, total returns: aggregates only); the Nov-Dec 2018 average
    demand per hour and day type, used as the simulated station's true rates; the twin's decision for 1 Nov.
web/bikeshare_replay.js (local only, git-ignored): the recorded rentals/returns of Nov-Dec 2018, hour by hour.
    Citi Bike data may not be redistributed as a stand-alone dataset, so this file is never committed or published.
results/ref_web_bikeshare.json: a simulated run on dtcore (3 days from 1 Nov; demand x1.5 from hour 30; a truck sets
    5 bikes before hour 44) with every simulated event, the twin's estimates and decisions, forecasts and what-ifs at
    a few hours, and model-only cases. The check feeds the same events to the page's code and compares.
About 15 s.
"""
import datetime
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT.parents[1] / "web"
sys.path.insert(0, str(ROOT))
from sdt import data, model  # noqa: E402
from sdt.station import ReplayStation, SimStation  # noqa: E402
from sdt.twin import StationTwin  # noqa: E402
from dtcore import Recorder, run  # noqa: E402

STATION, JAN1, NOV1 = 426, datetime.date(2018, 1, 1), datetime.date(2018, 11, 1)
C, ev, hourly = data.capacity(STATION), data.events(STATION), data.hourly(STATION)
DAY0 = (NOV1 - JAN1).days                                   # day index of 1 Nov (Jan 1 = 0, a Monday)

# 1. the twin learns from the real record, 1 Jan - 31 Oct, deciding every night
st, tw = ReplayStation(C, ev, JAN1, DAY0), StationTwin(C)
rec = Recorder("bikeshare-learn")
run(st, tw, recorder=rec)
s0 = rec.rows[-1]["action"]["set_bikes"]                    # decided at 23:00 on 31 Oct
base = {}
for wk in (True, False):
    rows = []
    for hr in range(24):
        dates, cp, cr = tw.hist[(wk, hr)]
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

# local-only replay of the recorded Nov-Dec days, one string per hour ("+" return, "-" rental)
by_hour = {(d, hr): g["change"].to_numpy() for (d, hr), g in ev[ev["date"] >= NOV1].groupby(["date", "hour"])}
hours = []
for i in range(61 * 24):
    d = NOV1 + datetime.timedelta(days=i // 24)
    hours.append("".join("+" if c > 0 else "-" for c in by_hour.get((d, i % 24), [])))
(WEB / "bikeshare_replay.js").write_text("window.BIKESHARE_REPLAY = " + json.dumps(dict(day0=DAY0, hours=hours)) + ";\n")

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
lost = sum(r["lost_rent"] + r["lost_return"] for r in rows)
print(f"wrote {WEB / 'bikeshare_model.js'}, {WEB / 'bikeshare_replay.js'} (local only)")
print(f"twin learned on {base['weekday'][8][0]} weekdays + {base['weekend'][8][0]} weekend days; decision for 1 Nov: {s0} bikes")
print(f"reference: {N} simulated hours, {lost} lost rentals + returns, nightly decisions "
      f"{[r['set_bikes'] for r in rows if r['set_bikes'] is not None]}")
