"""Predict and what-if during the day (beyond the paper, which only decides at midnight).

1. Consistency: predicting a whole day from midnight must give the same expected shortages as the
   model-only decision curve (model.udf) for every starting number of bikes.
2. What-if on the test day where the overnight decision went worst in the replay: one hour before the
   first lost rental/return, compare "no truck" with "a truck sets s bikes now" for every s, as the twin
   predicts it and as the replay of the recorded events then turns out.
Writes results/step5_whatif.json and results/step5_whatif.png.
"""
import copy
import datetime
import json
from pathlib import Path

import numpy as np

from dtcore import run
from twins.bikeshare import ReplayStation, StationTwin, data, model
from twins.bikeshare import experiment as X

ROOT = Path(__file__).resolve().parents[1]

S = X.load()
C, ev = S["C"], data.events(426)
JAN1 = datetime.date(2018, 1, 1)


def twin_at(day, hour):
    """Station and twin after replaying (with the twin's own decisions) up to and including `hour`."""
    st, tw = ReplayStation(C, ev, JAN1, 365), StationTwin(C)
    run(st, tw, max_steps=(day - JAN1).days * 24 + hour + 1)
    return st, tw


# 1. consistency with the model-only curve on one test day (from 23:00 the day before, every start)
day = X.TEST_DAYS[13]
st, tw = twin_at(day - datetime.timedelta(days=1), 23)
curve = model.udf(*X.rates(S, "overall_ha", day), C)
pred = np.array([tw.predict(24, set_bikes=s)["expected_lost"] for s in range(C + 1)])
gap = np.abs(pred - curve).max()
print(f"predict from midnight vs model-only curve ({day}): max diff {gap:.1e}")

# 2. the worst test day for the overnight decision, one hour before its first shortage
night = {d: X.run_day(S, d, sources=("overall_ha",))["overall_ha"] for d in X.TEST_DAYS}
worst = max(X.TEST_DAYS, key=lambda d: night[d]["lost_rent"] + night[d]["lost_return"])
day_ev = ev[ev.date == worst]
bikes, first = night[worst]["start"], None
for hr in range(24):
    lost_p, lost_r, bikes = model.replay(C, bikes, day_ev.loc[day_ev.hour == hr, "change"])
    if lost_p + lost_r and first is None:
        first = hr
at = max(first - 1, 0)                                   # decide at the end of this hour
st, tw = twin_at(worst, at)
horizon = 24 - at - 1                                    # rest of the day
rows = []
for s in [None] + list(range(C + 1)):
    p = tw.predict(horizon, set_bikes=s)
    st2 = copy.deepcopy(st)
    st2.advance()
    if s is not None:
        st2.act({"set_bikes": s})
    real = 0
    for _ in range(horizon):
        o = st2.observe()
        real += o["lost_rent"] + o["lost_return"]
        st2.advance()
    rows.append(dict(set_bikes=s, predicted=p["expected_lost"], replay=real))
none = rows[0]
best = min(rows[1:], key=lambda r: r["predicted"])
print(f"\nworst test day for the overnight decision: {worst} (start {night[worst]['start']} bikes, "
      f"{night[worst]['lost_rent']} lost rentals, {night[worst]['lost_return']} lost returns)")
print(f"at {at:02d}:59 the station has {tw.bikes} bikes; rest of the day ({horizon} h):")
print(f"  no truck            predicted {none['predicted']:5.1f} lost   replay {none['replay']:3d} lost")
print(f"  truck sets {best['set_bikes']:2d} bikes  predicted {best['predicted']:5.1f} lost   replay {best['replay']:3d} lost")
json.dump(dict(day=str(worst), hour=at, bikes=tw.bikes, horizon=horizon, rows=rows,
               consistency_max_diff=gap), open(ROOT / "results" / "step5_whatif.json", "w"), indent=0)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
hrs = np.arange(at + 1, 24) + 1
for s, c in ((None, "C0"), (best["set_bikes"], "C1")):
    p = tw.predict(horizon, set_bikes=s)
    lab = "no truck" if s is None else f"truck sets {s}"
    ax[0].plot(hrs, p["mean_bikes"], color=c, label=f"{lab}: mean")
    ax[0].fill_between(hrs, p["p10"], p["p90"], color=c, alpha=0.2)
ax[0].axhline(C, color="grey", lw=0.8)
ax[0].set(xlabel="hour (end of)", ylabel="bikes", title=f"twin forecast, {worst} from {at:02d}:59")
ax[0].legend()
xs = [r["set_bikes"] for r in rows[1:]]
ax[1].plot(xs, [r["predicted"] for r in rows[1:]], label="twin predicts")
ax[1].plot(xs, [r["replay"] for r in rows[1:]], "o", ms=3, label="replay of recorded events")
ax[1].axhline(none["predicted"], color="C0", ls=":", lw=1, label="no truck (predicted)")
ax[1].set(xlabel="bikes set by the truck", ylabel="lost rentals + returns, rest of day", title="what-if")
ax[1].legend()
fig.tight_layout()
fig.savefig(ROOT / "results" / "step5_whatif.png", dpi=130)
print("saved results/step5_whatif.json, results/step5_whatif.png")
