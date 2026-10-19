"""The paper's experiment (Gammelli et al. 2022, Section 4.2) for one station, shared by the scripts.

Every test day the operator sets the number of bikes at midnight (overnight rebalancing). Each rate
source gives hourly rental / return rates for that day; the queueing model turns them into a decision;
the decision is scored by replaying that day's recorded events ("real cost").
"""
import datetime

import numpy as np

from . import data, model

TEST_DAYS = [datetime.date(2018, 11, 1) + datetime.timedelta(days=i) for i in range(61)]   # Nov-Dec 2018
HOURS = list(range(24))
SOURCES = ("overall_ha", "specific_ha", "vp_rnn", "mo_vp_rnn", "true_count")


def load(station=426):
    h = data.hourly(station)
    ev = data.events(station)
    rnn = {"vp_rnn": data.rnn_rates("so_rnn", station), "mo_vp_rnn": data.rnn_rates("mo_rnn", station)}
    by_day = {d: g for d, g in h.groupby("date")}
    ev_day = {d: g["change"].to_numpy() for d, g in ev.groupby("date")}
    return dict(C=data.capacity(station), hourly=by_day, events=ev_day, rnn=rnn)


def rates(S, source, day):
    """Hourly (rental, return) rates for one day from one source."""
    if source in S["rnn"]:
        return S["rnn"][source](day, HOURS)
    g = S["hourly"][day].set_index("hour").loc[HOURS]
    col = {"overall_ha": "ha", "specific_ha": "ha30"}.get(source)
    if col is None:                                   # true_count: the day's own counts (perfect forecast)
        return g["pickups"].to_numpy(float), g["returns"].to_numpy(float)
    return g[f"{col}_pickup"].to_numpy(float), g[f"{col}_return"].to_numpy(float)


def run_day(S, day, sources=SOURCES):
    """Decision, expected and real shortages of each rate source on one day, plus the hindsight best."""
    C, ev = S["C"], S["events"].get(day, np.array([], int))
    out = {}
    for src in sources:
        lp, lr = rates(S, src, day)
        s, expected = model.best_start(lp, lr, C)
        lost_p, lost_r, _ = model.replay(C, s, ev)
        out[src] = dict(start=s, expected=expected, lost_rent=lost_p, lost_return=lost_r)
    real = [sum(model.replay(C, s, ev)[:2]) for s in range(C + 1)]
    s = int(np.argmin(real))                          # hindsight: knows the day's events in advance
    lost_p, lost_r, _ = model.replay(C, s, ev)
    out["best_possible"] = dict(start=s, expected=None, lost_rent=lost_p, lost_return=lost_r)
    return out
