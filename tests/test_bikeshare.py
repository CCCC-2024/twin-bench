"""Bike-share twin: what it reproduces of Gammelli et al. 2022. Needs twins/bikeshare/fetch_upstream.sh."""
import datetime
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dtcore import Recorder, run
from twins.bikeshare import HistoricalRates, ReplayStation, StationTwin, data, experiment, model

JAN1 = datetime.date(2018, 1, 1)
TOL = 1e-3          # the authors integrate with Runge-Kutta: relative error up to ~1e-3


@pytest.fixture(scope="module")
def paper():
    assert data.DATA.exists(), "run twins/bikeshare/fetch_upstream.sh first"
    return experiment.load()


@pytest.fixture(scope="module", params=[("overall_ha", None, "ha"), ("specific_ha", 30, "ha30")],
                ids=["all-earlier-days", "last-30-days"])
def year(request, paper):
    """The twin starts knowing nothing on 1 Jan 2018 and replays the year hour by hour, deciding every night."""
    source, window, column = request.param
    twin = StationTwin(paper["C"], HistoricalRates(window_days=window))
    rec = Recorder(f"bikeshare-{source}")
    run(ReplayStation(paper["C"], data.events(), JAN1, 365), twin, recorder=rec)
    return SimpleNamespace(source=source, column=column, twin=twin, by_hour={r["obs"]["t"]: r for r in rec.rows})


def test_learned_rates_equal_the_papers_benchmarks(year):
    hours = data.hourly().dropna(subset=[f"{year.column}_pickup"])
    learned = np.array([year.twin.rates.rate(day, hour)[0] for day, hour in zip(hours["date"], hours["hour"])])
    benchmark = hours[[f"{year.column}_pickup", f"{year.column}_return"]].to_numpy()
    assert np.abs(learned - benchmark).max() < 1e-9                  # equal up to rounding
    assert len(hours) == {"ha": 8712, "ha30": 8016}[year.column]       # every hour the paper's column is defined


def test_nightly_decisions_and_shortages_equal_the_papers_experiment(year, paper):
    for day in experiment.TEST_DAYS:
        expected = experiment.run_day(paper, day, sources=(year.source,))[year.source]
        k0 = (day - JAN1).days * 24
        assert year.by_hour[k0 - 1]["action"]["set_bikes"] == expected["start"]   # decided at 23:00 the day before
        lost = sum(year.by_hour[k0 + i]["obs"]["lost_rent"] + year.by_hour[k0 + i]["obs"]["lost_return"]
                   for i in range(24))
        assert lost == expected["lost_rent"] + expected["lost_return"]


def test_decisions_equal_the_authors_code_except_one_numerical_tie(paper):
    """Against the authors' answers saved by scripts/crosscheck_upstream.py (results/ref_upstream.json)."""
    saved = json.loads((Path(data.__file__).parent / "results" / "ref_upstream.json").read_text())
    ties, differences = [], []
    for day in experiment.TEST_DAYS:
        ours_by_source = experiment.run_day(paper, day)
        for source in experiment.SOURCES + ("best_possible",):
            theirs, ours = saved[str(day)][source], ours_by_source[source]
            if source == "best_possible":
                assert theirs["start"] == ours["start"] and theirs["real"] == ours["lost_rent"] + ours["lost_return"]
                continue
            cost = model.udf(*experiment.rates(paper, source, day), paper["C"])    # expected shortages per start
            differences.append(abs(theirs["expected"] - ours["expected"]) / theirs["expected"])
            if "curve" in theirs:
                differences.append(np.max(np.abs(np.array(theirs["curve"]) - cost) / np.array(theirs["curve"])))
            if theirs["start"] == ours["start"]:
                assert theirs["real"] == ours["lost_rent"] + ours["lost_return"]
            else:                                    # allowed only when both choices are equally good
                gap = abs(cost[theirs["start"]] - cost[ours["start"]]) / cost[ours["start"]]
                ties.append((day.isoformat(), source, gap))
    assert round(float(max(differences)), 5) == 0.00092 < TOL         # the authors' Runge-Kutta error
    assert [(day, source, round(float(gap), 7)) for day, source, gap in ties] == [("2018-11-02", "vp_rnn", 3.5e-6)]


@pytest.fixture(scope="module")
def nightly():
    """Lost rentals + returns per test day for each rate source (scripts/step1_model.py)."""
    saved = json.loads((Path(data.__file__).parent / "results" / "step1_decisions.json").read_text())
    lost = lambda day, source: saved[day][source]["lost_rent"] + saved[day][source]["lost_return"]
    return {source: np.array([lost(day, source) for day in sorted(saved)]) for source in experiment.SOURCES}


def test_rnn_forecasts_lose_fewer_than_the_all_history_average(nightly):
    def paired(a, b):                                                # mean and standard error of the daily difference
        d = nightly[a] - nightly[b]
        return round(float(d.mean()), 2), round(float(d.std(ddof=1) / np.sqrt(len(d))), 2)
    assert paired("overall_ha", "vp_rnn") == (1.66, 0.76)
    assert paired("mo_vp_rnn", "vp_rnn") == (0.34, 0.42)             # the multi-output RNN does not decide better


def test_the_multi_output_rnn_forecasts_more_accurately():
    """RMSE of the authors' saved forecasts against the recorded counts, test period (Nov-Dec 2018)."""
    test = data.hourly()[lambda h: h.date >= experiment.TEST_DAYS[0]]
    def rmse(name):
        rates = data.rnn_rates(name)
        forecast = np.array([np.concatenate(rates(day, [hour])) for day, hour in zip(test.date, test.hour)])
        return tuple(round(float(np.sqrt(np.mean((forecast[:, i] - test[k]) ** 2))), 2)
                     for i, k in enumerate(("pickups", "returns")))
    assert rmse("mo_rnn") == (3.61, 3.58) and rmse("so_rnn") == (3.85, 3.78)


def test_demand_varies_far_more_than_poisson_and_rentals_follow_returns():
    """The busy hours: those of a day type with at least 5 rentals and 5 returns on average over 2018."""
    by_hour = data.hourly().groupby(["weekday", "hour"])
    s = by_hour.agg(mp=("pickups", "mean"), vp=("pickups", "var"), mr=("returns", "mean"), vr=("returns", "var"))
    busy = s[(s.mp >= 5) & (s.mr >= 5)]
    corr = by_hour.apply(lambda g: g.pickups.corr(g.returns))[busy.index]
    dispersion = np.concatenate([busy.vp / busy.mp, busy.vr / busy.mr])
    assert len(busy) == 29 and (round(dispersion.min()), round(dispersion.max())) == (3, 26)
    assert (round(float(corr.min()), 2), round(float(corr.max()), 2)) == (0.78, 0.96)


def test_rates_learned_by_october_are_twice_the_test_period_demand():
    h = data.hourly()
    weekdays = h[h.weekday]
    start = experiment.TEST_DAYS[0]
    before, after = weekdays[weekdays.date < start], weekdays[weekdays.date >= start]
    per_day = lambda d: round(float(d.groupby("date").pickups.sum().mean()))
    assert (per_day(before), per_day(after)) == (292, 139)
    assert before.date.nunique() == 218                               # each new weekday moves a rate by about 1/220
