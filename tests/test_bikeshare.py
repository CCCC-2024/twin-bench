"""Bike-share twin: what it reproduces of Gammelli et al. 2022. Needs twins/bikeshare/fetch_upstream.sh."""
import datetime
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dtcore import Recorder, run
from twins.bikeshare import ReplayStation, StationTwin, data, experiment, model

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
    twin = StationTwin(paper["C"], window_days=window)
    rec = Recorder(f"bikeshare-{source}")
    run(ReplayStation(paper["C"], data.events(), JAN1, 365), twin, recorder=rec)
    return SimpleNamespace(source=source, column=column, twin=twin, by_hour={r["obs"]["t"]: r for r in rec.rows})


def test_learned_rates_equal_the_papers_benchmarks(year):
    hours = data.hourly().dropna(subset=[f"{year.column}_pickup"])
    learned = np.array([year.twin.rate(day, hour)[0] for day, hour in zip(hours["date"], hours["hour"])])
    benchmark = hours[[f"{year.column}_pickup", f"{year.column}_return"]].to_numpy()
    assert np.abs(learned - benchmark).max() < 1e-9


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
    ties = 0
    for day in experiment.TEST_DAYS:
        ours_by_source = experiment.run_day(paper, day)
        for source in experiment.SOURCES + ("best_possible",):
            theirs, ours = saved[str(day)][source], ours_by_source[source]
            if source == "best_possible":
                assert theirs["start"] == ours["start"] and theirs["real"] == ours["lost_rent"] + ours["lost_return"]
                continue
            cost = model.udf(*experiment.rates(paper, source, day), paper["C"])    # expected shortages per start
            assert abs(theirs["expected"] - ours["expected"]) / theirs["expected"] < TOL
            if "curve" in theirs:
                assert np.max(np.abs(np.array(theirs["curve"]) - cost) / np.array(theirs["curve"])) < TOL
            if theirs["start"] == ours["start"]:
                assert theirs["real"] == ours["lost_rent"] + ours["lost_return"]
            else:                                    # allowed only when both choices are equally good
                assert abs(cost[theirs["start"]] - cost[ours["start"]]) / cost[ours["start"]] < TOL
                ties += 1
    assert ties == 1                                 # 2 Nov, VP-RNN (README)
