"""Bike-share twin: what it reproduces of Gammelli et al. 2022. Needs twins/bikeshare/fetch_upstream.sh; about 30 s."""
import datetime
import json
from pathlib import Path

import numpy as np
import pytest

from dtcore import Recorder, run
from twins.bikeshare import ReplayStation, StationTwin, data, model
from twins.bikeshare import experiment as X

pytestmark = pytest.mark.skipif(not data.DATA.exists(), reason="run twins/bikeshare/fetch_upstream.sh first")
JAN1 = datetime.date(2018, 1, 1)
TOL = 1e-3          # the authors integrate with Runge-Kutta: relative error up to ~1e-3


@pytest.fixture(scope="module")
def paper():
    return X.load()


@pytest.fixture(scope="module", params=[("overall_ha", None, "ha"), ("specific_ha", 30, "ha30")],
                ids=["all-earlier-days", "last-30-days"])
def year(request, paper):
    """The twin starts knowing nothing on 1 Jan 2018 and replays the year hour by hour, deciding every night."""
    source, window, col = request.param
    twin = StationTwin(paper["C"], window_days=window)
    rec = Recorder(f"bikeshare-{source}")
    run(ReplayStation(paper["C"], data.events(), JAN1, 365), twin, recorder=rec)
    return source, col, twin, {r["obs"]["t"]: r for r in rec.rows}


def test_learned_rates_equal_the_papers_benchmarks(year):
    _, col, twin, _ = year
    h = data.hourly().dropna(subset=[f"{col}_pickup"])
    est = np.array([twin.rate(d, hr)[0] for d, hr in zip(h["date"], h["hour"])])
    assert np.abs(est - h[[f"{col}_pickup", f"{col}_return"]].to_numpy()).max() < 1e-9


def test_nightly_decisions_and_shortages_equal_the_papers_experiment(year, paper):
    source, _, _, by_t = year
    for day in X.TEST_DAYS:
        ref = X.run_day(paper, day, sources=(source,))[source]
        k0 = (day - JAN1).days * 24
        assert by_t[k0 - 1]["action"]["set_bikes"] == ref["start"]          # decided at 23:00 the day before
        lost = sum(by_t[k0 + i]["obs"]["lost_rent"] + by_t[k0 + i]["obs"]["lost_return"] for i in range(24))
        assert lost == ref["lost_rent"] + ref["lost_return"]


def test_decisions_equal_the_authors_code_except_one_numerical_tie(paper):
    """results/ref_upstream.json holds the authors' decision code run once on the same inputs (about 10 min;
    scripts/crosscheck_upstream.py in v0.2.0): 61 test days x (5 rate sources + hindsight best)."""
    up = json.loads((Path(data.__file__).parent / "results" / "ref_upstream.json").read_text())
    ties = 0
    for day in X.TEST_DAYS:
        ours = X.run_day(paper, day)
        for src in X.SOURCES + ("best_possible",):
            u, o = up[str(day)][src], ours[src]
            if src == "best_possible":
                assert u["start"] == o["start"] and u["real"] == o["lost_rent"] + o["lost_return"]
                continue
            f = model.udf(*X.rates(paper, src, day), paper["C"])
            assert abs(u["expected"] - o["expected"]) / u["expected"] < TOL
            if "curve" in u:
                assert np.max(np.abs(np.array(u["curve"]) - f) / np.array(u["curve"])) < TOL
            if u["start"] == o["start"]:
                assert u["real"] == o["lost_rent"] + o["lost_return"]
            else:                                    # allowed only when both choices are equally good
                assert abs(f[u["start"]] - f[o["start"]]) / f[o["start"]] < TOL
                ties += 1
    assert ties == 1                                 # 2 Nov, VP-RNN (README)
