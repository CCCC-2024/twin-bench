"""UAV twin: what it reproduces of Kapteyn, Pretorius & Willcox 2021. Needs twins/uav/fetch_upstream.sh."""
from types import SimpleNamespace

import numpy as np
import pytest

from dtcore import Recorder, run
from twins.uav import ScriptedUAV, UAVModel, UAVTwin

SEEDS, TURNS = range(1, 21), 60


@pytest.fixture(scope="module")
def uav_model():
    return UAVModel()


@pytest.fixture(scope="module")
def authors_asset(uav_model):
    """20 seeds x 60 turns on the authors' scripted asset (damage independent of the maneuvers)."""
    policy = uav_model.policy(2.5)
    err1, err2, mismatches = [], [], []
    for seed in SEEDS:
        np.random.seed(seed)                       # the authors' measurement generator draws from numpy's global RNG
        uav, twin = ScriptedUAV(), UAVTwin(uav_model)
        uav.act(twin.decide())                     # first maneuver from the prior, as in the authors' loop
        rec = Recorder("uav")
        run(uav, twin, max_steps=TURNS, recorder=rec)
        for r in rec.rows:
            obs, est = r["obs"], r["est"]
            err1.append(abs(est["est1"] - obs["true1"]))
            err2.append(abs(est["est2"] - obs["true2"]))
            nearest = (int(20 * round(obs["true1"] / 20)), int(20 * round(obs["true2"] / 20)))   # grid state
            best = int(policy[uav_model.S.index(nearest)])
            if obs["u"] != best:
                mismatches.append((obs["t"], obs["u"], best))
    return SimpleNamespace(err1=np.array(err1), err2=np.array(err2), mismatches=mismatches)


def test_policy_equals_the_authors_planner(uav_model):
    from digitaltwin.planner import Planner        # the authors' code, put on the path by twins.uav
    authors = Planner(uav_model.gm).getPolicy()
    assert all(uav_model.policy(2.5)[i] == authors(s) for i, s in enumerate(uav_model.S))


def test_estimate_is_off_by_more_than_10_points_a_third_of_the_time(authors_asset):
    assert round(float(np.mean((authors_asset.err1 > 10) | (authors_asset.err2 > 10))), 2) == 0.33


def test_large_errors_are_almost_all_in_region_2(authors_asset):
    assert round(float(np.mean(authors_asset.err2 > 10)), 2) == 0.31
    assert round(float(np.mean(authors_asset.err1 > 10)), 3) == 0.035


def test_maneuver_differs_only_when_switching_to_2g_late(authors_asset):
    """2.4 % of the turns, all at turns 31-32, right after region 1 jumps: the twin still flies 3g."""
    mismatches = authors_asset.mismatches
    assert round(len(mismatches) / (len(SEEDS) * TURNS), 3) == 0.024
    assert {(turn, flown, best) for turn, flown, best in mismatches} <= {(31, 1, 0), (32, 1, 0)}
