"""The template twin through dtcore: one hour of a cooling mug whose heater the twin controls."""
from types import SimpleNamespace

import numpy as np
import pytest

from dtcore import Recorder, run
from twins._template import MugTwin, SimMug


@pytest.fixture(scope="module")
def hour():
    mug, twin = SimMug(), MugTwin()
    rec = Recorder("template")
    run(mug, twin, recorder=rec)
    column = lambda part, key: np.array([r[part][key] for r in rec.rows])
    return SimpleNamespace(twin=twin, true=column("obs", "true_T"), measured=column("obs", "T"),
                           estimate=column("est", "T_est"))


def test_estimate_beats_the_raw_sensor(hour):
    rms = lambda x: np.sqrt(np.mean((x[10:] - hour.true[10:]) ** 2))
    assert rms(hour.estimate) < rms(hour.measured)


def test_decisions_hold_the_mug_near_the_target(hour):
    target = hour.twin.target
    reached = int(np.argmax(hour.true <= target + 2.0))
    assert reached > 0 and np.all(np.abs(hour.true[reached:] - target) < 2.0)


def test_prediction_starts_from_the_estimate(hour):
    assert hour.twin.predict(0)["T"][0] == hour.twin.T
