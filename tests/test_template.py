"""The template twin through dtcore: one hour of a cooling mug whose heater the twin controls."""
import numpy as np
import pytest

from dtcore import Recorder, run
from twins._template import MugTwin, SimMug


@pytest.fixture(scope="module")
def hour():
    mug, twin = SimMug(), MugTwin()
    rec = Recorder("template")
    run(mug, twin, recorder=rec)
    col = lambda part, key: np.array([r[part][key] for r in rec.rows])
    return twin, col("obs", "true_T"), col("obs", "T"), col("est", "T_est")


def test_estimate_beats_the_raw_sensor(hour):
    _, true, meas, est = hour
    rms = lambda x: np.sqrt(np.mean((x[10:] - true[10:]) ** 2))
    assert rms(est) < rms(meas)


def test_decisions_hold_the_mug_near_the_target(hour):
    twin, true, _, _ = hour
    reached = int(np.argmax(true <= twin.target + 2.0))
    assert reached > 0 and np.all(np.abs(true[reached:] - twin.target) < 2.0)


def test_prediction_starts_from_the_estimate(hour):
    twin = hour[0]
    assert twin.predict(0)["T"][0] == twin.T
