"""Incubator twin: what the model, the controller and the twin show on the real 2023 data and in closed loop.
Needs twins/incubator/fetch_upstream.sh."""
import numpy as np
import pytest

from dtcore import Recorder, run
from twins.incubator import Controller, IncubatorTwin, RecordedBox, SimBox, data, model

LID_OPEN, LID_CLOSED = 25 * 60, 26 * 60          # s, in the closed-loop run
NEW_SET_POINT_STEP = 800                         # minute 40, at 3 s per step


@pytest.fixture(scope="module", autouse=True)
def upstream_data():
    assert data.DATA.exists(), "run twins/incubator/fetch_upstream.sh first"


@pytest.fixture(scope="module")
def calibration():
    return data.load("calibration_2023")           # empty box, lid closed, 1.85 h


def test_published_model_reproduces_the_real_box(calibration):
    """Given only the recorded heater state and room temperature, the model alone tracks the box."""
    x0 = np.full(2, calibration["T"][0])          # heatbed starts at box temperature, as upstream assumes
    X = model.simulate(model.PARAMS_2023, calibration.t.to_numpy(), calibration.heater_on.to_numpy(),
                       calibration.T_room.to_numpy(), x0)
    error = X[:, 0] - calibration["T"].to_numpy()
    assert round(float(np.sqrt(np.mean(error ** 2))), 2) == 0.46
    assert round(float(np.abs(error).max()), 2) == 1.31


def test_controller_matches_the_recorded_heater(calibration):
    """Our controller, written from the dataset notes, driven by the recorded box temperature."""
    controller = Controller()
    ours = np.array([controller.step(t, T) for t, T in zip(calibration.t, calibration["T"])], float)
    recorded = calibration.heater_on.to_numpy()
    start = int(np.argmax(recorded == 1))         # before this the recording shows a cold box with the heater off:
    assert start == 31                            # the box's controller had not started yet
    assert round(float(np.mean(ours[start:] == recorded[start:])), 3) == 0.977
    our_switches = np.flatnonzero(np.diff(ours[start:]))
    recorded_switches = np.flatnonzero(np.diff(recorded[start:]))
    assert len(our_switches) == 17 and np.all(recorded_switches - our_switches == 3)   # recording 3 samples late


def test_twin_flags_an_opened_lid_and_nothing_else():
    """60 min closed loop on the simulated box, the page's scenario: the lid is open from minute 25 to 26 (the twin is
    not told), and at minute 40 the set point 35 °C is sent to the box."""
    def events(k, box, twin):
        box.lid_open = LID_OPEN <= box.t < LID_CLOSED
        if k == NEW_SET_POINT_STEP:
            box.act({"desired": 35.0})

    rec = Recorder("incubator")
    run(SimBox(T0=22.0, T_room=22.0, seed=1), IncubatorTwin(), max_steps=1200, recorder=rec, before_step=events)
    alarms = [r["obs"]["t"] for r in rec.rows if r["est"]["anomaly"]]
    assert alarms and LID_OPEN < alarms[0] <= LID_OPEN + 9          # within 9 s, none before
    assert alarms[-1] <= LID_CLOSED + 120                             # cleared within 2 min; none from the set point


def test_no_alarm_on_the_real_empty_box_experiment(calibration):
    rec = Recorder("incubator-replay")
    run(RecordedBox("calibration_2023"), IncubatorTwin(), recorder=rec)
    assert len(rec.rows) == len(calibration) and not any(r["est"]["anomaly"] for r in rec.rows)


def test_alarm_is_on_for_half_the_tempeh_run():
    """A batch of tempeh in the box: a load and heat the model does not have."""
    rec = Recorder("incubator-tempeh")
    run(RecordedBox("tempeh_2023"), IncubatorTwin(), recorder=rec)
    assert round(float(np.mean([r["est"]["anomaly"] for r in rec.rows])), 2) == 0.49
