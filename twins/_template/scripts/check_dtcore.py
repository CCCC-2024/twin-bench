"""The template twin on dtcore: one hour of a cooling mug whose heater the twin controls.

Passes when (1) the estimate tracks the true temperature better than the raw sensor does,
(2) the twin's decisions hold the mug within 2 K of the 60 °C target once it has cooled there, and
(3) predict(0) returns the current estimate (prediction starts from the updated state).
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mdt.twin import MugTwin, SimMug  # noqa: E402
from dtcore import Recorder, run  # noqa: E402

mug, tw = SimMug(), MugTwin()
rec = Recorder("template")
n = run(mug, tw, recorder=rec)
true = np.array([r["obs"]["true_T"] for r in rec.rows])
meas = np.array([r["obs"]["T"] for r in rec.rows])
est = np.array([r["est"]["T_est"] for r in rec.rows])
rms_est = np.sqrt(np.mean((est[10:] - true[10:]) ** 2))
rms_meas = np.sqrt(np.mean((meas[10:] - true[10:]) ** 2))
reached = int(np.argmax(true <= tw.target + 2.0))
held = bool(reached > 0 and np.all(np.abs(true[reached:] - tw.target) < 2.0))
starts = tw.predict(0)["T"][0] == tw.T
print(f"{n} steps; RMS error estimate {rms_est:.3f} K vs sensor {rms_meas:.3f} K; "
      f"held within 2 K of {tw.target:.0f} °C from {reached * mug.dt / 60:.0f} min: {held}; predict(0) = estimate: {starts}")
print("TEMPLATE OK" if rms_est < rms_meas and held and starts else "TEMPLATE MISMATCH")
