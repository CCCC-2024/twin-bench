"""Running the incubator through the shared skeleton (dtcore.run) must give exactly what a plain loop gives.

Both runs happen here, in the same environment, so the comparison is bit for bit:
  closed loop  simulated box + twin, 60 min, fixed sensor noise; lid open 25-26 min; set point 35 °C at 40 min
  replay       twin updated on the real 2023 experiment (needs ../fetch_upstream.sh)
The plain loop is the one scripts/export_reference.py uses to write the page's reference results.
"""
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from idt.data import load  # noqa: E402
from idt.plant_sim import SimBox  # noqa: E402
from idt.replay import RecordedBox  # noqa: E402
from idt.twin import IncubatorTwin  # noqa: E402
from dtcore import Recorder, run  # noqa: E402

N = 1200
noise = np.random.default_rng(1).normal(0, 0.05, (N, 3))
KEYS = ["T_est", "T_heater_est", "T_heater_sd", "innov", "anomaly"]


def lid_and_setpoint(k, box):
    box.lid_open = 25 * 60 <= box.t < 26 * 60
    if k == 800:
        box.act({"desired": 35.0})


# plain loop
box, tw = SimBox(T0=22.0, T_room=22.0, noise=noise), IncubatorTwin()
plain_obs, plain_est = [], []
for k in range(N):
    lid_and_setpoint(k, box)
    o = box.observe()
    plain_obs.append(o)
    plain_est.append(tw.update(o))
    box.advance()

# same experiment through dtcore. DT_DB: put the database on a local disk when the folder is a network mount.
rec = Recorder("incubator", path=os.environ.get("DT_DB", str(ROOT / "results" / "dtcore_runs.db")))
n = run(SimBox(T0=22.0, T_room=22.0, noise=noise), IncubatorTwin(), max_steps=N, recorder=rec,
        before_step=lambda k, source, twin: lid_and_setpoint(k, source))
rec.close()
same_sim = n == N and all(r["est"][k] == e[k] for r, e in zip(rec.rows, plain_est) for k in KEYS) \
    and all(r["obs"]["heater_on"] == o["heater_on"] and r["obs"]["T"] == o["T"] for r, o in zip(rec.rows, plain_obs))
print(f"closed loop via dtcore: {n} steps, identical to the plain loop: {same_sim}")

d = load("calibration_2023")
tw = IncubatorTwin()
plain_rep = [tw.update(dict(t=float(t), T=float(T), T_room=float(r), heater_on=float(h)))
             for t, T, r, h in zip(d.t, d["T"], d.T_room, d.heater_on)]
rec2 = Recorder("incubator-replay")
n2 = run(RecordedBox(), IncubatorTwin(), recorder=rec2)
same_rep = n2 == len(plain_rep) and all(r["est"][k] == e[k] for r, e in zip(rec2.rows, plain_rep) for k in KEYS)
print(f"replay via dtcore: {n2} steps, identical to the plain loop: {same_rep}")
print("DTCORE OK" if same_sim and same_rep else "DTCORE MISMATCH")
