"""Python reference runs, used to check the page's JavaScript step by step (web/checks/check_incubator.js).

results/ref_closed_loop.json  simulated box + twin, 60 min, fixed sensor noise:
                              lid open 25-26 min; at 40 min the set point 35 °C is sent to the box;
                              at 10 min a what-if (set point 35, 30 min ahead). Committed.
results/ref_replay.json       twin updated on the real 2023 experiment.             Local only (git-ignored).
web/replay_2023.js            the real experiment for the page's replay mode.        Local only (git-ignored):
                              the upstream repository has no license file, so the data are not redistributed.

Usage: python scripts/export_reference.py [--replay-only]   (--replay-only leaves ref_closed_loop.json alone)
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from idt.data import load  # noqa: E402
from idt.plant_sim import SimBox  # noqa: E402
from idt.twin import IncubatorTwin  # noqa: E402

if "--replay-only" not in sys.argv:
    N = 1200
    noise = np.random.default_rng(1).normal(0, 0.05, (N, 3))
    box, tw = SimBox(T0=22.0, T_room=22.0, noise=noise), IncubatorTwin()
    obs, est, whatif = [], [], None
    for k in range(N):
        box.lid_open = 25 * 60 <= box.t < 26 * 60
        if k == 800:
            box.act({"desired": 35.0})
        o = box.observe()
        s = tw.update(o)
        if k == 200:
            w = tw.predict(1800, box.ctrl, desired=35.0)
            whatif = dict(k=k, T=w["T"].tolist(), heater=w["heater"].tolist(), energy_Wh=w["energy_Wh"])
        obs.append({k2: (float(v) if not isinstance(v, bool) else v) for k2, v in o.items()})
        est.append(s)
        box.advance()
    json.dump(dict(noise=noise.tolist(), obs=obs, est=est, whatif=whatif),
              open(ROOT / "results" / "ref_closed_loop.json", "w"))
    an = [o["t"] for o, s in zip(obs, est) if s["anomaly"]]
    print(f"closed loop: alarms {len(an)} (first {an[0]:.0f} s, lid opened 1500 s); "
          f"what-if energy {whatif['energy_Wh']:.2f} Wh")

d = load("calibration_2023")
tw = IncubatorTwin()
rep = [tw.update(dict(t=float(t), T=float(T), T_room=float(r), heater_on=float(h)))
       for t, T, r, h in zip(d.t, d["T"], d.T_room, d.heater_on)]
json.dump(rep, open(ROOT / "results" / "ref_replay.json", "w"))
print(f"replay: {len(rep)} samples, alarms {sum(r['anomaly'] for r in rep)}, "
      f"max |innov| {max(abs(r['innov']) for r in rep):.3f}")

payload = {"t": [float(x) for x in d.t], "T": [float(x) for x in d["T"]],
           "T_room": [float(x) for x in d.T_room],
           "t1": [float(x) for x in d.t1], "t2": [float(x) for x in d.t2], "heater_on": d.heater_on.astype(int).tolist(),
           "source": "Aarhus University incubator, experiment 20230501_calibration_empty_system "
                     "(INTO-CPS example_digital-twin_incubator @ 0a851c0)"}
(ROOT.parents[1] / "web" / "replay_2023.js").write_text("window.REPLAY_2023 = " + json.dumps(payload) + ";\n")
print("wrote web/replay_2023.js")
