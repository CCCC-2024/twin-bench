"""The UAV twin on the shared skeleton must reproduce the earlier reproduction (experiments/2026-10-02_noros_repro)
exactly, and our parameterised value iteration must give the authors' planner policy at alpha2 = 2.5.

The reproduction is re-run here rather than read from its stored run_seed0.json: the authors' measurement
generator interpolates with scipy's LinearNDInterpolator, and once the true damage drifts off the grid by
floating-point accumulation (step 11, 32.000000000000014) different scipy versions give slightly different
readings. Both runs therefore have to happen in the same environment to be compared bit for bit.
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from udt.assets import ScriptedUAV  # noqa: E402
from udt.model import UAVModel  # noqa: E402
from udt.twin import UAVTwin  # noqa: E402
from dtcore import Recorder, run  # noqa: E402
from digitaltwin.planner import Planner  # noqa: E402

STEPS = 60
mod = UAVModel()
author_pol = Planner(mod.gm).getPolicy()
pol_ok = all(mod.policy(2.5)[i] == author_pol(s) for i, s in enumerate(mod.S))
print("policy(2.5) equals the authors' planner:", pol_ok)

exp = ROOT / "experiments" / "2026-10-02_noros_repro"
with tempfile.TemporaryDirectory() as out:
    subprocess.run([sys.executable, "kapteyn_noros_driver.py", "--repo", str(ROOT / "upstream" / "UAV-Digital-Twin"),
                    "--steps", str(STEPS), "--seed", "0", "--out", out],
                   cwd=exp, check=True, capture_output=True, env={**os.environ, "MPLBACKEND": "Agg"})
    ref = json.load(open(Path(out) / "run_seed0.json"))

np.random.seed(0)
src, tw = ScriptedUAV(), UAVTwin(mod)
src.act(tw.decide())                       # first maneuver from the prior, as in the authors' loop
rec = Recorder("uav")
run(src, tw, max_steps=STEPS, recorder=rec)
same_u = all(mod.C[r["obs"]["u"]] == x["u"] for r, x in zip(rec.rows, ref["rows"]))
diff = max(abs(r["est"][k] - x[k]) for r, x in zip(rec.rows, ref["rows"]) for k in ("est1", "est2", "pmax"))
same = len(rec.rows) == len(ref["rows"]) and same_u and diff < 1e-12
print(f"{len(rec.rows)} steps via dtcore vs noros repro: maneuvers identical {same_u}, "
      f"max |difference| in posterior means / max prob {diff:.1e}")
print("UAV DTCORE OK" if pol_ok and same else "UAV DTCORE MISMATCH")
