"""Export what the page needs for the UAV twin (web/uav_model.js) and a Python reference run
(twins/uav/results/ref_web_uav.json) that web/checks/check_uav.js replays step by step.

Reference run: ControlledUAV (seed 3) on the skeleton, 120 turns; region 1 damaged +40 % at turn 40,
region 2 +20 % at turn 70; at turn 90 the twin's policy is switched to alpha2 = 0.5.
"""
import json
from pathlib import Path

import numpy as np

from dtcore import Recorder, run
from twins.uav import ControlledUAV, UAVModel, UAVTwin
from twins.uav.model import SIG_ASSET, SIG_TWIN

ROOT = Path(__file__).resolve().parents[1]

mod = UAVModel()
web = dict(S=[list(s) for s in mod.S], C=mod.C, p=mod.p, gamma=mod.gamma, sigTwin=SIG_TWIN, sigAsset=SIG_ASSET,
           raw={c: mod.raw[c].tolist() for c in mod.C}, SR=mod.SR.tolist(), policy=mod.policy(2.5).tolist(),
           source="Kapteyn, Pretorius & Willcox 2021, UAV-Digital-Twin @ b51e42d (BSD 3-Clause, (c) 2021 Michael Kapteyn)")
out = ROOT.parents[1] / "web" / "uav_model.js"
out.write_text("window.UAV_MODEL = " + json.dumps(web, separators=(",", ":")) + ";\n")

N = 120
src, tw = ControlledUAV(mod, seed=3), UAVTwin(mod)
src.act(tw.decide())


def events(k, source, twin):
    if k == 40:
        source.damage(0); source.damage(0)
    if k == 70:
        source.damage(1)
    if k == 90:
        twin.set_policy(0.5)


rec = Recorder("uav")
run(src, tw, max_steps=N, recorder=rec, before_step=events)
alphas = [0.0, 0.5, 1.0, 2.5, 4.0, 8.0]
roll = []
for k in (0, 50, 100):
    b = np.array(rec.rows[k]["est"]["belief"])
    for a in alphas:
        roll.append(dict(k=k, alpha2=a, **mod.rollout(b, mod.policy(a), 30)))
ref = dict(noise=src.noise[:N].tolist(), growth=src.growth[:N].tolist(),
           rows=[dict(u=r["obs"]["u"], true1=r["obs"]["true1"], true2=r["obs"]["true2"], est1=r["est"]["est1"],
                      est2=r["est"]["est2"], pmax=r["est"]["pmax"], next_u=r["action"]["u"]) for r in rec.rows],
           policies={str(a): mod.policy(a).tolist() for a in alphas}, rollouts=roll)
json.dump(ref, open(ROOT / "results" / "ref_web_uav.json", "w"))
us = [r["obs"]["u"] for r in rec.rows]
print(f"wrote {out} ({out.stat().st_size // 1024} KB); reference: {N} turns, 3g turns {sum(us)}, "
      f"final true damage ({rec.rows[-1]['obs']['true1']}, {rec.rows[-1]['obs']['true2']}), "
      f"twin estimate ({rec.rows[-1]['est']['est1']:.1f}, {rec.rows[-1]['est']['est2']:.1f})")
for a in alphas:
    print(f"  alpha2 {a}: 3g in {sum(mod.policy(a))} of 25 states")
