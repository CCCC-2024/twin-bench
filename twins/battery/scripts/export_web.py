"""Export three cells for the shared page (web/battery_cells.js), computed on the shared skeleton.

The twin runs through dtcore.run over every measured cycle (the export source records the
retirement decision but keeps replaying, so the page can also show a later, user-chosen retirement;
states up to any cycle are identical either way). Every 5th cycle we also store the capacity band
predicted from the particles. results/ref_web_battery.json holds what-if decisions for the page check.
Data: Severson et al. 2019 cells, as packaged in Thelen's MIT-licensed repository.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bdt.data import load_cells, split  # noqa: E402
from bdt.decision import decide_online  # noqa: E402
from bdt.dt import BatteryDT, CellReplay  # noqa: E402
from dtcore import Recorder, run  # noqa: E402

CELLS = [(27, "Paper's example cell"), (4, "Long-lived cell"), (24, "Short-lived cell")]
CELLS_ZH = {27: "论文示例电池", 4: "长寿命电池", 24: "短寿命电池"}


class KeepReplaying(CellReplay):
    def act(self, action):
        self.received = action                      # remember the decision, keep the data coming


class BandRecorder(Recorder):
    def __init__(self, twin):
        super().__init__("battery")
        self.twin, self.bands = twin, {}

    def add(self, k, obs, est, action):
        super().add(k, obs, est, action)
        if est.get("started") and est["cycle"] % 5 == 0:
            h = max(int(min(est["cycle"] + est["rul_hi"] + 150, 2500) - est["cycle"]), 50)   # to 95 % EOL + 150
            p = self.twin.predict(h)
            sel = slice(19, None, 20)
            self.bands[est["cycle"]] = dict(c=p["cycles"][sel].tolist(), lo=np.round(p["q_lo"][sel], 5).tolist(),
                                            hi=np.round(p["q_hi"][sel], 5).tolist())


cells = split(load_cells())[1]
out, ref = [], []
for idx, label in CELLS:
    c = cells[idx]
    src, tw = KeepReplaying(c), BatteryDT(c["q0"], seed=0)
    rec = BandRecorder(tw)
    run(src, tw, recorder=rec)
    S = [r["est"] for r in rec.rows if r["est"]["started"]]
    n = c["n_measured"]
    out.append(dict(
        idx=idx, label=label, label_zh=CELLS_ZH[idx], barcode=c["barcode"], q0=c["q0"],
        policy=f'{c["c1"]:g}C to {c["percent"] * 100:g}%, then {c["c2"]:g}C',
        cycles=[int(x) for x in c["cycles"][:n]], q=[float(x) for x in c["q"][:n]],
        k_fpt=int(c["k_fpt"]),
        st={k: [float(s[k]) for s in S] for k in ["log10_a", "b", "log10_a_std", "b_std", "eol_pred", "rul_lo", "rul_hi"]},
        bands=rec.bands,
        decision=dict(cycle_decided=int(tw.decision["cycle_decided"]), x_twin=int(tw.decision["x_twin"]))))
    # what-if references: every 10th twin cycle, five weights
    q = np.array(c["q"][:n])
    for j in range(0, len(S), 10):
        k = S[j]["cycle"]
        i = k - 2
        for w in [0.0, 0.25, 0.5, 0.75, 1.0]:
            d = decide_online(q[:i + 1], c["q0"], S[j]["log10_a"], S[j]["b"], S[j]["eol_pred"], weights=(w, 1 - w))
            u = d["utility"]
            ref.append(dict(idx=idx, cycle=k, w=w, x=int(d["x_twin"]), horizon=int(d["horizon"]),
                            umax=None if np.all(np.isnan(u)) else float(np.nanmax(u))))
    print(f"cell {idx}: {n} measured cycles, twin from {c['k_fpt']}, decided at {tw.decision['cycle_decided']} "
          f"-> retire at {tw.decision['x_twin']}, {len(rec.bands)} bands")

web = ROOT.parents[1] / "web" / "battery_cells.js"
web.write_text("window.BATTERY_CELLS = " + json.dumps(out, separators=(",", ":")) + ";\n")
json.dump(ref, open(ROOT / "results" / "ref_web_battery.json", "w"))
print(f"wrote {web} ({web.stat().st_size // 1024} KB), {len(ref)} what-if references")
