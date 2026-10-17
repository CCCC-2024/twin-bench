"""The battery twin on the shared skeleton (dtcore.run) must reproduce the existing battery code exactly.

States: identical to the batch particle filter run_pf (same seed) on every cycle the twin saw.
Decision: identical to calling decide_online directly on the batch states at the first cycle <= 0.95.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bdt.data import load_cells, split  # noqa: E402
from bdt.decision import decide_online  # noqa: E402
from bdt.dt import BatteryDT, CellReplay  # noqa: E402
from bdt.pf import run_pf  # noqa: E402
from dtcore import Recorder, run  # noqa: E402

cells = split(load_cells())[1]
ok_all = True
for idx in [int(a) for a in sys.argv[1:]] or [27]:
    cell, seed = cells[idx], 0
    src, tw = CellReplay(cell), BatteryDT(cell["q0"], seed=seed)
    rec = Recorder("battery")
    n = run(src, tw, recorder=rec)
    states = [r["est"] for r in rec.rows if r["est"]["started"]]
    ref = run_pf(cell, rng=np.random.RandomState(seed))
    m = len(states)
    same = all(states[i]["cycle"] == ref["cycles"][i] and states[i]["log10_a"] == ref["log10_a"][i]
               and states[i]["b"] == ref["b"][i] and states[i]["eol_pred"] == ref["eol_pred"][i] for i in range(m))
    # independent decision reference
    q = cell["q"][:cell["n_measured"]]
    j = int(np.where(q <= 0.95)[0][0])
    jj = int(np.where(ref["cycles"] == cell["cycles"][j])[0][0])
    dec_ref = decide_online(q[:j + 1], cell["q0"], ref["log10_a"][jj], ref["b"][jj], ref["eol_pred"][jj])
    dec_ok = tw.decision is not None and tw.decision["x_twin"] == dec_ref["x_twin"]
    stop_ok = src.retire_at is not None and rec.rows[-1]["obs"]["cycle"] == min(src.retire_at, cell["last_measured_cycle"])
    print(f"cell {idx} ({cell['barcode']}): {n} observations, {m} twin states identical to batch PF: {same}; "
          f"decided at cycle {tw.decision['cycle_decided']} -> retire at {tw.decision['x_twin']} "
          f"(reference {dec_ref['x_twin']}); replay stopped at cycle {rec.rows[-1]['obs']['cycle']}: {stop_ok}")
    ok_all &= same and dec_ok and stop_ok
print("BATTERY DTCORE OK" if ok_all else "BATTERY DTCORE MISMATCH")
