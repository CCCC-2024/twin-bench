"""End-to-end check of the live pipeline (broker -> replayer -> twin service -> SQLite).

Passes if every twin state stored by the live service equals the batch particle filter
(same seed) on the measured cycles, and prints the online decision next to the paper-mode one.
"""
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "live"))
from bdt.data import load_cells, split  # noqa: E402
from bdt.pf import run_pf  # noqa: E402
from bdt.decision import decide  # noqa: E402
import store  # noqa: E402

CELL, SEED = 27, 0
py = sys.executable
procs = [subprocess.Popen(["amqtt"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)]
time.sleep(2)
procs.append(subprocess.Popen([py, str(ROOT / "live/twin_service.py"), str(SEED)]))
time.sleep(2)
subprocess.run([py, str(ROOT / "live/replayer.py"), "--cell", str(CELL), "--rate", "1000"], check=True)
time.sleep(2)
for p in procs:
    p.terminate()

cell = split(load_cells())[1][CELL]
con = store.connect()
rows = con.execute("SELECT cycle, log10_a, b, eol_pred FROM state WHERE cell=? ORDER BY cycle",
                   (cell["barcode"],)).fetchall()
dec = con.execute("SELECT cycle_decided, x_twin, horizon FROM decisions WHERE cell=?", (cell["barcode"],)).fetchone()
n_meas = con.execute("SELECT COUNT(*), MAX(cycle) FROM measurements WHERE cell=?", (cell["barcode"],)).fetchone()

ref = run_pf(cell, rng=np.random.RandomState(SEED))
m = len(rows)
live = np.array(rows)
same = (np.array_equal(live[:, 0], ref["cycles"][:m]) and np.array_equal(live[:, 1], ref["log10_a"][:m])
        and np.array_equal(live[:, 2], ref["b"][:m]) and np.array_equal(live[:, 3], ref["eol_pred"][:m]))
paper = decide(cell, ref, threshold=0.95)
print(f"measurements stored: {n_meas[0]} (cell measured {cell['n_measured']}, last cycle {n_meas[1]})")
print(f"twin states: {m}; identical to batch PF on these cycles: {same}")
print(f"online decision: at cycle {dec[0]} -> retire at {dec[1]} (twin-predicted EOL {dec[2]})")
print(f"paper mode     : at cycle {paper['k_decide']} -> retire at {paper['x_twin']} (true EOL {cell['k_eol']})")
print("LIVE OK" if same and n_meas[0] == cell["n_measured"] else "LIVE MISMATCH")
