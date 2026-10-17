"""Check that our refactor gives the authors' result on their example cell.

Upstream (seed 2, first random test cell = Primary Test Cell 28) prints
"Optimal Cycle: 542, End of Life: 691" and draws Figures/utility_panel_cell27.pdf.
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bdt.data import load_cells, split  # noqa: E402
from bdt.pf import run_pf  # noqa: E402
from bdt.decision import decide  # noqa: E402

cells = load_cells()
train, test = split(cells)

np.random.seed(2)                       # same RNG sequence as upstream
idx = int(np.random.permutation(len(test))[0])
cell = test[idx]
t = time.time()
res = run_pf(cell, check_eol=True)   # also asserts closed-form EOL == upstream grid search
d = decide(cell, res, threshold=0.95)
print(f"test cell index {idx} ({cell['group']} cell {cell['idx_in_group'] + 1}, {cell['barcode']})")
print(f"twin retirement cycle {d['x_twin']} (upstream 542), EOL {cell['k_eol']} (upstream 691), "
      f"PF time {time.time() - t:.1f}s")
ok = d["x_twin"] == 542 and cell["k_eol"] == 691

ref = Path(__file__).resolve().parents[1] / "results" / "upstream_cell.pkl"   # optional deep check (made with upstream code)
if ref.exists():
    import pickle
    up = pickle.load(open(ref, "rb"))["res"]
    same = all(np.allclose(np.ravel(up[k]), np.ravel(res[k2])) for k, k2 in
               [("eol_pred", "eol_pred"), ("log10_a_pred", "log10_a"), ("b_pred", "b")])
    print("PF trajectories identical to upstream run:", same)
    ok = ok and same
print("REPRO OK" if ok else "REPRO MISMATCH")
