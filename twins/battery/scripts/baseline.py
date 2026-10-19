"""Baseline: how often is the original twin's retirement decision wrong, and how does
that change with how much of the cell's own data the twin has seen?

For every test cell (82) and PF seed, the twin decides when normalized capacity first
drops to each threshold (later threshold = more own data). Output: results/baseline.csv
Besides decide()'s columns: x_global / u_global, the best retirement cycle over the whole life
(true curve, cycles 3 .. EOL-2); u_twin_true, the true utility of the twin's choice; regret_global;
too_late, the best cycle had already passed when the twin decided (plot_tradeoff.py uses the last two).
"""
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bdt.data import load_cells, split  # noqa: E402
from bdt.pf import run_pf  # noqa: E402
from bdt.decision import decide, utility_curve  # noqa: E402

THRESHOLDS = [0.97, 0.95, 0.92, 0.90, 0.85]
SEEDS = [0, 1, 2]
_, TEST = split(load_cells())


def one(args):
    ci, seed = args
    cell = TEST[ci]
    res = run_pf(cell, rng=np.random.RandomState(seed))
    xs = np.arange(3, cell["k_eol"] - 1)
    u_true = utility_curve(cell["cap"][:cell["k_eol"] - 2], xs)
    x_global, u_global = int(xs[np.argmax(u_true)]), float(u_true.max())
    rows = []
    for thr in THRESHOLDS:
        d = decide(cell, res, threshold=thr)
        # was the decision triggered by a one-cycle dip (next 5 cycles back above threshold)?
        k = np.where(cell["q"] <= thr)[0][0]
        spurious = bool(np.all(cell["q"][k + 1:k + 6] > thr))
        rows.append(dict(cell=ci, barcode=cell["barcode"], seed=seed, threshold=thr,
                         k_fpt=cell["k_fpt"], k_eol=cell["k_eol"],
                         last_measured=cell["last_measured_cycle"], spurious_trigger=spurious, **d))
        u_twin_true = float(u_true[d["x_twin"] - 3])
        rows[-1].update(x_global=x_global, u_global=u_global, u_twin_true=u_twin_true,
                        regret_global=u_global - u_twin_true, too_late=x_global < d["k_decide"])
    return rows


if __name__ == "__main__":
    jobs = [(ci, s) for ci in range(len(TEST)) for s in SEEDS]
    with Pool(2) as p:
        rows = [r for rs in p.map(one, jobs) for r in rs]
    df = pd.DataFrame(rows)
    out = ROOT / "results" / "baseline.csv"
    df.to_csv(out, index=False)
    print("saved", out, len(df))
