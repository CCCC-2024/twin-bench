"""Baseline: how often is the original twin's retirement decision wrong, and how does
that change with how much of the cell's own data the twin has seen?

For every test cell (82) and PF seed, the twin decides when normalized capacity first
drops to each threshold (later threshold = more own data). Output: results/baseline.csv
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
from bdt.decision import decide  # noqa: E402

THRESHOLDS = [0.97, 0.95, 0.92, 0.90, 0.85]
SEEDS = [0, 1, 2]
_, TEST = split(load_cells())


def one(args):
    ci, seed = args
    cell = TEST[ci]
    res = run_pf(cell, rng=np.random.RandomState(seed))
    rows = []
    for thr in THRESHOLDS:
        d = decide(cell, res, threshold=thr)
        # was the decision triggered by a one-cycle dip (next 5 cycles back above threshold)?
        k = np.where(cell["q"] <= thr)[0][0]
        spurious = bool(np.all(cell["q"][k + 1:k + 6] > thr))
        rows.append(dict(cell=ci, barcode=cell["barcode"], seed=seed, threshold=thr,
                         k_fpt=cell["k_fpt"], k_eol=cell["k_eol"],
                         last_measured=cell["last_measured_cycle"], spurious_trigger=spurious, **d))
    return rows


if __name__ == "__main__":
    jobs = [(ci, s) for ci in range(len(TEST)) for s in SEEDS]
    with Pool(2) as p:
        rows = [r for rs in p.map(one, jobs) for r in rs]
    df = pd.DataFrame(rows)
    out = ROOT / "results" / "baseline.csv"
    df.to_csv(out, index=False)
    print("saved", out, len(df))
