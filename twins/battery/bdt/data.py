"""Physical side of the twin: the 123 LFP cells (Severson et al. 2019 data, as packaged by Thelen et al.).

Rebuilds the authors' dataset from the raw CSV files (same steps as
upstream/create_capacity_dataset_from_raw_files.py) so we do not depend on their
pickle, which only loads with pandas < 2. Also keeps how many cycles were really
measured, because the authors extend every curve with a straight line for 3000
cycles so it reaches the 50% end-of-life threshold.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

UPSTREAM = Path(__file__).resolve().parents[1] / "upstream" / "battery_digital_twin"   # ../fetch_upstream.sh
RAW = UPSTREAM / "124 LFP Capacity Data"

FPT_THRESHOLD = 0.985  # prediction starts once normalized capacity <= this
EOL_THRESHOLD = 0.50   # authors' end of life (normalized capacity)
N_TRAIN = 41           # first 41 cells = authors' training set; remaining 82 = test


def _linear(x, a, b):
    return a * x + b


def _extrapolate(cap, extrap_len=3000):
    x = np.arange(1, len(cap) + 1)[-30:]
    y = cap[-30:]
    (a, b), _ = curve_fit(_linear, x, y)
    nx = np.linspace(x[-1] + 1, x[-1] + extrap_len, extrap_len)
    return np.concatenate([cap, _linear(nx, a, b)])


def load_cells():
    """Return a list of 123 dicts in the authors' order (train, test1, test2)."""
    conds = pd.read_excel(UPSTREAM / "124 LFP Cell Conditions.xlsx")
    tr = conds[conds["Dataset"] == "Train"]
    t1 = conds[conds["Dataset"] == "Prim. Test"].drop(index=42)  # short-lived cell removed by authors
    t2 = conds[conds["Dataset"] == "Sec. test"]
    groups = [("train", tr, 41), ("test1", t1, 42), ("test2", t2, 40)]

    cells = []
    for folder, cdf, n in groups:
        cdf = cdf.reset_index(drop=True)
        for i in range(n):
            raw = np.loadtxt(RAW / folder / f"cell{i + 1}.csv", delimiter=",")
            cap_meas = raw[:, 1]
            cap = _extrapolate(cap_meas)
            cyc = np.arange(2, len(cap) + 2)
            q = cap / cap[0]
            idx_fpt = int(np.where(q <= FPT_THRESHOLD)[0][0])
            below = np.where(q <= EOL_THRESHOLD)[0]
            k_eol = int(cyc[below[0]]) if len(below) else int(cyc[-1])
            row = cdf.iloc[i]
            cells.append(dict(
                group=folder, idx_in_group=i, barcode=row["Cell barcode"],
                c1=float(row["c1"]), percent=float(row["percent"]) / 100, c2=float(row["c2"]),
                cycles=cyc, cap=cap, q=q, q0=float(cap[0]),
                n_measured=len(cap_meas), last_measured_cycle=int(cyc[len(cap_meas) - 1]),
                idx_fpt=idx_fpt, k_fpt=int(cyc[idx_fpt]), k_eol=k_eol,
            ))
    return cells


def split(cells):
    return cells[:N_TRAIN], cells[N_TRAIN:]
