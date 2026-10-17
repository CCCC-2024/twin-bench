"""Merge chunk files sims_<seed>_<start>-<end>.npz into sims_<seed>.npz (rows in design order).
Usage: python scripts/merge_sims.py --seed 0 [--patient HGG_demo_001]
"""
import argparse
import re
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--patient", default="HGG_demo_001")
a = ap.parse_args()
d = Path(__file__).resolve().parents[1] / "results" / a.patient
parts = []
for f in d.glob(f"sims_{a.seed}_*-*.npz"):
    s, e = map(int, re.search(r"_(\d+)-(\d+)\.npz$", f.name).groups())
    z = np.load(f)
    parts.append((s, z))
    if len(z["X"]) != e - s:
        print(f"warning: {f.name} has {len(z['X'])} of {e - s} runs (job unfinished?)")
parts.sort(key=lambda p: p[0])
X = np.concatenate([z["X"] for _, z in parts])
Y = np.concatenate([z["Y"] for _, z in parts])
z0 = parts[0][1]
np.savez(d / f"sims_{a.seed}.npz", X=X, Y=Y, days=z0["days"], lo=z0["lo"], hi=z0["hi"])
print(f"merged {len(parts)} chunks -> {len(X)} runs, {int(np.isfinite(Y).all(1).sum())} complete")
