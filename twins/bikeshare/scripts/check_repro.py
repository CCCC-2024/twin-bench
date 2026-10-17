"""Our queueing model vs the upstream code (Gammelli et al. 2022) on the same inputs.

Compares, for 61 test days x (5 rate sources + hindsight best): the chosen number of bikes at midnight
and the replayed lost rentals + returns; the expected shortages; and on 7 days the whole expected-shortage
curve over all 31 starting levels. Our side is computed now; the upstream side is the saved
results/ref_upstream.json (crosscheck_upstream.py, slow). Upstream integrates with Runge-Kutta (error up
to ~1e-3 relative), so a different decision passes only if its two options are closer than that.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sdt import model  # noqa: E402
from sdt import experiment as X  # noqa: E402

TOL = 1e-3
up = json.load(open(ROOT / "results" / "ref_upstream.json"))
S = X.load()

n = same_start = same_real = ties = 0
rel, differ = [], []
for d in X.TEST_DAYS:
    ours = X.run_day(S, d)
    for src in X.SOURCES + ("best_possible",):
        u, o = up[str(d)][src], ours[src]
        n += 1
        same_start += u["start"] == o["start"]
        same_real += u["real"] == o["lost_rent"] + o["lost_return"]
        if "expected" in u:
            rel.append(abs(u["expected"] - o["expected"]) / u["expected"])
        if "curve" in u or u["start"] != o["start"]:
            f = model.udf(*X.rates(S, src, d), S["C"]) if src != "best_possible" else None
            if "curve" in u:
                rel.append(np.max(np.abs(np.array(u["curve"]) - f) / np.array(u["curve"])))
            if u["start"] != o["start"]:
                gap = abs(f[u["start"]] - f[o["start"]]) / f[o["start"]] if f is not None else np.inf
                ties += gap < TOL
                differ.append(f"  {d} {src}: upstream {u['start']}, ours {o['start']}; exact expected shortages "
                              f"differ by {gap:.0e} relative -> {'numerical tie' if gap < TOL else 'REAL DIFFERENCE'}")

print(f"decisions identical: {same_start}/{n}   replayed lost identical: {same_real}/{n}")
print(f"expected shortages: largest relative difference {max(rel):.1e}")
for line in differ:
    print(line)
ok = ties == len(differ) and same_real >= n - len(differ) and max(rel) < TOL
print("BIKESHARE REPRO OK" if ok else "BIKESHARE REPRO MISMATCH")
