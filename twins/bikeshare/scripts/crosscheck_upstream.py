"""Run the upstream decision code (Gammelli et al. 2022, MIT) on the same inputs and save its answers.

Slow (Runge-Kutta), so it runs only when the inputs change; tests/test_bikeshare.py compares against the saved
results/ref_upstream.json. Uses the upstream functions directly with our inputs, so only the queueing
model is compared.
"""
import json
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from twins.bikeshare import experiment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "upstream" / "variational-poisson-rnn" / "src" / "algos"))   # the authors' module
warnings.filterwarnings("ignore", category=SyntaxWarning)

CURVE_DAYS = experiment.TEST_DAYS[::10]          # full expected-shortage curve (all 31 starts) on these days
paper = None


def one_day(day):
    global paper
    import inventory_decision_hourly as up
    if paper is None:
        paper = experiment.load()
    C, ev = paper["C"], paper["events"].get(day, [])
    out = {}
    for src in experiment.SOURCES:
        lp, lr = experiment.rates(paper, src, day)
        s, cost = up.calInventoryDecisionOptExpectedCost(C, list(lp), list(lr), 24)
        out[src] = dict(start=int(s), expected=float(cost), real=float(up.calRealCost(C, s, ev)))
        if day in CURVE_DAYS:
            out[src]["curve"] = [float(up.calExpectedCost(list(lp), list(lr), C, k, 24)) for k in range(C + 1)]
    best = [up.calRealCost(C, k, ev) for k in range(C + 1)]
    s = min(range(C + 1), key=lambda k: best[k])
    out["best_possible"] = dict(start=s, real=float(best[s]))
    return str(day), out


if __name__ == "__main__":
    with ProcessPoolExecutor() as pool:
        res = dict(pool.map(one_day, experiment.TEST_DAYS))
    path = ROOT / "results" / "ref_upstream.json"
    json.dump(res, open(path, "w"))
    print(f"saved {len(res)} days -> {path}")
