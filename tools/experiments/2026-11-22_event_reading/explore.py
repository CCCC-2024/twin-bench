"""Experiment 3, offline, on the readings saved in runs.json; no model is run.

  A. Other choice rules on the same range readings (exploratory: chosen after the results were seen, not registered).
     frozen       advise's rule: of the best actions at the low/middle/high of every range (every combination), the
                  one with the smallest worst regret; recomputed from each reading and checked against the saved
                  extra
     minimax_all  the same, over every action rather than those best actions
     mean         the smallest mean regret over the same points
     linked       worst regret over three points only: every range at its low, middle or high together
     middle       the best action at the middle of every range: one number again
     A reading the twin could not run (no action) stays without one under every rule.
     Baseline without any reading: the frozen rule on a range that says nothing about the event (incubator: the
     room's whole input range, -20 to 50 °C; bike share: both factors 0 to 4 all day, which spans every reference).
  B. Sensitivity of the registered comparison (planned before the runs): every reference moved down and up
     (incubator 3 °C; bike share factors x0.8 and x1.25), the saved choice of every arm and of the not-told twin
     rescored.
Rebuilding the twins' states takes a few seconds; A takes about 15 minutes.

Usage: python tools/experiments/2026-11-22_event_reading/explore.py   (from the repository root)
"""
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
src = (ROOT / "tools" / "advise_events.py").read_text()            # its states() and events, without running it
lib = {"__file__": str(ROOT / "tools" / "advise_events.py"), "__name__": "advise_events"}
exec(compile(src.split("\nargs = parser.parse_args()")[0], "advise_events.py", "exec"), lib)
TWINS = ("incubator", "bikeshare")
EVENTS = lib["EVENTS"]
STATES = {n: {label: tw for label, tw, _ in lib["states"](n, extra=True)} for n in TWINS}
RUNS = [r for r in json.loads((HERE / "runs.json").read_text()) if r["experiment"] == 3]

_loss, _best = {}, {}
take = lambda tw, a: {k: a[k] for k in tw.actions}


def loss(tw, action, c):
    k = (id(tw), tuple(take(tw, action).items()), tuple(sorted(c.items())))
    if k not in _loss:
        key, sign = tw.objective
        _loss[k] = sign * tw.whatif(**take(tw, action), **c)[key]
    return _loss[k]


def best(tw, c):
    k = (id(tw), tuple(sorted(c.items())))
    if k not in _best:
        _best[k] = tw.best(**c)
    return _best[k]


def every_action(tw):
    if tw.actions == ("set_bikes",):
        return [dict(set_bikes=s) for s in range(tw.C + 1)]
    lo, hi = tw.band                                                # the set points the incubator's best() tries
    return [dict(desired=d) for d in np.arange(lo - 1, hi + 1.01, 0.25).tolist()]


def points(tw, cond, linked):
    p = {}
    for k, v in cond.items():
        if isinstance(v, dict):
            mid = (v["low"] + v["high"]) / 2
            p[k] = [v["low"], int(round(mid)) if tw.conditions.get(k, {}).get("type") == "integer" else mid, v["high"]]
        else:
            p[k] = [v] * 3
    if linked:
        return [dict(zip(p, vals)) for vals in zip(*p.values())]
    return [dict(zip(p, vals)) for vals in itertools.product(*[sorted(set(v)) for v in p.values()])]


def choose(tw, cond, rule):
    grid = points(tw, cond, rule in ("linked", "middle"))
    if rule == "middle":
        return best(tw, grid[1])
    bests = [best(tw, c) for c in grid]
    base = [loss(tw, b, c) for b, c in zip(bests, grid)]
    aggregate = np.mean if rule == "mean" else max
    return min(every_action(tw) if rule == "minimax_all" else bests,
               key=lambda a: aggregate([loss(tw, a, c) - b for c, b in zip(grid, base)]))


def extra(tw, action, refs):
    return float(np.mean([loss(tw, action, r) - loss(tw, best(tw, r), r) for r in refs]))


def moved(twin, refs, s):
    if twin == "incubator":
        return [dict(T_room=r["T_room"] + 3 * s) for r in refs]
    return [{k: v * {-1: 0.8, 0: 1, 1: 1.25}[s] if k.endswith("factor") else v for k, v in r.items()} for r in refs]


def references(twin, event):
    return EVENTS[twin][event][1]


RULES = ("frozen", "minimax_all", "mean", "linked", "middle")
A = defaultdict(lambda: defaultdict(list))
B = defaultdict(lambda: defaultdict(list))
mismatch = 0
for run in RUNS:
    config = (run["model"], run["reasoning"], run["arm"])
    for c in run["cells"]:
        if c["kind"] != "vague" or c["twin"] not in TWINS:
            continue
        tw, refs = STATES[c["twin"]][c["state"]], references(c["twin"], c["event"])
        usual = best(tw, {})
        action = take(tw, c) if all(k in c for k in tw.actions) and c.get("said") else take(tw, usual)
        for s in (-1, 0, 1):
            B[config + (c["twin"],)][s].append((extra(tw, action, moved(c["twin"], refs, s)),
                                                extra(tw, usual, moved(c["twin"], refs, s))))
        if run["arm"] == "range":
            cond = c.get("conditions") or {}
            ranged = all(k in c for k in tw.actions) and any(isinstance(v, dict) for v in cond.values())
            for rule in RULES:
                A[config + (c["twin"],)][rule].append(extra(tw, choose(tw, cond, rule) if ranged else action, refs))
            mismatch += abs(A[config + (c["twin"],)]["frozen"][-1] - c["extra"]) > 1e-3

print("The frozen rule, recomputed, gives the saved extra in every cell:",
      "yes" if not mismatch else f"no ({mismatch} differ)")
print("\n## A. Choice rules on the saved range readings (exploratory, not registered): mean extra, vague cells\n")
print("| model | reasoning | twin | cells | " + " | ".join(RULES) + " | not told |")
print("|---|---|---|---|" + "---|" * (len(RULES) + 1))
for (model, level, arm, twin), d in sorted(A.items()):
    nt = np.mean([x[1] for x in B[(model, level, arm, twin)][0]])
    print(f"| {model} | {level} | {twin} | {len(d['frozen'])} | " + " | ".join(f"{np.mean(d[r]):.2f}" for r in RULES)
          + f" | {nt:.2f} |")
UNINFORMED = dict(incubator={"T_room": {"low": -20, "high": 50}},
                  bikeshare={"rent_factor": {"low": 0, "high": 4}, "return_factor": {"low": 0, "high": 4}})
for twin in TWINS:
    cells = [(label, ev) for label in STATES[twin] for ev in EVENTS[twin]
             if ev.startswith("vague-") and 11 <= int(ev[6:]) < 21]
    told = [extra(STATES[twin][l], choose(STATES[twin][l], UNINFORMED[twin], "frozen"), references(twin, e))
            for l, e in cells]
    nt = [extra(STATES[twin][l], best(STATES[twin][l], {}), references(twin, e)) for l, e in cells]
    print(f"| no reading: {UNINFORMED[twin]} | | {twin} | {len(cells)} | {np.mean(told):.2f} | | | | "
          f"| {np.mean(nt):.2f} |")
print("\n## B. Sensitivity (registered): mean extra, told / not told, references moved down, as written, up\n")
print("| model | reasoning | reply | twin | cells | down | as written | up |\n|---|---|---|---|---|---|---|---|")
for (model, level, arm, twin), d in sorted(B.items()):
    print(f"| {model} | {level} | {arm} | {twin} | {len(d[0])} | " + " | ".join(
        f"{np.mean([x[0] for x in d[s]]):.2f} / {np.mean([x[1] for x in d[s]]):.2f}" for s in (-1, 0, 1)) + " |")
