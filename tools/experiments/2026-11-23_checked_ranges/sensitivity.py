"""The registered sensitivity check: every reference moved down and up, the saved choices rescored; no model is run.

Incubator references move by 3 °C, bike-share factors by x0.8 and x1.25. For every configuration, arm and twin: the
mean extra of the saved choices and of the twin that is not told, with the references down, as written and up.

Usage: python tools/experiments/2026-11-23_checked_ranges/sensitivity.py > .../sensitivity.md   (from the repository
       root; a few minutes)
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
src = (ROOT / "tools" / "advise_events.py").read_text()            # its states() and events, without running it
lib = {"__file__": str(ROOT / "tools" / "advise_events.py"), "__name__": "advise_events"}
exec(compile(src.split("\nparser = argparse")[0], "advise_events.py", "exec"), lib)
TWINS = ("incubator", "bikeshare")
STATES = {n: {label: tw for label, tw, _ in lib["states"](n, extra=True)} for n in TWINS}
FACTOR = {-1: 0.8, 0: 1, 1: 1.25}


def moved(twin, refs, s):
    if twin == "incubator":
        return [dict(T_room=r["T_room"] + 3 * s) for r in refs]
    return [{k: round(v * FACTOR[s], 4) if k.endswith("factor") else v for k, v in r.items()} for r in refs]


_extra = {}


def extra(twin, state, action, refs):
    key = (twin, state, json.dumps(action, sort_keys=True), json.dumps(refs, sort_keys=True))
    if key not in _extra:
        _extra[key] = lib["extra"](STATES[twin][state], action, refs)
    return _extra[key]


table = defaultdict(lambda: defaultdict(list))
for run in json.loads((HERE / "runs.json").read_text()):
    for c in run["cells"]:
        if c["kind"] != "vague":
            continue
        tw, refs = STATES[c["twin"]][c["state"]], lib["EVENTS"][c["twin"]][c["event"]][1]
        usual = {k: tw.best()[k] for k in tw.actions}
        action = {k: c[k] for k in tw.actions} if all(k in c for k in tw.actions) else usual
        for s in FACTOR:
            r = moved(c["twin"], refs, s)
            table[(run["model"], run["reasoning"], c["twin"], run["arm"])][s].append(
                (extra(c["twin"], c["state"], action, r), extra(c["twin"], c["state"], usual, r)))

print("Mean extra over the vague cells, told / not told, with every reference moved down, as written, and up.\n")
print("| model | reasoning | twin | arm | cells | down | as written | up | below not told in all three |")
print("|---|---|---|---|---|---|---|---|---|")
for (model, level, twin, arm), d in sorted(table.items()):
    means = {s: np.mean(d[s], axis=0) for s in FACTOR}
    print(f"| {model} | {level} | {twin} | {arm} | {len(d[0])} | " + " | ".join(
        f"{means[s][0]:.2f} / {means[s][1]:.2f}" for s in FACTOR)
        + f" | {'yes' if all(m[0] < m[1] for m in means.values()) else 'no'} |")
