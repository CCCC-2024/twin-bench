"""Development of `checked` on test2: saved range readings rescored with agent.robust_best as it is now; no model
is run.

Readings: the range runs of ../2026-11-22_event_reading/runs.json, and any run folders given (a development run of
tools/advise_events.py --ranges --split test2 with another prompt; its rows are read from
<folder>/twins/<twin>/results/advise_events.json). For each configuration and twin, over the vague test2 cells:
  as run    the mean extra the run itself scored, with the rule it ran
  checked   the mean extra of robust_best now on the same readings
  harmed    the share of cells more than 0.5 (minutes, riders) worse than the twin that is not told
A reading without ranges, or without an action, keeps the run's own choice. The last line counts the readings the
twin refused under the rule now.

Usage: python tools/experiments/2026-11-23_checked_ranges/develop.py [RUN_FOLDER ...]   (from the repository root;
       about 20 minutes the first time, most of it the bike station's what-ifs)
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from dtcore import agent

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
src = (ROOT / "tools" / "advise_events.py").read_text()            # its states() and events, without running it
lib = {"__file__": str(ROOT / "tools" / "advise_events.py"), "__name__": "advise_events"}
exec(compile(src.split("\nparser = argparse")[0], "advise_events.py", "exec"), lib)
TWINS = ("incubator", "bikeshare")
STATES = {n: {label: tw for label, tw, _ in lib["states"](n, extra=True)} for n in TWINS}


def kept(f):
    """f, remembering its answers: the same what-if is asked many times."""
    memo = {}

    def g(*args, **kw):
        key = (args, tuple(sorted(kw.items())))
        if key not in memo:
            memo[key] = f(*args, **kw)
        return memo[key]
    return g


for states in STATES.values():
    for tw in states.values():
        tw.whatif, tw.best = kept(tw.whatif), kept(tw.best)


def cells():
    """(configuration, twin, state, event, the run's row, not told's extra): every vague test2 cell of a range arm."""
    for run in json.loads((HERE.parent / "2026-11-22_event_reading" / "runs.json").read_text()):
        if run["experiment"] == 3 and run["arm"] == "range":
            for c in run["cells"]:
                if c["kind"] == "vague" and c["twin"] in TWINS:
                    yield f"{run['model']} {run['reasoning']}", c["twin"], c["state"], c["event"], c, c["not_told"]
    for folder in map(Path, sys.argv[1:]):
        for twin in TWINS:
            res = json.loads((folder / "twins" / twin / "results" / "advise_events.json").read_text())
            for state, events in res["states"].items():
                for event, row in events.items():
                    if res["events"][event]["split"] == "test2" and event.startswith("vague"):
                        for name, r in row.items():
                            if ", ranges" in name:
                                yield f"{folder.name}: {name}", twin, state, event, r, row["not_told"]["extra"]


table, refused = defaultdict(lambda: defaultdict(list)), 0
for config, twin, state, event, row, not_told in cells():
    tw, refs = STATES[twin][state], lib["EVENTS"][twin][event][1]
    cond = row.get("conditions") or {}
    checked = row["extra"]
    if all(k in row for k in tw.actions) and any(isinstance(v, dict) for v in cond.values()):
        try:
            checked = lib["extra"](tw, agent.robust_best(tw, cond)[0], refs)
        except (TypeError, ValueError):             # advise would have told the model; here the twin keeps its usual
            checked, refused = not_told, refused + 1
    t = table[(config, twin)]
    t["as run"].append(row["extra"]), t["checked"].append(checked), t["not told"].append(not_told)

print("| readings | twin | cells | as run | harmed | checked | harmed | not told |\n|---|---|---|---|---|---|---|---|")
for (config, twin), t in sorted(table.items(), key=lambda kv: (kv[0][1], kv[0][0])):
    harmed = lambda k: f"{100 * np.mean(np.array(t[k]) - np.array(t['not told']) > 0.5):.0f} %"
    print(f"| {config} | {twin} | {len(t['checked'])} | {np.mean(t['as run']):.2f} | {harmed('as run')} | "
          f"{np.mean(t['checked']):.2f} | {harmed('checked')} | {np.mean(t['not told']):.2f} |")
print(f"\nReadings the twin refused: {refused}")
