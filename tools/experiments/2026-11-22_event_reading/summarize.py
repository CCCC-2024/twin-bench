"""The study's tables from runs.json: per configuration, the mean over its runs and [min-max].

  stated / nothing / cannot   cells read right, of all four twins' cells of that kind
  vague better/same/worse     the vague cells (incubator and bike share) where acting on the reading is better, the
                              same or worse than the twin that is not told the event
  incubator, bike share       mean extra over the vague cells (minutes of 60 outside the band; lost riders a day),
                              and the twin that is not told

Usage: python summarize.py
"""
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

RUNS = json.loads(Path(__file__).with_name("runs.json").read_text())
REFERENCES = json.loads(Path(__file__).with_name("references.json").read_text())   # twin -> event -> references


def stats(values):
    mean = f"{st.mean(values):.1f}"
    return mean if len(set(values)) == 1 else f"{mean} [{min(values):.1f}–{max(values):.1f}]"


def table(experiment):
    by = defaultdict(list)
    for r in RUNS:
        if r["experiment"] == experiment:
            by[(r["model"], r["reasoning"], r["arm"], r["machine"])].append(r["cells"])
    print(f"\n## Experiment {experiment} ({'first test set' if experiment == 1 else 'second test set, test2'})\n")
    print("| model | reasoning | reply | runs | stated | nothing | cannot | vague better / same / worse | "
          "incubator vague (not told) | bike share vague (not told) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for (model, reasoning, arm, machine), runs in sorted(by.items()):
        right = lambda kind: [sum(bool(c.get("right")) for c in cells if c["kind"] == kind) for cells in runs]
        of = lambda kind: sum(c["kind"] == kind for c in runs[0])
        vague = [[c for c in cells if c["kind"] == "vague" and c["twin"] in ("incubator", "bikeshare")]
                 for cells in runs]
        bsw = [stats([sum(f(c["extra"] - c["not_told"]) for c in v) for v in vague])
               for f in (lambda d: d < 0, lambda d: d == 0, lambda d: d > 0)]
        mean = lambda twin, key: [st.mean(c[key] for c in v if c["twin"] == twin) for v in vague]
        print(f"| {model} | {reasoning} | {arm} | {len(runs)} | {stats(right('stated'))} / {of('stated')} | "
              f"{stats(right('nothing'))} / {of('nothing')} | "
              f"{stats(right('inexpressible'))} / {of('inexpressible')} | "
              f"{' / '.join(bsw)} of {len(vague[0])} | {stats(mean('incubator', 'extra'))} "
              f"({mean('incubator', 'not_told')[0]:.1f}) | {stats(mean('bikeshare', 'extra'))} "
              f"({mean('bikeshare', 'not_told')[0]:.2f}) |")


def covers(reading, reference, defaults):
    """Whether a reading holds the middle reference for every input: inside its range, or equal to its number (an
    input left out is at its default)."""
    for k in set(reference) | set(reading):
        v, want = reading.get(k, defaults.get(k)), reference.get(k, defaults.get(k))
        if not (v["low"] <= want <= v["high"] if isinstance(v, dict) else v == want):
            return False
    return True


def coverage():
    """Vague cells of experiment 3 where the reading holds the middle reference, of those with a reading."""
    defaults = dict(bikeshare=dict(rent_factor=1, return_factor=1, from_hour=0, to_hour=24), incubator={})
    by = defaultdict(lambda: defaultdict(list))
    for r in RUNS:
        if r["experiment"] != 3:
            continue
        for c in r["cells"]:
            if c["kind"] == "vague" and c["twin"] in defaults and c.get("said") == "conditions" and c.get("conditions"):
                ref = REFERENCES[c["twin"]][c["event"]]
                by[(r["model"], r["reasoning"], r["arm"])][c["twin"]].append(
                    covers(c["conditions"], ref[len(ref) // 2], defaults[c["twin"]]))
    print("\n## Experiment 3: vague readings that hold the middle reference, of the cells with a reading\n")
    print("| model | reasoning | reply | incubator | bike share |\n|---|---|---|---|---|")
    for (model, reasoning, arm), d in sorted(by.items()):
        print(f"| {model} | {reasoning} | {arm} | " + " | ".join(
            f"{sum(d[t])} / {len(d[t])}" for t in ("incubator", "bikeshare")) + " |")


for e in (1, 3):
    table(e)
coverage()
print("\nMachines: every gpt-oss run on one NVIDIA B200 (HiPerGator); Qwen3.5-9B on an Apple M-series laptop.")
