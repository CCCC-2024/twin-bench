"""The registered tables of the study, from runs.json: python summarize.py > results.md

Means are over vague cells and runs. A cell is harmed if its extra is more than 0.5 above not told's. The interval is
the 2.5 to 97.5 % span of the mean difference from not told over 10,000 bootstrap draws of the twin's events (an
event's states and runs stay together). Questions 1 to 4 are answered by the means, as registered in README.md.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

RUNS = json.loads(Path(__file__).with_name("runs.json").read_text())
ARMS, TWINS = ("number", "range", "checked"), ("incubator", "bikeshare")
CONFIGS = sorted({(r["model"], r["reasoning"]) for r in RUNS})
rng = np.random.default_rng(0)


def cells(config, arm, twin, events=None, kind="vague"):
    return [c for r in RUNS if (r["model"], r["reasoning"]) == config and r["arm"] == arm for c in r["cells"]
            if c["twin"] == twin and c["kind"] == kind and (events is None or int(c["event"].split("-")[1]) in events)]


def mean(cs, key="extra"):
    return np.mean([c[key] for c in cs]) if cs else float("nan")


def harmed(cs):
    return 100 * np.mean([c["extra"] - c["not_told"] > 0.5 for c in cs]) if cs else float("nan")


def interval(cs):
    by = defaultdict(list)
    for c in cs:
        by[c["event"]].append(c["extra"] - c["not_told"])
    sums, counts = np.array([sum(v) for v in by.values()]), np.array([len(v) for v in by.values()])
    draws = rng.integers(0, len(by), (10000, len(by)))
    lo, hi = np.quantile(sums[draws].sum(1) / counts[draws].sum(1), [0.025, 0.975])
    return f"{lo:+.2f} to {hi:+.2f}"


def runs_of(config, arm):
    return sum((r["model"], r["reasoning"]) == config and r["arm"] == arm for r in RUNS)


print("## Vague events: mean extra, share of cells harmed, and the difference from not told with its interval\n")
print("| model | reasoning | twin | arm | runs | extra | not told | harmed | difference | 95 % interval |")
print("|---|---|---|---|---|---|---|---|---|---|")
for config in CONFIGS:
    for twin in TWINS:
        for arm in ARMS:
            cs = cells(config, arm, twin)
            if cs:
                print(f"| {config[0]} | {config[1]} | {twin} | {arm} | {runs_of(config, arm)} | {mean(cs):.2f} | "
                      f"{mean(cs, 'not_told'):.2f} | {harmed(cs):.0f} % | {mean(cs) - mean(cs, 'not_told'):+.2f} | "
                      f"{interval(cs)} |")

print("\n## The registered questions (by the means)\n")
print("| model | reasoning | twin | 1. checked below not told | 2. checked below range | "
      "3. checked harms fewer cells than range |")
print("|---|---|---|---|---|---|")
answers = defaultdict(list)
for config in CONFIGS:
    for twin in TWINS:
        ch, ra = cells(config, "checked", twin), cells(config, "range", twin)
        if not ch or not ra:
            continue
        q = (mean(ch) < mean(ch, "not_told"), mean(ch) < mean(ra), harmed(ch) < harmed(ra))
        for n, a in enumerate(q, 1):
            answers[n].append(a)
        print(f"| {config[0]} | {config[1]} | {twin} | " + " | ".join(
            f"{'yes' if a else 'no'} ({x})" for a, x in zip(q, (
                f"{mean(ch):.2f} / {mean(ch, 'not_told'):.2f}", f"{mean(ch):.2f} / {mean(ra):.2f}",
                f"{harmed(ch):.0f} % / {harmed(ra):.0f} %"))) + " |")

print("\n## Bike share by kind of event: mean extra (share harmed)\n")
print("| model | reasoning | arm | 21-24, one vague input | 25-28, the same situations, all vague | 29-31, whole day | "
      "not told: 21-28, 29-31 |\n|---|---|---|---|---|---|---|")
for config in CONFIGS:
    for arm in ARMS:
        one, all_, day = (cells(config, arm, "bikeshare", e) for e in (range(21, 25), range(25, 29), range(29, 32)))
        if not one:
            continue
        if arm != "number":
            answers[4].append(mean(one) < mean(all_))
        print(f"| {config[0]} | {config[1]} | {arm} | "
              + " | ".join(f"{mean(x):.2f} ({harmed(x):.0f} %)" for x in (one, all_, day))
              + f" | {mean(one, 'not_told'):.2f}, {mean(day, 'not_told'):.2f} |")

print("\n## Answers\n")
texts = ("`checked` beats not telling the twin", "`checked` beats `range`", "`checked` harms fewer cells than `range`",
         "one vague input is easier than several (21-24 below 25-28, for `range` and `checked`)")
for n, text in enumerate(texts, 1):
    a = answers[n]
    print(f"{n}. {text}: **{'yes' if a and all(a) else 'no'}** ({sum(a)} of {len(a)} comparisons).")

print("\n## Reading the other events (right, of the cells; mean over runs)\n")
print("| model | reasoning | arm | stated | nothing | inexpressible | vague cells without an action |")
print("|---|---|---|---|---|---|---|")
for config in CONFIGS:
    for arm in ARMS:
        n = runs_of(config, arm)
        if not n:
            continue
        count = lambda kind: (sum(c["right"] for t in TWINS for c in cells(config, arm, t, kind=kind)) / n,
                              sum(len(cells(config, arm, t, kind=kind)) for t in TWINS) // n)
        vague = [c for t in TWINS for c in cells(config, arm, t)]
        idle = sum(not any(k in c for k in ("desired", "set_bikes")) for c in vague) / n
        print(f"| {config[0]} | {config[1]} | {arm} | " + " | ".join("%.1f / %d" % count(k) for k in
              ("stated", "nothing", "inexpressible")) + f" | {idle:.1f} / {len(vague) // n} |")
