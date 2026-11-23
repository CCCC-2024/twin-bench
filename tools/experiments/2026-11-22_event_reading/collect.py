"""Collect every run of the study into runs.json: one record per run, its rows of tools/advise_events.py.

Each run wrote twins/<name>/results/advise_events.json in its own copy of the repository. The raw folders are not in
the repository (about 30 MB, mostly rows of other runs); this keeps only the run's own row in each cell, with the
not-told row it is compared with. A run counts only if its summary (the script's printed output) is complete: a run
that crashed has none, and its folder may still hold another run's files.

Usage: python collect.py RAW_DIR [QWEN_DIR]   (RAW_DIR: the cluster's run folder, copied back; see README.md)
"""
import json
import os
import sys
from pathlib import Path

RAW = Path(sys.argv[1])
QWEN = Path(sys.argv[2]) if len(sys.argv) > 2 else None
B200, MAC = "NVIDIA B200 (HiPerGator)", "Apple M-series laptop"
TWINS = ("battery", "bikeshare", "incubator", "uav")
KEPT = ("said", "conditions", "regret", "replies", "error", "extra", "right",
        "desired", "set_bikes", "retire", "maneuver", "reply")          # a row's fields, the action among them


def row_name(model, reasoning, arm):
    return model + {"range": ", ranges", "tools": ", tools", "turns1": ", 1 reply"}.get(arm, "") + \
        (f", reasoning {reasoning}" if reasoning in ("medium", "high") else "")


def complete(summary, split):
    text = summary.read_text(errors="replace") if summary.exists() else ""
    return all(f"{split}{k}" in text.replace(" ", "") for k in ("stated", "vague", "nothing", "inexpressible"))


def record(experiment, model, reasoning, arm, run, machine, folder, summary, split):
    if not complete(summary, split):
        return None
    name, cells = row_name(model, reasoning, arm), []
    for twin in TWINS:
        path = folder / twin / "advise_events.json"
        if not path.exists():
            path = folder / "twins" / twin / "results" / "advise_events.json"
        res = json.loads(path.read_text())
        for state, events in res["states"].items():
            for event, row in events.items():
                if res["events"][event]["split"] != split or name not in row:
                    continue
                r = row[name]
                cells.append(dict(twin=twin, state=state, event=event, kind=event.split("-")[0],
                                  not_told=row.get("not_told", {}).get("extra"),
                                  **{k: r.get(k) for k in KEPT if k in r}))
    return dict(experiment=experiment, model=model, reasoning=reasoning, arm=arm, run=run, machine=machine,
                split=split, summary=summary.name, cells=cells)


runs = []
# Experiment 1: the first test set ("test"), the reply as one number. Run 1 of medium and high went into the
# cluster's clone of the repository (clone_results/), the others into copies; 120b low: arm_default and rep2-4.
for i, f in enumerate(["arm_default", "rep2", "rep3", "rep4"], 1):
    runs.append(record(1, "gpt-oss:120b", "low", "number", i, B200, RAW / ("clone_results" if i == 1 else f"{f}"),
                       RAW / f"{f}.txt", "test"))
for arm in ("tools", "turns1"):
    runs.append(record(1, "gpt-oss:120b", "low", arm, 1, B200, RAW / "clone_results", RAW / f"arm_{arm}.txt", "test"))
for summary in sorted(RAW.glob("r_*_*_[0-9].txt")):
    tag, level, i = summary.stem[2:].split("_")
    runs.append(record(1, f"gpt-oss:{tag}", level, "number", int(i), B200,
                       RAW / ("clone_results" if i == "1" else summary.stem), summary, "test"))
# Experiment 3: the second test set ("test2"), one number or a range: each run in its own folder (run_array.sbatch),
# except run 1 of each configuration, made earlier in one folder per model, exp3_<model>/.
for summary in sorted(RAW.glob("exp3_*_*_*_[0-9].txt")):
    tag, level, arm, i = summary.stem[5:].split("_")
    folder = RAW / summary.stem if (RAW / summary.stem).is_dir() else RAW / f"exp3_{tag}"
    runs.append(record(3, f"gpt-oss:{tag}", level, arm, int(i), B200, folder, summary, "test2"))
if QWEN:
    for summary in sorted(QWEN.glob("*_[0-9].txt")):
        arm, i = summary.stem.split("_")
        runs.append(record(3, "qwen3.5:9b", "none", arm, int(i), MAC, QWEN / f"run_{arm}_{i}", summary, "test2"))

kept = [r for r in runs if r]
print(f"{len(kept)} runs complete, {len(runs) - len(kept)} not")
out = Path(__file__).with_name("runs.json")
out.write_text("[\n" + ",\n".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) for r in kept) + "\n]\n")
print("wrote", out, f"{os.path.getsize(out) / 1e6:.1f} MB")
