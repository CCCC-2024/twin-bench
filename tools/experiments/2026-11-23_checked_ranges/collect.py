"""Collect every run of the study into runs.json: one record per run, its rows of tools/advise_events.py on test3.

Each run wrote twins/<name>/results/advise_events.json in its own copy of the code, RAW/<code>_<model>_<reasoning>_
<arm>_<run>/, and its printed summary next to it as .txt (run_array.sbatch; the laptop's runs have the same names).
A run counts only if its summary is complete. The arm of a record is number, range (the code as at 6363443) or
checked (the code as frozen for this study, asked for ranges).

Usage: python collect.py RAW_DIR [RAW_DIR ...]
"""
import json
import os
import sys
from pathlib import Path

MODELS = {"120b": "gpt-oss:120b", "20b": "gpt-oss:20b", "qwen": "qwen3.5:9b"}
MACHINES = {"qwen": "Apple M-series laptop"}
TWINS = ("incubator", "bikeshare")
KEPT = ("said", "conditions", "regret", "replies", "error", "extra", "right", "desired", "set_bikes", "reply")
KINDS = ("stated", "vague", "nothing", "inexpressible")

runs, dropped = [], []
for raw in map(Path, sys.argv[1:]):
    for summary in sorted(raw.glob("*_*_*_*_[0-9].txt")):
        code, tag, level, arm, i = summary.stem.split("_")
        text = summary.read_text(errors="replace").replace(" ", "")
        if not all(f"test3{k}" in text for k in KINDS):
            dropped.append(summary.name)
            continue
        name = MODELS[tag] + (", ranges" if arm == "range" else "") + \
            (f", reasoning {level}" if level == "medium" else "")
        cells = []
        for twin in TWINS:
            res = json.loads((raw / summary.stem / "twins" / twin / "results" / "advise_events.json").read_text())
            for state, events in res["states"].items():
                for event, row in events.items():
                    if res["events"][event]["split"] == "test3" and name in row:
                        cells.append(dict(twin=twin, state=state, event=event, kind=event.split("-")[0],
                                          not_told=row.get("not_told", {}).get("extra"),
                                          **{k: row[name][k] for k in KEPT if k in row[name]}))
        runs.append(dict(model=MODELS[tag], reasoning=level, arm="checked" if code == "checked" else arm, run=int(i),
                         machine=MACHINES.get(tag, "NVIDIA B200 (HiPerGator)"), summary=summary.name, cells=cells))

print(f"{len(runs)} runs complete; not complete: {dropped or 'none'}")
out = Path(__file__).with_name("runs.json")
out.write_text("[\n" + ",\n".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) for r in runs) + "\n]\n")
print("wrote", out, f"{os.path.getsize(out) / 1e6:.1f} MB")
