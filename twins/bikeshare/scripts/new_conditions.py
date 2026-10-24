"""The LLM agent told, in words, of an event the data never showed: does it adapt the nightly decision?

The twin replays 2018 with the rule; at 23:00 before every sixth test day the agent is told one event below and asked
for the number of bikes to start the next day with. It can run the twin's what-if with the event's demand
(rent_factor / return_factor in some hours) or without. The rule does not know the event. No record holds the event,
so each choice is scored by the twin's own forecast with the event as its text says ("truth" below), against the best
start under that forecast. This measures whether the agent turns the words into the right simulation and decision,
not what the event would cost in the real city. Events A and B change the best start; C (in the evening) barely does,
so the right answer there is close to the rule's.
With --check the request also asks the agent to run the event on the what-if first; its rows are named MODEL+check.
Writes results/new_conditions.json, adding to it model by model.
The console summary covers only the models requested in this run.

Usage: python twins/bikeshare/scripts/new_conditions.py MODEL [MODEL ...] [--url URL] [--turns N]
"""
import argparse
import datetime
import json
from pathlib import Path

import numpy as np

from dtcore import agent, run
from twins.bikeshare import ReplayStation, StationTwin, data, experiment, model
from twins.bikeshare.twin import NIGHTLY_PROMPT

ROOT = Path(__file__).resolve().parents[1]
JAN1 = datetime.date(2018, 1, 1)
C = data.capacity()
EVENTS = {   # text for the agent; truth = (rent factor, return factor, from hour, to hour)
    "A": ("Tomorrow a subway line near the station is closed from 7:00 to 10:00. In those hours about three times the "
          "usual number of people will rent bikes here; returns stay as usual.", (3, 1, 7, 10)),
    "B": ("Tomorrow a charity ride starts near the station. From 6:00 to 9:00 about twice the usual number of people "
          "will rent bikes here; returns stay as usual.", (2, 1, 6, 9)),
    "C": ("Tomorrow evening a concert ends near the station at 21:00. From 21:00 to 23:00 about four times the usual "
          "number of people will rent bikes here; returns stay as usual.", (4, 1, 21, 23)),
}
DAYS = experiment.TEST_DAYS[::6]
CHECK_PROMPT = " Before you choose, run the what-if with this event and compare several numbers of bikes."

parser = argparse.ArgumentParser(description="The LLM agent on events the data never showed.")
parser.add_argument("models", nargs="+", help="language models, as the server names them")
parser.add_argument("--url", default="http://localhost:11434/v1", help="the server's OpenAI chat API (default: Ollama)")
parser.add_argument("--turns", type=int, default=15, help="the most replies a model may give a decision")
parser.add_argument("--reasoning", help="reasoning_effort for the models, e.g. none or low (default: theirs)")
parser.add_argument("--check", action="store_true", help="ask the agent to run the event on the what-if first")
args = parser.parse_args()

out = ROOT / "results" / "new_conditions.json"
res = json.loads(out.read_text()) if out.exists() else dict(models={}, events={k: v[0] for k, v in EVENTS.items()},
                                                            days={})
for model_name in args.models:
    llm = agent.chat_model(model_name, args.url, reasoning=args.reasoning)
    name = model_name + "+check" if args.check else model_name

    def at_midnight(k, source, twin):                                 # the twin has seen 23:00 the day before
        day = JAN1 + datetime.timedelta(days=k // 24)
        if k % 24 or day not in DAYS:
            return
        lp, lr = twin.rates.ahead(24)
        for key, (text, (fp, fr, a, b)) in EVENTS.items():
            event = (np.arange(24) >= a) & (np.arange(24) < b)
            truth = model.udf(np.where(event, lp * fp, lp), np.where(event, lr * fr, lr), twin.C)  # every start
            rule, best = model.best_start(lp, lr, twin.C)[0], int(np.argmin(truth))
            said = agent.ask(llm, twin, f"{NIGHTLY_PROMPT} {text}" + (CHECK_PROMPT if args.check else ""),
                             [dict(set_bikes=s) for s in range(twin.C + 1)], args.turns)
            s = None if said["choice"] is None else said["choice"]["set_bikes"]
            row = res["days"].setdefault(str(day), {}).setdefault(key, {})
            row.update(best=dict(start=best, expected=float(truth[best])),
                       rule=dict(start=rule, expected=float(truth[rule])),
                       **{name: dict(start=s, expected=None if s is None else float(truth[s]), asked=said["asked"],
                                     answer=said["answer"] or said["error"])})

    run(ReplayStation(C, data.events(), JAN1, 365), StationTwin(C), max_steps=365 * 24 - 1, before_step=at_midnight)
    res["models"][name] = dict(url=args.url, turns=args.turns, reasoning=args.reasoning)
    json.dump(res, open(out, "w"), indent=0)                          # after each model: a slow run keeps what it has

print(f"{'event':<6}{'policy':<14}{'start (median)':>15}{'above best / night':>20}{'no answer':>10}"
      f"{'ran the event':>15}")
for key in EVENTS:
    rows = [d[key] for d in res["days"].values()]
    for p in ["rule", *(n + "+check" if args.check else n for n in args.models)]:
        chose = [r for r in rows if r[p]["expected"] is not None]
        above = np.mean([r[p]["expected"] - r["best"]["expected"] for r in chose]) if chose else np.nan
        ran = "" if p == "rule" else sum(any(w.get("rent_factor", 1) != 1 for w in r[p]["asked"]) for r in rows)
        print(f"{key:<6}{p:<14}{np.median([r[p]['start'] for r in chose]) if chose else np.nan:>15.0f}{above:>20.2f}"
              f"{len(rows) - len(chose):>10}{ran:>15}")
best = {k: np.median([d[k]["best"]["start"] for d in res["days"].values()]) for k in EVENTS}
print("\nbest start (median): " + ", ".join(f"{k} {s:.0f}" for k, s in best.items()) + f"; {len(DAYS)} nights")
