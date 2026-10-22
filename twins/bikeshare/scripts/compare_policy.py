"""Compare the twin's policies on the paper's test days (Nov-Dec 2018); nothing else in the twin changes.

The same StationTwin (rate model: HistoricalRates, the default) replays 2018 on dtcore. Until 31 Oct the rule decides;
from the night before 1 Nov each policy decides every night, and each decision is scored by replaying the next day's
recorded events, as in compare_rates.py. Policies:
  rule     RulePolicy(): the paper's decision, the start with the fewest expected lost rentals + returns
  <model>  AgentPolicy with that language model (dtcore.agent), choosing with the twin's what-if; with no valid choice
           no truck goes out, and the station starts the day with the bikes it has
The rule is exact for the twin's own forecast, so an agent can at best equal its expected shortages; how far it stays
above them measures its search. The replay also counts the twin's forecast errors: on these days the all-history
rates are about twice the demand (README, Notes), and a fixed start can lose fewer than the rule.
The models are served with the OpenAI chat API, by default Ollama's (--url http://localhost:1234/v1 for LM Studio);
not yet run against a real model server. Writes results/compare_policy.json.

Usage: python twins/bikeshare/scripts/compare_policy.py MODEL [MODEL ...] [--url URL]
"""
import argparse
import datetime
import json
from pathlib import Path

import numpy as np

from dtcore import Recorder, run
from dtcore.agent import chat_model
from twins.bikeshare import AgentPolicy, ReplayStation, RulePolicy, StationTwin, data, experiment, model

ROOT = Path(__file__).resolve().parents[1]
JAN1 = datetime.date(2018, 1, 1)
paper = experiment.load()
C, events = paper["C"], data.events()
first = (experiment.TEST_DAYS[0] - JAN1).days * 24                     # hour index of 1 Nov, 00:00


def on_the_test_days(policy):
    """Replay 2018 with the policy deciding from 31 Oct, 23:00 (the decision for 2019 is not needed, so not the last
    hour). Returns each test day's start, the policy's expected shortages and what it recorded, and the replay's."""
    twin, rec, out = StationTwin(C), Recorder("compare-policy"), {}

    def from_31_oct(k, source, twin):                  # earlier decisions do not change the test days' rates, so the
        if k == first - 1:                              # model is asked only on the 61 nights that are scored
            twin.policy = policy
    run(ReplayStation(C, events, JAN1, 365), twin, max_steps=365 * 24 - 1, recorder=rec, before_step=from_31_oct)
    for day in experiment.TEST_DAYS:
        night = rec.rows[(day - JAN1).days * 24 - 1]                    # 23:00 the day before
        action = night["action"]
        start = action.get("set_bikes", night["obs"]["bikes"])          # no truck: the bikes the station has
        lost_rent, lost_return, _ = model.replay(C, start, paper["events"].get(day, np.array([], int)))
        out[str(day)] = dict(start=start, expected=action.get("expected_lost"), lost_rent=lost_rent,
                             lost_return=lost_return, **{k: action[k] for k in ("asked", "reason") if k in action})
    return out


parser = argparse.ArgumentParser(description="Compare the bike-share twin's policies on the paper's test days.")
parser.add_argument("models", nargs="+", help="language models for AgentPolicy, as the server names them")
parser.add_argument("--url", default="http://localhost:11434/v1", help="the server's OpenAI chat API (default: Ollama)")
parser.add_argument("--turns", type=int, default=10, help="the most replies a model may give a night")
args = parser.parse_args()

agents = {name: AgentPolicy(chat_model(name, args.url), args.turns) for name in args.models}
policies = dict(rule=RulePolicy(), **agents)
res = dict(url=args.url, turns=args.turns, days={str(day): {} for day in experiment.TEST_DAYS})
for name, policy in policies.items():
    for day, scored in on_the_test_days(policy).items():
        res["days"][day][name] = scored
json.dump(res, open(ROOT / "results" / "compare_policy.json", "w"), indent=0)

days = list(res["days"].values())
lost = {p: np.array([d[p]["lost_rent"] + d[p]["lost_return"] for d in days]) for p in policies}
print(f"{'policy':<20}{'start (median)':>16}{'expects':>9}{'lost/day':>10}{'no truck':>10}{'what-ifs/night':>16}")
for p in policies:
    rows = [d[p] for d in days]
    expected = [r["expected"] for r in rows if r["expected"] is not None]          # nights a truck went out
    asked = np.mean([len(r.get("asked", [])) for r in rows])
    print(f"{p:<20}{np.median([r['start'] for r in rows]):>16.0f}{np.mean(expected) if expected else np.nan:>9.1f}"
          f"{lost[p].mean():>10.2f}{len(rows) - len(expected):>10}{asked:>16.1f}")
print("\nagainst the rule, 61 days: paired daily difference in lost rentals + returns (mean +/- standard error)")
for p in args.models:
    d = lost[p] - lost["rule"]
    same = sum(row[p]["start"] == row["rule"]["start"] for row in days)
    above = [row[p]["expected"] - row["rule"]["expected"] for row in days if row[p]["expected"] is not None]
    print(f"  {p} - rule: {d.mean():+.2f} +/- {d.std(ddof=1) / np.sqrt(len(d)):.2f}; same start on {same} days; "
          f"expects {np.mean(above) if above else np.nan:.2f} more per night than the rule (twin's forecast)")
