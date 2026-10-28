"""An event typed in words, live: the LLM extracts the what-if's demand inputs, the twin compares every start.

The same steps as "Type your own" on the page's bike-share panel, in the terminal: a fallback when the page cannot
reach the model. The twin replays 2018 with the rule up to 23:00 the night before --day; there the model reads the
event (twin.EXTRACT_EVENT_PROMPT and the twin's brief, as on the page) and the twin scores every start with the inputs
it read, next to the rule, which is not told the event.

Usage (from code/, with `ollama serve` running):
  python twins/bikeshare/scripts/live_event.py "Big concert at the arena near the station tomorrow night."
  [--model qwen3.5:9b] [--url http://localhost:11434/v1] [--reasoning none] [--day 2018-11-01]
"""
import argparse
import datetime
import time

import numpy as np

from dtcore import agent, run
from twins.bikeshare import ReplayStation, StationTwin, data, model
from twins.bikeshare.twin import EXTRACT_EVENT_PROMPT

parser = argparse.ArgumentParser(description="An event in words: the LLM extracts, the twin compares.")
parser.add_argument("text", help="the event, in words")
parser.add_argument("--model", default="qwen3.5:9b", help="the language model, as the server names it")
parser.add_argument("--url", default="http://localhost:11434/v1", help="the server's OpenAI chat API (default: Ollama)")
parser.add_argument("--reasoning", default="none", help="reasoning_effort (none stops Qwen3.5 thinking; '' for the model's own)")
parser.add_argument("--day", default="2018-11-01", help="the day the event is on; decided at 23:00 the night before")
args = parser.parse_args()

JAN1, C = datetime.date(2018, 1, 1), data.capacity()
DAY = datetime.date.fromisoformat(args.day)
llm = agent.chat_model(args.model, args.url, reasoning=args.reasoning or None)


class Done(Exception):
    pass


def at_midnight(k, source, twin):                                     # the twin has seen 23:00 the day before
    if k % 24 or JAN1 + datetime.timedelta(days=k // 24) != DAY:
        return
    lp, lr = twin.rates.ahead(24)
    rule = model.best_start(lp, lr, C)[0]
    t0 = time.time()
    reply = llm([dict(role="system", content=EXTRACT_EVENT_PROMPT + twin.brief()), dict(role="user", content=args.text)])
    said = agent._parse(reply)
    print(f"Event:            {args.text}")
    print(f"Agent reads it:   {reply.strip() or '(empty reply)'}   [{args.model}, {time.time() - t0:.1f} s]")
    if "rent_factor" not in said and "return_factor" not in said:
        print("No inputs read, so nothing to compare.")
        raise Done
    fp, fr = float(said.get("rent_factor", 1)), float(said.get("return_factor", 1))
    a, b = int(said.get("from_hour", 0)), int(said.get("to_hour", 24))
    event = (np.arange(24) >= a) & (np.arange(24) < b)
    cost = model.udf(np.where(event, lp * fp, lp), np.where(event, lr * fr, lr), C)   # every start at once
    s = int(np.argmin(cost))
    print(f"Twin compares all {C + 1} starts: start with {s} bikes, expected lost {cost[s]:.1f}")
    print(f"Rule, event not told:   {rule} bikes, expected lost {cost[rule]:.1f} with the event as read")
    raise Done


try:
    run(ReplayStation(C, data.events(), JAN1, 365), StationTwin(C), max_steps=365 * 24 - 1, before_step=at_midnight)
    print(f"{DAY} is not in 2018.")
except Done:
    pass
