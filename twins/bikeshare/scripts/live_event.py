"""An event typed in words, live: the LLM extracts the what-if's conditions, the twin compares every start.

The same steps as "Type your own" on the page's bike-share panel, in the terminal: a fallback when the page cannot
reach the model. The twin replays 2018 with the rule up to 23:00 the night before --day; there dtcore.agent.advise
has the model read the event and the twin choose its start, shown next to the rule, which is not told the event.

Usage (from code/, with `ollama serve` running):
  python twins/bikeshare/scripts/live_event.py "Big concert at the arena near the station tomorrow night."
  [--model qwen3.5:9b] [--url http://localhost:11434/v1] [--reasoning none] [--day 2018-11-01]
"""
import argparse
import datetime
import time

from dtcore import agent, run
from twins.bikeshare import ReplayStation, StationTwin, data

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
    rule = twin.best()["set_bikes"]
    t0 = time.time()
    a = agent.advise(llm, twin, args.text)
    print(f"Event:            {args.text}")
    print(f"Agent reads it:   {a['reply'].strip() or '(empty reply)'}   [{args.model}, {time.time() - t0:.1f} s]")
    if a["best"] is None:
        print(f"No action: {a['error']}")
        raise Done
    lost = twin.whatif(rule, **a["conditions"])["expected_lost"]
    print(f"Twin compares all {C + 1} starts: start with {a['best']['set_bikes']} bikes, expected lost {a['best']['expected_lost']:.1f}")
    print(f"Rule, event not told:   {rule} bikes, expected lost {lost:.1f} with the event as read")
    raise Done


try:
    run(ReplayStation(C, data.events(), JAN1, 365), StationTwin(C), max_steps=365 * 24 - 1, before_step=at_midnight)
    print(f"{DAY} is not in 2018.")
except Done:
    pass
