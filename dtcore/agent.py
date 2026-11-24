"""An LLM agent that works with any twin through the twin's what-if (docs/architecture.md).

ask(llm, twin, request, choices=None, turns=10) holds one conversation with a language model about one twin:
  llm      llm(messages, tools=None) -> reply (str): the model. messages are {"role": "system" | "user" |
           "assistant", "content": str} and tools are function definitions, both as in the OpenAI chat API; a
           tool call is replied as the JSON {"name": ..., "arguments": {...}}, and a model without tools may ignore
           them. chat_model(name, url) is such a model, served that way (Ollama and LM Studio serve local models so)
  twin     a Twin with brief() and whatif(**scenario) (contract.py): the system in words, and its what-if
  request  what the agent is asked, in words: a decision, or a question such as how to adapt to a new condition
  choices  when the answer is to be used as an action, the scenarios it may choose from; otherwise None. Only their
           keys count: a choice that also states the conditions the model assumed (an event) is still that action
  turns    the most replies the model may give
The model may run the what-if as often as its turns allow, then answers. Returns dict(answer, choice, asked, error):
its answer in words, the scenario it chose (with choices), the scenarios it ran, and why there is no answer, if so.

advise(llm, twin, event, turns=2) splits the work the other way: the model only extracts, from an event in words, the
conditions it means for the twin's what-if; the twin then compares every action itself, twin.best(**conditions). The
model chooses nothing and computes nothing. It answers with one of three calls, as a tool call where the server has
tools (llm(messages, tools)) and as the same JSON in its reply where not:
  conditions  the event changes these inputs (twin.conditions: name -> the JSON schema of its value)
  no_effect   the event changes nothing the choice depends on: the twin's usual choice, twin.best()
  cannot      the event matters but the inputs cannot express it: no action
A call that is not one of these, or conditions the twin cannot run (best raises ValueError or TypeError), is told
back to the model with the reason, up to `turns` replies. Returns dict(said, conditions, best, regret, reply, replies,
error): the call it ended with (None if never a valid one), the conditions, the twin's best action under them, the
worst regret of that action over a range (below; else None), the last reply, the number of replies, and why there is
no action, if so.

With ranges=True the model gives a condition the event states no number for as a range, {"low": a, "high": b}. An
event has one unknown size, so the twin moves the ranged inputs together, from each one's value now (its default in
twin.conditions) to the far end of its range, finds its best action at each step, and keeps the one whose worst regret
over the steps is smallest (robust_best): regret is how much worse an action is than the best one at that point, in the
twin's objective (twin.objective). The value now is always one of the points: the event may do less than the model
thinks, down to nothing. A large regret is a reason to ask a person before acting.
"""
import json
import urllib.error
import urllib.request

SYSTEM_PROMPT = """You work with a digital twin of a real system: a model kept in step with the system's data that can \
simulate it. Its what-if takes a scenario and returns what the twin predicts. Reply with one JSON object and nothing \
else, either
  {"whatif": {<scenario>}}   to run the what-if, or
  {"answer": "<your answer and why>"%s}   to finish.
You have at most %d replies.

The twin: %s"""


ADVISE_PROMPT = """You read events for a digital twin of a real system: a model kept in step with the system's data that \
can simulate it. The twin chooses its own action by comparing every one; you only tell it what the event the user \
describes changes. Answer with exactly one call:
  conditions  the event changes inputs of the twin's what-if. Give only the inputs it changes, as numbers; %s
  no_effect   the event changes neither how the system behaves nor what the decision is trying to achieve, such as \
a new name, colour or label. Argument: why.
  cannot      the event does change how the system behaves, what it can do, or what the decision is trying to \
achieve, but none of these inputs describes that change. Such an event is never no_effect. Argument: why.
Call it as a tool; without tools, reply with one JSON object and nothing else: {"name": "<call>", "arguments": {...}}.

The inputs of conditions: %s

The twin: %s"""
ESTIMATE = "where the event states no number, give your best estimate."
RANGE = """where the event states a number, even an approximate one such as "about three times", give that \
number. Where it states none, start from the input's value now (its default) and give a range you are confident \
holds the input under the event, as {"low": <number>, "high": <number>}: wide enough that you would be surprised to \
find the true value outside it."""


def _calls(twin):
    """The three calls of advise as tool definitions; the twin's conditions are the parameters of the first."""
    why = dict(type="object", properties=dict(why=dict(type="string")), required=["why"])
    described = (("conditions", "The event changes these inputs of the twin's what-if.",
                  dict(type="object", properties=twin.conditions, additionalProperties=False)),
                 ("no_effect", "The event changes neither how the system behaves nor what the decision is for.", why),
                 ("cannot", "The event changes the system or its goal in a way the inputs cannot express.", why))
    return [dict(type="function", function=dict(name=n, description=d, parameters=p)) for n, d, p in described]


def advise(llm, twin, event, turns=2, ranges=False):
    prompt = ADVISE_PROMPT % (RANGE if ranges else ESTIMATE, json.dumps(twin.conditions), twin.brief())
    messages = [dict(role="system", content=prompt), dict(role="user", content=event)]
    tools = _calls(twin)
    for n in range(1, turns + 1):
        reply = llm(messages, tools)
        said = _parse(reply)
        name, given = said.get("name"), said.get("arguments")
        out = dict(said=None, conditions=None, best=None, regret=None)
        if name == "cannot":                                    # only its name counts; why is for a person
            return dict(out, said=name, reply=reply, replies=n, error=_why(given))
        if name == "no_effect" or (name == "conditions" and isinstance(given, dict)):
            conditions = {k: v.get("low") if isinstance(v, dict) and v.get("low") == v.get("high") else v
                          for k, v in given.items() if k not in twin.actions} if name == "conditions" else {}
            out = dict(said=name, conditions=conditions, best=None, regret=None)
            try:
                if ranges and any(isinstance(v, dict) for v in conditions.values()):
                    best, regret = robust_best(twin, conditions)
                    return dict(out, best=best, regret=regret, reply=reply, replies=n, error=None)
                return dict(out, best=twin.best(**conditions), reply=reply, replies=n, error=None)
            except (TypeError, ValueError) as e:
                error = f"the twin cannot run {json.dumps(conditions)}: {e}"
        else:
            error = "not one of the calls conditions, no_effect, cannot"
        messages += [dict(role="assistant", content=reply), dict(role="user", content=f"No: {error}. Answer again.")]
    return dict(out, reply=reply, replies=turns, error=error)


def robust_best(twin, conditions):
    """The twin's action for conditions of which some are ranges {"low": a, "high": b}, and its worst regret.

    An event has one unknown size, so the ranged inputs move together: from each one's value now (the default in
    twin.conditions) to the end of its range farther from that, in four even steps (five points; whole-number inputs
    rounded). The range is thus widened to now: the event may do less than the model thinks, down to nothing. A range
    that reaches past now starts at its other end instead, and now is a sixth point. Of the twin's best actions at
    these points, the one whose worst regret over them is smallest is chosen. Only best() and whatif() are used, so
    the candidates are those best actions, not every action. Every condition needs its default. A point the twin cannot
    run raises, as in best(): two hour ranges that meet, such as from 7-9 and to 9-11, end at the same hour, and advise
    then tells the model."""
    fixed, paths = {}, {}
    for k, v in conditions.items():
        if not isinstance(v, dict):
            fixed[k] = v
            continue
        if k not in twin.conditions:
            raise TypeError(f"{k} is not an input of the twin")
        lo, hi, now = v.get("low"), v.get("high"), twin.conditions[k]["default"]
        if not all(type(x) in (int, float) for x in (lo, hi)) or lo > hi:
            raise ValueError(f'the range of {k} must be {{"low": a, "high": b}} with numbers a <= b')
        near, far = sorted((lo, hi), key=lambda x: abs(x - now))
        start = near if (near - now) * (far - now) < 0 else now
        path = [start + (far - start) * step / 4 for step in range(5)] + [now]
        paths[k] = [int(round(x)) for x in path] if twin.conditions[k].get("type") == "integer" else path
    grid = [dict(fixed, **{k: path[j] for k, path in paths.items()}) for j in range(6)]
    grid = [c for j, c in enumerate(grid) if c not in grid[:j]]
    key, sign = twin.objective
    take = lambda a: {k: a[k] for k in twin.actions}
    score = lambda action, c: sign * twin.whatif(**take(action), **c)[key]
    bests = [twin.best(**c) for c in grid]
    worst = lambda action: max(score(action, c) - score(b, c) for b, c in zip(bests, grid))
    chosen = min(bests, key=worst)
    return chosen, round(float(worst(chosen)), 4) + 0.0


def ask(llm, twin, request, choices=None, turns=10):
    allowed = None if choices is None else {_key(c) for c in choices}
    keys = set().union(*choices) if choices else set()
    choose = ', "choice": {<the scenario you choose>}' if choices else ""
    messages = [dict(role="system", content=SYSTEM_PROMPT % (choose, turns, twin.brief())),
                dict(role="user", content=request)]
    asked = []
    for _ in range(turns):
        reply = llm(messages)
        messages.append(dict(role="assistant", content=reply))
        said = _parse(reply)
        if isinstance(said.get("whatif"), dict):
            try:
                note = json.dumps(dict(whatif=said["whatif"], result=twin.whatif(**said["whatif"])))
                asked.append(said["whatif"])
            except (TypeError, ValueError) as e:                        # a scenario the twin cannot run
                note = f"The twin cannot run {json.dumps(said['whatif'])}: {e}"
        elif "answer" in said and (allowed is None or _key(_only(said.get("choice"), keys)) in allowed):
            return dict(answer=str(said["answer"]), choice=_only(said["choice"], keys) if choices else None,
                        asked=asked, error=None)
        elif "answer" in said:
            note = f"{json.dumps(said.get('choice'))} is not one of the choices. Reply with one JSON object as " \
                   "described."
        else:
            note = 'Not understood: reply with one JSON object with "whatif" or "answer", as described.'
        messages.append(dict(role="user", content=note))
    return dict(answer=None, choice=None, asked=asked, error=f"no valid answer in {turns} replies")


def _why(given):
    return str(next(iter(given.values()), "") if isinstance(given, dict) else given)


def _parse(reply):
    """The JSON object in a reply (a model may wrap it in text or a code block), or {}."""
    try:
        return json.loads(reply[reply.find("{"):reply.rfind("}") + 1])
    except ValueError:
        return {}


def _only(choice, keys):
    return {k: v for k, v in choice.items() if k in keys} if isinstance(choice, dict) else choice


def _key(scenario):
    return json.dumps(scenario, sort_keys=True)                         # so 3.0 or true is not the setting 3 or 1


def chat_model(name, url, max_tokens=4096, reasoning=None, use_tools=False):
    """llm(messages) -> reply from the model `name` served with the OpenAI chat API at `url` (Ollama:
    http://localhost:11434/v1), with temperature 0: its most likely reply. max_tokens caps each reply, thinking
    included: Qwen3.5-9B once thought for over 30,000 tokens. A reply cut there is empty, which ask() does not accept,
    and so is one the server answers with an error (Ollama could not parse some of gpt-oss-20b's output). reasoning is
    sent as reasoning_effort: "none" stops Qwen3.5-9B thinking; gpt-oss-20b thinks least at "low". The tools
    are sent only with use_tools=True, and the model's first tool call is then the reply, as {"name": ...,
    "arguments": {...}}. Off by default: in advise, both models read events as well without tool calls and said
    cannot more reliably (tools/advise_events.py --tools)."""
    def llm(messages, tools=None):
        options = dict(model=name, messages=messages, temperature=0, max_tokens=max_tokens)
        if reasoning is not None:
            options["reasoning_effort"] = reasoning
        if tools and use_tools:
            options["tools"] = tools
        request = urllib.request.Request(f"{url}/chat/completions", json.dumps(options).encode(),
                                         {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                message = json.load(response)["choices"][0]["message"]
        except urllib.error.HTTPError:
            return ""
        for call in message.get("tool_calls") or []:
            try:
                return json.dumps(dict(name=call["function"]["name"], arguments=json.loads(call["function"]["arguments"])))
            except ValueError:                                              # arguments that are not JSON
                return call["function"]["arguments"]
        return message.get("content") or ""
    return llm
