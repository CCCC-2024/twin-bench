"""An LLM agent that makes a twin's decision, using the twin's what-if as its tool.

ask(llm, task, options, whatif) holds one conversation with a language model: the model may ask the twin's what-if
about any of the options, as often as its turns allow, and then chooses one.
  llm(messages) -> str   the model; messages are {"role": ..., "content": ...} dicts as in the OpenAI chat API
  task                   what the twin asks: the system, the decision, the goal, what the what-if returns
  options                the allowed choices
  whatif(choice) -> dict what the twin predicts for that choice (numbers the model can compare)
chat_model(name, url) is such an llm: a model served with the OpenAI chat API (Ollama and LM Studio serve local
models this way).
Nothing here is about bike share; it moves to dtcore when a second twin uses it (docs/architecture.md).
"""
import json
import urllib.request

PROTOCOL = ("You make a decision for a digital twin of a real system. Before deciding you can ask the twin's what-if: "
            "it simulates the system with one of your choices and returns what it predicts. Reply with one JSON "
            "object and nothing else, either\n"
            '  {{"whatif": <choice>}}   to see the twin\'s prediction for that choice, or\n'
            '  {{"choose": <choice>, "reason": "<one sentence>"}}   to decide.\n'
            "You have at most {turns} replies; decide before they run out.")


def ask(llm, task, options, whatif, turns=10):
    """Returns dict(choice, reason, whatifs): the model's choice, its reason, and the (choice, prediction) pairs it
    asked for."""
    options = list(options)
    messages = [dict(role="system", content=PROTOCOL.format(turns=turns)), dict(role="user", content=task)]
    asked = []
    for _ in range(turns):
        reply = llm(messages)
        messages.append(dict(role="assistant", content=reply))
        answer, kind = _read(reply)
        if kind is None:
            problem = 'the reply is not a JSON object with "whatif" or "choose"'
        elif answer[kind] not in options:
            problem = f"{answer[kind]!r} is not one of the choices"
        elif kind == "choose":
            return dict(choice=answer["choose"], reason=answer.get("reason", ""), whatifs=asked)
        else:
            outcome = whatif(answer["whatif"])
            asked.append((answer["whatif"], outcome))
            messages.append(dict(role="user", content=json.dumps(dict(whatif=answer["whatif"], **outcome))))
            continue
        messages.append(dict(role="user", content=f"Not understood: {problem}. Reply with one JSON object as "
                                                  "described."))
    return dict(choice=None, reason=f"no valid choice in {turns} replies", whatifs=asked)


def _read(reply):
    """The JSON object in a reply (a model may wrap it in text or a code block) and which of the two answers it is."""
    try:
        answer = json.loads(reply[reply.find("{"):reply.rfind("}") + 1])
    except ValueError:
        return None, None
    return answer, next((k for k in ("choose", "whatif") if k in answer), None)


def chat_model(name, url):
    """llm(messages) -> reply from the model `name` served at `url` (e.g. http://localhost:11434/v1, Ollama), with
    temperature 0: its most likely reply."""
    def llm(messages):
        body = json.dumps(dict(model=name, messages=messages, temperature=0)).encode()
        request = urllib.request.Request(f"{url}/chat/completions", body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(request) as response:
            return json.load(response)["choices"][0]["message"]["content"]
    return llm
