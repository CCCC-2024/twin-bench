"""An LLM agent that works with any twin through the twin's what-if (docs/architecture.md).

ask(llm, twin, request, choices=None, turns=10) holds one conversation with a language model about one twin:
  llm      llm(messages) -> reply (str): the model. messages are {"role": "system" | "user" | "assistant",
           "content": str}, as in the OpenAI chat API; chat_model(name, url) is such a model, served that way
           (Ollama and LM Studio serve local models so)
  twin     a Twin with brief() and whatif(**scenario) (contract.py): the system in words, and its what-if
  request  what the agent is asked, in words: a decision, or a question such as how to adapt to a new condition
  choices  when the answer is to be used as an action, the scenarios it may choose from; otherwise None
  turns    the most replies the model may give
The model may run the what-if as often as its turns allow, then answers. Returns dict(answer, choice, asked, error):
its answer in words, the scenario it chose (with choices), the scenarios it ran, and why there is no answer, if so.
"""
import json
import urllib.error
import urllib.request

PROTOCOL = """You work with a digital twin of a real system: a model kept in step with the system's data that can \
simulate it. Its what-if takes a scenario and returns what the twin predicts. Reply with one JSON object and nothing \
else, either
  {"whatif": {<scenario>}}   to run the what-if, or
  {"answer": "<your answer and why>"%s}   to finish.
You have at most %d replies.

The twin: %s"""


def ask(llm, twin, request, choices=None, turns=10):
    allowed = None if choices is None else {_key(c) for c in choices}
    choose = ', "choice": {<the scenario you choose>}' if choices else ""
    messages = [dict(role="system", content=PROTOCOL % (choose, turns, twin.brief())),
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
        elif "answer" in said and (allowed is None or _key(said.get("choice")) in allowed):
            return dict(answer=str(said["answer"]), choice=said.get("choice") if choices else None, asked=asked,
                        error=None)
        elif "answer" in said:
            note = f"{json.dumps(said.get('choice'))} is not one of the choices. Reply with one JSON object as " \
                   "described."
        else:
            note = 'Not understood: reply with one JSON object with "whatif" or "answer", as described.'
        messages.append(dict(role="user", content=note))
    return dict(answer=None, choice=None, asked=asked, error=f"no valid answer in {turns} replies")


def _parse(reply):
    """The JSON object in a reply (a model may wrap it in text or a code block), or {}."""
    try:
        return json.loads(reply[reply.find("{"):reply.rfind("}") + 1])
    except ValueError:
        return {}


def _key(scenario):
    return json.dumps(scenario, sort_keys=True)                         # so 3.0 or true is not the setting 3 or 1


def chat_model(name, url, max_tokens=4096):
    """llm(messages) -> reply from the model `name` served with the OpenAI chat API at `url` (Ollama:
    http://localhost:11434/v1), with temperature 0: its most likely reply. max_tokens caps each reply, thinking
    included: Qwen3.5-9B once thought for over 30,000 tokens. A reply cut there is empty, which ask() does not accept,
    and so is one the server answers with an error (Ollama could not parse some of gpt-oss-20b's output)."""
    def llm(messages):
        body = json.dumps(dict(model=name, messages=messages, temperature=0, max_tokens=max_tokens)).encode()
        request = urllib.request.Request(f"{url}/chat/completions", body, {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                return json.load(response)["choices"][0]["message"]["content"]
        except urllib.error.HTTPError:
            return ""
    return llm
