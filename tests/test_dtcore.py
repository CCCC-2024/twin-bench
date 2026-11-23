"""Unit tests for the skeleton itself, with a toy counter instead of a real system."""
import json
import sqlite3

import pytest

from dtcore import Recorder, Source, Twin, agent, run


class Counter(Source):
    """Physical side: counts up by `step`; an action {"step": s} changes the step size."""

    kind = "sim"

    def __init__(self, n=5):
        self.x, self.step, self.n, self.k, self.actions = 0, 1, n, 0, []

    def observe(self):
        return None if self.k >= self.n else {"t": float(self.k), "x": self.x}

    def act(self, action):
        self.actions.append(action)
        self.step = action["step"]

    def advance(self):
        self.x += self.step
        self.k += 1


class Echo(Twin):
    capabilities = ("update", "decide")

    def __init__(self):
        self.last = None

    def update(self, obs):
        self.last = obs["x"]
        return {"x_est": obs["x"]}

    def decide(self):
        return {"step": 2} if self.last == 1 else None


def test_run_stops_when_source_is_exhausted():
    assert run(Counter(n=4), Echo()) == 4


def test_max_steps():
    assert run(Counter(n=100), Echo(), max_steps=3) == 3


def test_decisions_reach_the_source_and_change_it():
    src = Counter(n=4)
    run(src, Echo())
    assert src.actions == [{"step": 2}]          # decided once, at x == 1
    assert src.x == 0 + 1 + 2 + 2 + 2


def test_decide_is_skipped_without_the_capability():
    class NoDecide(Echo):
        capabilities = ("update",)

    src = Counter(n=4)
    run(src, NoDecide())
    assert src.actions == []


def test_before_step_can_change_the_world():
    src = Counter(n=3)
    run(src, Echo(), before_step=lambda k, s, t: setattr(s, "x", 10) if k == 1 else None)
    assert src.x == 10 + 1 + 1                   # set to 10 before step 1, then two more advances


def test_recorder_keeps_rows_and_writes_sqlite(tmp_path):
    db = tmp_path / "runs.db"
    rec = Recorder("counter", path=str(db))
    run(Counter(n=3), Echo(), recorder=rec)
    rec.close()
    assert [r["obs"]["x"] for r in rec.rows] == [0, 1, 3]
    assert [r["action"] for r in rec.rows] == [None, {"step": 2}, None]
    rows = sqlite3.connect(db).execute("SELECT k, kind, data FROM records ORDER BY rowid").fetchall()
    assert [(k, kind) for k, kind, _ in rows] == [(0, "obs"), (0, "est"), (1, "obs"), (1, "est"), (1, "action"),
                                                 (2, "obs"), (2, "est")]
    assert json.loads(rows[4][2]) == {"step": 2}


def test_recorder_converts_numpy_values():
    np = pytest.importorskip("numpy")
    rec = Recorder("np")
    rec.add(0, {"t": np.float64(1.5), "v": np.arange(3)}, {"m": np.int64(2)}, None)
    assert rec.rows[0]["obs"] == {"t": 1.5, "v": [0, 1, 2]} and rec.rows[0]["est"] == {"m": 2}
    json.dumps(rec.rows)


def test_an_action_for_a_source_without_act_is_not_dropped():
    class NoAct(Counter):
        act = Source.act

    with pytest.raises(NotImplementedError, match="does not accept actions"):
        run(NoAct(n=4), Echo())                   # Echo decides at x == 1


def test_base_classes_must_be_implemented():
    with pytest.raises(NotImplementedError):
        Source().observe()
    with pytest.raises(NotImplementedError):
        Twin().update({})
    assert Twin().decide() is None


class Dial(Twin):
    """A toy twin for the agent: the what-if of turning a dial to n says how far n is from the best setting, 3."""

    def brief(self):
        return "A dial. whatif(n=...) turns it to n, a whole number from 0 to 4, and returns miss: how far off n is."

    actions = ("n",)
    objective = ("miss", +1)
    conditions = dict(shift=dict(type="integer", minimum=0, maximum=1, description="the best setting is this much higher"))

    def whatif(self, n, shift=0):
        if type(n) is not int or not 0 <= n <= 4:
            raise ValueError("n must be a whole number from 0 to 4")
        return {"miss": abs(n - 3 - shift)}

    def best(self, shift=0):
        if shift not in (0, 1):
            raise ValueError("shift must be 0 or 1")
        return {"n": 3 + shift, "miss": 0}


class FixedModel:
    """A fake language model with fixed answers: its i-th reply in every conversation is answers[i]. It keeps every
    conversation it was sent."""

    def __init__(self, *answers):
        self.answers, self.sent = answers, []

    def __call__(self, messages, tools=None):
        self.sent.append(list(messages))
        return self.answers[len(messages) // 2 - 1]                  # system + request, then one exchange per reply



def test_the_agent_runs_the_twins_whatif_then_answers():
    llm = FixedModel('{"whatif": {"n": 4}}', '{"answer": "Turning it to 4 misses by 1."}')
    answer = agent.ask(llm, Dial(), "What happens at 4?")
    assert answer == dict(answer="Turning it to 4 misses by 1.", choice=None, asked=[{"n": 4}], error=None)
    assert "A dial." in llm.sent[0][0]["content"] and llm.sent[0][1]["content"] == "What happens at 4?"
    assert json.loads(llm.sent[1][-1]["content"]) == {"whatif": {"n": 4}, "result": {"miss": 1}}
    chose = FixedModel('{"answer": "4", "choice": {"n": 4, "windy": true}}')        # the conditions it assumed, too
    assert agent.ask(chose, Dial(), "Set the dial.", choices=[{"n": n} for n in range(5)])["choice"] == {"n": 4}


def test_the_agent_is_told_what_was_wrong_and_gives_no_answer_when_its_turns_run_out():
    llm = FixedModel("Three.", '{"whatif": {"n": 9}}', '{"answer": "3", "choice": {"n": 3.0}}', "3")
    answer = agent.ask(llm, Dial(), "Set the dial.", choices=[{"n": n} for n in range(5)], turns=4)
    assert answer == dict(answer=None, choice=None, asked=[], error="no valid answer in 4 replies")
    told = [conversation[-1]["content"] for conversation in llm.sent[1:]]
    assert "Not understood" in told[0] and "n must be a whole number from 0 to 4" in told[1]
    assert '{"n": 3.0} is not one of the choices' in told[2]                     # 3.0 is not the setting 3


def call(name, **arguments):
    return json.dumps(dict(name=name, arguments=arguments))


def test_advise_has_the_model_extract_the_conditions_and_the_twin_choose():
    llm = FixedModel("The event shifts it: " + call("conditions", shift=1, n=0))   # an action it should not give, too
    advice = agent.advise(llm, Dial(), "The best setting moved up by one.")
    assert advice == dict(said="conditions", conditions={"shift": 1}, best={"n": 4, "miss": 0}, regret=None,
                          reply=llm.answers[0], replies=1, error=None)
    assert '"shift": {"type": "integer"' in llm.sent[0][0]["content"] and "A dial." in llm.sent[0][0]["content"]
    polished = agent.advise(FixedModel('{"name": "no_effect", "arguments": "polish changes nothing"}'), Dial(), "x")
    assert polished["conditions"] == {} and polished["best"] == {"n": 3, "miss": 0}     # the twin's usual choice
    cannot = agent.advise(FixedModel(call("cannot", why="the dial has no colour input")), Dial(), "Paint it red.")
    assert cannot["said"] == "cannot" and cannot["best"] is None and cannot["error"] == "the dial has no colour input"


def test_advise_tells_the_model_what_was_wrong_and_gives_no_action_when_its_turns_run_out():
    llm = FixedModel('{"conditions": {"shift": 1}}', call("conditions", shift=5), call("conditions", shift=1))
    advice = agent.advise(llm, Dial(), "x", turns=3)
    assert advice["best"] == {"n": 4, "miss": 0} and advice["replies"] == 3
    told = [conversation[-1]["content"] for conversation in llm.sent[1:]]
    assert "not one of the calls" in told[0] and "shift must be 0 or 1" in told[1]
    for reply in (call("conditions", shift=5), call("conditions", tilt=2), "It moves up."):   # twice the same
        advice = agent.advise(FixedModel(reply, reply), Dial(), "x")
        assert advice["best"] is None and advice["replies"] == 2 and advice["error"]


def test_with_ranges_the_twin_takes_the_action_whose_worst_regret_over_the_range_is_smallest():
    llm = FixedModel(call("conditions", shift={"low": 0, "high": 1}))
    advice = agent.advise(llm, Dial(), "The best setting may have moved up.", ranges=True)
    assert advice["best"] == {"n": 3, "miss": 0} and advice["regret"] == 1          # n=3 and n=4 each miss by 1 at worst
    assert "range" in llm.sent[0][0]["content"]
    bad = agent.advise(FixedModel(call("conditions", shift={"low": 1, "high": 0}), call("conditions", shift=1)), Dial(),
                       "x", ranges=True)
    assert bad["best"] == {"n": 4, "miss": 0} and bad["replies"] == 2               # told the range was backwards
    unknown = agent.advise(FixedModel(call("conditions", tilt={"low": 0, "high": 1}), call("conditions", shift=1)),
                           Dial(), "x", ranges=True)
    assert unknown["best"] == {"n": 4, "miss": 0} and unknown["replies"] == 2       # told the dial has no tilt


def test_the_conditions_a_twin_declares_are_the_inputs_of_its_best():
    import inspect
    from twins.battery import BatteryTwin
    from twins.bikeshare import StationTwin
    from twins.incubator import IncubatorTwin
    from twins.uav import UAVTwin
    for twin in (Dial, BatteryTwin, StationTwin, IncubatorTwin, UAVTwin):
        assert list(twin.conditions) == list(inspect.signature(twin.best).parameters)[1:]      # after self
        assert not set(twin.actions) & set(twin.conditions) and twin.objective[1] in (1, -1)


def test_an_event_matters_where_the_decision_is_sensitive_to_it():
    """tools/advise_events.py with a perfect reader (twins/*/results/advise_events.json): how much worse the twin's
    usual choice is than its best when the event's conditions are known, averaged over the states."""
    from pathlib import Path
    cost = {}
    for name in ("bikeshare", "incubator", "battery", "uav"):
        saved = json.loads((Path(__file__).parents[1] / "twins" / name / "results" / "advise_events.json").read_text())
        rows = [r for r in saved["states"].values() if "stated" in r]       # the states every event is scored in
        cost[name] = round(sum(r["stated"]["not_told"]["extra"] for r in rows) / len(rows), 4)
        assert all(r["stated"]["reference"]["extra"] == 0 and r["nothing"]["not_told"]["extra"] == 0 for r in rows)
        assert all(r["inexpressible"]["reference"]["right"] for r in rows)
    assert cost == dict(bikeshare=15.24, incubator=16.5, battery=0.0004, uav=0.0027)


def test_local_models_read_stated_events_but_their_guess_at_a_vague_one_can_cost_more_than_not_knowing():
    """tools/advise_events.py with Qwen3.5-9B, gpt-oss-20b and gpt-oss-120b on the test events, the 20 cells of each kind over the
    four twins (30 stated): the right readings, and for the incubator's vague events the minutes outside the band."""
    from pathlib import Path
    cells = []
    for name in ("bikeshare", "incubator", "battery", "uav"):
        saved = json.loads((Path(__file__).parents[1] / "twins" / name / "results" / "advise_events.json").read_text())
        cells += [(name, event.split("-")[0], row) for state in saved["states"].values() for event, row in state.items()
                  if saved["events"][event]["split"] == "test"]
    right = lambda model, kind: sum(row[model]["right"] for _, k, row in cells if k == kind)
    count = lambda model: [right(model, kind) for kind in ("stated", "nothing", "inexpressible")]
    assert count("reference") == [30, 20, 20]
    assert count("qwen3.5:9b") == [27, 20, 14] and count("gpt-oss:20b") == [30, 19, 15]
    assert count("qwen3.5:9b, tools") == [30, 20, 8] and count("gpt-oss:20b, tools") == [30, 19, 14]   # cannot: worse
    assert count("gpt-oss:120b") == [30, 20, 11]           # a larger model (on a B200): no better at saying cannot
    vague = [row for name, kind, row in cells if (name, kind) == ("incubator", "vague")]
    mean = lambda who: round(sum(row[who]["extra"] for row in vague) / len(vague), 1)
    assert [mean(w) for w in ("reference", "not_told", "gpt-oss:20b", "qwen3.5:9b")] == [0.9, 8.0, 29.6, 53.5]
    assert mean("gpt-oss:120b") == 29.6                     # it too gave the room an outdoor temperature
