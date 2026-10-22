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

    def whatif(self, n):
        if type(n) is not int or not 0 <= n <= 4:
            raise ValueError("n must be a whole number from 0 to 4")
        return {"miss": abs(n - 3)}


class FixedModel:
    """A fake language model with fixed answers: its i-th reply in every conversation is answers[i]. It keeps every
    conversation it was sent."""

    def __init__(self, *answers):
        self.answers, self.sent = answers, []

    def __call__(self, messages):
        self.sent.append(list(messages))
        return self.answers[len(messages) // 2 - 1]                  # system + request, then one exchange per reply



def test_the_agent_runs_the_twins_whatif_then_answers():
    llm = FixedModel('{"whatif": {"n": 4}}', '{"answer": "Turning it to 4 misses by 1."}')
    answer = agent.ask(llm, Dial(), "What happens at 4?")
    assert answer == dict(answer="Turning it to 4 misses by 1.", choice=None, asked=[{"n": 4}], error=None)
    assert "A dial." in llm.sent[0][0]["content"] and llm.sent[0][1]["content"] == "What happens at 4?"
    assert json.loads(llm.sent[1][-1]["content"]) == {"whatif": {"n": 4}, "result": {"miss": 1}}


def test_the_agent_is_told_what_was_wrong_and_gives_no_answer_when_its_turns_run_out():
    llm = FixedModel("Three.", '{"whatif": {"n": 9}}', '{"answer": "3", "choice": {"n": 3.0}}', "3")
    answer = agent.ask(llm, Dial(), "Set the dial.", choices=[{"n": n} for n in range(5)], turns=4)
    assert answer == dict(answer=None, choice=None, asked=[], error="no valid answer in 4 replies")
    told = [conversation[-1]["content"] for conversation in llm.sent[1:]]
    assert "Not understood" in told[0] and "n must be a whole number from 0 to 4" in told[1]
    assert '{"n": 3.0} is not one of the choices' in told[2]                     # 3.0 is not the setting 3
