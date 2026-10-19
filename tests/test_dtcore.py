"""Unit tests for the skeleton itself, with a toy counter instead of a real system."""
import json
import sqlite3

import pytest

from dtcore import Recorder, Source, Twin, run


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
