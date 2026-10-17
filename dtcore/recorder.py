"""Keeps every observation, estimate and action of a run, in memory and optionally in SQLite."""
import json
import sqlite3
import time


def _plain(v):
    if hasattr(v, "tolist"):                # numpy array or scalar -> list or Python number
        return v.tolist()
    return v


class Recorder:
    def __init__(self, system, path=None):
        self.system, self.rows = system, []
        self.con = None
        if path is not None:
            self.con = sqlite3.connect(path)
            self.con.executescript(
                "CREATE TABLE IF NOT EXISTS records (system TEXT, run TEXT, k INTEGER, t REAL, kind TEXT, data TEXT);")
            self.run_id = time.strftime("%Y%m%dT%H%M%S")

    def add(self, k, obs, est, action):
        row = dict(k=k, obs={a: _plain(b) for a, b in obs.items()}, est={a: _plain(b) for a, b in (est or {}).items()},
                   action=action)
        self.rows.append(row)
        if self.con is not None:
            t = row["obs"].get("t")
            items = [("obs", row["obs"]), ("est", row["est"])] + ([("action", action)] if action is not None else [])
            self.con.executemany("INSERT INTO records VALUES (?,?,?,?,?,?)",
                                 [(self.system, self.run_id, k, t, kind, json.dumps(d)) for kind, d in items])

    def close(self):
        if self.con is not None:
            self.con.commit()
            self.con.close()
