"""Adapter: the existing battery twin on the shared skeleton (dtcore).

Nothing in data.py, pf.py or decision.py changes; this file only maps them onto the contract.
  CellReplay   physical side: one cell's *measured* cycles (not the authors' extrapolation).
               act({"retire_at": x}) retires the cell: the replay stops after cycle x.
  BatteryDT    digital side: particle filter from first-prediction time (capacity <= 0.985),
               automatic retirement decision once capacity first drops to 0.95 (as in Thelen et al.),
               predict() projects capacity; predict(weights=...) is the what-if on the decision.
"""
import numpy as np
from dtcore import Source, Twin

from .data import FPT_THRESHOLD
from .decision import WEIGHTS, decide_online
from .pf import BatteryTwin, power_law

DECIDE_AT = 0.95


class CellReplay(Source):
    kind = "replay"

    def __init__(self, cell):
        self.cell, self.i, self.retire_at = cell, 0, None

    def observe(self):
        c = self.cell
        if self.i >= c["n_measured"]:
            return None
        cycle = int(c["cycles"][self.i])
        if self.retire_at is not None and cycle > self.retire_at:
            return None                                   # retired: no more use, no more data
        return dict(t=cycle, cycle=cycle, q=float(c["q"][self.i]), cap_ah=float(c["cap"][self.i]))

    def act(self, action):
        self.retire_at = int(action["retire_at"])

    def advance(self):
        self.i += 1


class BatteryDT(Twin):
    capabilities = ("update", "predict", "whatif", "decide")

    def __init__(self, q0, seed=0, decide_at=DECIDE_AT):
        self.q0, self.seed, self.decide_at = q0, seed, decide_at
        self.hist, self.pf, self.state, self.last, self.decision = [], None, None, None, None

    def update(self, obs):
        self.hist.append(obs["q"])
        self.last = obs
        if self.pf is None and obs["q"] <= FPT_THRESHOLD:          # first-prediction time reached
            self.pf = BatteryTwin(rng=np.random.RandomState(self.seed))
        if self.pf is not None:
            self.state = self.pf.step(obs["cycle"], obs["q"])
        out = dict(self.state) if self.pf is not None else dict(cycle=obs["cycle"])
        out["started"] = self.pf is not None
        return out

    def decide(self):
        if self.decision is None and self.state is not None and self.last["q"] <= self.decide_at:
            s = self.state
            self.decision = decide_online(np.array(self.hist), self.q0, s["log10_a"], s["b"], s["eol_pred"])
            self.decision["cycle_decided"] = self.last["cycle"]
            return {"retire_at": self.decision["x_twin"]}
        return None

    def predict(self, horizon, weights=None):
        """Capacity projection for the next `horizon` cycles (mean and 5-95 % over particles).
        With weights: what-if on the retirement decision (utility over candidate cycles)."""
        if self.state is None:
            return None
        k = self.last["cycle"]
        cyc = np.arange(k + 1, k + 1 + horizon)
        P = self.pf.particles
        band = power_law(cyc[None, :], P[:, :1], P[:, 1:])
        out = dict(cycles=cyc, q_mean=power_law(cyc, self.state["log10_a"], self.state["b"]),
                   q_lo=np.percentile(band, 5, axis=0), q_hi=np.percentile(band, 95, axis=0))
        if weights is not None:
            s = self.state
            out["decision"] = decide_online(np.array(self.hist), self.q0, s["log10_a"], s["b"], s["eol_pred"],
                                            weights=tuple(weights))
        return out


__all__ = ["CellReplay", "BatteryDT", "WEIGHTS"]
