"""BatteryTwin: the battery twin on the dtcore contract (docs/architecture.md).

Particle filter (pf.py) from first-prediction time (capacity <= 0.985); automatic retirement decision once capacity
first drops to 0.95 (as in Thelen et al.), with decide_online; predict() projects capacity, and predict(weights=...)
is the what-if on the decision. The filter is the original algorithm, unchanged; decide_online is the authors' utility
with the candidate range ending at the twin's predicted end of life, which an online twin can know (decision.py).
"""
import numpy as np
from dtcore import Twin

from .data import FPT_THRESHOLD
from .decision import decide_online
from .pf import ParticleFilter, power_law

DECIDE_AT = 0.95


class BatteryTwin(Twin):
    capabilities = ("update", "predict", "whatif", "decide")

    def __init__(self, q0, seed=0, decide_at=DECIDE_AT):
        self.q0, self.seed, self.decide_at = q0, seed, decide_at
        self.hist, self.pf, self.state, self.last, self.decision = [], None, None, None, None

    def update(self, obs):
        self.hist.append(obs["q"])
        self.last = obs
        if self.pf is None and obs["q"] <= FPT_THRESHOLD:          # first-prediction time reached
            self.pf = ParticleFilter(rng=np.random.RandomState(self.seed))
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
        q_lo, q_hi = self.pf.capacity_band(cyc)
        out = dict(cycles=cyc, q_mean=power_law(cyc, self.state["log10_a"], self.state["b"]), q_lo=q_lo, q_hi=q_hi)
        if weights is not None:
            s = self.state
            out["decision"] = decide_online(np.array(self.hist), self.q0, s["log10_a"], s["b"], s["eol_pred"],
                                            weights=tuple(weights))
        return out
