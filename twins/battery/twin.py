"""BatteryTwin: the battery twin on the dtcore contract (docs/architecture.md).

Particle filter (pf.py) from first-prediction time (capacity <= 0.985); automatic retirement decision once capacity
first drops to 0.95 (as in Thelen et al.), with decide_online; predict() projects capacity, and predict(weights=...)
is the what-if on the decision. The filter is the original algorithm, unchanged; decide_online is the authors' utility
with the candidate range ending at the twin's predicted end of life, which an online twin can know (decision.py).
brief(), whatif(), best(): for an LLM agent (dtcore.agent): retiring at a cycle, optionally with the cell fading faster
or slower from now on (fade_factor), as a few numbers; and the best retirement cycle under that fade.
"""
import numpy as np
from dtcore import Twin

from .data import FPT_THRESHOLD
from .decision import decide_online
from .pf import ParticleFilter, power_law

DECIDE_AT = 0.95


class BatteryTwin(Twin):
    capabilities = ("update", "predict", "whatif", "decide")
    actions = ("retire_at",)
    conditions = dict(fade_factor=dict(type="number", minimum=0.2, maximum=5, description=
                                       "the cell loses capacity this many times as fast from now on (1: as now)"))

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

    # ---- for an LLM agent -------------------------------------------------------------------------
    def brief(self):
        return (f"A battery cell in its first life, at cycle {self.last['cycle']} with {100 * self.last['q']:.1f} % of "
                f"its first capacity; at the present fade it reaches end of life (50 %) near cycle "
                f"{int(self.state['eol_pred'])}. Retiring it later delivers more energy in total but shorter runs per "
                "charge. whatif(retire_at=x, fade_factor=1): the cell is retired after cycle x, a whole number, "
                "now or later; optionally it loses capacity fade_factor times as fast from now on, a number "
                "from 0.2 to 5 (1: as now). It returns utility, from 0 to 1, higher is better.")

    def _decision(self, fade_factor):
        if self.state is None:
            raise ValueError("no forecast yet: the twin predicts from 98.5 % capacity on")
        if type(fade_factor) not in (int, float) or not 0.2 <= fade_factor <= 5:
            raise ValueError("fade_factor must be a number from 0.2 to 5")
        s = self.state
        return decide_online(np.array(self.hist), self.q0, s["log10_a"], s["b"], s["eol_pred"], fade_factor=float(fade_factor))

    def whatif(self, retire_at, fade_factor=1.0):
        d = self._decision(fade_factor)
        if type(retire_at) is not int or retire_at < d["xs"][0]:
            raise ValueError(f"retire_at must be a whole number, {int(d['xs'][0])} or more")
        x = min(retire_at, int(d["xs"][-1]))               # past its end of life the cell has already stopped
        return dict(utility=round(float(d["utility"][x - d["xs"][0]]), 4))

    def best(self, fade_factor=1.0):
        d = self._decision(fade_factor)
        return dict(retire_at=d["x_twin"], utility=round(float(np.nanmax(d["utility"])), 4), end_of_life=d["horizon"])
