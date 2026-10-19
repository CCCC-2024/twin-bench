"""What the twin knows: states, strain tables, transition model, rewards and the planner's policy.

States: 25 = damage of wing region 1 x region 2, each 0/20/40/60/80 %. Controls: 2g or 3g turn.
Observations: 24 strain gauges; table means are divided by the load factor (as the authors do).
Policy: the authors' value iteration (planner.py). planning reward = state reward + alpha2 x control reward,
alpha2 = 2.5 in the paper; policy(alpha2) recomputes it for the what-if.
"""
from pathlib import Path

import numpy as np
from digitaltwin.planner import Planner
from digitaltwin.utils import flatten_list, read_json_file

CFG = Path(__file__).resolve().parent / "upstream/UAV-Digital-Twin/src/digitaltwin/inputfiles/UAVconfig.json"
LOAD = {"2g": 2.0, "3g": 3.0}
SIG_TWIN = 125.0           # twin's noise model: N(mean/load, 125/sqrt(load))  (noros repro, as the authors' GM)
SIG_ASSET = 150.0          # authors' measurement generator noise (before dividing by load)


class _GM:
    """The bits of the authors' GraphicalModel the planner needs (same formulas)."""
    def __init__(self, cfg):
        self.config = cfg
        self.config["flat_states"] = flatten_list(cfg["states"])
        self.n_samples = 30

    def health_reward_function(self, ref_obs):
        return 1.0 - (np.max(ref_obs) / 2500)

    def control_reward_function(self, control):
        return 0.2 * control - 0.1


class UAVModel:
    def __init__(self):
        self.cfg = read_json_file(str(CFG))
        self.gm = _GM(self.cfg)
        self.S = self.cfg["flat_states"]                     # list of 25 (d1, d2) tuples in percent
        self.C = self.cfg["controls"]                        # ['2g', '3g']
        self.p = self.cfg["transition_probabilities"]        # {'2g': 0.05, '3g': 0.1}
        self.gamma = self.cfg["gamma"]
        obs = self.cfg["observations"]
        self.raw = {c: np.array([obs[str(s[0])][str(s[1])][c]["mean"] for s in self.S]) for c in self.C}
        self.means = {c: self.raw[c] / LOAD[c] for c in self.C}
        self.T = [self._T(c) for c in self.C]
        planner = Planner(self.gm)
        self.SR = np.array([[planner.state_reward_function(s, c) for c in self.C] for s in self.S])   # 25 x 2

    def _T(self, control):
        p, S = self.p[control], self.S
        T = np.zeros((25, 25))
        for i, s1 in enumerate(S):
            for j, s2 in enumerate(S):
                d1, d2 = s2[0] - s1[0], s2[1] - s1[1]
                if s1 == (80, 80) and s2 == (80, 80): T[i, j] = 1.0
                elif d1 == d2 == 0: T[i, j] = (1 - p) * (1 - p)
                elif d1 == 20 and d2 == 20: T[i, j] = p * p
                elif d1 == 20 and d2 == 0: T[i, j] = p * (1 - p)
                elif d1 == 0 and d2 == 20: T[i, j] = p * (1 - p)
        return T

    def policy(self, alpha2=2.5):
        """Value iteration exactly as planner.getPolicy, with the control-reward weight as a parameter."""
        R = self.SR + alpha2 * np.array([[self.gm.control_reward_function(ci) for ci in range(2)]])
        V = np.zeros(25)
        for _ in range(2, 100):
            v = np.stack([R[:, ci] + self.T[ci] @ V for ci in range(2)], axis=1)
            V = self.gamma * v.max(axis=1)
        Q = np.stack([R[:, ci] + self.T[ci] @ V for ci in range(2)], axis=1)
        return np.argmax(Q, axis=1)

    def likelihood(self, m, ci):
        from scipy.stats import norm
        c = self.C[ci]
        ll = norm.logpdf(np.asarray(m)[None, :], self.means[c], SIG_TWIN / np.sqrt(LOAD[c])).sum(1)
        w = np.exp(ll - ll.max())
        return w / w.sum()

    def rollout(self, belief, pol, horizon=30):
        """Expected outcome if this policy is followed for `horizon` turns, starting from `belief`
        and (optimistically) acting on the true state: damage distribution propagated with T."""
        b = np.array(belief, float)
        n3g = 0.0
        for _ in range(horizon):
            n3g += b[pol == 1].sum()
            b = sum(b[pol == ci] @ self.T[ci][pol == ci] for ci in range(2))
            b = b / b.sum()
        d1 = np.array([s[0] for s in self.S]); d2 = np.array([s[1] for s in self.S])
        return dict(n3g=float(n3g), mean_d1=float(b @ d1), mean_d2=float(b @ d2), p_d1_60=float(b[d1 >= 60].sum()))
