"""UAV digital twin on the skeleton: discrete Bayes filter over 25 damage states + the planner's policy.

update(obs)   posterior over damage from the 24 strains under the maneuver flown, then predict the next step.
decide()      the maneuver for the next turn: the control most of the predicted belief mass maps to
              under the policy (the authors' ControlP marginal argmax).
predict(h, alpha2=None)  expected outcome of following a policy for h turns (what-if on the reward weight).
brief(), whatif(), best()  for an LLM agent (dtcore.agent): the planner's value of a maneuver, optionally with damage
              growing faster or slower (growth_factor), and the maneuver the planner then picks.
"""
import numpy as np
from dtcore import Twin


class UAVTwin(Twin):
    capabilities = ("update", "predict", "whatif", "decide")
    actions = ("maneuver",)
    objective = ("value", -1)
    conditions = dict(growth_factor=dict(type="number", minimum=0, maximum=3, default=1, description=
                                         "wing damage grows this many times as often as usual from now on (1: as usual)"))

    def __init__(self, model, alpha2=2.5):
        self.m, self.alpha2 = model, alpha2
        self.pol = model.policy(alpha2)
        self.b_pred = np.zeros(25); self.b_pred[0] = 1.0       # prior: undamaged
        self.b_post = self.b_pred.copy()

    def set_policy(self, alpha2):
        self.alpha2, self.pol = alpha2, self.m.policy(alpha2)

    def decide(self):
        pu = np.array([self.b_pred[self.pol == u].sum() for u in range(2)])
        return {"u": int(np.argmax(pu)), "p3g": float(pu[1])}

    def update(self, obs):
        u = obs["u"]
        b = self.b_pred * self.m.likelihood(obs["m"], u)
        self.b_post = b / b.sum()
        nxt = self.b_post @ self.m.T[u]
        self.b_pred = nxt / nxt.sum()
        S = self.m.S
        return dict(t=obs["t"], est1=float(sum(self.b_post[i] * S[i][0] for i in range(25))),
                    est2=float(sum(self.b_post[i] * S[i][1] for i in range(25))),
                    map=list(S[int(np.argmax(self.b_post))]), pmax=float(self.b_post.max()), belief=self.b_post.tolist())

    def predict(self, horizon=30, alpha2=None):
        pol = self.pol if alpha2 is None else self.m.policy(alpha2)
        return dict(policy=pol.tolist(), **self.m.rollout(self.b_post, pol, horizon))

    # ---- for an LLM agent -------------------------------------------------------------------------
    def brief(self):
        S = self.m.S
        d1, d2 = (sum(self.b_pred[i] * S[i][r] for i in range(25)) for r in range(2))
        return (f"A drone whose wing has two regions that can be damaged; the twin estimates their damage at {d1:.0f} % "
                f"and {d2:.0f} %. Every turn it flies a 2g or a 3g turn: 3g is faster, loads the wing more, and damage "
                "grows twice as often. whatif(maneuver=m, growth_factor=1): the next turn is m, \"2g\" or \"3g\"; "
                "optionally damage grows growth_factor times as often as usual from now on, a number from 0 to 3 (1: as "
                "usual). It returns value, the planner's expected reward (speed against wing health), higher is better.")

    def _values(self, growth_factor):
        if type(growth_factor) not in (int, float) or not 0 <= growth_factor <= 3:
            raise ValueError("growth_factor must be a number from 0 to 3")    # beyond, the planner's last state rules
        return self.b_pred @ self.m.values(self.alpha2, float(growth_factor)), self.m.policy(self.alpha2, float(growth_factor))

    def whatif(self, maneuver, growth_factor=1.0):
        if maneuver not in self.m.C:
            raise ValueError('maneuver must be "2g" or "3g"')
        return dict(value=round(float(self._values(growth_factor)[0][self.m.C.index(maneuver)]), 3))

    def best(self, growth_factor=1.0):
        """The maneuver most of the belief maps to under the policy for this growth: decide()'s rule."""
        value, pol = self._values(growth_factor)
        u = int(np.argmax([self.b_pred[pol == u].sum() for u in range(2)]))
        return dict(maneuver=self.m.C[u], value=round(float(value[u]), 3))
