"""UAV digital twin on the skeleton: discrete Bayes filter over 25 damage states + the planner's policy.

update(obs)   posterior over damage from the 24 strains under the maneuver flown, then predict the next step.
decide()      the maneuver for the next turn: the control most of the predicted belief mass maps to
              under the policy (the authors' ControlP marginal argmax).
predict(h, alpha2=None)  expected outcome of following a policy for h turns (what-if on the reward weight).
"""
import numpy as np
from dtcore import Twin


class UAVTwin(Twin):
    capabilities = ("update", "predict", "whatif", "decide")

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
