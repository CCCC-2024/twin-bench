"""Digital side of the mug: MugTwin."""
import math

from dtcore import Twin

from .model import TAU, step


class MugTwin(Twin):
    """Scalar Kalman filter on the model (update), model roll-out (predict) and a heater rule (decide)."""

    capabilities = ("update", "predict", "whatif", "decide")

    def __init__(self, T_room=20.0, dt=10.0, target=60.0, lookahead=10.0, q=0.01, r=0.09):
        self.T_room, self.dt, self.target, self.lookahead, self.q, self.r = T_room, dt, target, lookahead, q, r
        self.T, self.P = 50.0, 100.0          # vague prior: we do not know how hot the mug starts
        self.heater_on, self.started = 0, False

    def update(self, obs):
        self.heater_on = int(obs["heater_on"])  # the heater state during the interval that just ended
        if self.started:                      # carry the estimate forward over that interval
            a = math.exp(-self.dt / TAU)
            self.T = step(self.T, self.T_room, self.heater_on, self.dt)
            self.P = a * a * self.P + self.q
        self.started = True
        K = self.P / (self.P + self.r)
        innov = obs["T"] - self.T
        self.T += K * innov
        self.P *= 1.0 - K
        return {"t": obs["t"], "T_est": self.T, "T_sd": math.sqrt(self.P), "innov": innov}

    def predict(self, horizon, heater_on=None):
        """Temperatures every dt for `horizon` seconds; heater_on given = what-if, else keep the current state."""
        u = self.heater_on if heater_on is None else int(heater_on)
        T, out = self.T, [self.T]
        for _ in range(int(round(horizon / self.dt))):
            T = step(T, self.T_room, u, self.dt)
            out.append(T)
        return {"T": out}

    def decide(self):
        """Heater on when the mug would fall below the target within the look-ahead, off otherwise."""
        will_be = self.predict(self.lookahead, heater_on=0)["T"][-1]
        return {"heater_on": int(will_be < self.target)}
