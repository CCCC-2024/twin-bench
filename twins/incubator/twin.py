"""Incubator digital twin on the dtcore contract (docs/architecture.md).

update(obs)            Kalman step; returns the estimate (incl. the unmeasured heatbed temperature)
                       and an anomaly flag when measurements keep disagreeing with the model.
predict(horizon, ...)  what-if: simulate forward from the current estimate with a copy of the
                       controller, optionally with a different set point; returns T, heater, energy.
decide()               not automatic yet: the user decides with the what-if (decision support);
                       this is where an AI agent will plug in later.
"""
import numpy as np
from dtcore import Twin

from .controller import Controller
from .kalman import Kalman
from .model import PARAMS_2023, Stepper

ANOMALY_C = 0.5     # |innovation| above this (°C) ...
ANOMALY_N = 3       # ... for this many samples in a row -> anomaly


class IncubatorTwin(Twin):
    capabilities = ("update", "predict", "whatif")       # no automatic decide (yet)

    def __init__(self, p=PARAMS_2023, kalman_kw=None):
        self.p = dict(p)
        self.kf, self.kalman_kw = None, kalman_kw or {}
        self.model = Stepper(self.p)
        self.last, self.streak, self.anomaly = None, 0, False

    def update(self, obs):
        if self.kf is None:                                    # first observation: start from it
            self.kf = Kalman(self.p, [obs["T"], obs["T"]], **self.kalman_kw)
            innov, S = 0.0, self.kf.P[0, 0] + self.kf.R
        else:
            self.kf.predict(self.last["heater_on"], self.last["T_room"], obs["t"] - self.last["t"])
            innov, S = self.kf.update(obs["T"])
        self.streak = self.streak + 1 if abs(innov) > ANOMALY_C else 0
        self.anomaly = self.streak >= ANOMALY_N
        self.last = obs
        return dict(t=obs["t"], T_est=float(self.kf.x[0]), T_heater_est=float(self.kf.x[1]),
                    T_heater_sd=float(np.sqrt(self.kf.P[1, 1])), innov=innov, anomaly=self.anomaly)

    def predict(self, horizon_s, controller, desired=None, T_room=None, dt=3.0):
        """Forward simulation from the current estimate; does not change the twin or the box."""
        c = controller.copy()
        c.set(desired)
        x = self.kf.x.copy()
        Tr = self.last["T_room"] if T_room is None else T_room
        n = int(horizon_s / dt)
        T, H = np.empty(n + 1), np.zeros(n + 1)
        T[0], t = x[0], self.last["t"]
        for i in range(n):
            H[i] = c.step(t, x[0])
            x = self.model.step(x, H[i], Tr, dt)
            t += dt
            T[i + 1] = x[0]
        energy_Wh = float(H[:n].sum() * dt * self.p["V"] * self.p["I"] / 3600)
        return dict(T=T, heater=H, energy_Wh=energy_Wh)
