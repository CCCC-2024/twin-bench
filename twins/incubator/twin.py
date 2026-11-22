"""Incubator digital twin on the dtcore contract (docs/architecture.md).

update(obs)            Kalman step; returns the estimate (incl. the unmeasured heatbed temperature)
                       and an anomaly flag when measurements keep disagreeing with the model.
predict(horizon, ...)  what-if: simulate forward from the current estimate with a copy of the
                       controller, optionally with a different set point; returns T, heater, energy.
decide()               not automatic: the user decides with the what-if (decision support).
brief(), whatif(), best()  for an LLM agent (dtcore.agent), once the twin has its copy of the box's controller and
                       the band the air must stay in: the set point's what-if as a few numbers, and the set point
                       that keeps the air in the band longest over the next hour (ties: least energy).
"""
import numpy as np
from dtcore import Twin

from .controller import Controller
from .kalman import Kalman
from .model import PARAMS_2023, Stepper

ANOMALY_C = 0.5     # |innovation| above this (°C) ...
ANOMALY_N = 3       # ... for this many samples in a row -> anomaly


class IncubatorTwin(Twin):
    capabilities = ("update", "predict", "whatif")       # no automatic decide
    actions = ("desired",)
    objective = ("minutes_outside", +1)
    conditions = dict(T_room=dict(type="number", minimum=-20, maximum=50,
                                  description="temperature of the room around the box for the next hour, in °C"))

    def __init__(self, p=PARAMS_2023, kalman_kw=None, controller=None, band=None):
        """controller: the twin's copy of the box's controller (kept in step by the caller); band: (low, high) °C the
        air should stay in. Both are needed only for brief / whatif / best."""
        self.controller, self.band = controller, band
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

    # ---- for an LLM agent -------------------------------------------------------------------------
    def brief(self):
        lo, hi = self.band
        return (f"An incubator: a box whose air a heater keeps warm. The air is at {self.kf.x[0]:.1f} °C, the room at "
                f"{self.last['T_room']:.1f} °C, the set point is {self.controller.desired:.2f} °C; the air should stay "
                f"between {lo:g} and {hi:g} °C. whatif(desired=d, T_room=r): the set point becomes d °C now, from "
                f"{lo - 1:g} to {hi + 1:g} in steps of 0.25; optionally the room is at r °C for the next hour, from "
                "-20 to 50. It returns minutes_outside (minutes of the next 60 with the air outside the band), "
                "energy_Wh, and the lowest and highest air temperature.")

    def whatif(self, desired, T_room=None):
        lo, hi = self.band
        if type(desired) not in (int, float) or not lo - 1 <= desired <= hi + 1 or (desired * 4) % 1:
            raise ValueError(f"desired must be from {lo - 1:g} to {hi + 1:g} in steps of 0.25")
        if T_room is not None and (type(T_room) not in (int, float) or not -20 <= T_room <= 50):
            raise ValueError("T_room must be a number from -20 to 50")
        p = self.predict(3600, self.controller, desired=desired, T_room=T_room)
        outside = float(((p["T"] < lo) | (p["T"] > hi)).mean() * 60)
        return dict(minutes_outside=round(outside, 1), energy_Wh=round(p["energy_Wh"], 1),
                    T_min=round(float(p["T"].min()), 1), T_max=round(float(p["T"].max()), 1))

    def best(self, T_room=None):
        lo, hi = self.band
        tried = [(d, self.whatif(d, T_room)) for d in np.arange(lo - 1, hi + 1.01, 0.25).tolist()]
        d, w = min(tried, key=lambda t: (t[1]["minutes_outside"], t[1]["energy_Wh"]))
        return dict(desired=d, minutes_outside=w["minutes_outside"], energy_Wh=w["energy_Wh"])
