"""Twin update (physical -> digital): Kalman filter on the linear incubator model.

State x = [T, T_heater]; only T (mean of the two box sensors) is measured.
Each sample: predict with the inputs applied since the last sample, then correct with the new T.
The innovation (measured - predicted T) is what the anomaly check looks at.
"""
import numpy as np

from .model import Stepper


class Kalman:
    def __init__(self, p, x0, P0=(0.5, 5.0), q=(0.002, 0.002), r=0.05):
        """P0: initial std of [T, T_heater] (°C); q: process noise std per sqrt(s); r: measurement std (°C)."""
        self.st = Stepper(p)
        self.x = np.array(x0, dtype=float)
        self.P = np.diag(np.square(P0)).astype(float)
        self.Qrate = np.diag(np.square(q))
        self.R = r ** 2

    def predict(self, heater_on, T_room, dt):
        Ad, Bd = self.st.mats(dt)
        self.x = Ad @ self.x + Bd @ np.array([float(heater_on), float(T_room)])
        self.P = Ad @ self.P @ Ad.T + self.Qrate * dt

    def update(self, y):
        innov = y - self.x[0]
        S = self.P[0, 0] + self.R
        K = self.P[:, 0] / S
        self.x = self.x + K * innov
        self.P = self.P - np.outer(K, self.P[0, :])
        return float(innov), float(S)
