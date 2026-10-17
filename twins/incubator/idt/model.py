"""Incubator plant model, written from the published description.

Description: Feng et al. 2021 (arXiv 2102.10390) and the four-parameter model documented in
INTO-CPS example_digital-twin_incubator. Our own code; no upstream code is imported.

States x = [T, T_heater] (box air, heatbed). Inputs u = [heater_on (0/1), T_room].
    P_in                   = heater_on * V * I
    C_heater * dT_heater/dt = P_in - G_heater * (T_heater - T)
    C_air    * dT/dt        = G_heater * (T_heater - T) - G_box * (T - T_room)
The model is linear, so with inputs held constant over each sample (zero-order hold,
the same convention as the data logger) the discrete step below is exact.
"""
import numpy as np
from scipy.linalg import expm

# Upstream best_parameters.py, calibrated on 20230501_calibration_empty_system (2023 hardware).
PARAMS_2023 = dict(C_air=267.55929458, G_box=0.5763498, C_heater=329.25376821,
                   G_heater=1.67053237, V=12.15579391, I=1.53551347)
NAMES = ("C_air", "G_box", "C_heater", "G_heater")


def continuous(p):
    """A, B of dx/dt = A x + B u, x = [T, T_heater], u = [heater_on, T_room]."""
    Ca, Gb, Ch, Gh, P = p["C_air"], p["G_box"], p["C_heater"], p["G_heater"], p["V"] * p["I"]
    A = np.array([[-(Gh + Gb) / Ca, Gh / Ca],
                  [Gh / Ch, -Gh / Ch]])
    B = np.array([[0.0, Gb / Ca],
                  [P / Ch, 0.0]])
    return A, B


def discretize(p, dt):
    """Exact zero-order-hold discretization: x[k+1] = Ad x[k] + Bd u[k]."""
    A, B = continuous(p)
    M = np.zeros((4, 4))
    M[:2, :2], M[:2, 2:] = A, B
    E = expm(M * dt)
    return E[:2, :2], E[:2, 2:]


class Stepper:
    """Caches the discrete matrices per step length (samples are ~3 s apart, not exactly)."""

    def __init__(self, p):
        self.p, self._cache = dict(p), {}

    def mats(self, dt):
        key = round(float(dt), 3)
        if key not in self._cache:
            self._cache[key] = discretize(self.p, key)
        return self._cache[key]

    def step(self, x, heater_on, T_room, dt):
        Ad, Bd = self.mats(dt)
        return Ad @ x + Bd @ np.array([float(heater_on), float(T_room)])


def simulate(p, t, heater_on, T_room, x0):
    """Open-loop simulation along the data's time stamps. Returns X with columns [T, T_heater]."""
    st = Stepper(p)
    X = np.empty((len(t), 2))
    X[0] = x0
    for k in range(len(t) - 1):
        X[k + 1] = st.step(X[k], heater_on[k], T_room[k], t[k + 1] - t[k])
    return X
