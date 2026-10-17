"""Physical side for closed-loop runs: a simulated box (the real box is in Aarhus).

True dynamics = the same four-parameter model with the published parameters, plus:
  - lid: when open, heat loss to the room uses G_open_lid instead of G_box
    (14.39 W/K, upstream's calibration from its 2021 lid-opening experiment, older hardware;
    used here only to simulate the effect of opening the lid);
  - sensors: two box sensors (offsets +-0.5 °C, as the real t1/t2 differ by about 1 °C) and one room
    sensor, each with Gaussian noise, quantized to 0.0625 °C like the real DS18 sensors.
The controller runs here (on the box), on the measured box temperature, every 3 s.
noise can be passed in (n x 3 array) so other implementations can be checked step by step.
"""
import numpy as np
from dtcore import Source

from .controller import Controller
from .model import PARAMS_2023, Stepper

G_OPEN_LID = 14.3873949
DT = 3.0
Q = 0.0625


def quantize(v):
    return np.round(v / Q) * Q


class SimBox(Source):
    kind = "sim"

    def __init__(self, p=PARAMS_2023, T0=22.0, T_room=22.0, controller=None, sensor_sd=0.05, seed=0, noise=None):
        self.p = dict(p)
        self.st_closed, self.st_open = Stepper(self.p), Stepper(dict(self.p, G_box=G_OPEN_LID))
        self.x = np.array([T0, T0], dtype=float)
        self.T_room, self.lid_open, self.t, self.k = float(T_room), False, 0.0, 0
        self.ctrl = controller or Controller()
        self.noise = noise if noise is not None else np.random.default_rng(seed).normal(0, sensor_sd, (10 ** 6, 3))
        self.heater_on = False

    def observe(self):
        """Read sensors, let the controller decide the heater for the next 3 s, return the observation."""
        n = self.noise[self.k]
        t1, t2 = quantize(self.x[0] + 0.5 + n[0]), quantize(self.x[0] - 0.5 + n[1])
        t3 = quantize(self.T_room + n[2])
        T_meas = (t1 + t2) / 2
        self.heater_on = self.ctrl.step(self.t, T_meas)
        return dict(t=self.t, t1=t1, t2=t2, t3=t3, T=T_meas, T_room=t3, heater_on=float(self.heater_on),
                    true_T=float(self.x[0]), true_T_heater=float(self.x[1]), lid_open=self.lid_open)

    def act(self, action):
        """Digital -> physical: new controller settings, e.g. {"desired": 35.0}."""
        self.ctrl.set(**action)

    def advance(self, dt=DT):
        st = self.st_open if self.lid_open else self.st_closed
        self.x = st.step(self.x, self.heater_on, self.T_room, dt)
        self.t += dt
        self.k += 1
