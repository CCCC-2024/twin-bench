"""Physical side of the mug: SimMug, a simulated mug the twin's heater decisions act on."""
import numpy as np

from dtcore import Source

from .model import step


class SimMug(Source):
    """A simulated mug: the true temperature follows the model; the sensor adds Gaussian noise.

    Observations also carry the true temperature ("true_T") so checks can score the twin; the twin never reads it.
    """

    kind = "sim"

    def __init__(self, T0=80.0, T_room=20.0, dt=10.0, noise_sd=0.3, n_steps=360, seed=0):
        self.T, self.T_room, self.dt, self.noise_sd, self.n_steps = T0, T_room, dt, noise_sd, n_steps
        self.rng = np.random.default_rng(seed)
        self.k, self.heater_on = 0, 0

    def observe(self):
        if self.k >= self.n_steps:
            return None
        return {"t": self.k * self.dt, "T": self.T + self.rng.normal(0.0, self.noise_sd),
                "heater_on": self.heater_on, "true_T": self.T}

    def act(self, action):
        self.heater_on = int(action["heater_on"])

    def advance(self):
        self.T = step(self.T, self.T_room, self.heater_on, self.dt)
        self.k += 1
