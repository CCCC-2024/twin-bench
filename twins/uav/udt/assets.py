"""Physical side: the UAV (simulated; the authors also simulate it).

ScriptedUAV    the authors' default asset ('custom' damage script, independent of the maneuvers)
               with their measurement generator: used to check we reproduce their run exactly.
ControlledUAV  our closed-loop asset for the page: damage on the 20 % grid grows with the maneuver
               (each region +20 % with probability 0.05 per 2g turn, 0.10 per 3g turn: the same rates
               the twin assumes), and can be damaged on purpose. Strain = table mean + N(0, 150), >= 1,
               divided by the load factor (authors' generator, without interpolation since the state is on the grid).
"""
import numpy as np
from dtcore import Source
from digitaltwin.UAV import UAV

from .model import CFG, LOAD, SIG_ASSET


def _custom_step(x, t):                     # authors' nodes/asset.py, state_transition='custom'
    x = list(x)
    if t in [20, 30]: x[0] += 1
    if 10 <= t < 20: x[0] += 0.1
    if t in [5, 25]: x[1] += 1
    if 5 <= t < 15: x[1] += 0.1
    return x


class ScriptedUAV(Source):
    kind = "sim"

    def __init__(self):
        self.asset, self.x, self.t, self.u, self.k = UAV(config_fpath=str(CFG)), [0.0, 0.0], -1, 1, 0

    def observe(self):
        m = self.asset.measurementGenerator.getMeasurement(self.x, [self.u])
        return dict(t=self.k, m=list(m), u=self.u, true1=20 * self.x[0], true2=20 * self.x[1])

    def act(self, action):
        self.u = action["u"]

    def advance(self):
        self.t += 1; self.k += 1
        self.x = _custom_step(self.x, self.t)


class ControlledUAV(Source):
    kind = "sim"

    def __init__(self, model, seed=0, noise=None, growth=None):
        self.mod, self.g, self.u, self.k = model, [0, 0], 1, 0
        rng = np.random.default_rng(seed)
        self.noise = noise if noise is not None else rng.normal(0, SIG_ASSET, (100000, 24))
        self.growth = growth if growth is not None else rng.random((100000, 2))
        self.events = []

    def damage(self, region):
        """Damage one region on purpose (+20 %, up to 80 %). The twin is not told."""
        self.g[region] = min(4, self.g[region] + 1)

    def observe(self):
        s = self.g[0] * 5 + self.g[1]
        c = self.mod.C[self.u]
        m = np.clip(self.mod.raw[c][s] + self.noise[self.k], 1, None) / LOAD[c]
        return dict(t=self.k, m=m.tolist(), u=self.u, true1=20 * self.g[0], true2=20 * self.g[1])

    def act(self, action):
        self.u = action["u"]

    def advance(self):
        p = self.mod.p[self.mod.C[self.u]]
        for r in range(2):
            if self.growth[self.k][r] < p and self.g[r] < 4:
                self.g[r] += 1
        self.k += 1
