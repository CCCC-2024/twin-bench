"""Physical side for replay: the real 2023 experiment, one sample at a time. It cannot obey actions."""
from dtcore import Source

from .data import load


class RecordedBox(Source):
    kind = "replay"

    def __init__(self, name="calibration_2023"):
        self.d, self.i, self.received = load(name), 0, []

    def observe(self):
        if self.i >= len(self.d):
            return None
        r = self.d.iloc[self.i]
        return dict(t=float(r.t), T=float(r["T"]), T_room=float(r.T_room), heater_on=float(r.heater_on),
                    t1=float(r.t1), t2=float(r.t2))

    def act(self, action):
        self.received.append(action)        # recorded only: the experiment already happened

    def advance(self):
        self.i += 1
