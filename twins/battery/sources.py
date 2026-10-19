"""Physical side of the battery twin: CellReplay, one cell's measured cycles (not the authors' extrapolation).

act({"retire_at": x}) retires the cell: the replay stops after cycle x. The recorded capacities never change.
"""
from dtcore import Source


class CellReplay(Source):
    kind = "replay"

    def __init__(self, cell):
        self.cell, self.i, self.retire_at = cell, 0, None

    def observe(self):
        c = self.cell
        if self.i >= c["n_measured"]:
            return None
        cycle = int(c["cycles"][self.i])
        if self.retire_at is not None and cycle > self.retire_at:
            return None                                   # retired: no more use, no more data
        return dict(t=cycle, cycle=cycle, q=float(c["q"][self.i]), cap_ah=float(c["cap"][self.i]))

    def act(self, action):
        self.retire_at = int(action["retire_at"])

    def advance(self):
        self.i += 1
