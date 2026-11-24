"""Physical side of the tumor twin: PatientReplay, one patient's MRI visits in order."""
from dtcore import Source


class PatientReplay(Source):
    """Each observation is one visit: the day since the first MRI, its number, and the tumor cell count measured
    from the MRI. The 3D density itself stays in the Glioma, where the twin reads visit `visit`."""

    def __init__(self, glioma):
        self.glioma, self.n = glioma, 0

    def observe(self):
        g = self.glioma
        if self.n >= len(g.dates):
            return None
        return dict(t=g.day(g.dates[self.n]), visit=self.n, cell_count=g.cell_count(g.maps[self.n]))

    def advance(self):
        self.n += 1
