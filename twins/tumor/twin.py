"""Tumor digital twin on the dtcore contract (docs/architecture.md).

D (digital state)  the four parameters of the patient's reaction-diffusion model (model.PARAMS), a point estimate.
update(obs)        one MRI visit: first the count the twin expected for this visit with the parameters it had, then
                   the parameters calibrated again on every MRI so far, starting from the last ones. The observation
                   names the visit; its 3D density is read from the Glioma, which holds the patient's images.
predict(h, ...)    the tumor cell count every 5 days up to h days after the last visit; chemo_total makes it a
                   what-if: the adjuvant chemotherapy scaled so that the whole course sums to that dose.
whatif()           the what-if as one number.
No decide(): the model has no cost of a dose (toxicity), so more chemotherapy is always better and there is nothing
to choose. That is why the twin has no actions, objective or best() either.
"""
import numpy as np

from dtcore import Twin

from .model import PARAMS, START


class TumorTwin(Twin):
    capabilities = ("update", "predict", "whatif")

    def __init__(self, glioma, steps=20):
        """steps: Levenberg-Marquardt steps at each visit, from the last parameters (the tutorial's one calibration
        takes 20)."""
        self.g, self.steps = glioma, steps
        self.params, self.visits, self.day = START, 0, 0.0

    def update(self, obs):
        self.visits, self.day = obs["visit"] + 1, obs["t"]
        expected = None
        if self.visits > 1:
            expected = self.g.cell_count(self.g.solve(self.params, [0.0, self.day])[-1])
            self.params = self.g.calibrate(self.params, self.visits, self.steps)
        return dict(t=obs["t"], expected_cell_count=expected, **dict(zip(PARAMS, self.params)))

    def predict(self, horizon, chemo_total=None):
        if horizon < 0:
            raise ValueError("horizon must be 0 or more days")
        end = self.day + horizon
        days = sorted({*np.arange(0.0, end, 5.0).tolist(), end, 0.0})
        if end == 0:                                    # only the first MRI so far: nothing to solve
            return dict(day=[0.0], cell_count=[self.g.cell_count(self.g.maps[0])])
        maps = self.g.solve(self.params, days, chemo_total)
        return dict(day=days, cell_count=[self.g.cell_count(m) for m in maps])

    def whatif(self, horizon, chemo_total=None):
        if type(horizon) not in (int, float) or horizon <= 0:
            raise ValueError("horizon must be a number of days above 0")
        if chemo_total is not None and type(chemo_total) not in (int, float):
            raise ValueError("chemo_total must be a number")
        maps = self.g.solve(self.params, [0.0, self.day + horizon], chemo_total)
        return dict(cell_count=self.g.cell_count(maps[-1]))
