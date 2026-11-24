"""The authors' model of one patient, reduced to what the twin needs: solve and calibrate.

A 3D reaction-diffusion model of the tumor cell density with radiotherapy and chemotherapy (tumortwin, upstream at
bedf90a), set up as in the authors' tutorials/HGG_Demo.ipynb. The density always starts from the first MRI. As in
the tutorial, the grid is cropped around the tumor seen at the last visit, so the region modeled is known from the
start; the densities of later visits are not.
"""
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
from tumortwin.models import ReactionDiffusion3D
from tumortwin.optimizers import LMoptimizer, LMoptions
from tumortwin.postprocessing import compute_total_cell_count
from tumortwin.preprocessing import ADC_to_cellularity, compute_carrying_capacity
from tumortwin.solvers import TorchDiffEqSolver, TorchDiffEqSolverOptions
from tumortwin.types import ChemotherapySpecification, CropSettings, CropTarget, RadiotherapySpecification
from tumortwin.types.hgg_data import HGGPatientData

INPUT = Path(__file__).resolve().parent / "upstream" / "TumorTwin" / "input_files"
PARAMS = ("d", "k", "alpha_RT", "chemo_sensitivity")       # diffusion, growth, radiotherapy and chemotherapy effect
START = (0.025, 0.05, 0.025, 0.5)                           # the tutorial's initial guess
BOUNDS = ((0.0, 2), (0.0, 0.5), (0.001, 0.1), (0.0, 1.0))


class Glioma:
    def __init__(self, patient_id="HGG_demo_001"):
        folder = INPUT / patient_id
        crop = CropSettings(crop_to=CropTarget.ROI_ENHANCE, padding=10, visit_index=-1)
        patient = HGGPatientData.from_file(folder / f"{patient_id}.json", image_dir=folder, crop_settings=crop)
        self.capacity = compute_carrying_capacity(patient.brainmask_image)
        self.dates = [v.time for v in patient.visits]
        self.maps = [ADC_to_cellularity(v.adc_image, v.roi_enhance_image, v.roi_nonenhance_image).array
                     for v in patient.visits]                # the tumor cell density measured at each visit
        rt = RadiotherapySpecification(alpha=0.025, alpha_beta_ratio=10, times=[r.time for r in patient.radiotherapy],
                                       doses=[r.dose for r in patient.radiotherapy])
        self.chemo = ChemotherapySpecification(sensitivity=0.5, decay_rate=9.2420,
                                               times=[c.time for c in patient.chemotherapy],
                                               doses=[c.dose for c in patient.chemotherapy])
        self.doses = list(self.chemo.doses)                  # as planned; the constructor has rescaled them
        gaps = [b - a > timedelta(days=1) for a, b in zip(self.chemo.times, self.chemo.times[1:])]
        self.adjuvant = gaps.index(True) + 1                 # the doses after the first break in the daily course
        model = ReactionDiffusion3D(k=torch.tensor(0.05), d=torch.tensor(0.025), theta=torch.tensor(1.0),
                                    patient_data=patient, initial_time=self.dates[0],
                                    chemotherapy_specifications=[self.chemo], radiotherapy_specification=rt)
        options = TorchDiffEqSolverOptions(step_size=timedelta(days=0.5), use_adjoint=True,
                                           device=torch.device("cpu"), method="rk4")
        self.solver = TorchDiffEqSolver(model, options)

    def day(self, date):
        return (date - self.dates[0]) / timedelta(days=1)

    def cell_count(self, density):
        return float(compute_total_cell_count(density, self.capacity))

    def _set(self, params):
        d, k, alpha, sensitivity = torch.nn.Parameter(torch.as_tensor(params, dtype=torch.float32))
        m = self.solver.model
        m.d, m.k = torch.nn.Parameter(d), torch.nn.Parameter(k)
        m.radiotherapy_specification.alpha = torch.nn.Parameter(alpha)
        m.chemotherapy_specifications[0].sensitivity = torch.nn.Parameter(sensitivity)

    def _solve(self, params, days):
        self._set(params)
        dates = [self.dates[0] + timedelta(days=float(x)) for x in days]
        return self.solver.solve(timepoints=dates, u_initial=torch.from_numpy(self.maps[0]))[1]

    def solve(self, params, days, chemo_total=None):
        """The density at each of `days` (from the first MRI; the first must be 0) under the parameters; chemo_total:
        what if the adjuvant doses are scaled so that the whole course sums to this (the tutorial's what-if)."""
        if chemo_total is not None:
            rest = float(np.sum(self.doses[self.adjuvant:]))
            scale = (chemo_total - float(np.sum(self.doses)) + rest) / rest
            if scale < 0:
                raise ValueError(f"chemo_total must be at least {float(np.sum(self.doses)) - rest:g}")
            self.chemo.doses = self.doses[:self.adjuvant] + [x * scale for x in self.doses[self.adjuvant:]]
        try:
            with torch.no_grad():
                return self._solve(params, days)
        finally:
            self.chemo.doses = self.doses

    def calibrate(self, start, n_visits, steps):
        """Levenberg-Marquardt from `start` on the first n_visits MRIs, `steps` steps: the parameters."""
        days = [self.day(t) for t in self.dates[:n_visits]]
        measured = torch.stack([torch.from_numpy(m) for m in self.maps[:n_visits]])
        optimizer = LMoptimizer(model=lambda p: self._solve(p, days), y_data=measured, options=LMoptions(),
                                initial_guess=torch.tensor(start, dtype=torch.float32), bounds=torch.tensor(BOUNDS))
        for _ in range(steps):
            optimizer.step()
        return tuple(float(x) for x in optimizer.parameters[-1].detach())
