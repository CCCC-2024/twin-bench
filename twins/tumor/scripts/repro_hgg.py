"""Reproduce TumorTwin's high-grade glioma (HGG) demo end to end and save the numbers.

Same steps and settings as upstream tutorials/HGG_Demo.ipynb (commit bedf90a):
load patient -> cellularity from ADC -> reaction-diffusion model -> forward solve ->
calibrate (d, k, alpha_RT, chemo sensitivity) on visits 1-5 with Levenberg-Marquardt ->
predict all visits -> what-if over adjuvant chemo total dose.
Usage: python scripts/repro_hgg.py [patient_id]   (default HGG_demo_001)
"""
import json
import sys
import time
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
from tumortwin.utils import daterange, days_since_first

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "upstream" / "TumorTwin" / "input_files"
PID = sys.argv[1] if len(sys.argv) > 1 else "HGG_demo_001"
OUT = ROOT / "results" / PID
OUT.mkdir(parents=True, exist_ok=True)
T0 = time.time()
log = lambda *a: print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)  # noqa: E731

device = torch.device("cpu")
patient = HGGPatientData.from_file(INPUT / PID / f"{PID}.json", image_dir=INPUT / PID,
                                   crop_settings=CropSettings(crop_to=CropTarget.ROI_ENHANCE, padding=10, visit_index=-1))
meas = [ADC_to_cellularity(v.adc_image, v.roi_enhance_image, v.roi_nonenhance_image) for v in patient.visits]
cap = compute_carrying_capacity(patient.brainmask_image)
visit_times = [v.time for v in patient.visits]
log(f"{PID}: {len(visit_times)} visits, grid {meas[0].array.shape}, carrying capacity {cap:.0f}")

rt = RadiotherapySpecification(alpha=0.025, alpha_beta_ratio=10,
                               times=[r.time for r in patient.radiotherapy], doses=[r.dose for r in patient.radiotherapy])
ct = ChemotherapySpecification(sensitivity=0.5, decay_rate=9.2420,
                               times=[c.time for c in patient.chemotherapy], doses=[c.dose for c in patient.chemotherapy])
model = ReactionDiffusion3D(k=torch.tensor(0.05), d=torch.tensor(0.025), theta=torch.tensor(1.0),
                            patient_data=patient, initial_time=visit_times[0],
                            chemotherapy_specifications=[ct], radiotherapy_specification=rt)
solver = TorchDiffEqSolver(model, TorchDiffEqSolverOptions(step_size=timedelta(days=0.5), use_adjoint=True,
                                                           device=device, method="rk4"))
u0 = torch.from_numpy(meas[0].array)
tcc = lambda maps: [float(compute_total_cell_count(m, cap)) for m in maps]  # noqa: E731


def set_params(p):
    d, k, alpha, ct_sens = torch.nn.Parameter(p)   # order used by the tutorial: (d, k, alpha_RT, chemo sensitivity)
    solver.model.d = torch.nn.Parameter(d)
    solver.model.k = torch.nn.Parameter(k)
    solver.model.radiotherapy_specification.alpha = torch.nn.Parameter(alpha)
    solver.model.chemotherapy_specifications[0].sensitivity = torch.nn.Parameter(ct_sens)


def predict(p, tps):
    set_params(p)
    with torch.no_grad():
        _, maps = solver.solve(timepoints=tps, u_initial=u0)
    return maps


N_CAL = 5
cal_t = visit_times[:N_CAL]
y = torch.stack([torch.from_numpy(m.array) for m in meas[:N_CAL]])
init = torch.tensor((0.025, 0.05, 0.025, 0.5))
optim = LMoptimizer(model=lambda p: (set_params(p), solver.solve(timepoints=cal_t, u_initial=u0)[1])[1],
                    initial_guess=init, bounds=torch.tensor(((0.0, 2), (0.0, 0.5), (0.001, 0.1), (0.0, 1.0))),
                    y_data=y, options=LMoptions())
for i in range(20):
    optim.step()
    log(f"LM step {i + 1}/20 params {[round(float(v), 4) for v in optim.parameters[-1]]}")
best = optim.parameters[-1].detach()
truth = torch.tensor((0.1, 0.05, 0.05, 0.2))   # ground truth stated in the tutorial, same (d, k, alpha, ct) order

full_t = visit_times
pred_init = tcc(predict(init, full_t))
pred_best = tcc(predict(best, full_t))
log("predictions done")

# What-if: total chemo dose, scaling only the adjuvant doses (index 49 onward), as in the tutorial
base_doses = list(ct.doses)
whatif = {}
fine_t = list(daterange(visit_times[0], visit_times[-1], timedelta(days=5)))
for total in [30, 40, 50, 60, 70]:
    adj = np.sum(base_doses[49:])
    mult = (total - np.sum(base_doses) + adj) / adj
    ct.doses = base_doses[:49] + [d * mult for d in base_doses[49:]]
    solver.model.chemotherapy_specifications = [ct]
    whatif[total] = tcc(predict(best, fine_t))
    log(f"what-if total dose {total}: final TCC {whatif[total][-1]:.3e}")
ct.doses = base_doses

res = dict(patient=PID, n_calibration_visits=N_CAL,
           visit_days=[float(days_since_first(t, visit_times[0])) for t in visit_times],
           measured_tcc=tcc([m.array for m in meas]),
           init_params=init.tolist(), calibrated_params=best.tolist(), true_params=truth.tolist(),
           param_order=["d", "k", "alpha_RT", "chemo_sensitivity"],
           rel_error_pct=(100 * (best - truth).abs() / truth).tolist(),
           lm_loss=[float(e) for e in optim.error],
           pred_tcc_initial_guess=pred_init, pred_tcc_calibrated=pred_best,
           whatif_days=[float(days_since_first(t, visit_times[0])) for t in fine_t],
           whatif_tcc={str(k): v for k, v in whatif.items()},
           runtime_s=time.time() - T0)
(OUT / "repro.json").write_text(json.dumps(res, indent=1))
log("saved", OUT / "repro.json")
