"""Minimal-effective-dose decision for HGG_demo_001: smallest total chemo dose with day-270 TCC <= TARGET.
TCC(270) is ~linear in total dose, so simulate doses 40 and 70 and interpolate. Usage: python decide.py d k aRT aCT"""
import sys, numpy as np, torch
from datetime import timedelta
from pathlib import Path
from tumortwin.models import ReactionDiffusion3D
from tumortwin.solvers import TorchDiffEqSolver, TorchDiffEqSolverOptions
from tumortwin.preprocessing import ADC_to_cellularity, compute_carrying_capacity
from tumortwin.postprocessing import compute_total_cell_count
from tumortwin.types import ChemotherapySpecification, CropSettings, CropTarget, RadiotherapySpecification
from tumortwin.types.hgg_data import HGGPatientData
torch.set_num_threads(1)
p = [float(x) for x in sys.argv[1:5]]; TARGET = 2.5e11
INPUT = Path(__file__).parents[2] / "upstream/TumorTwin/input_files"; PID = "HGG_demo_001"
pt = HGGPatientData.from_file(INPUT/PID/f"{PID}.json", image_dir=INPUT/PID, crop_settings=CropSettings(crop_to=CropTarget.ROI_ENHANCE, padding=10, visit_index=-1))
u0 = torch.from_numpy(ADC_to_cellularity(pt.visits[0].adc_image, pt.visits[0].roi_enhance_image, pt.visits[0].roi_nonenhance_image).array)
cap = compute_carrying_capacity(pt.brainmask_image); vt = [v.time for v in pt.visits]
base = list(ChemotherapySpecification(sensitivity=0.2, decay_rate=9.2420, times=[c.time for c in pt.chemotherapy], doses=[c.dose for c in pt.chemotherapy]).doses); out = {}; print("total base dose", sum(base))
for total in [40, 70]:
    adj = np.sum(base[49:]); mult = (total - np.sum(base) + adj) / adj
    ct = ChemotherapySpecification(sensitivity=p[3], decay_rate=9.2420, times=[c.time for c in pt.chemotherapy], doses=[c.dose for c in pt.chemotherapy]); ct.doses = base[:49] + [d * mult for d in base[49:]]  # set after construction: the constructor re-normalizes by max dose
    rt = RadiotherapySpecification(alpha=p[2], alpha_beta_ratio=10, times=[r.time for r in pt.radiotherapy], doses=[r.dose for r in pt.radiotherapy])
    m = ReactionDiffusion3D(k=torch.tensor(p[1]), d=torch.tensor(p[0]), theta=torch.tensor(1.0), patient_data=pt, initial_time=vt[0], chemotherapy_specifications=[ct], radiotherapy_specification=rt)
    s = TorchDiffEqSolver(m, TorchDiffEqSolverOptions(step_size=timedelta(days=0.5), use_adjoint=False, device=torch.device("cpu"), method="rk4"))
    with torch.no_grad():
        _, maps = s.solve(timepoints=vt, u_initial=u0)
    out[total] = float(compute_total_cell_count(maps[-1], cap))
slope = (out[70] - out[40]) / 30; dose = 40 + (TARGET - out[40]) / slope
print(f"params={p} TCC270@40={out[40]:.3e} @70={out[70]:.3e} min_dose_for_target={dose:.1f}")
