"""Generate virtual patients with TumorTwin's HGG model: sample (d,k,aRT,aCT), simulate from HGG_demo_001's day-0 map,
save per-visit summary features. Usage: python gen.py N seed outfile"""
import sys, time, numpy as np, torch
from datetime import timedelta
from pathlib import Path
from tumortwin.models import ReactionDiffusion3D
from tumortwin.solvers import TorchDiffEqSolver, TorchDiffEqSolverOptions
from tumortwin.preprocessing import ADC_to_cellularity, compute_carrying_capacity
from tumortwin.types import ChemotherapySpecification, CropSettings, CropTarget, RadiotherapySpecification
from tumortwin.types.hgg_data import HGGPatientData
torch.set_num_threads(1)
N, seed, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
INPUT = Path(sys.argv[4]) if len(sys.argv) > 4 else Path(__file__).parents[2] / "upstream/TumorTwin/input_files"
PID = "HGG_demo_001"
patient = HGGPatientData.from_file(INPUT/PID/f"{PID}.json", image_dir=INPUT/PID,
          crop_settings=CropSettings(crop_to=CropTarget.ROI_ENHANCE, padding=10, visit_index=-1))
meas = [ADC_to_cellularity(v.adc_image, v.roi_enhance_image, v.roi_nonenhance_image).array for v in patient.visits]
cap = compute_carrying_capacity(patient.brainmask_image)
vt = [v.time for v in patient.visits]
rt = RadiotherapySpecification(alpha=0.05, alpha_beta_ratio=10, times=[r.time for r in patient.radiotherapy], doses=[r.dose for r in patient.radiotherapy])
ct = ChemotherapySpecification(sensitivity=0.2, decay_rate=9.2420, times=[c.time for c in patient.chemotherapy], doses=[c.dose for c in patient.chemotherapy])
model = ReactionDiffusion3D(k=torch.tensor(0.05), d=torch.tensor(0.1), theta=torch.tensor(1.0), patient_data=patient,
        initial_time=vt[0], chemotherapy_specifications=[ct], radiotherapy_specification=rt)
solver = TorchDiffEqSolver(model, TorchDiffEqSolverOptions(step_size=timedelta(days=0.5), use_adjoint=False, device=torch.device("cpu"), method="rk4"))
u0 = torch.from_numpy(meas[0])
idx = np.indices(meas[0].shape).reshape(3, -1).T * np.array([1.5, 1.5, 3.0])
def feats(m):
    m = np.asarray(m); f = m.ravel(); w = f.sum()
    c = (idx * f[:, None]).sum(0) / max(w, 1e-9)
    rg = np.sqrt((((idx - c) ** 2).sum(1) * f).sum() / max(w, 1e-9))
    return [w * cap, (f > 0.05).sum(), (f > 0.3).sum(), (f > 0.6).sum(), rg]
def run(p):
    solver.model.d = torch.nn.Parameter(torch.tensor(p[0])); solver.model.k = torch.nn.Parameter(torch.tensor(p[1]))
    solver.model.radiotherapy_specification.alpha = torch.nn.Parameter(torch.tensor(p[2]))
    solver.model.chemotherapy_specifications[0].sensitivity = torch.nn.Parameter(torch.tensor(p[3]))
    with torch.no_grad():
        _, maps = solver.solve(timepoints=vt, u_initial=u0)
    return [feats(m.detach().numpy() if hasattr(m, "detach") else m) for m in maps]
rng = np.random.default_rng(seed)
lo, hi = np.log([0.02, 0.02, 0.01, 0.05]), np.log([0.5, 0.1, 0.09, 0.6])
P, F = [], []
if seed == 0:  # first row: the measured data of HGG_demo_001 (truth 0.1,0.05,0.05,0.2)
    P.append([0.1, 0.05, 0.05, 0.2]); F.append([feats(m) for m in meas])
t0 = time.time()
for i in range(N):
    p = np.exp(rng.uniform(lo, hi)).tolist()
    P.append(p); F.append(run(p))
    print(f"{i+1}/{N} {time.time()-t0:.0f}s p={np.round(p,4).tolist()} finalTCC={F[-1][-1][0]:.2e}", flush=True)
    np.savez(out, P=np.array(P), F=np.array(F))
