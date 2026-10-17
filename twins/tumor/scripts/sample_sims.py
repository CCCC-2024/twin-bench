"""Generate training data for the AI surrogate: run the TumorTwin simulator many times.

Inputs per run (Latin hypercube over plausible ranges):
  d (diffusion), k (proliferation), alpha_RT, alpha_CT (chemo sensitivity), adj_mult (adjuvant chemo dose multiplier)
Output per run: total tumor cell count (TCC) every 10 days from day 0 to day 270 (28 values).
Same patient, same initial condition (visit-1 cellularity) and same solver settings as the reproduction.

Usage:
  python scripts/sample_sims.py --n 256 --seed 0                    # all runs in one process
  python scripts/sample_sims.py --n 256 --seed 0 --start 0 --end 8  # one chunk (for a SLURM job array)
  python scripts/sample_sims.py --check                             # true parameters once; must match the data
  python scripts/sample_sims.py --truth                             # true parameters at each dose multiplier
  add --device cuda to run on a GPU (untested so far), --patient HGG_demo_002 for another virtual patient
Output: results/<patient>/sims_<tag>.npz   (merge chunks with scripts/merge_sims.py)
"""
import argparse
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
from scipy.stats import qmc

from tumortwin.models import ReactionDiffusion3D
from tumortwin.postprocessing import compute_total_cell_count
from tumortwin.preprocessing import ADC_to_cellularity, compute_carrying_capacity
from tumortwin.solvers import TorchDiffEqSolver, TorchDiffEqSolverOptions
from tumortwin.types import ChemotherapySpecification, CropSettings, CropTarget, RadiotherapySpecification
from tumortwin.types.hgg_data import HGGPatientData
from tumortwin.utils import daterange

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=256)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--start", type=int, default=0)
ap.add_argument("--end", type=int, default=None)
ap.add_argument("--device", default="cpu")
ap.add_argument("--threads", type=int, default=2)
ap.add_argument("--patient", default="HGG_demo_001")
ap.add_argument("--check", action="store_true")
ap.add_argument("--truth", action="store_true")
a = ap.parse_args()

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "upstream" / "TumorTwin" / "input_files"
OUTDIR = ROOT / "results" / a.patient
OUTDIR.mkdir(parents=True, exist_ok=True)

# parameter ranges: (d, k, alpha_RT, alpha_CT, adj_mult)
LO = np.array([0.02, 0.02, 0.01, 0.05, 0.25])
HI = np.array([0.20, 0.10, 0.10, 0.50, 2.00])
TRUE = [0.1, 0.05, 0.05, 0.2]   # ground truth of the demo patient, as stated in the tutorial

if a.check:
    X, tag = np.array([TRUE + [1.0]]), "check"
elif a.truth:
    X, tag = np.array([TRUE + [m] for m in np.arange(0.25, 2.001, 0.25)]), "truth"
else:
    X = LO + qmc.LatinHypercube(d=5, seed=a.seed).random(a.n) * (HI - LO)   # same design for every chunk
    end = a.n if a.end is None else min(a.end, a.n)
    X = X[a.start:end]
    tag = f"{a.seed}" if (a.start == 0 and end == a.n) else f"{a.seed}_{a.start}-{end}"
OUT = OUTDIR / f"sims_{tag}.npz"

torch.set_num_threads(a.threads)
device = torch.device(a.device)
patient = HGGPatientData.from_file(INPUT / a.patient / f"{a.patient}.json", image_dir=INPUT / a.patient,
                                   crop_settings=CropSettings(crop_to=CropTarget.ROI_ENHANCE, padding=10, visit_index=-1))
v0 = patient.visits[0]
meas0 = ADC_to_cellularity(v0.adc_image, v0.roi_enhance_image, v0.roi_nonenhance_image)
cap = compute_carrying_capacity(patient.brainmask_image)
t0 = v0.time
rt = RadiotherapySpecification(alpha=0.025, alpha_beta_ratio=10, times=[r.time for r in patient.radiotherapy],
                               doses=[r.dose for r in patient.radiotherapy])
ct = ChemotherapySpecification(sensitivity=0.5, decay_rate=9.2420, times=[c.time for c in patient.chemotherapy],
                               doses=[c.dose for c in patient.chemotherapy])
base_doses = list(ct.doses)
N_CONC = 49   # first 49 doses = concurrent chemo; the rest = adjuvant (HGG demo protocol)
model = ReactionDiffusion3D(k=torch.tensor(0.05), d=torch.tensor(0.025), theta=torch.tensor(1.0), patient_data=patient,
                            initial_time=t0, chemotherapy_specifications=[ct], radiotherapy_specification=rt,
                            require_grad=False, device=device)
solver = TorchDiffEqSolver(model, TorchDiffEqSolverOptions(step_size=timedelta(days=0.5), use_adjoint=False,
                                                           device=device, method="rk4"))
u0 = torch.from_numpy(meas0.array).to(device)
tps = list(daterange(t0, t0 + timedelta(days=270), timedelta(days=10)))
days = np.arange(0, 271, 10)


def simulate(x):
    d, k, a_rt, a_ct, mult = [float(v) for v in x]
    solver.model.d = torch.nn.Parameter(torch.tensor(d, device=device), requires_grad=False)
    solver.model.k = torch.nn.Parameter(torch.tensor(k, device=device), requires_grad=False)
    solver.model.radiotherapy_specification.alpha = a_rt
    ct.doses = base_doses[:N_CONC] + [v * mult for v in base_doses[N_CONC:]]
    ct.sensitivity = torch.nn.Parameter(torch.tensor(a_ct), requires_grad=False)
    solver.model.chemotherapy_specifications = [ct]
    with torch.no_grad():
        _, maps = solver.solve(timepoints=tps, u_initial=u0)
    return np.array([float(compute_total_cell_count(m.cpu() if torch.is_tensor(m) else m, cap)) for m in maps])


Y = np.full((len(X), len(days)), np.nan)
T0 = time.time()
for i, x in enumerate(X):
    Y[i] = simulate(x)
    if i == 0 or (i + 1) % 10 == 0 or i == len(X) - 1:
        np.savez(OUT, X=X[: i + 1], Y=Y[: i + 1], days=days, lo=LO, hi=HI)
        print(f"[{time.time() - T0:7.1f}s] {i + 1}/{len(X)} done -> {OUT.name}", flush=True)
