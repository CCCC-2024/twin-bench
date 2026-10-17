"""AI surrogate for the TumorTwin simulator, and what it makes possible.

1. Train a Gaussian-process surrogate: (d, k, alpha_RT, alpha_CT, adjuvant dose multiplier) -> log total tumor
   cells every 10 days (days 0-270). Check it on held-out simulations.
2. Calibrate with the surrogate instead of the simulator (fit to measured tumor totals at the first n MRI visits;
   the original LM fits every voxel of the maps, so this twin sees less information),
   compare time and accuracy with the original voxel-level Levenberg-Marquardt calibration.
3. Data experiment the original twin is too slow for: noisy tumor totals, n = 2..5 pre-adjuvant visits (days 0-120;
   n < 4 gives fewer data points than the 4 parameters, so the fit there depends on the starting guess),
   200 noise draws each -> error in predicted day-270 tumor burden and in a simple dose choice.
Usage: python scripts/surrogate_experiments.py   -> results/HGG_demo_001/surrogate.json
"""
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results" / "HGG_demo_001"
z = np.load(R / "sims_0.npz")
X, Y, days, LO, HI = z["X"], z["Y"], z["days"], z["lo"], z["hi"]
ok = np.all(np.isfinite(Y), axis=1) & np.all(Y > 0, axis=1)
X, Y = X[ok], Y[ok]
repro = json.loads((R / "repro.json").read_text())
truth = np.array(repro["true_params"])            # (d, k, alpha_RT, alpha_CT)
measured = np.array(repro["measured_tcc"])        # visits at days 0, 30, ..., 270
visit_idx = np.arange(0, 28, 3)                   # positions of visit days in the 10-day grid
rng = np.random.default_rng(0)

scale = lambda x: (x - LO) / (HI - LO)            # noqa: E731
logY = np.log(Y)
mu, sd = logY.mean(0), logY.std(0)

# ---- 1. surrogate + held-out check
perm = rng.permutation(len(X))
n_test = max(20, len(X) // 5)
te, tr = perm[:n_test], perm[n_test:]
kernel = ConstantKernel(1.0) * RBF(length_scale=np.ones(5), length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(1e-4, (1e-8, 1e-1))
t = time.time()
gp = GaussianProcessRegressor(kernel, normalize_y=False, n_restarts_optimizer=3, random_state=0)
gp.fit(scale(X[tr]), (logY[tr] - mu) / sd)
train_s = time.time() - t
pred = np.exp(gp.predict(scale(X[te])) * sd + mu)
rel = np.abs(pred - Y[te]) / Y[te]
holdout = dict(n_train=int(len(tr)), n_test=int(len(te)), median_rel_err_pct=float(100 * np.median(rel)),
               p95_rel_err_pct=float(100 * np.percentile(rel, 95)),
               day270_median_rel_err_pct=float(100 * np.median(rel[:, -1])), train_seconds=train_s)
# refit on all simulations for the experiments
gp.fit(scale(X), (logY - mu) / sd)
f = lambda p, mult=1.0: np.exp(gp.predict(scale(np.append(p, mult))[None])[0] * sd + mu)  # noqa: E731

# ---- 2. surrogate calibration on the noise-free demo data (first 5 visits, as the tutorial)
def calibrate(obs, n_vis, starts=8, seed=0):
    r = np.random.default_rng(seed)
    best = None
    for _ in range(starts):
        p0 = LO[:4] + r.random(4) * (HI[:4] - LO[:4])
        sol = least_squares(lambda p: np.log(f(p))[visit_idx[:n_vis]] - np.log(obs[:n_vis]), p0,
                            bounds=(LO[:4], HI[:4]))
        if best is None or sol.cost < best.cost:
            best = sol
    return best.x

t = time.time()
p5 = calibrate(measured, 5)
cal_s = time.time() - t
calib = dict(params=p5.tolist(), rel_err_pct=(100 * np.abs(p5 - truth) / truth).tolist(), seconds=cal_s,
             day270_pred=float(f(p5)[-1]), day270_measured=float(measured[-1]),
             original_lm=dict(params=repro["calibrated_params"], minutes=27))

# ---- 3. data experiment: noisy scans, n pre-adjuvant visits
# Dose decision on a continuous scale. A 0.25-step grid put the right answer on a knife edge: at the true
# parameters, multiplier 1.5 misses the target by only 0.25%, so tiny errors flipped "right" to "wrong".
zt = np.load(R / "sims_truth.npz")             # simulator (not surrogate) at the true parameters, 8 multipliers
tm, tf = zt["X"][:, 4], zt["Y"][:, -1]         # day-270 tumor falls almost linearly with the multiplier
TARGET = float(np.interp(1.0, tm, tf)) * 0.9   # placeholder goal: 10% less tumor at day 270 than the standard plan
FINE = np.round(np.arange(0.25, 2.0001, 0.01), 2)
def choose(p):                                 # lowest multiplier whose predicted day-270 tumor <= TARGET
    q = scale(np.column_stack([np.tile(p, (len(FINE), 1)), FINE]))
    fin = np.exp(gp.predict(q)[:, -1] * sd[-1] + mu[-1])
    ok_m = FINE[fin <= TARGET]
    return float(ok_m.min()) if ok_m.size else float("inf")
right_dose = float(np.interp(-TARGET, -tf, tm))  # invert the (decreasing) true curve
exp = {}
for noise in [0.05, 0.10]:
    for n_vis in [2, 3, 4, 5]:
        errs, doses = [], []
        for s in range(200):
            obs = measured * np.exp(rng.normal(0, noise, size=measured.shape))
            p = calibrate(obs, n_vis, starts=4, seed=s)
            errs.append(abs(f(p)[-1] - measured[-1]) / measured[-1])
            doses.append(choose(p))
        doses = np.array(doses)
        exp[f"noise{int(noise*100)}_n{n_vis}"] = dict(
            noise=noise, n_visits=n_vis,
            day270_median_rel_err_pct=float(100 * np.median(errs)),
            day270_p90_rel_err_pct=float(100 * np.percentile(errs, 90)),
            dose_err_median=float(np.median(np.abs(doses - right_dose))),   # in multiplier units (0.04 = 1 total-dose unit)
            dose_err_p90=float(np.percentile(np.abs(doses - right_dose), 90)),
            too_low_dose_pct=float(100 * np.mean(doses < right_dose - 0.05)),  # under-dosed by more than 0.05
            no_dose_found_pct=float(100 * np.mean(np.isinf(doses))))
        print(noise, n_vis, exp[f"noise{int(noise*100)}_n{n_vis}"], flush=True)

out = dict(n_sims=int(len(X)), holdout=holdout, calibration=calib,
           decision=dict(rule="lowest adjuvant dose multiplier (0.25-2.0, step 0.01) whose predicted day-270 tumor <= 90% of the standard plan's",
                         right_multiplier=float(right_dose), target=TARGET),
           data_experiment=exp)
(R / "surrogate.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: out[k] for k in ["n_sims", "holdout", "calibration", "decision"]}, indent=1))
