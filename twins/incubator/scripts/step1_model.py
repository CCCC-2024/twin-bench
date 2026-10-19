"""Step 1: model only. Does our model, written from the description, reproduce the real box?

1. Simulate the 2023 empty-box experiment with the published parameters (heater and room
   temperature taken from the data; box temperature predicted by the model alone).
2. Identifiability: fix C_heater at a range of values, fit the other three, compare the error.
   Only the box temperature is measured, so we check which parameters the data can pin down.
3. Our own calibration with C_heater fixed at the published value: do we recover the rest?
Writes results/step1_model.json, results/step1_model.png, results/step1_identifiability.png.
"""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from twins.incubator.data import load
from twins.incubator.model import NAMES, PARAMS_2023, simulate

ROOT = Path(__file__).resolve().parents[1]

d = load("calibration_2023")
t, u_h, u_room, y = d["t"].to_numpy(), d["heater_on"].to_numpy(), d["T_room"].to_numpy(), d["T"].to_numpy()
x0 = np.array([y[0], y[0]])          # heatbed starts at box temperature, as upstream assumes


def run(p):
    return simulate(p, t, u_h, u_room, x0)


def errors(X):
    e = X[:, 0] - y
    return dict(rmse=float(np.sqrt(np.mean(e ** 2))), max_abs=float(np.max(np.abs(e))),
                max_heatbed_T=float(X[:, 1].max()))


def fit_with_fixed_C_heater(C_heater, start):
    def residual(th):
        return run(dict(PARAMS_2023, C_air=th[0], G_box=th[1], C_heater=C_heater, G_heater=th[2]))[:, 0] - y
    f = least_squares(residual, start, bounds=(1e-6, np.inf), x_scale="jac")
    p = dict(PARAMS_2023, C_air=f.x[0], G_box=f.x[1], C_heater=C_heater, G_heater=f.x[2])
    return p, errors(run(p))


# 1) published parameters
X_pub = run(PARAMS_2023)
res = {"n_samples": len(t), "hours": float(t[-1] / 3600),
       "published": {"params": PARAMS_2023, **errors(X_pub)}}

# 2) identifiability profile over C_heater
start = [PARAMS_2023["C_air"], PARAMS_2023["G_box"], PARAMS_2023["G_heater"]]
profile = []
for ch in [50, 100, 200, PARAMS_2023["C_heater"], 500, 800]:
    p, e = fit_with_fixed_C_heater(ch, start)
    profile.append({k: float(p[k]) for k in NAMES} | e)
res["profile_C_heater"] = profile

# 3) our calibration, C_heater fixed at the published value, deliberately wrong start
p_fit, e_fit = fit_with_fixed_C_heater(PARAMS_2023["C_heater"],
                                       [PARAMS_2023["C_air"] * 0.5, PARAMS_2023["G_box"] * 2, PARAMS_2023["G_heater"] * 0.5])
res["our_calibration_fixed_C_heater"] = {
    "params": {k: float(p_fit[k]) for k in NAMES},
    "rel_diff_vs_published": {k: float(p_fit[k] / PARAMS_2023[k] - 1) for k in NAMES}, **e_fit}

(ROOT / "results").mkdir(exist_ok=True)
json.dump(res, open(ROOT / "results" / "step1_model.json", "w"), indent=1)
print("published:", {k: round(v, 3) for k, v in res["published"].items() if k != "params"})
print("profile (C_heater fixed -> best fit of the rest):")
for r in profile:
    print(f"  C_heater {r['C_heater']:6.1f}  C_air {r['C_air']:6.1f}  G_box {r['G_box']:.3f}  G_heater {r['G_heater']:.3f}"
          f"  RMSE {r['rmse']:.3f}  max heatbed T {r['max_heatbed_T']:5.1f}")
print("our calibration (C_heater fixed):", {k: f"{v:+.1%}" for k, v in res["our_calibration_fixed_C_heater"]["rel_diff_vs_published"].items()},
      "RMSE %.3f" % e_fit["rmse"])

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

h = t / 60
fig, ax = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True, gridspec_kw=dict(height_ratios=[3, 1]))
ax[0].plot(h, y, color="#222", lw=1.6, label="measured box T (real data)")
ax[0].plot(h, X_pub[:, 0], color="#2a6fdb", lw=1.2, ls="--", label="model T (published params)")
ax[0].plot(h, X_pub[:, 1], color="#d9822b", lw=1, alpha=.8, label="model heatbed T (not measured)")
ax[0].plot(h, u_room, color="#888", lw=1, label="room T (measured)")
ax[0].set_ylabel("°C"); ax[0].grid(alpha=.3); ax[0].legend(fontsize=8, loc="upper left")
ax[0].set_title(f"Step 1 — incubator model vs real data (RMSE {res['published']['rmse']:.2f} °C)", fontsize=10)
ax[1].step(h, u_h, where="post", color="#c0392b", lw=1)
ax[1].set_yticks([0, 1]); ax[1].set_yticklabels(["off", "on"]); ax[1].set_ylabel("heater")
ax[1].set_xlabel("minutes"); ax[1].grid(alpha=.3)
fig.tight_layout(); fig.savefig(ROOT / "results" / "step1_model.png", dpi=150); plt.close(fig)

ch = [r["C_heater"] for r in profile]
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
ax[0].plot(ch, [r["rmse"] for r in profile], "o-", color="#2a6fdb")
ax[0].set_xlabel("C_heater fixed at (J/K)"); ax[0].set_ylabel("box T RMSE (°C)"); ax[0].grid(alpha=.3)
ax[1].plot(ch, [r["max_heatbed_T"] for r in profile], "o-", color="#d9822b")
ax[1].set_xlabel("C_heater fixed at (J/K)"); ax[1].set_ylabel("max heatbed T (°C)"); ax[1].grid(alpha=.3)
for a in ax:
    a.axvline(PARAMS_2023["C_heater"], color="#888", ls=":", lw=1)
fig.suptitle("Box T fits almost equally well for C_heater 50–329 J/K, but the heatbed temperature differs", fontsize=9)
fig.tight_layout(); fig.savefig(ROOT / "results" / "step1_identifiability.png", dpi=150)
print("saved figures")
