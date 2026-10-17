"""Model only: reproduce the paper's experiment for station 426 (Nov-Dec 2018), without the twin.

For each test day and each rate source: choose the number of bikes at midnight with the queueing model,
then replay that day's recorded rentals/returns to count real shortages. Also checks the model math
against a Monte Carlo simulation of the same Poisson model (does the code solve its own equations?).
Writes results/step1_decisions.json and results/step1_model.png. About 20 s.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sdt import data, model  # noqa: E402
from sdt import experiment as X  # noqa: E402

S = X.load()
C = S["C"]
print(f"station 426 ({data.name(426)}), {C} docks; {len(X.TEST_DAYS)} test days")

res = {str(d): X.run_day(S, d) for d in X.TEST_DAYS}
json.dump(res, open(ROOT / "results" / "step1_decisions.json", "w"), indent=0)

print(f"\n{'rate source':<14}{'start (median)':>15}{'expected':>10}{'real lost/day':>15}{'rent':>7}{'return':>8}")
for src in X.SOURCES + ("best_possible",):
    rows = [r[src] for r in res.values()]
    exp = np.mean([r["expected"] for r in rows]) if rows[0]["expected"] is not None else float("nan")
    rp, rr = np.mean([r["lost_rent"] for r in rows]), np.mean([r["lost_return"] for r in rows])
    print(f"{src:<14}{np.median([r['start'] for r in rows]):>15.0f}{exp:>10.2f}{rp + rr:>15.2f}{rp:>7.2f}{rr:>8.2f}")

# --- verification: analytic expectation vs Monte Carlo of the same model (one day, a few starts) ---
rng = np.random.default_rng(0)
day = X.TEST_DAYS[13]
lp, lr = X.rates(S, "overall_ha", day)


def simulate(start):
    bikes, lost = start, 0
    for a, b in zip(lp, lr):
        n = rng.poisson(a + b)                            # merged stream, each event a rental w.p. a/(a+b)
        for is_rent in rng.random(n) < a / (a + b):
            if is_rent:
                lost += bikes == 0
                bikes -= bikes > 0
            else:
                lost += bikes == C
                bikes += bikes < C
    return lost


f = model.udf(lp, lr, C)
worst = 0.0
for s in (0, 10, 20, 30):
    mc = np.array([simulate(s) for _ in range(4000)])
    se = mc.std() / np.sqrt(len(mc))
    worst = max(worst, abs(mc.mean() - f[s]) / se)
    print(f"start {s:2d}: model {f[s]:6.2f}   Monte Carlo {mc.mean():6.2f} +/- {se:.2f}")
print(f"largest gap = {worst:.1f} standard errors (< 3 expected)")

# --- figure: the decision on one day ---
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
for src, ls in (("overall_ha", "-"), ("true_count", "--")):
    a, b = X.rates(S, src, day)
    u = model.udf(a, b, C)
    ax[0].plot(range(C + 1), u, ls, label=src)
    ax[0].plot(int(np.argmin(u)), u.min(), "o", color=ax[0].lines[-1].get_color())
ax[0].set(xlabel="bikes at midnight", ylabel="expected lost rentals + returns", title=f"decision, {day}")
ax[0].legend()
ev = S["events"][day]
for src in ("overall_ha", "best_possible"):
    s = res[str(day)][src]["start"]
    bikes, traj = s, [s]
    for c in ev:
        bikes = min(C, bikes + 1) if c > 0 else max(0, bikes - 1)
        traj.append(bikes)
    ax[1].step(range(len(traj)), traj, where="post", label=f"{src} (start {s})")
ax[1].axhline(C, color="grey", lw=0.8)
ax[1].set(xlabel="recorded event number that day", ylabel="bikes in station", title="replay of that day's events")
ax[1].legend()
fig.tight_layout()
fig.savefig(ROOT / "results" / "step1_model.png", dpi=130)
print("saved results/step1_decisions.json, results/step1_model.png")
