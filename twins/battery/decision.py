"""Twin decision (virtual -> physical): when to retire the cell from first life.

Utility functions, bounds and weights are copied from the upstream retirement script.
The twin decides once, when the measured normalized capacity first drops to
`threshold` (authors: 0.95). Future capacity = PF mean projection at that cycle.
Two versions, because they know different things:
  decide_paper   the authors' evaluation: candidates run to the cell's true end of life, and an oracle
                 decision on the true capacity curve scores the twin's (its decision error);
  decide_online  what an online twin can do: candidates run to the twin's predicted end of life.
"""
import numpy as np

from .pf import power_law, HORIZON as HORIZON_CAP

AH_BOUNDS = (300, 1000)
MTBC_BOUNDS = (0.210, 0.250)
WEIGHTS = (0.5, 0.5)


def _exp_utility(v, lb, ub, R):
    if v < lb:
        return 0.0
    if v > ub:
        return 1.0
    a = np.exp(-lb / R) / (np.exp(-lb / R) - np.exp(-ub / R))
    b = 1 / (np.exp(-lb / R) - np.exp(-ub / R))
    return a - b * np.exp(-v / R)


def utility_curve(cap_ah, x_candidates, weights=WEIGHTS):
    """Overall utility of retiring at each cycle x (cap_ah[j] = capacity at cycle j+2)."""
    csum = np.concatenate([[0.0], np.cumsum(cap_ah)])
    out = []
    for x in x_candidates:
        n = int(x) - 2                       # cycles 2 .. x-1, as upstream
        total_ah = csum[n]
        mean_dchg_h = csum[n] / n / 4.4
        u = weights[0] * _exp_utility(total_ah, *AH_BOUNDS, R=200) + \
            weights[1] * _exp_utility(mean_dchg_h, *MTBC_BOUNDS, R=0.015)
        out.append(u)
    return np.array(out)


def decide_paper(cell, pf_res, threshold=0.95, q_obs=None):
    """Return the twin's retirement cycle, the oracle's, and the true-utility regret.

    Knows what an online twin cannot: candidates run to the cell's true end of life, and the oracle and the regret use
    the true capacity curve (the authors' extrapolation past the last measured cycle).
    q_obs: the measurement stream the twin saw (default: the cell's own data).
    """
    q_obs = cell["q"] if q_obs is None else q_obs
    k_eol = cell["k_eol"]
    online_idx = int(np.where(q_obs <= threshold)[0][0])          # first cycle at/below threshold
    k_decide = int(cell["cycles"][online_idx])
    j = int(np.where(pf_res["capacities"] <= threshold)[0][0])    # same cycle inside PF output
    la, b = pf_res["log10_a"][j], pf_res["b"][j]

    n = k_eol - 2                                                  # cycles 2 .. k_eol-1
    cyc = cell["cycles"][:n]
    true_ah = cell["cap"][:n]
    twin_ah = np.concatenate([q_obs[:online_idx], power_law(cyc[online_idx:], la, b)]) * cell["q0"]

    xs = np.arange(k_decide, k_eol - 1)
    u_twin_view = utility_curve(twin_ah, xs)
    u_true = utility_curve(true_ah, xs)
    x_twin = int(xs[np.argmax(u_twin_view)])
    x_oracle = int(xs[np.argmax(u_true)])
    regret = float(u_true.max() - u_true[np.argmax(u_twin_view)])
    return dict(k_decide=k_decide, x_twin=x_twin, x_oracle=x_oracle, regret=regret,
                u_oracle=float(u_true.max()), log10_a=float(la), b=float(b),
                oracle_in_measured=x_oracle <= cell["last_measured_cycle"])


def decide_online(q_hist, q0, log10_a, b, eol_pred, weights=WEIGHTS):
    """Retirement decision using only what the twin knows now.

    Unlike `decide_paper` (and the upstream script), the candidate range ends at the twin's
    *predicted* end of life, not the cell's true one, which an online twin cannot know.
    q_hist: normalized capacities measured so far, starting at cycle 2.
    Returns the suggested retirement cycle and the utility curve the twin sees.
    """
    k_now = len(q_hist) + 1                       # cycle of the latest measurement
    horizon = int(min(eol_pred, HORIZON_CAP))
    if horizon - 1 <= k_now:
        return dict(x_twin=k_now, xs=np.array([k_now]), utility=np.array([np.nan]), horizon=horizon)
    future = np.arange(k_now + 1, horizon)
    cap_ah = np.concatenate([q_hist, power_law(future, log10_a, b)]) * q0
    xs = np.arange(k_now, horizon - 1)
    u = utility_curve(cap_ah, xs, weights)
    return dict(x_twin=int(xs[np.argmax(u)]), xs=xs, utility=u, horizon=horizon)
