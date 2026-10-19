"""Station model (Raviv & Kolka 2013, as used by Gammelli et al. 2022).

A station has C docks. Bikes leave by rentals and arrive by returns; the two are independent Poisson
streams whose rates are constant within each hour (lp rentals/h, lr returns/h). A rental at an empty
station and a return at a full station are lost: these are the shortages the operator wants to avoid.

The number of bikes is then a birth-death Markov chain on 0..C. For one hour with generator Q:
  P = expm(Q)                 moves the distribution of bikes forward by one hour
  M = integral_0^1 expm(Q s)  expected fraction of the hour spent in each state (Van Loan block trick)
so the expected lost rentals in that hour are lp * (pi @ M)[0] and lost returns lr * (pi @ M)[C].
Both are exact for this model (the upstream code integrates the same ODE with Runge-Kutta).
"""
import numpy as np
from scipy.linalg import expm


def generator(lp, lr, C):
    """Rate matrix of the bikes count: i -> i+1 at rate lr (return), i -> i-1 at rate lp (rental)."""
    Q = np.zeros((C + 1, C + 1))
    i = np.arange(C)
    Q[i, i + 1] = lr            # a return adds a bike unless the station is full
    Q[i + 1, i] = lp            # a rental removes a bike unless the station is empty
    Q[np.arange(C + 1), np.arange(C + 1)] = -Q.sum(axis=1)
    return Q


def hour(lp, lr, C):
    """(P, M) for one hour; see the module docstring. Only the columns of M for "empty" and "full"
    are needed, so M is returned as an (C+1, 2) array: expected fraction of the hour empty, full."""
    n = C + 1
    A = np.zeros((n + 2, n + 2))
    A[:n, :n] = generator(lp, lr, C)
    A[0, n] = A[C, n + 1] = 1.0              # Van Loan: top-right block = integral of expm(Q s) @ [e_0, e_C]
    E = expm(A)
    return E[:n, :n], E[:n, n:]


def forecast(lp, lr, C, start):
    """Propagate distributions of bikes through consecutive hours.

    lp, lr  rates for each hour (same length)
    start   one distribution over 0..C, or a (k, C+1) array of them (e.g. np.eye(C+1): every start)
    Returns a dict of per-hour arrays (hour axis first): dist (end of hour), lost_rent, lost_return,
    p_empty / p_full (fraction of the hour spent empty / full).
    """
    pi = np.atleast_2d(np.asarray(start, float))
    out = dict(dist=[], lost_rent=[], lost_return=[], p_empty=[], p_full=[])
    for a, b in zip(lp, lr):
        P, M = hour(a, b, C)
        avg = pi @ M                         # (k, 2): fraction of the hour spent empty, full
        out["lost_rent"].append(a * avg[:, 0])
        out["lost_return"].append(b * avg[:, 1])
        out["p_empty"].append(avg[:, 0])
        out["p_full"].append(avg[:, 1])
        pi = pi @ P
        out["dist"].append(pi)
    return {k: np.array(v) for k, v in out.items()}


def udf(lp, lr, C, w_rent=1.0, w_return=1.0):
    """User dissatisfaction function: expected weighted shortages over the period, for start = 0..C."""
    f = forecast(lp, lr, C, np.eye(C + 1))
    return w_rent * f["lost_rent"].sum(axis=0) + w_return * f["lost_return"].sum(axis=0)


def best_start(lp, lr, C, **w):
    """Starting number of bikes that minimises the expected shortages (the paper's decision)."""
    f = udf(lp, lr, C, **w)
    s = int(np.argmin(f))
    return s, float(f[s])


def replay(C, start, changes):
    """Play recorded events (+1 return, -1 rental) on a station that starts with `start` bikes.

    Returns (lost rentals, lost returns, bikes at the end). Deterministic: the demand is what was
    recorded; only the starting number of bikes is ours.
    """
    bikes, lost_p, lost_r = int(start), 0, 0
    for c in changes:
        if c < 0:
            if bikes == 0:
                lost_p += 1
            else:
                bikes -= 1
        elif c > 0:
            if bikes == C:
                lost_r += 1
            else:
                bikes += 1
    return lost_p, lost_r, bikes
