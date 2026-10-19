"""Twin update (physical -> virtual): particle filter on a power-law fade model.

Same algorithm and same random-number order as upstream/run_pf_and_optimize_retirement_six_cells.py:
  capacity(k) = 1 - 10**log10_a * k**b, parameters follow a random walk,
  one Gaussian likelihood update per measured cycle.
Only change: the end-of-life search over 500 sampled trajectories is solved in
closed form instead of a 10,000-cycle grid search. No randomness is involved, so the
random stream and all outputs are identical (checked at every cycle of the paper's
example cell by scripts/check_repro.py in v0.2.0).
"""
import numpy as np
from scipy import stats

from .data import EOL_THRESHOLD

PRIOR = (-15.77, 5.45)        # authors' initial log10_a, b
MEAS_STD = 0.01               # authors' value in the retirement script
PROC_STD = (0.05, 0.05)
N_PARTICLES = 200
N_SAMPLES = 500
HORIZON = 10000


def power_law(k, log10_a, b):
    return 1 - (10 ** log10_a) * k ** b


def _eol_samples(cycle, la, b, eol_threshold):
    """First integer cycle in [cycle, HORIZON) with predicted capacity <= threshold, else 100000.

    Closed form of upstream's grid search: the curve is monotone in k, so solve
    1 - a k^b = thr and correct the rounding by checking the neighbours.
    """
    with np.errstate(all="ignore"):
        kstar = ((1 - eol_threshold) / 10 ** la) ** (1 / b)
    k = np.where(b > 0, np.maximum(cycle, np.ceil(np.nan_to_num(kstar, nan=np.inf, posinf=np.inf))), cycle)
    k = np.minimum(k, HORIZON)
    for _ in range(2):  # fix float rounding at the boundary
        up = (k < HORIZON) & (power_law(k, la, b) > eol_threshold) & (b > 0)
        k = np.where(up, k + 1, k)
        down = (k > cycle) & (power_law(k - 1, la, b) <= eol_threshold)
        k = np.where(down, k - 1, k)
    hit = (k < HORIZON) & (power_law(k, la, b) <= eol_threshold)
    return np.where(hit, k, 100000.0)


class ParticleFilter:
    """Particle filter on the fade-model parameters, updated one measured cycle at a time.

    Same algorithm and random-number order as the original batch loop, so feeding the
    measurements one by one gives exactly the batch result (run_pf below uses this class).
    """

    def __init__(self, prior=PRIOR, prior_std=PROC_STD, meas_std=MEAS_STD, proc_std=PROC_STD,
                 n_particles=N_PARTICLES, eol_threshold=EOL_THRESHOLD, rng=np.random):
        self.meas_std, self.proc_std = meas_std, proc_std
        self.n, self.eol_threshold, self.rng = n_particles, eol_threshold, rng
        self.particles = np.column_stack([prior[k] + rng.randn(n_particles) * prior_std[k] for k in range(2)])
        self.weights = np.full_like(self.particles, 1 / n_particles)

    def step(self, cycle, cap):
        """Ingest one measurement (cycle number, normalized capacity) and update the state."""
        rng, n = self.rng, self.n
        # predict (random walk)
        particles = np.column_stack([self.particles[:, k] + rng.randn(n) * self.proc_std[k] for k in range(2)])
        # update
        pdf = stats.norm(power_law(cycle, particles[:, 0], particles[:, 1]), self.meas_std).pdf(cap)
        w = self.weights * pdf[:, None]
        weights = w / w.sum(axis=0)
        # estimate
        m = np.array([np.average(particles[:, k], weights=weights[:, k]) for k in range(2)])
        s = np.array([np.sqrt(np.average((particles[:, k] - m[k]) ** 2, weights=weights[:, k])) for k in range(2)])
        la_s = rng.choice(particles[:, 0], N_SAMPLES, p=weights[:, 0])
        b_s = rng.choice(particles[:, 1], N_SAMPLES, p=weights[:, 1])
        # resample
        new = np.zeros_like(particles)
        for k in range(2):
            cdf = np.cumsum(weights[:, k]); cdf[-1] = 1.0
            new[:, k] = particles[np.searchsorted(cdf, rng.random(n)), k]
        self.particles, self.weights = new, np.full_like(particles, 1 / n)

        eol = _eol_samples(cycle, la_s, b_s, self.eol_threshold)
        lo, hi = np.percentile(eol - cycle, (5, 95))
        return dict(cycle=int(cycle), capacity=float(cap), log10_a=m[0], b=m[1],
                    log10_a_std=s[0], b_std=s[1], eol_pred=float(np.median(eol)),
                    rul_lo=float(lo), rul_hi=float(hi))

    def capacity_band(self, cycles):
        """5th and 95th percentile of the projected normalized capacity over the particles, at each cycle."""
        band = power_law(np.asarray(cycles)[None, :], self.particles[:, :1], self.particles[:, 1:])
        return np.percentile(band, 5, axis=0), np.percentile(band, 95, axis=0)


def run_pf(cell, prior=PRIOR, prior_std=PROC_STD, meas_std=MEAS_STD, proc_std=PROC_STD,
           n_particles=N_PARTICLES, eol_threshold=EOL_THRESHOLD, q=None, rng=np.random):
    """Filter cycle by cycle from first-prediction time to end of life (batch use of ParticleFilter).

    `q` lets experiments feed a modified measurement stream (e.g. noisier sensors);
    by default the cell's normalized capacity is used. `prior_std` defaults to the
    process noise, as in the original code.
    """
    q = cell["q"] if q is None else q
    i0, k_eol = cell["idx_fpt"], cell["k_eol"]
    cycles = np.arange(cell["k_fpt"], k_eol + 1, dtype=int)
    caps = q[i0:]
    filt = ParticleFilter(prior, prior_std, meas_std, proc_std, n_particles, eol_threshold, rng)
    rows = [filt.step(c, caps[i]) for i, c in enumerate(cycles)]
    col = lambda k: np.array([r[k] for r in rows])
    return dict(cycles=cycles, capacities=caps, log10_a=col("log10_a"), b=col("b"),
                log10_a_std=col("log10_a_std"), b_std=col("b_std"), eol_pred=col("eol_pred"),
                rul_lo=col("rul_lo"), rul_hi=col("rul_hi"), true_rul=k_eol - cycles)
