"""Station digital twin, following the skeleton contract in docs/dt_template.md.

D (digital state)  the current number of bikes (observed) and, for every hour of the day and
                   day type (weekday / weekend), a Gamma posterior over the rental and return rates.
update(obs)        Gamma-Poisson update with one hour of counts (conjugate: just add counts and days).
                   With the flat prior (0, 0) the posterior mean is exactly the paper's
                   "historical average"; with window_days=30 it is its "monthly historical average".
predict(h, ...)    distribution of bikes over the next h hours, chance of being empty / full and the
                   expected lost rentals / returns; set_bikes=s makes it a what-if (a truck now).
decide()           before midnight: the number of bikes to start the next day with (the paper's
                   overnight rebalancing, minimising expected lost rentals + returns).
"""
import bisect
import datetime

import numpy as np

from dtcore import Twin

from . import model


class StationTwin(Twin):
    capabilities = ("update", "predict", "whatif", "decide")

    def __init__(self, C, window_days=None, prior=(0.0, 0.0)):
        self.C, self.window, self.prior = C, window_days, prior
        # (weekday, hour) -> [dates], [running total of rentals], [running total of returns]
        self.hist = {}
        self.bikes, self.date, self.hour = None, None, None

    # ---- update ---------------------------------------------------------------------------------
    def update(self, obs):
        day = datetime.date.fromisoformat(obs["date"])
        dates, cp, cr = self.hist.setdefault((obs["weekday"], obs["hour"]), ([], [0], [0]))
        dates.append(day)                                  # the conjugate update: add one day of counts
        cp.append(cp[-1] + obs["pickups"])
        cr.append(cr[-1] + obs["returns"])
        self.bikes, self.date, self.hour = obs["bikes"], day, obs["hour"]
        nd, nh = self._next()
        (mp, mr), (sp, sr), n = self.rate(nd, nh)
        return dict(t=obs["t"], bikes=self.bikes, next_rent_rate=mp, next_return_rate=mr,
                    next_rent_sd=sp, next_return_sd=sr, days_seen=n)

    def rate(self, day, hour):
        """Posterior mean and sd of (rental, return) rates for `hour` on `day`, from earlier days only."""
        dates, cum_p, cum_r = self.hist.get((day.weekday() < 5, hour), ([], [0], [0]))
        end = bisect.bisect_left(dates, day)               # days strictly before `day`
        start = 0 if self.window is None else bisect.bisect_left(dates, day - datetime.timedelta(days=self.window))
        a0, b0 = self.prior
        n = end - start
        if n + b0 == 0:
            return (np.nan, np.nan), (np.nan, np.nan), 0
        cp, cr = cum_p[end] - cum_p[start] + a0, cum_r[end] - cum_r[start] + a0
        b = n + b0
        return (cp / b, cr / b), (np.sqrt(cp) / b, np.sqrt(cr) / b), n

    def rates_ahead(self, horizon):
        """Mean rates for the next `horizon` hours, starting with the hour after the last observation."""
        out, (day, hour) = [], self._next()
        for _ in range(horizon):
            out.append(self.rate(day, hour)[0])
            day, hour = (day, hour + 1) if hour < 23 else (day + datetime.timedelta(days=1), 0)
        lp, lr = np.array(out).T
        return lp, lr

    def _next(self):
        if self.hour < 23:
            return self.date, self.hour + 1
        return self.date + datetime.timedelta(days=1), 0

    # ---- predict / what-if ------------------------------------------------------------------------
    def predict(self, horizon, set_bikes=None):
        """Forecast the next `horizon` hours from now; set_bikes=s: what if a truck sets s bikes now."""
        lp, lr = self.rates_ahead(horizon)
        start = np.zeros(self.C + 1)
        start[self.bikes if set_bikes is None else int(set_bikes)] = 1.0
        f = model.forecast(lp, lr, self.C, start)
        dist = f["dist"][:, 0, :]
        cdf = dist.cumsum(axis=1)
        return dict(rent_rate=lp, return_rate=lr,
                    mean_bikes=dist @ np.arange(self.C + 1),
                    p10=(cdf < 0.1).sum(axis=1), p90=(cdf < 0.9).sum(axis=1),
                    p_empty=f["p_empty"][:, 0], p_full=f["p_full"][:, 0],
                    lost_rent=f["lost_rent"][:, 0], lost_return=f["lost_return"][:, 0],
                    expected_lost=float(f["lost_rent"].sum() + f["lost_return"].sum()))

    # ---- decide -----------------------------------------------------------------------------------
    def decide(self):
        """At the last hour of a day: bikes to start tomorrow with. Otherwise no action."""
        if self.hour != 23:
            return None
        lp, lr = self.rates_ahead(24)
        if np.isnan(lp).any() or np.isnan(lr).any():      # no history yet for some hour
            return None
        s, cost = model.best_start(lp, lr, self.C)
        return dict(set_bikes=s, expected_lost=cost)
