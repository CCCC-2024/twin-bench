"""Rate models: what the station twin knows about hourly rental and return demand.

StationTwin takes a rate model as an argument, feeds it one hour at a time, and asks it for the hours ahead:
  update(obs)      take one hour of counts (date, hour, weekday, pickups, returns); return the next hour's
                   rates, which the twin passes on with its estimate
  ahead(horizon)   mean (rental, return) rates for the next `horizon` hours, after the last hour seen
A rate model sees the data only through update, in time order, so its forecasts use only what the twin has seen.

HistoricalRates  the paper's statistical model: for every hour of the day and day type (weekday / weekend), a
                 Gamma posterior over the rental and return rates, updated by adding counts (conjugate). With
                 the flat prior (0, 0) the posterior mean is exactly the paper's "historical average"; with
                 window_days=30 it is its "monthly historical average".
"""
import bisect
import datetime

import numpy as np


class HistoricalRates:
    def __init__(self, window_days=None, prior=(0.0, 0.0)):
        self.window, self.prior = window_days, prior
        # (weekday, hour) -> [dates], [running total of rentals], [running total of returns]
        self.hist = {}
        self.date, self.hour = None, None

    def update(self, obs):
        day = datetime.date.fromisoformat(obs["date"])
        dates, cp, cr = self.hist.setdefault((obs["weekday"], obs["hour"]), ([], [0], [0]))
        dates.append(day)                                  # the conjugate update: add one day of counts
        cp.append(cp[-1] + obs["pickups"])
        cr.append(cr[-1] + obs["returns"])
        self.date, self.hour = day, obs["hour"]
        (mp, mr), (sp, sr), n = self.rate(*next_hour(day, self.hour))
        return dict(next_rent_rate=mp, next_return_rate=mr, next_rent_sd=sp, next_return_sd=sr, days_seen=n)

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

    def ahead(self, horizon):
        out, (day, hour) = [], next_hour(self.date, self.hour)
        for _ in range(horizon):
            out.append(self.rate(day, hour)[0])
            day, hour = next_hour(day, hour)
        lp, lr = np.array(out).T
        return lp, lr


def next_hour(day, hour):
    return (day, hour + 1) if hour < 23 else (day + datetime.timedelta(days=1), 0)
