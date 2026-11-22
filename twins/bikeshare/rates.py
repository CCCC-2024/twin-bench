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
DiscountedRates  the same posterior with exponential forgetting (a discount factor, West & Harrison 1997): before a
                 day's counts are added, what was known is weighted down, by half every half_life_days. Old days
                 fade instead of dropping out at the edge of a window.
SelectedRates    several rate models side by side; forecasts with the one whose day-ahead forecasts had the smallest
                 squared error over the last judge_days days. The twin chooses how much to forget from what it sees.
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
        return _ahead(self, horizon)


class DiscountedRates:
    def __init__(self, half_life_days):
        self.keep = 0.5 ** (1.0 / half_life_days)          # weight left after one day
        self.post = {}                                     # (weekday, hour) -> [rentals, returns, days, last date]
        self.date, self.hour = None, None

    def update(self, obs):
        day = datetime.date.fromisoformat(obs["date"])
        p = self.post.setdefault((obs["weekday"], obs["hour"]), [0.0, 0.0, 0.0, day])
        w = self.keep ** (day - p[3]).days                 # forget, then add one day of counts
        p[:] = [w * p[0] + obs["pickups"], w * p[1] + obs["returns"], w * p[2] + 1, day]
        self.date, self.hour = day, obs["hour"]
        (mp, mr), (sp, sr), n = self.rate(*next_hour(day, self.hour))
        return dict(next_rent_rate=mp, next_return_rate=mr, next_rent_sd=sp, next_return_sd=sr, days_seen=n)

    def rate(self, day, hour):
        """Posterior mean and sd of (rental, return) rates for `hour` on `day`; n is the effective number of days."""
        cp, cr, n, _ = self.post.get((day.weekday() < 5, hour), (0.0, 0.0, 0.0, None))
        if n == 0:
            return (np.nan, np.nan), (np.nan, np.nan), 0
        return (cp / n, cr / n), (np.sqrt(cp) / n, np.sqrt(cr) / n), n

    def ahead(self, horizon):
        return _ahead(self, horizon)


class SelectedRates:
    def __init__(self, candidates, judge_days=28):
        """candidates: {name: rate model}. chosen is the name in use; it changes only at 23:00."""
        self.candidates, self.judge = candidates, judge_days
        self.errors = {name: [] for name in candidates}    # squared error of each day's forecast, newest last
        self.forecast, self.seen, self.chosen = {}, [], next(iter(candidates))

    def update(self, obs):
        est = {name: r.update(obs) for name, r in self.candidates.items()}
        self.seen.append((obs["pickups"], obs["returns"]))
        if obs["hour"] == 23:
            if len(self.seen) == 24:                       # a whole day seen: judge last night's forecasts of it
                for name, f in self.forecast.items():
                    self.errors[name].append(float(((f - np.array(self.seen)) ** 2).sum()))
            self.seen = []
            self.forecast = {name: np.stack(r.ahead(24), axis=1) for name, r in self.candidates.items()}
            judged = {name: sum(e[-self.judge:]) for name, e in self.errors.items()
                      if e and not np.isnan(e[-self.judge:]).any()}
            if judged:
                self.chosen = min(judged, key=judged.get)
        return dict(est[self.chosen], rates_chosen=self.chosen)

    def ahead(self, horizon):
        return self.candidates[self.chosen].ahead(horizon)


def _ahead(rates, horizon):
    out, (day, hour) = [], next_hour(rates.date, rates.hour)
    for _ in range(horizon):
        out.append(rates.rate(day, hour)[0])
        day, hour = next_hour(day, hour)
    lp, lr = np.array(out).T
    return lp, lr


def next_hour(day, hour):
    return (day, hour + 1) if hour < 23 else (day + datetime.timedelta(days=1), 0)
