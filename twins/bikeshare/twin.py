"""Station digital twin on the dtcore contract (docs/architecture.md).

D (digital state)  the current number of bikes (observed) and a rate model (rates.py): what the twin knows
                   about hourly rental and return demand. Default HistoricalRates, the paper's statistical
                   model (a Gamma posterior per hour and day type).
update(obs)        keeps the bikes and the hour, and feeds the hour's counts to the rate model.
predict(h, ...)    distribution of bikes over the next h hours, chance of being empty / full and the
                   expected lost rentals / returns; set_bikes=s makes it a what-if (a truck now).
decide()           before midnight: the number of bikes to start the next day with (the paper's
                   overnight rebalancing, minimising expected lost rentals + returns).
"""
import numpy as np

from dtcore import Twin

from . import model
from .rates import HistoricalRates


class StationTwin(Twin):
    capabilities = ("update", "predict", "whatif", "decide")

    def __init__(self, C, rates=None):
        self.C = C
        self.rates = HistoricalRates() if rates is None else rates
        self.bikes, self.hour = None, None

    # ---- update ---------------------------------------------------------------------------------
    def update(self, obs):
        self.bikes, self.hour = obs["bikes"], obs["hour"]
        return dict(t=obs["t"], bikes=self.bikes, **self.rates.update(obs))

    # ---- predict / what-if ------------------------------------------------------------------------
    def predict(self, horizon, set_bikes=None):
        """Forecast the next `horizon` hours from now; set_bikes=s: what if a truck sets s bikes now."""
        lp, lr = self.rates.ahead(horizon)
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
        lp, lr = self.rates.ahead(24)
        if np.isnan(lp).any() or np.isnan(lr).any():      # no history yet for some hour
            return None
        s, cost = model.best_start(lp, lr, self.C)
        return dict(set_bikes=s, expected_lost=cost)
