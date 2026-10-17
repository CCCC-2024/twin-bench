"""Physical side of the station twin: two sources with the same observe / act / advance interface.

ReplayStation  replays the recorded rentals and returns of 2018 hour by hour (kind "replay").
               The demand is fixed by the record, but our actions change the station: setting the
               number of bikes changes the inventory and how many rentals / returns are lost.
SimStation     the same station driven by simulated Poisson demand (kind "sim").

One step = one hour. observe() returns what happened during the current hour:
  t, date, hour, weekday, pickups / returns (demand recorded or simulated in that hour),
  bikes (inventory at the end of the hour), lost_rent / lost_return (shortages on OUR station).
Caveat: a real station cannot observe lost rentals (people just walk away); the twin does not use them.
"""
import datetime

import numpy as np

from dtcore import Source

from . import model


class _Station(Source):
    def __init__(self, C, start_date, n_days, bikes=None):
        self.C, self.k, self.n = C, 0, 24 * n_days
        self.start = start_date
        self.bikes = C // 2 if bikes is None else int(bikes)
        self.moved = 0                                     # bikes added (+) or removed (-) by actions

    def _when(self):
        day = self.start + datetime.timedelta(days=self.k // 24)
        return day, self.k % 24

    def act(self, action):
        if "set_bikes" in action:                         # a truck brings the station to this level
            target = int(min(max(action["set_bikes"], 0), self.C))
            self.moved += target - self.bikes
            self.bikes = target

    def _play(self, changes):
        self.last_changes = [int(c) for c in changes]      # kept so the page can replay the same hour
        before = self.bikes
        lost_p, lost_r, self.bikes = model.replay(self.C, before, changes)
        day, hour = self._when()
        return dict(t=self.k, date=str(day), hour=hour, weekday=day.weekday() < 5,
                    pickups=int(np.sum(np.asarray(changes) < 0)), returns=int(np.sum(np.asarray(changes) > 0)),
                    bikes=self.bikes, lost_rent=lost_p, lost_return=lost_r)

    def advance(self):
        self.k += 1


class ReplayStation(_Station):
    kind = "replay"

    def __init__(self, C, events, start_date, n_days, bikes=None):
        """events: DataFrame from sdt.data.events (time-ordered changes with date and hour)."""
        super().__init__(C, start_date, n_days, bikes)
        self.by_hour = {(d, h): g["change"].to_numpy() for (d, h), g in events.groupby(["date", "hour"])}

    def observe(self):
        if self.k >= self.n:
            return None
        return self._play(self.by_hour.get(self._when(), np.array([], int)))


class SimStation(_Station):
    kind = "sim"

    def __init__(self, C, rate_fn, start_date, n_days, bikes=None, seed=0):
        """rate_fn(date, hour) -> (rentals/h, returns/h) of the simulated demand."""
        super().__init__(C, start_date, n_days, bikes)
        self.rate_fn, self.rng = rate_fn, np.random.default_rng(seed)
        self.surge = 1.0                                   # experiments can scale demand (e.g. an event)

    def observe(self):
        if self.k >= self.n:
            return None
        a, b = (self.surge * r for r in self.rate_fn(*self._when()))
        n = self.rng.poisson(a + b)
        changes = np.where(self.rng.random(n) < a / (a + b), -1, 1) if n else np.array([], int)
        return self._play(changes)
