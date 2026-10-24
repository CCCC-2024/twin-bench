"""Station digital twin on the dtcore contract (docs/architecture.md).

D (digital state)  the current number of bikes (observed) and a rate model (rates.py): what the twin knows
                   about hourly rental and return demand. Default HistoricalRates, the paper's statistical
                   model (a Gamma posterior per hour and day type).
update(obs)        keeps the bikes and the hour, and feeds the hour's counts to the rate model.
predict(h, ...)    distribution of bikes over the next h hours, chance of being empty / full and the
                   expected lost rentals / returns; set_bikes=s makes it a what-if (a truck now), and so do
                   rent_factor / return_factor: demand times that factor from from_hour to to_hour (clock hours),
                   an event the data never showed.
decide()           before midnight: the number of bikes to start the next day with (the paper's
                   overnight rebalancing), chosen by the twin's policy:
                     RulePolicy   the paper's decision, the fewest expected lost rentals + returns (default)
                     AgentPolicy  the LLM agent (dtcore.agent) chooses, running the twin's what-if
                   A policy is asked only at 23:00 once the next day can be forecast: decide(twin) -> action.
brief(), whatif()  the twin for an LLM agent: the station in words, and the what-if of a truck as three numbers.
"""
import numpy as np

from dtcore import Twin, agent

from . import model
from .rates import HistoricalRates

NIGHTLY_PROMPT = ("Tonight a truck sets the number of bikes the station starts tomorrow with. Choose it so that tomorrow has "
                  "the fewest expected lost rentals plus returns.")
# extraction: the model only extracts whatif's demand inputs from an event in words; the twin then compares every start
# Sent as the system message, followed by the twin's brief(); the page sends it too (web/bikeshare_events.js)
EXTRACT_EVENT_PROMPT = ("Extract, from the event the user describes, the what-if's demand inputs for tomorrow. Reply with "
                        'one JSON object and nothing else: {"rent_factor": x, "return_factor": y, "from_hour": a, "to_hour": '
                        'b} (rentals and returns are x and y times the usual from clock hour a up to, not including, hour b), '
                        'or {"cannot": "<why>"} if the event cannot be written with these inputs.\n\nThe twin: ')


class RulePolicy:
    def decide(self, twin):
        s, cost = model.best_start(*twin.rates.ahead(24), twin.C)       # every start at once
        return dict(set_bikes=s, expected_lost=cost)


class AgentPolicy:
    def __init__(self, llm, turns=10):
        """llm(messages) -> reply, a language model (dtcore.agent); turns: the most replies it may give a night."""
        self.llm, self.turns = llm, turns

    def decide(self, twin):
        """The agent's choice, with the scenarios it ran and its reason. With no valid choice the action records why
        and has no set_bikes: no truck goes out."""
        a = agent.ask(self.llm, twin, NIGHTLY_PROMPT, [dict(set_bikes=s) for s in range(twin.C + 1)], self.turns)
        if a["choice"] is None:
            return dict(asked=a["asked"], reason=a["error"])
        s = a["choice"]["set_bikes"]
        return dict(set_bikes=s, expected_lost=twin.predict(24, set_bikes=s)["expected_lost"], asked=a["asked"],
                    reason=a["answer"])


class StationTwin(Twin):
    capabilities = ("update", "predict", "whatif", "decide")

    def __init__(self, C, rates=None, policy=None):
        self.C = C
        self.rates = HistoricalRates() if rates is None else rates
        self.policy = RulePolicy() if policy is None else policy
        self.bikes, self.hour = None, None

    # ---- update ---------------------------------------------------------------------------------
    def update(self, obs):
        self.bikes, self.hour = obs["bikes"], obs["hour"]
        return dict(t=obs["t"], bikes=self.bikes, **self.rates.update(obs))

    # ---- predict / what-if ------------------------------------------------------------------------
    def predict(self, horizon, set_bikes=None, rent_factor=1.0, return_factor=1.0, from_hour=0, to_hour=24):
        """Forecast the next `horizon` hours from now; set_bikes=s: what if a truck sets s bikes now; the factors:
        what if rentals / returns are that many times the forecast in clock hours from_hour <= h < to_hour."""
        lp, lr = self.rates.ahead(horizon)
        clock = (self.hour + 1 + np.arange(horizon)) % 24
        event = (clock >= from_hour) & (clock < to_hour)
        lp, lr = np.where(event, lp * rent_factor, lp), np.where(event, lr * return_factor, lr)
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
        """At the last hour of a day: the policy's action, bikes to start tomorrow with. Otherwise no action."""
        if self.hour != 23:
            return None
        lp, lr = self.rates.ahead(24)
        if np.isnan(lp).any() or np.isnan(lr).any():      # no history yet for some hour
            return None
        return self.policy.decide(self)

    # ---- for an LLM agent -------------------------------------------------------------------------
    def brief(self):
        return (f"A bike-share station with {self.C} docks. It is {self.hour}:00 and the station has {self.bikes} "
                "bikes. A rental at an empty station or a return at a full station is lost. whatif(set_bikes=s, "
                "rent_factor=1, return_factor=1, from_hour=0, to_hour=24): a truck sets the station to s bikes now, a "
                f"whole number from 0 to {self.C}; optionally, rentals and returns are rent_factor and return_factor "
                "times the usual from clock hour from_hour up to (not including) to_hour, whole numbers from 0 to 24. "
                "It returns the expected lost rentals, lost returns and their sum, expected_lost, over the next 24 "
                "hours.")

    def whatif(self, set_bikes, rent_factor=1.0, return_factor=1.0, from_hour=0, to_hour=24):
        if type(set_bikes) is not int or not 0 <= set_bikes <= self.C:
            raise ValueError(f"set_bikes must be a whole number from 0 to {self.C}")
        if any(type(f) not in (int, float) or f < 0 for f in (rent_factor, return_factor)):
            raise ValueError("rent_factor and return_factor must be numbers, 0 or more")
        if any(type(h) is not int for h in (from_hour, to_hour)) or not 0 <= from_hour < to_hour <= 24:
            raise ValueError("from_hour and to_hour must be whole numbers with 0 <= from_hour < to_hour <= 24")
        p = self.predict(24, set_bikes, rent_factor, return_factor, from_hour, to_hour)
        return dict(lost_rentals=round(float(p["lost_rent"].sum()), 2),
                    lost_returns=round(float(p["lost_return"].sum()), 2), expected_lost=round(p["expected_lost"], 2))
