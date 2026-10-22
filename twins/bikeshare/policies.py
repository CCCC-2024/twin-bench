"""Policies: how the station twin chooses the number of bikes the station starts the next day with.

StationTwin takes a policy as an argument and asks it at 23:00, once its rate model can forecast every hour of the
next day (the twin decides when; the policy, what):
  decide(twin)   -> {"set_bikes": s, "expected_lost": ...} plus anything worth recording; without "set_bikes" no
                    truck goes out that night
A policy may read the twin's capacity (twin.C), its rate model (twin.rates) and its what-if,
twin.predict(24, set_bikes=s).

RulePolicy   the paper's decision: the start with the fewest expected lost rentals + returns, found by computing every
             start at once (model.best_start). The default and the baseline.
AgentPolicy  a language model chooses, asking the twin's what-if as often as it likes within its turns (agent.py).
             It records the starts it asked about and its reason; with no valid choice, it records why, and no truck
             goes out.
"""
from . import agent, model

TASK = ("It is 23:00 at a bike-share station with {C} docks. Tonight a truck sets the number of bikes the station "
        "starts tomorrow with: any whole number from 0 to {C}. A rental at an empty station or a return at a full "
        "station is lost. Choose the number with the fewest expected lost rentals plus returns tomorrow. For a number "
        "of bikes, the twin's what-if gives tomorrow's expected lost rentals, lost returns and their sum, "
        "expected_lost.")


class RulePolicy:
    def decide(self, twin):
        s, cost = model.best_start(*twin.rates.ahead(24), twin.C)
        return dict(set_bikes=s, expected_lost=cost)


class AgentPolicy:
    def __init__(self, llm, turns=10):
        """llm(messages) -> reply, a language model (agent.py); turns: the most replies it may give in one night."""
        self.llm, self.turns = llm, turns

    def decide(self, twin):
        def whatif(s):
            p = twin.predict(24, set_bikes=s)
            return dict(lost_rentals=round(float(p["lost_rent"].sum()), 2),
                        lost_returns=round(float(p["lost_return"].sum()), 2),
                        expected_lost=round(p["expected_lost"], 2))

        answer = agent.ask(self.llm, TASK.format(C=twin.C), range(twin.C + 1), whatif, self.turns)
        asked = [s for s, _ in answer["whatifs"]]
        if answer["choice"] is None:
            return dict(asked=asked, reason=answer["reason"])
        s = answer["choice"]
        return dict(set_bikes=s, expected_lost=twin.predict(24, set_bikes=s)["expected_lost"], asked=asked,
                    reason=answer["reason"])
