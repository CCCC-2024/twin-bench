"""Bike-share station twin: one Citi Bike station, reproducing Gammelli et al. 2022 on dtcore (see README.md).

Two entry points, both exported here:
  running the twin   StationTwin (digital side) with a rate model (rates.py: HistoricalRates, the default; RNNRates), and a
                     policy, RulePolicy (default) or AgentPolicy (the LLM agent of dtcore.agent); ReplayStation or
                     SimStation (physical side);
  the paper          experiment, the paper's set-up without the twin.
model.py (the station model) and data.py (the upstream files) are the implementation behind both.
"""
from . import experiment
from .rates import DiscountedRates, HistoricalRates, SelectedRates
from .sources import ReplayStation, SimStation
from .twin import AgentPolicy, RulePolicy, StationTwin

__all__ = ["StationTwin", "HistoricalRates", "DiscountedRates", "SelectedRates", "RulePolicy", "AgentPolicy", "ReplayStation", "SimStation",
           "experiment"]   # and RNNRates, below


def __getattr__(name):
    """RNNRates needs torch (pip install -e ".[rnn]"), so it is imported only when asked for (it is not in __all__,
    so `from twins.bikeshare import *` works without torch)."""
    if name == "RNNRates":
        from .rnn import RNNRates
        return RNNRates
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
