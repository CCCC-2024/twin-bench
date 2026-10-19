"""Bike-share station twin: one Citi Bike station, reproducing Gammelli et al. 2022 on dtcore (see README.md).

Two entry points, both exported here:
  running the twin   StationTwin (digital side) with its rate model HistoricalRates, ReplayStation and
                     SimStation (physical side);
  the paper          experiment, the paper's set-up without the twin.
model.py (the station model) and data.py (the upstream files) are the implementation behind both.
"""
from . import experiment
from .rates import HistoricalRates
from .sources import ReplayStation, SimStation
from .twin import StationTwin

__all__ = ["StationTwin", "HistoricalRates", "ReplayStation", "SimStation", "experiment"]
