"""Bike-share station twin: one Citi Bike station, reproducing Gammelli et al. 2022 on dtcore (see README.md).

Two entry points:
  running the twin   StationTwin (digital side), ReplayStation and SimStation (physical side), exported here;
  the paper          experiment.py, the paper's set-up without the twin (scripts/ and tests/ use it).
model.py (the station model) and data.py (the upstream files) are the implementation behind both.
"""
from .sources import ReplayStation, SimStation
from .twin import StationTwin

__all__ = ["StationTwin", "ReplayStation", "SimStation"]
