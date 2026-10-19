"""Bike-share station twin: one Citi Bike station, reproducing Gammelli et al. 2022 on dtcore (see README.md).

Public names: StationTwin (digital side), ReplayStation and SimStation (physical side). The paper's experiment
(experiment.py), the station model (model.py) and the data readers (data.py) are this twin's internals.
"""
from .sources import ReplayStation, SimStation
from .twin import StationTwin

__all__ = ["StationTwin", "ReplayStation", "SimStation"]
