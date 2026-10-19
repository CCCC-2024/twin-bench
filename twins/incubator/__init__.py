"""Incubator twin, built from the paper's description (Feng et al. 2021) and run on dtcore (see README.md).

Public names: IncubatorTwin (digital side), SimBox and RecordedBox (physical side), and Controller, the box's
controller, which IncubatorTwin.predict needs as an argument. model.py, kalman.py and data.py are the implementation.
"""
from .controller import Controller
from .sources import RecordedBox, SimBox
from .twin import IncubatorTwin

__all__ = ["IncubatorTwin", "SimBox", "RecordedBox", "Controller"]
