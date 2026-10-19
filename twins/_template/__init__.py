"""Template twin: a mug of water with a heater. Copy twins/_template to start a new twin (see its README).

Public names: MugTwin (digital side) and SimMug (physical side); everything else here is internal.
"""
from .sources import SimMug
from .twin import MugTwin

__all__ = ["MugTwin", "SimMug"]
