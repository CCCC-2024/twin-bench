"""Battery twin: Thelen et al. 2023 reproduced on dtcore (see README.md).

Two entry points, both exported here:
  running the twin   BatteryTwin (digital side) and CellReplay (physical side);
  the paper          run_pf, the authors' particle filter over a whole cell, and decide_paper, their decision and its
                     evaluation, which know the cell's true end of life.
data.py (the cells), pf.py (the filter) and decision.py (the retirement utility) are the implementation behind both.
"""
from .decision import decide_paper
from .pf import run_pf
from .sources import CellReplay
from .twin import BatteryTwin

__all__ = ["BatteryTwin", "CellReplay", "run_pf", "decide_paper"]
