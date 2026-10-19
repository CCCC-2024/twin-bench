"""UAV twin: Kapteyn, Pretorius & Willcox 2021 reproduced on dtcore without ROS (see README.md).

Public names: UAVTwin (digital side), ScriptedUAV and ControlledUAV (physical side), and UAVModel, which UAVTwin and
ControlledUAV take as an argument. It reuses the authors' configuration, measurement generator and planner.
"""
import sys
from pathlib import Path

# The authors' package `digitaltwin` (UAV-Digital-Twin @ b51e42d, fetched by fetch_upstream.sh) cannot be installed;
# this is the one place it is put on the import path (appended, so its other folders shadow nothing).
_UPSTREAM = Path(__file__).resolve().parent / "upstream" / "UAV-Digital-Twin" / "src" / "digitaltwin"
if not _UPSTREAM.exists():
    raise ImportError("twins.uav needs the authors' code: run twins/uav/fetch_upstream.sh")
if str(_UPSTREAM) not in sys.path:
    sys.path.append(str(_UPSTREAM))

from .model import UAVModel  # noqa: E402
from .sources import ControlledUAV, ScriptedUAV  # noqa: E402
from .twin import UAVTwin  # noqa: E402

__all__ = ["UAVTwin", "ScriptedUAV", "ControlledUAV", "UAVModel"]
