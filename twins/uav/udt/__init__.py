"""Kapteyn UAV twin on dtcore. Reuses the authors' config, measurement generator and planner
(UAV-Digital-Twin @ b51e42d, BSD 3-Clause; ../fetch_upstream.sh); the Bayes filter mirrors
experiments/2026-10-02_noros_repro."""
import sys
from pathlib import Path

_REPO = str(Path(__file__).resolve().parents[1] / "upstream" / "UAV-Digital-Twin" / "src" / "digitaltwin")
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
