"""Tumor twin: TumorTwin (Kapteyn et al. 2026) on dtcore (see README.md).

Public names: TumorTwin (digital side), PatientReplay (physical side: one patient's MRI visits in order), and Glioma,
the authors' model of that patient, which both take as an argument. It needs the authors' package in its own
environment: README.md, "Setup".
"""
from pathlib import Path

if not (Path(__file__).resolve().parent / "upstream" / "TumorTwin" / "input_files").exists():
    raise ImportError("twins.tumor needs the authors' code and data: run twins/tumor/fetch_upstream.sh")

from .model import Glioma  # noqa: E402
from .sources import PatientReplay  # noqa: E402
from .twin import TumorTwin  # noqa: E402

__all__ = ["TumorTwin", "PatientReplay", "Glioma"]
