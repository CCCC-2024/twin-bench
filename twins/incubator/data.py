"""Load the public incubator datasets (run ../fetch_upstream.sh first).

2023 hardware: t1 and t2 are inside the box, t3 measures the room.
average_temperature in the files = mean(t1, t2); this is the observed box temperature T.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = (Path(__file__).resolve().parent / "upstream" / "example_digital-twin_incubator"
        / "software" / "incubator" / "datasets")
FILES = {
    "calibration_2023": "20230501_calibration_empty_system/20230501_calibration_empty_system.csv",
    "tempeh_2023": "20230501_tempeh_batch/rec_2023-05-01__08_09_06.csv",
}


def load(name):
    df = pd.read_csv(DATA / FILES[name])
    t_ns = df["time"].to_numpy()
    out = pd.DataFrame({
        "t": (t_ns - t_ns[0]) / 1e9,                     # seconds from start
        "time_ns": t_ns,
        "T": df["average_temperature"].to_numpy(),        # box (observed)
        "T_room": df["t3"].to_numpy(),
        "heater_on": df["heater_on"].astype(str).str.lower().eq("true").astype(float).to_numpy(),
        "t1": df["t1"].to_numpy(), "t2": df["t2"].to_numpy(),
    })
    assert np.allclose(out["T"], (df["t1"] + df["t2"]) / 2, atol=1e-3), "sensor mapping changed"
    return out
