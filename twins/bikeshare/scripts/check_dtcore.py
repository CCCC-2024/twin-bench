"""The twin through dtcore.run vs the same functions called directly, in one process.

The twin starts knowing nothing on 2018-01-01 and replays the whole year hour by hour.
  update  its rate estimates must equal the paper's two benchmark columns for every hour where those are
          defined: the twin learned the paper's inputs itself from the observations;
  decide  its overnight decisions on the test days (Nov-Dec), and the lost rentals / returns they lead to
          in the replay, must equal the model-only experiment (sdt.experiment.run_day) computed here.
About 10 s.
"""
import datetime
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sdt import data  # noqa: E402
from sdt import experiment as X  # noqa: E402
from sdt.station import ReplayStation  # noqa: E402
from sdt.twin import StationTwin  # noqa: E402
from dtcore import Recorder, run  # noqa: E402

S = X.load()
C, ev, hourly = S["C"], data.events(426), data.hourly(426)
ok = True

for source, window, col in (("overall_ha", None, "ha"), ("specific_ha", 30, "ha30")):
    ref = {d: X.run_day(S, d, sources=(source,))[source] for d in X.TEST_DAYS}
    station = ReplayStation(C, ev, datetime.date(2018, 1, 1), 365)
    twin = StationTwin(C, window_days=window)
    rec = Recorder(f"bikeshare-{source}", path=os.environ.get("DT_DB") if source == "overall_ha" else None)
    n = run(station, twin, recorder=rec)
    rec.close()

    h = hourly.dropna(subset=[f"{col}_pickup"])
    est = np.array([twin.rate(d, hr)[0] for d, hr in zip(h["date"], h["hour"])])
    diff = np.abs(est - h[[f"{col}_pickup", f"{col}_return"]].to_numpy()).max()

    by_t = {r["obs"]["t"]: r for r in rec.rows}
    same_dec = same_cost = 0
    for d in X.TEST_DAYS:
        k0 = (d - datetime.date(2018, 1, 1)).days * 24
        act = by_t[k0 - 1]["action"]                                  # decided at 23:00 the day before
        lost = sum(by_t[k0 + i]["obs"]["lost_rent"] + by_t[k0 + i]["obs"]["lost_return"] for i in range(24))
        same_dec += act is not None and act["set_bikes"] == ref[d]["start"]
        same_cost += lost == ref[d]["lost_rent"] + ref[d]["lost_return"]
    good = diff < 1e-9 and same_dec == same_cost == len(X.TEST_DAYS)
    ok &= good
    print(f"{source:12s} {n} hourly steps | update vs paper column: max diff {diff:.1e} over {len(h)} hours | "
          f"decisions {same_dec}/{len(X.TEST_DAYS)}, lost {same_cost}/{len(X.TEST_DAYS)} same as model only")

print("BIKESHARE DTCORE OK" if ok else "BIKESHARE DTCORE MISMATCH")
