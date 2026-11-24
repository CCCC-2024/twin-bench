"""The tumor twin through dtcore: the patient's MRI visits arrive one by one, and the twin calibrates at each.

For every visit: the cell count measured from the MRI, the count the twin expected before seeing it, and the
parameters after calibrating on every MRI so far. Then, with the parameters after the last visit run, the what-if
over the total chemotherapy dose up to the patient's last visit (none when every visit was run). Writes
results/<patient>/twin_run_<visits>.json and .png.

With --steps other than 20 the file names end in _steps<S>.

Usage: python twins/tumor/scripts/run_twin.py [--visits N] [--steps S]   (from the repository root; all 10 visits
       take about 35 minutes on a laptop CPU, the first 5 about 13)
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                         # noqa: E402

from dtcore import Recorder, run                                        # noqa: E402
from twins.tumor import Glioma, PatientReplay, TumorTwin                 # noqa: E402
from twins.tumor.model import PARAMS                                    # noqa: E402

PATIENT = "HGG_demo_001"
OUT = Path(__file__).resolve().parents[1] / "results" / PATIENT

parser = argparse.ArgumentParser(description="The tumor twin on one patient's MRI visits.")
parser.add_argument("--visits", type=int, help="stop after this many visits (default: all)")
parser.add_argument("--steps", type=int, default=20, help="Levenberg-Marquardt steps at each visit (default 20)")
args = parser.parse_args()
TRUE = dict(zip(PARAMS, (0.1, 0.05, 0.05, 0.2)))                        # stated in the authors' tutorial
DOSES = (30, 40, 50, 60, 70)

start = time.time()
glioma = Glioma(PATIENT)
twin, recorder = TumorTwin(glioma, steps=args.steps), Recorder("tumor")
run(PatientReplay(glioma), twin, max_steps=args.visits, recorder=recorder)
visits = [dict(day=r["obs"]["t"], measured=r["obs"]["cell_count"], expected=r["est"]["expected_cell_count"],
               **{p: r["est"][p] for p in PARAMS}) for r in recorder.rows]
last_day = glioma.day(glioma.dates[-1])
ahead = twin.predict(last_day - twin.day)                               # to the patient's last visit
whatif = {dose: twin.whatif(last_day - twin.day, chemo_total=dose)["cell_count"] for dose in DOSES} \
    if twin.day < last_day else {}
name = f"twin_run_{twin.visits}" + (f"_steps{args.steps}" if args.steps != 20 else "")
(OUT / f"{name}.json").write_text(json.dumps(dict(
    patient=PATIENT, steps=args.steps, true_params=TRUE, visits=visits, forecast=ahead,
    later_visits=[dict(day=glioma.day(t), measured=glioma.cell_count(m))
                  for t, m in zip(glioma.dates[twin.visits:], glioma.maps[twin.visits:])],
    whatif_chemo_total=whatif, runtime_s=round(time.time() - start)), indent=1) + "\n")

fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
a.plot(ahead["day"], ahead["cell_count"], color="tab:blue", label=f"twin, after visit {twin.visits}")
a.plot([glioma.day(t) for t in glioma.dates], [glioma.cell_count(m) for m in glioma.maps], "ko", label="MRI")
seen = [v for v in visits if v["expected"] is not None]
a.plot([v["day"] for v in seen], [v["expected"] for v in seen], "x", color="tab:red",
       label="expected before the visit")
a.axvline(twin.day, color="gray", linestyle=":")
a.set(xlabel="days since the first MRI", ylabel="tumor cells", title="Tumor cell count")
a.legend()
for p in PARAMS:
    b.plot([v["day"] for v in visits], [v[p] / TRUE[p] for v in visits], marker="o", label=p)
b.axhline(1, color="gray", linestyle=":")
b.set(xlabel="days since the first MRI", ylabel="estimate / true value", title="Parameters after each visit")
b.legend()
fig.tight_layout()
fig.savefig(OUT / f"{name}.png", dpi=150)
print(f"wrote {OUT / name}.json and .png in {time.time() - start:.0f} s")
