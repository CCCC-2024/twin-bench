"""Cross-check our model against the upstream implementation (their code and solver), same data.

Needs a full clone of the upstream repo and its model package:
  git clone https://github.com/INTO-CPS-Association/example_digital-twin_incubator /some/dir/inc
  git -C /some/dir/inc checkout 0a851c027587c96664ff319d82fff8e8d68cf80d
  pip install oomodelling pika pandas scipy
  (pyhocon may fail to build; the model code does not use it, a stub module with ConfigFactory is enough)
Run step1a_model.py first (it saves results/step1a_ours_T.npy), then:
  python scripts/crosscheck_upstream.py /some/dir/inc/software
Result on 2026-10-17: max |ours - upstream| = 0.0034 °C, mean 0.0011 °C over 2216 samples.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, sys.argv[1])
from idt.data import DATA, FILES  # noqa: E402
from incubator.models.plant_models.model_functions import run_experiment_four_parameter_model  # noqa: E402
from incubator.models.plant_models.four_parameters_model.best_parameters import four_param_model_params as P  # noqa: E402

df = pd.read_csv(DATA / FILES["calibration_2023"])
df["time"] = df["time"] / 1e9                      # ns -> s, as upstream load_data(convert_to_seconds=True)
df = df.rename(columns={"t3": "T_room"})
df["heater_on"] = df["heater_on"].astype(str).str.lower().eq("true")
m, sol = run_experiment_four_parameter_model(df, P)  # upstream default step h = 3 s
T_up = sol.y[list(m.state_names()).index("T"), :]
ours = np.load(ROOT / "results" / "step1a_ours_T.npy")
y = df["average_temperature"].to_numpy()
print("upstream RMSE vs data %.4f" % np.sqrt(np.mean((T_up - y) ** 2)))
print("ours     RMSE vs data %.4f" % np.sqrt(np.mean((ours - y) ** 2)))
diff = np.abs(ours - T_up)
print("ours vs upstream: max %.4f °C, mean %.4f °C" % (diff.max(), diff.mean()))
print("CROSSCHECK OK" if diff.max() < 0.01 else "CROSSCHECK FAILED")
