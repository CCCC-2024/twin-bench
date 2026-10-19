# Incubator twin

A twin built from scratch from a paper's description: the Aarhus University incubator, a small insulated box with a
heater and a fan, the standard introductory case in digital-twin engineering. The model is our own, checked against
public real experiments; the upstream code is used only for comparison.

## Sources
- Description: Feng et al. 2021, *The Incubator Case Study for Digital Twin Engineering* (arXiv 2102.10390).
- Data and published parameters: INTO-CPS-Association/example_digital-twin_incubator @ `0a851c0`.
  The repository has no license file, so `fetch_upstream.sh` downloads only the five data files used, and nothing
  that reproduces the data is committed or published (`web/incubator_replay.js` and
  `results/ref_web_incubator_replay.json` are local only).

## Six elements

| S | O | D | Q | U | R |
|---|---|---|---|---|---|
| air temperature, heater-plate temperature | 2 box + 1 room sensors every 3 s; heater on/off | estimated temperatures + 4 thermal parameters | future temperature curve | heater on / off (set point) | stay in the target band, use little energy (ours; the paper gives none) |

### Observations and actions

One step is one sample, about 3 s. What a Source returns, and what the twin uses:

| Field | Meaning | A real box has it? | `IncubatorTwin.update` reads it |
|---|---|---|---|
| `t` | seconds since the start (SimBox: steps of 3 s; RecordedBox: the recorded time stamps) | yes | yes: the time between two samples drives the Kalman prediction |
| `T` | box air temperature, the mean of the two box sensors | yes | yes |
| `T_room` | room temperature (sensor `t3`) | yes | yes, from the previous sample |
| `heater_on` | heater on (1) or off (0) | yes | yes, from the previous sample |
| `t1`, `t2` (and `t3` from SimBox) | the sensors themselves | yes | no |
| `true_T`, `true_T_heater`, `lid_open` | the simulated box's true temperatures and lid (SimBox only) | no | no; for scoring only |

Action on `SimBox`: `{"desired": x}` (or `"lower_bound"`) changes the set point of the box's controller, which runs on
the box. `RecordedBox` only records actions: the experiment already happened. The twin has no `decide`: a person
chooses the set point with the what-if, `predict(horizon_s, controller, desired=x)`, which needs the box's controller.

## Files
- `__init__.py`: the public names, `from twins.incubator import IncubatorTwin, SimBox, RecordedBox, Controller`.
- `twin.py` — `IncubatorTwin`: `update` (Kalman step and anomaly flag) and `predict` (what-if on the set point).
- `sources.py` — `SimBox`, a simulated box: same model + sensor noise and 0.0625 °C quantisation; `RecordedBox`, a
  replay of a real experiment.
- `model.py` model (linear, discretised exactly); `kalman.py`; `controller.py` (as in the dataset notes: set point
  38 °C, lower band 2 °C); `data.py` loads the experiments.
- `scripts/step1_model.py`: the model alone against the 2023 data, identifiability and our own calibration
  (`results/step1_*`).
- `scripts/export_web.py`: the page's reference run (`results/ref_web_incubator.json`) and, locally only, the real
  experiment for the page (`web/incubator_replay.js`, `results/ref_web_incubator_replay.json`).
- `tests/test_incubator.py` (in the repository's `tests/`): the results below marked *(test)*.

## Run
```bash
twins/incubator/fetch_upstream.sh
pytest tests/test_incubator.py                       # under a second
python twins/incubator/scripts/step1_model.py        # a few seconds; figures in results/
python twins/incubator/scripts/export_web.py         # page references
node web/checks/check_incubator.js twins/incubator/results web    # WEB CHECK OK
```

## Results
On the 2023 empty-box experiment (1.85 h, 2216 samples) unless noted.

- Given only the heater state and room temperature, the model's box temperature has RMSE 0.46 °C, max error
  1.31 °C (`results/step1_model.png`) *(test)*.
- Against the upstream implementation the largest difference is 0.0034 °C: the model written from the description
  matches the authors'. Compared once (2026-10-17) with the authors' code and solver, which need a full clone of
  their repository; the script is `twins/incubator/scripts/crosscheck_upstream.py` in v0.2.0.
- With only the box temperature measured, the heater parameters are not identifiable: fixing C_heater anywhere in
  50–329 J/K and fitting the rest gives 0.41–0.44 °C error, while the inferred peak plate temperature ranges 47–70 °C
  (`results/step1_identifiability.png`). Box loss G_box comes out 0.576 every time. The twin therefore uses the
  published parameters. Fixing C_heater at the published value and calibrating from a wrong start: C_air −7 %,
  G_box −0.1 %, G_heater +1.6 %, RMSE 0.437.
- The Kalman filter raises an alarm 9 s after the lid opens, none before, and clears it within 2 minutes of the lid
  closing; a new set point sent later raises none (closed loop on the simulated box) *(test)*. On the real empty-box
  experiment it raises none *(test)*.
- Our controller, driven by the recorded box temperature, agrees with the recorded heater state 97.7 % of the time
  from the first recorded heating on; every one of the 17 switches appears in the recording exactly 3 samples (about
  9 s) after ours *(test)*. Before that (31 samples) the recording shows a cold box with the heater off: the box's
  controller had not started.

## Notes
- 2023 hardware: t1 and t2 are inside the box, t3 measures the room. In the 2020–2021 datasets t1 is the room; do not mix them.
- The hardware list says 11–13 A heater current; the 2023 measurements show about 1.5 A. Parameters use the measured value.
- Lid-open heat loss (14.39 W/K) comes from the upstream calibration on the older 2021 hardware and is used only to simulate an open lid.
- "Box and twin disagree" scenario: the simulated box uses another parameter set that fits the data equally well
  (C_heater 100 J/K). The box temperature is still tracked; the heater-plate estimate drifts.
- On the tempeh experiment (a batch of fermenting beans in the box; set point lowered to 30 °C after 11 h) the alarm is
  on for 49 % of the samples *(test)*. The model has no load and no heat from the fungus, so this is what the alarm is
  for ("something in the box"), not a false alarm; a twin for loaded runs would need those terms.
