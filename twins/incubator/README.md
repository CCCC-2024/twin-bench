# Incubator twin

A twin built from scratch from a paper's description: the Aarhus University incubator, a small insulated box with a
heater and a fan, the standard introductory case in digital-twin engineering. The model is our own, checked against
public real experiments; the upstream code is used only for comparison.

## Sources
- Description: Feng et al. 2021, *The Incubator Case Study for Digital Twin Engineering* (arXiv 2102.10390).
- Data and published parameters: INTO-CPS-Association/example_digital-twin_incubator @ `0a851c0`.
  The repository has no license file, so `fetch_upstream.sh` downloads only the five data files used, and nothing
  that reproduces the data is committed or published (`web/replay_2023.js` and `results/ref_replay.json` are local only).

## Six elements

| S | O | D | Q | U | R |
|---|---|---|---|---|---|
| air temperature, heater-plate temperature | 2 box + 1 room sensors every 3 s; heater on/off | estimated temperatures + 4 thermal parameters | future temperature curve | heater on / off (set point) | stay in the target band, use little energy (ours; the paper gives none) |

## Files
- `idt/model.py` model (linear, discretised exactly); `idt/kalman.py`; `idt/controller.py` (as in the dataset notes:
  set point 38 °C, lower band 2 °C); `idt/data.py` loads the experiments.
- `idt/plant_sim.py` — `SimBox`, a simulated box: same model + sensor noise and 0.0625 °C quantisation.
  `idt/replay.py` — `RecordedBox`, a replay of a real experiment. Both implement `observe / act / advance`.
- `idt/twin.py` — `IncubatorTwin`: `update` and `predict`; no automatic `decide` (a person decides, using the what-if).
- `scripts/step1a_model.py` the model alone against the 2023 data; `scripts/crosscheck_upstream.py` point-by-point
  against the upstream implementation; `scripts/check_dtcore.py`; `scripts/export_reference.py` references for the page.

## Run
```bash
twins/incubator/fetch_upstream.sh
python twins/incubator/scripts/step1a_model.py        # ~15 s; figures in results/
python twins/incubator/scripts/check_dtcore.py        # DTCORE OK
```

## Results (2023 empty-box experiment, 1.85 h, 2216 samples)
- Given only the heater state and room temperature, the model's box temperature has RMSE 0.46 °C, max error
  1.31 °C (`results/step1a_model.png`).
- Against the upstream implementation the largest difference is 0.0034 °C: the model written from the description
  matches the authors'.
- With only the box temperature measured, the heater parameters are not identifiable: fixing C_heater anywhere in
  50–329 J/K and fitting the rest gives 0.41–0.44 °C error, while the inferred peak plate temperature ranges 47–70 °C
  (`results/step1a_identifiability.png`). Box loss G_box comes out 0.576 every time. The twin therefore uses the
  published parameters. Fixing C_heater at the published value and calibrating from a wrong start: C_air −7 %,
  G_box −0.1 %, G_heater +1.6 %, RMSE 0.437.
- The Kalman filter raises an alarm within 9 s of the lid opening, with no false alarms on the real data or in
  simulation. The controller agrees with the recorded heater state 97.7 % of the time; all differences are at
  switching instants, where the recording lags 2–3 samples.

## Notes
- 2023 hardware: t1 and t2 are inside the box, t3 measures the room. In the 2020–2021 datasets t1 is the room; do not mix them.
- The hardware list says 11–13 A heater current; the 2023 measurements show about 1.5 A. Parameters use the measured value.
- Lid-open heat loss (14.39 W/K) comes from the upstream calibration on the older 2021 hardware and is used only to simulate an open lid.
- "Box and twin disagree" scenario: the simulated box uses another parameter set that fits the data equally well
  (C_heater 100 J/K). The box temperature is still tracked; the heater-plate estimate drifts.
