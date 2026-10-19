# Template twin: a mug with a heater

The smallest complete twin, and the folder to copy when starting a new one (see
[docs/adding-a-twin.md](../../docs/adding-a-twin.md)). A mug of water cools towards room temperature; a heater can
warm it. The twin estimates the temperature from a noisy sensor and switches the heater to hold 60 °C.

## Six elements

| S | O | D | Q | U | R |
|---|---|---|---|---|---|
| water temperature | noisy thermometer every 10 s; heater on/off | temperature estimate and its variance | temperature over the next minutes | heater on / off | stay near 60 °C |

### Observations and actions

One step is 10 s. What `SimMug` returns, and what the twin uses:

| Field | Meaning | A real mug has it? | `MugTwin.update` reads it |
|---|---|---|---|
| `t` | seconds since the start | yes | no (passed through) |
| `T` | the thermometer reading, °C | yes | yes |
| `heater_on` | heater on (1) or off (0) during the last 10 s | yes | yes |
| `true_T` | the simulated true temperature | no | no; for scoring only |

Action `{"heater_on": 0 or 1}`: switches the heater for the next 10 s.

## Files
- `__init__.py` — the public names, `MugTwin` and `SimMug`: `from twins._template import MugTwin, SimMug`.
- `model.py` — Newton's law of cooling with a heater, one exact step for a given dt.
- `sources.py` — `SimMug`, the physical side: a simulated mug with sensor noise.
- `twin.py` — `MugTwin`: scalar Kalman filter as `update`, model roll-out as `predict` and what-if, a heater rule as
  `decide`.
- `tests/test_template.py` (in the repository's `tests/`) — one hour through `dtcore.run`: the estimate beats the raw
  sensor, the mug is held within 2 K of the target, and a prediction starts from the current estimate.

## Run
```bash
pytest tests/test_template.py    # 3 passed, under a second
```

## Making it yours
Replace the model, the Source (a replay of real data, a simulator or a live feed) and the Twin's methods; keep the
structure. Add `fetch_upstream.sh` when you use third-party code or data, `results/` and `experiments/` when you
produce outputs, and rewrite this README with the sections in [docs/conventions.md](../../docs/conventions.md#4-readme-structure).
