# Template twin: a mug with a heater

The smallest complete twin, and the folder to copy when starting a new one (see
[docs/adding-a-twin.md](../../docs/adding-a-twin.md)). A mug of water cools towards room temperature; a heater can
warm it. The twin estimates the temperature from a noisy sensor and switches the heater to hold 60 °C.

## Six elements

| S | O | D | Q | U | R |
|---|---|---|---|---|---|
| water temperature | noisy thermometer every 10 s; heater on/off | temperature estimate and its variance | temperature over the next minutes | heater on / off | stay near 60 °C |

## Files
- `mdt/model.py` — Newton's law of cooling with a heater, one exact step for a given dt.
- `mdt/twin.py` — `SimMug` (the physical side: a simulated mug with sensor noise) and `MugTwin` (scalar Kalman
  filter as `update`, model roll-out as `predict` and what-if, a heater rule as `decide`).
- `scripts/check_dtcore.py` — one hour through `dtcore.run`; checks the estimate beats the raw sensor and the mug is
  held within 2 K of the target.

## Run
```bash
python twins/_template/scripts/check_dtcore.py    # TEMPLATE OK
```

## Making it yours
Replace the model, the Source (a replay of real data, a simulator or a live feed) and the Twin's methods; keep the
structure. Add `fetch_upstream.sh` when you use third-party code or data, `results/` and `experiments/` when you
produce outputs, and rewrite this README with the sections in [docs/conventions.md](../../docs/conventions.md#4-readme-structure).
