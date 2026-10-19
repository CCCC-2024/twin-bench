# Battery twin

Reproduces the battery twin of Thelen et al. (2023): a particle filter tracks capacity fade of an LFP cell, and when
capacity first drops to 95 % a multi-attribute utility decides at which cycle to retire it. Research question:
**how much data does the twin need before its retirement decision is right?**

## Sources
- Thelen et al. 2023, *A comprehensive review of digital twin — part 2* (Structural and Multidisciplinary Optimization).
- Code and data: acthelen/battery_digital_twin @ `869a1f6` (MIT), fetched by `fetch_upstream.sh`.
- Cells: Severson et al. 2019, *Nature Energy* — 123 LFP cells as packaged in the upstream repository.

## Six elements

| S | O | D | Q | U | R |
|---|---|---|---|---|---|
| true capacity | capacity of each measured cycle | fade-model parameters (log10 a, b), 200 particles | capacity curve, end of life with interval | retirement cycle | utility: total Ah throughput + mean discharge time, 0.5 each |

### Observations and actions

One step is one measured cycle. What `CellReplay` returns, and what the twin uses (a replay: there are no simulated
true values):

| Field | Meaning | A real cell has it? | `BatteryTwin.update` reads it |
|---|---|---|---|
| `t`, `cycle` | the cycle number (`t` = `cycle`) | yes | `cycle` |
| `q` | normalized capacity: the cycle's capacity over the first cycle's | yes | yes |
| `cap_ah` | the cycle's capacity in Ah | yes | no |

Action `{"retire_at": x}`: the cell is retired after cycle x and the replay stops there; the recorded capacities never
change. The twin decides once, the first time capacity is at or below 95 %.

## Files
- `__init__.py`: the public names, `from twins.battery import BatteryTwin, CellReplay` to run the twin, and
  `run_pf`, `decide_paper` for the paper's evaluation.
- `twin.py` — `BatteryTwin`: the particle filter as `update` (from first-prediction time, capacity ≤ 98.5 %), the
  retirement decision with `decide_online` as `decide`, the capacity projection and the what-if on the decision
  weights as `predict`.
- `sources.py` — `CellReplay`: a cell's measured cycles, stopping after `{"retire_at": x}`.
- `data.py` — rebuilds the dataset from the raw CSV files (identical to the authors' pickle, which only loads
  with pandas < 2) and records which cycles were really measured.
- `pf.py` — `ParticleFilter` + power-law fade model, step-by-step identical to the original (same random-number
  order); end of life in closed form instead of the authors' grid search, with the same results at every cycle of the
  example cell and about 100× faster there (both against the grid search in v0.2.0's `scripts/check_repro.py`, timed
  2026-10-19). `run_pf` runs it over a whole cell, as the authors' batch loop does.
- `decision.py` — the authors' utility and two decisions that know different things: `decide_paper`, the authors'
  evaluation (candidates up to the cell's true end of life, scored against an oracle on the true capacity curve), and
  `decide_online` (candidates up to the twin's predicted end of life), which `BatteryTwin` uses.
- `live/` — real-time version: replayer → MQTT → twin service → SQLite → Streamlit page.
- `scripts/baseline.py` → `results/baseline.csv` (82 test cells × 3 seeds × 5 thresholds);
  `scripts/plot_tradeoff.py` → `results/tradeoff_baseline.png`; `scripts/export_web.py` → the page data;
  `scripts/check_live.py` checks the live pipeline.
- `tests/test_battery.py` (in the repository's `tests/`): the results and notes below marked *(test)*.

## Run
```bash
twins/battery/fetch_upstream.sh
pytest tests/test_battery.py                           # a few seconds
python twins/battery/scripts/baseline.py               # under a minute on 2 cores -> results/baseline.csv
python twins/battery/scripts/plot_tradeoff.py          # -> results/tradeoff_baseline.png, a few seconds
python twins/battery/scripts/export_web.py             # page data and reference, a few seconds
node web/checks/check_battery.js twins/battery/results web    # BATTERY WEB CHECK OK
```

### Live demo
```
replayer ──MQTT──▶ twin service ──▶ SQLite ──▶ Streamlit page (http://localhost:8501)
(measured cycles     (update per message,
 only)                decide)
         ◀──────── retirement decision (digital → physical) ────────
```
```bash
pip install -e ".[twins,live]"
twins/battery/run_demo.sh                      # the paper's example cell (test index 27), 20 cycles per second
python twins/battery/scripts/check_live.py     # LIVE OK: message-by-message states equal the batch run, about 10 s
```
The service decides the first time capacity is ≤ 95 %, using only what it has seen so far. The database defaults to
`results/live.db`; on a network or VM-mounted folder SQLite may fail with "disk I/O error": set `DT_DB=/local/path/live.db`.

## Results
- The paper's example cell (test index 27, Primary Test Cell 28) retires at cycle 542 with end of life 691, as in the
  authors' script *(test)*.
- On `dtcore` the twin's states equal the original batch particle filter at every cycle, its decision equals
  `decide_online` on those states, and the replay stops at the retirement cycle (cells 27, 4 and 24) *(test)*.
- Waiting for more of the cell's own data trades large decision errors for late decisions
  (`results/tradeoff_baseline.png`, from `results/baseline.csv`).
- The page's JavaScript gives the same retirement cycle in all 615 what-if cases of the three page cells
  (`web/checks/check_battery.js`).

## Notes
- The raw data stop near 80 % capacity. The authors extrapolate the last 30 cycles linearly for 3000 cycles to reach
  the 50 % end of life, so for about 21 % of cells the optimal retirement falls in the extrapolated part *(test)*.
- Some cells show a single-cycle capacity dip. About 8.5 % of decisions at the 95 % threshold are triggered by such a
  dip (21 of 246, from 7 cells), and those decisions have a large error (utility loss > 0.05) twice as often: 3 of 21
  against 15 of 225, but all three from one cell, so the ratio is not robust *(test)*. The paper's example
  cell is one: a reading of 0.890 at cycle 249 (neighbours ≈ 0.985) triggers both the start of prediction and the
  decision *(test)*.
- The authors' candidate retirement range runs to the cell's *true* end of life, which an online twin cannot know;
  we use the predicted end of life. At the 95 % threshold this changes 4 of 246 decisions (82 cells × 3 seeds), each
  by one cycle *(test)*.
- The authors' prior (log10 a = −15.77, b = 5.45, std 0.05) is narrow: power laws fitted to every measured cycle of
  the training cells have mean (−12.9, 4.35) and std (3.2, 1.2) *(test)*.
- The model has no usage inputs (charging rate is fixed per cell), so it cannot answer "what if we use it differently".
- Open: `live/twin_service.py` runs `ParticleFilter` and `decide_online` itself rather than `BatteryTwin` through
  `dtcore.run`, a second copy of the twin's loop; it stays until the live demo is redesigned.
