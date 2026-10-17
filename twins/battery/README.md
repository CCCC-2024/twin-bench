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

## Files
- `bdt/data.py` — rebuilds the dataset from the raw CSV files (identical to the authors' pickle, which only loads
  with pandas < 2) and records which cycles were really measured.
- `bdt/pf.py` — particle filter + power-law fade model, step-by-step identical to the original (same random-number
  order); end of life in closed form instead of a grid search, about 40× faster.
- `bdt/decision.py` — the authors' utility; also an oracle decision on the true curve, the difference being the decision error.
- `bdt/dt.py` — the dtcore adapter (~90 lines, the only addition): `CellReplay` replays a cell's measured cycles and
  stops at cycle x after `{"retire_at": x}`; `BatteryDT` wraps the particle filter and `decide_online`.
- `live/` — real-time version: replayer → MQTT → twin service → SQLite → Streamlit page.
- `scripts/` — `check_repro.py`, `check_dtcore.py`, `check_live.py`, `baseline.py`, `plot_tradeoff.py`, `export_web.py`.

## Run
```bash
twins/battery/fetch_upstream.sh
python twins/battery/scripts/check_repro.py            # ~1 min; REPRO OK (example cell retires at cycle 542, as in the paper)
python twins/battery/scripts/check_dtcore.py 27 4 24   # BATTERY DTCORE OK
python twins/battery/scripts/baseline.py               # 82 test cells × 3 seeds, ~5 min on 2 cores -> results/baseline.csv
python twins/battery/scripts/plot_tradeoff.py          # -> results/tradeoff_baseline.png
python twins/battery/scripts/export_web.py             # -> web/battery_cells.js, results/ref_web_battery.json
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
python twins/battery/scripts/check_live.py     # LIVE OK: message-by-message states equal the batch run
```
The service decides the first time capacity is ≤ 95 %, using only what it has seen so far. The database defaults to
`results/live.db`; on a network or VM-mounted folder SQLite may fail with "disk I/O error": set `DT_DB=/local/path/live.db`.

## Notes
- The raw data stop near 80 % capacity. The authors extrapolate the last 30 cycles linearly for 3000 cycles to reach
  the 50 % end of life, so for about 21 % of cells the optimal retirement falls in the extrapolated part.
- Some cells show a single-cycle capacity dip. About 8.5 % of decisions are triggered by such a dip at the 95 %
  threshold, and those decisions are wrong about twice as often. The paper's example cell is one: a reading of
  0.890 at cycle 249 (neighbours ≈ 0.985) triggers both the start of prediction and the decision.
- The authors' candidate retirement range runs to the cell's *true* end of life, which an online twin cannot know;
  we use the predicted end of life. This changes one decision out of 82, by one cycle.
- The authors' prior (log10 a = −15.77, b = 5.45, std 0.05) is narrow; parameters fitted on the training cells have
  mean (−12.9, 4.35) and std (3.2, 1.2).
- The model has no usage inputs (charging rate is fixed per cell), so it cannot answer "what if we use it differently".
