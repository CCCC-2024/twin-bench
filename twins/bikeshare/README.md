# Bike-share station twin

One Citi Bike station (West St & Chambers St, 30 docks). Reproduces the overnight inventory decision of
Gammelli et al. 2022 (the queueing model of Raviv & Kolka 2013) and runs it as a twin on dtcore: it learns hourly
rental and return rates from the record, forecasts an empty or full station, sets the number of bikes each night,
and answers "what if a truck sets s bikes now".

## Sources
- Raviv & Kolka 2013, *Optimal inventory management of a bike-sharing station*, IISE Transactions 45(10),
  1077–1093: the station model and the user dissatisfaction function (expected shortages).
- Gammelli, Wang, Prak, Rodrigues, Minner & Pereira 2022, *Predictive and prescriptive performance of bike-sharing
  demand forecasts for inventory management*, Transportation Research Part C 138, 103571: the experiment reproduced
  here. Code: [DanieleGammelli/variational-poisson-rnn](https://github.com/DanieleGammelli/variational-poisson-rnn)
  @ `abf77f7` (MIT): the decision code, per-station events and hourly counts, the authors' saved forecasts.
- Data: Citi Bike System Data 2018 (NYC Bike Share), as aggregated per station in that repository. The NYCBS Data
  Use Policy forbids redistribution as a stand-alone dataset, so `fetch_upstream.sh` downloads it and nothing that
  reproduces it is committed.

## Six elements

| S | O | D | Q | U | R |
|---|---|---|---|---|---|
| bikes in the station (0–30); true hourly rental and return demand (not observed) | rentals and returns recorded each hour; bikes now | bikes now; a Gamma posterior over rental and return rates for each hour × weekday / weekend | distribution of bikes over the next hours, chance of empty / full, expected lost rentals and returns | each night: the number of bikes for the next day (automatic, as in the paper); a daytime truck setting s bikes (what-if, a person decides) | fewest expected lost rentals + lost returns (weight 1 each, as in the paper) |

Update: Gamma–Poisson conjugate update (add each hour's counts). With a flat prior the posterior mean is exactly the
paper's two benchmarks (all earlier days, or the last 30, same hour and day type), so the twin learns the paper's
inputs from its observations. Physical side: a replay of the recorded events. The demand is fixed by the record but
the number of bikes is ours, so actions change the inventory and the shortages.

### Observations and actions

One step is one hour. What a Source returns, and what the twin uses:

| Field | Meaning | A real station has it? | `StationTwin.update` reads it |
|---|---|---|---|
| `t` | hour index since the Source's start date (0, 1, 2 …) | — | no (passed through) |
| `date`, `hour`, `weekday` | the calendar hour | yes | yes |
| `pickups`, `returns` | rentals and returns in that hour: the recorded (or simulated) demand, including those our station could not serve | only the served ones | yes |
| `bikes` | bikes in the station at the end of the hour | yes | yes |
| `lost_rent`, `lost_return` | rentals and returns that failed at our station in that hour | no: people who leave are not recorded | no; for scoring only |

Action `{"set_bikes": s}`: a truck brings the station to s bikes (clipped to 0–30). The demand stays as recorded.
`decide()` also returns `expected_lost`, the twin's own forecast; the Source ignores it.

## Files
- `__init__.py`: the public names, `from twins.bikeshare import StationTwin, ReplayStation, SimStation`.
- `twin.py`: `StationTwin` with update, predict / what-if and decide.
- `sources.py`: `ReplayStation` (recorded events, hour by hour) and `SimStation` (Poisson demand).
- `model.py`: the number of bikes as a birth–death chain; one hour is a matrix exponential, and the time spent
  empty / full comes from the same exponential (Van Loan); expected shortages for every start; replay of events.
- `data.py` reads the upstream files; `experiment.py` is the paper's set-up (Nov–Dec 2018, five rate sources).
- `scripts/step1_model.py`: model only, the paper's table and a Monte Carlo check of the model;
  `scripts/step5_whatif.py`: the daytime-truck what-if.
- `scripts/export_web.py` writes the page data (`web/bikeshare_model.js`, and the local-only `web/bikeshare_replay.js`) and
  the reference run `results/ref_web_bikeshare.json` that `web/checks/check_bikeshare.js` replays.
- `results/ref_upstream.json`: the authors' decision code run once on the same inputs (about 10 min;
  `twins/bikeshare/scripts/crosscheck_upstream.py` in v0.2.0).
- `tests/test_bikeshare.py` (in the repository's `tests/`): the learned rates equal the paper's benchmarks, the
  nightly decisions equal the paper's experiment, and the decisions equal the authors' code except one numerical tie.

## Run
```bash
twins/bikeshare/fetch_upstream.sh                    # about 11 MB
pytest tests/test_bikeshare.py                       # what the twin reproduces of the paper, a few seconds
python twins/bikeshare/scripts/step1_model.py        # model only -> results/step1_*, a few seconds
python twins/bikeshare/scripts/step5_whatif.py       # daytime-truck what-if -> results/step5_*, a few seconds
python twins/bikeshare/scripts/export_web.py         # page data and reference run, about 5 s
node web/checks/check_bikeshare.js twins/bikeshare/results web    # BIKESHARE WEB CHECK OK
```

## Results
Station 426, 1 Nov – 31 Dec 2018 (61 days), one decision per night (`results/step1_decisions.json`):

| Rates from | Start (median) | Twin expects lost / day | Replay: lost / day |
|---|---|---|---|
| all-history average (paper benchmark) | 2 | 42.8 | 8.4 |
| last-30-days average (paper benchmark) | 7 | 24.5 | 7.8 |
| VP-RNN (paper's model, authors' saved forecasts) | 12 | 15.6 | 6.8 |
| MOVP-RNN (multi-output version) | 15 | 9.4 | 7.1 |
| the day's true counts (perfect forecast) | 11 | 13.4 | 5.1 |
| hindsight best (knows every event of the day) | 5 | — | 4.7 |

- Against the authors' code: 365 of 366 decisions identical, with identical replayed shortages. The other one
  (2 Nov, VP-RNN) is a numerical tie: its two options differ by 4e-6 relative, below the upstream Runge–Kutta error
  (up to 9.2e-4). The model also agrees with a Monte Carlo simulation of itself within 0.3 standard errors.
- On dtcore, starting from nothing on 1 Jan, the twin's rate estimates equal the paper's benchmark columns (largest
  difference 7e-15 over 8,712 hours), and its 61 overnight decisions and replayed shortages equal the model-only run.
- Learned forecasts: the RNN forecasts lose 1.66 ± 0.76 fewer per day than the all-history average (paired, 61 days).
  MOVP-RNN forecasts more accurately (test RMSE 3.66 vs 3.82 for rentals, 3.57 vs 3.79 for returns, from the authors'
  saved results) but does not decide better here (+0.34 ± 0.42 per day). One station is consistent with the paper's
  conclusion that better forecasts need not give better decisions; it cannot confirm it.
- What-if (`results/step5_whatif.png`): on 2 Nov, the worst day for the overnight decision, the station holds 14 bikes
  at 06:59. A truck setting 1 bike: predicted 65.7 → 54.0 lost for the rest of the day; replay 99 → 86.
- Page (Twin Bench, `web/`): a simulated station online, the recorded Nov–Dec days in the local copy. The JavaScript
  matches Python over 72 simulated hours (decisions, station states and 10–90 % bands identical; numbers within 1.3e-11).

## Notes
- The twin's expected shortages and the replay disagree a lot, in both directions (42.8 vs 8.4 per day on average;
  66 vs 99 on 2 Nov after 07:00). The model assumes independent Poisson rentals and returns at average rates; the
  replay holds only trips that happened (the operator's rebalancing trips are removed from the data, and returns that
  went elsewhere when the station was full are missing). Without the station's recorded inventory it is not possible
  to say which is closer. Differences between actions agree better than absolute levels.
- Day-to-day variation is much larger than Poisson (variance 4–11 times the mean in the hours checked) and rentals
  and returns in the same hour are correlated (0.7–0.9). The posterior sd reflects estimation error only.
- Censored demand: people who find the station empty or full leave no record, so the estimated rates are below the
  true demand. Neither the paper nor this twin corrects for it.
- The twin's all-history rates, learned from Jan–Oct, are about twice the Nov–Dec demand (weekday rentals 292 vs 139 a
  day), and each new day moves them by about 1/220, so the twin adapts slowly. The page's simulated station uses the
  Nov–Dec average, so the gap shows there.
- The capacity (30) comes from the station table in the upstream repository; its date is unknown.
- Where a learned forecast would plug in: every rate the twin uses comes from `StationTwin.rate(day, hour)` (through
  `rates_ahead`, used by `predict` and `decide`). The paper's RNN forecasts are read by `data.rnn_rates` and used only
  in `experiment.py`; they are not yet a rate source of the twin.
- "Twin expects" is the twin's forecast with its own rates. The paper's evaluation code instead computes expected cost
  with the day's true counts.
