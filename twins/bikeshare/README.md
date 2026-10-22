# Bike-share station twin

One Citi Bike station (West St & Chambers St, 30 docks). Reproduces the overnight inventory decision of
Gammelli et al. 2022 (the queueing model of Raviv & Kolka 2013) and runs it as a twin on dtcore: it learns hourly
rental and return rates from the record, forecasts an empty or full station, sets the number of bikes each night,
and answers "what if a truck sets s bikes now". Its rate model can be swapped: the paper's trained RNN (VP-RNN)
runs in the twin too, forecasting from what the twin has seen. So can its nightly decision: an LLM agent chooses by
asking the twin's what-if.

## Sources
- Raviv & Kolka 2013, *Optimal inventory management of a bike-sharing station*, IISE Transactions 45(10),
  1077–1093: the station model and the user dissatisfaction function (expected shortages).
- Gammelli, Wang, Prak, Rodrigues, Minner & Pereira 2022, *Predictive and prescriptive performance of bike-sharing
  demand forecasts for inventory management*, Transportation Research Part C 138, 103571: the experiment reproduced
  here. Code: [DanieleGammelli/variational-poisson-rnn](https://github.com/DanieleGammelli/variational-poisson-rnn)
  @ `abf77f7` (MIT): the decision code, per-station events and hourly counts, the authors' saved forecasts, and
  their trained VP-RNN for this station with the hourly weather it reads (origin not stated).
- Data: Citi Bike System Data 2018 (NYC Bike Share), as aggregated per station in that repository. The NYCBS Data
  Use Policy forbids redistribution as a stand-alone dataset, so `fetch_upstream.sh` downloads it and nothing that
  reproduces it is committed.

## Six elements

| S | O | D | Q | U | R |
|---|---|---|---|---|---|
| bikes in the station (0–30); true hourly rental and return demand (not observed) | rentals and returns recorded each hour; bikes now | bikes now; the rate model: a Gamma posterior over rental and return rates for each hour × weekday / weekend (default), or the VP-RNN's state | distribution of bikes over the next hours, chance of empty / full, expected lost rentals and returns | each night: the number of bikes for the next day (automatic: the paper's rule by default, or an LLM agent asking the what-if); a daytime truck setting s bikes (what-if, a person decides) | fewest expected lost rentals + lost returns (weight 1 each, as in the paper) |

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
`decide()` also returns `expected_lost`, the twin's own forecast; the Source ignores it. With the LLM agent as the
policy the action also records the scenarios it ran on the what-if (`asked`) and its `reason`; when the model gives
no valid choice the action has no `set_bikes` and the reason says why: no truck goes out.

## Files
- `__init__.py`: the public names: `StationTwin`, its rate models `HistoricalRates` and `RNNRates`, its policies
  `RulePolicy` and `AgentPolicy`, `ReplayStation`, `SimStation`, `experiment`.
- `twin.py`: `StationTwin` with update, predict / what-if and decide, and `brief` / `whatif` for the LLM agent; it
  takes its rate model and its policy as arguments: `RulePolicy` (the paper's decision, the default) or `AgentPolicy`
  (the LLM agent of `dtcore.agent`).
- `rates.py`: rate models, what the twin knows about demand: `HistoricalRates` (Gamma posterior per hour and day
  type, the default); `rnn.py`: `RNNRates`, the paper's trained VP-RNN (needs the `rnn` extra, PyTorch).
- `sources.py`: `ReplayStation` (recorded events, hour by hour) and `SimStation` (Poisson demand).
- `model.py`: the number of bikes as a birth–death chain; one hour is a matrix exponential, and the time spent
  empty / full comes from the same exponential (Van Loan); expected shortages for every start; replay of events.
- `data.py` reads the upstream files; `experiment.py` is the paper's set-up (Nov–Dec 2018, five rate sources).
- `scripts/step1_model.py`: model only, the paper's table and a Monte Carlo check of the model;
  `scripts/step5_whatif.py`: the daytime-truck what-if;
  `scripts/compare_rates.py`: the twin with each rate model on the test days (needs the `rnn` extra);
  `scripts/compare_policy.py`: the twin with the rule and with the LLM agent on the test days (needs a model
  server).
- `scripts/export_web.py` writes the page data (`web/bikeshare_model.js`, and the local-only `web/bikeshare_replay.js`) and
  the reference run `results/ref_web_bikeshare.json` that `web/checks/check_bikeshare.js` replays.
- `scripts/crosscheck_upstream.py` runs the authors' decision code on the same inputs and saves its answers in
  `results/ref_upstream.json`, which the tests compare against.
- `tests/test_bikeshare.py` (in the repository's `tests/`): the results and notes below marked *(test)*.

## Run
```bash
twins/bikeshare/fetch_upstream.sh                    # about 11 MB
pytest tests/test_bikeshare.py                       # what the twin reproduces of the paper, ~7 s (RNN tests: .[rnn])
python twins/bikeshare/scripts/step1_model.py        # model only -> results/step1_*, a few seconds
python twins/bikeshare/scripts/step5_whatif.py       # daytime-truck what-if -> results/step5_*, a few seconds
python twins/bikeshare/scripts/compare_rates.py      # rate models in the twin -> results/compare_rates.json, ~10 s
python twins/bikeshare/scripts/compare_policy.py MODEL  # rule vs LLM agent -> results/compare_policy.json; MODEL served
                                                     # with the OpenAI chat API (Ollama's by default); 61 nights of it
python twins/bikeshare/scripts/export_web.py         # page data and reference run, ~10 s (--replay-only: local file only)
python twins/bikeshare/scripts/crosscheck_upstream.py  # the authors' code -> results/ref_upstream.json, ~1 min
node web/checks/check_bikeshare.js twins/bikeshare/results web    # BIKESHARE WEB CHECK OK
```

## Results
Station 426, 1 Nov – 31 Dec 2018 (61 days), one decision per night (`results/step1_decisions.json`; rows marked †
from `results/compare_rates.json`):

| Rates from | Start (median) | Twin expects lost / day | Replay: lost / day |
|---|---|---|---|
| all-history average (paper benchmark) | 2 | 42.8 | 8.4 |
| last-30-days average (paper benchmark) | 7 | 24.5 | 7.8 |
| VP-RNN, the paper's protocol, authors' saved forecasts | 12 | 15.6 | 6.8 |
| VP-RNN, the paper's protocol, our run of the authors' saved weights † | 13 | 13.7 | 5.8 |
| VP-RNN in the twin, forecasting the day from what is known at 23:00 † | 17 | 16.2 | 8.8 |
| MOVP-RNN (multi-output version) | 15 | 9.4 | 7.1 |
| the day's true counts (perfect forecast) | 11 | 13.4 | 5.1 |
| hindsight best (knows every event of the day) | 5 | — | 4.7 |

- Against the authors' code: 365 of 366 decisions identical, with identical replayed shortages. The other one
  (2 Nov, VP-RNN) is a numerical tie: its two options differ by 3.5e-6 relative, below the upstream Runge–Kutta error
  (up to 9.2e-4) *(test)*. The model also agrees with a Monte Carlo simulation of itself within 0.3 standard errors
  (printed by `scripts/step1_model.py`).
- On dtcore, starting from nothing on 1 Jan, the twin's rate estimates equal the paper's benchmark columns up to
  rounding (over the 8,712 hours where the all-history column is defined), and its 61 overnight decisions and replayed
  shortages equal the model-only run *(test)*.
- Learned forecasts, the paper's protocol: the VP-RNN forecasts lose 1.66 ± 0.76 fewer per day than the all-history
  average (mean ± standard error of the paired daily difference, 61 days) *(test)*. MOVP-RNN forecasts more
  accurately (RMSE of the authors' saved forecasts against the recorded counts, Nov–Dec: 3.61 vs 3.85 for rentals,
  3.58 vs 3.78 for returns) but does not decide better here (+0.34 ± 0.42 per day) *(test)*. One station is
  consistent with the paper's conclusion that better forecasts need not give better decisions; it cannot confirm it.
- The VP-RNN in the twin (`RNNRates`, `results/compare_rates.json`): with the authors' weights its next-hour forecasts
  have the accuracy the authors recorded, all 18 numbers *(test)*. At 23:00 the twin has seen only the counts before
  the next day, so for the rest of that day the network reads its own forecasts: test RMSE 4.69 / 4.67 (rentals /
  returns) instead of 3.82 / 3.79, still better than the last-30-days average's 6.17 / 6.04 *(test)*. It does not
  decide better: 8.8 lost per day against 7.8 (paired difference 1.02 ± 0.96) *(test)*. The same weights under the
  paper's protocol lose 3.00 ± 0.64 fewer *(test)*: that protocol decides each day from forecasts that use the
  previous hour's count of the same day, which is not known when the bikes are set at night.
- Training runs: the authors' saved forecasts come from another training run than their saved weights (the accuracy
  they recorded matches the weights). With about the same accuracy (3.85 / 3.78 against 3.82 / 3.79), the two
  choose the same start on 5 of 61 days, and the saved forecasts lose 0.97 ± 0.29 more per day *(test)*. The VP-RNN
  rows describe one trained network, not the method.
- LLM agent as the nightly policy (`AgentPolicy`, `results/compare_policy.json`), two local models through Ollama,
  at most 10 replies a night, same twin and same days as the rule. The rule is exact for the twin's forecast, so an
  agent can at best equal its expected shortages. Qwen3.5-9B ran the what-if 6 times a night on average and chose the
  rule's start on 16 of 61 nights; on 17 nights it was still exploring when its replies ran out, so no truck went
  out. gpt-oss-20b ran no what-if on 43 nights and mostly chose 15 bikes ("the midpoint of capacity"), calling it
  optimal; it never chose the rule's start. In the replay neither differs from the rule beyond noise (Qwen −0.13 ±
  0.73, gpt-oss +0.79 ± 1.05 lost per day) *(test)*. About 1 h 20 min (Qwen) and 30 min (gpt-oss) on an Apple-silicon
  Mac with 24 GB.
- What-if (`results/step5_whatif.json`, `.png`): on 2 Nov, the worst day for the overnight decision, the station holds
  14 bikes at 06:59. A truck setting 1 bike: predicted 65.7 → 54.0 lost for the rest of the day; replay 99 → 86.
- Page (Twin Bench, `web/`): a simulated station online, the recorded Nov–Dec days in the local copy. The JavaScript
  matches Python over 72 simulated hours: decisions, station states and 10–90 % bands identical, numbers within
  ~1e-11 (`web/checks/check_bikeshare.js`). On the recorded days a person can switch the demand forecast between
  the historical average and the VP-RNN (its rates computed in Python for those days, so local copy only); the page
  then takes the same nightly decisions and loses the same rentals and returns as the twin in
  `results/compare_rates.json`, every day: 61 with the average, 60 with the VP-RNN, whose weather input ends with
  2018 (also `check_bikeshare.js`).

## Notes
- The twin's expected shortages and the replay disagree a lot, in both directions (42.8 vs 8.4 per day on average,
  `results/step1_decisions.json`; 66 vs 99 on 2 Nov after 07:00, `results/step5_whatif.json`). The model assumes
  independent Poisson rentals and returns at average rates; the replay holds only trips that happened (the operator's
  rebalancing trips are removed from the data, and returns that went elsewhere when the station was full are missing).
  Without the station's recorded inventory it is not possible to say which is closer. Differences between actions
  agree better than absolute levels.
- Day-to-day variation is much larger than Poisson, and rentals and returns in the same hour move together: in the 29
  busy hours (an hour of a day type with at least 5 rentals and 5 returns on average over 2018) the variance is 3–26
  times the mean and the correlation 0.78–0.96 *(test)*. The posterior sd reflects estimation error only.
- Censored demand: people who find the station empty or full leave no record, so the estimated rates are below the
  true demand. Neither the paper nor this twin corrects for it.
- The twin's all-history rates, learned from Jan–Oct, are about twice the Nov–Dec demand (weekday rentals 292 vs 139 a
  day), and after 218 weekdays each new one moves them by about 1/220, so the twin adapts slowly *(test)*. The page's
  simulated station uses the Nov–Dec average, so the gap shows there.
- The capacity (30) comes from the station table in the upstream repository; its date is unknown.
- Where a learned forecast plugs in: every rate the twin uses comes from its rate model (`StationTwin(C, rates)`,
  `rates.py`), which `predict` and `decide` ask for the hours ahead. `RNNRates` runs the authors' trained VP-RNN
  there; the forecasts they saved (`data.rnn_rates`) are used only in `experiment.py`, the paper's set-up.
- `RNNRates` reads the recorded precipitation probability of the hours ahead, i.e. a perfect weather forecast.
  Without any weather (probability 0) its test RMSE is 5.13 / 5.01 and its decisions lose the same (0.00 ± 0.16 per
  day) *(test)*.
- Running the local models: Qwen3.5-9B once thought for over 30,000 tokens without answering, so each reply is capped
  (`chat_model`, 4,096 tokens) and a cut reply costs a turn; Ollama could not parse some of gpt-oss-20b's output (it
  writes OpenAI's harmony format) and answered with a server error, which also costs a turn. A model's reply at
  temperature 0 can still change between runs, so `results/compare_policy.json` is a record of one run, not
  regenerable exactly.
- "Twin expects" is the twin's forecast with its own rates. The paper's evaluation code instead computes expected cost
  with the day's true counts.
