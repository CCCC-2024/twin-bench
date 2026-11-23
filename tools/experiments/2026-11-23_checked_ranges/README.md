# Events in words: a third test of acting on a range (2026-11-23)

Follows `../2026-11-22_event_reading/`. There a range with a robust choice helped the incubator on average and never
the bike station, and three explanations came out of the saved readings after the fact (its `explore.md`; the rest
is in "Why" below). None was registered, so this study tests them on new events.

## Registered before any language model was run on the new events
This section was committed with the events, before the method below was changed and before any model read a `test3`
event. Only the perfect reader, which is no language model, had been run on them, to apply the keeping rule.

**Why.** On `test2`, looked at after the runs:
1. Ranges were too narrow. gpt-oss-120b's incubator ranges were 5 to 7 °C wide and held the middle reference in 43 to
   58 % of cells; Qwen's were 31 °C wide, held it in 80 %, and did nearly as well as the best gpt-oss configuration.
   Widening each saved range to include the input's value now (the event may do less than the model thinks, down to
   nothing) gave 2.4 to 3.8 minutes in all seven configurations (frozen rule 3.2 to 9.5; not told 6.0).
2. The bike station's loss sat in a few events: one read in the wrong direction (riders arriving read as rentals),
   and whole-day events where not knowing costs almost nothing, so a reading can only do harm.
3. Some bike-share references of mine were off. I had a snowstorm at 0.6, 0.4 or 0.2 of the usual demand; the
   station's two snow days of 2018 were at 0.09 and 0.19, and the models' readings (0.1 to 0.4) were nearer the
   records than my references were.

**Events** (`tools/advise_events.py`, split `test3`, numbered 21 and up; states as for `test2`).
- Incubator: 8 vague events, 4 states, 32 cells. References are my estimates.
- Bike share, 6 nights:
  - 21 to 24: four situations with the hours and the side (rentals or returns) stated and the size left vague, so
    one input is vague.
  - 25 to 28: the same four situations in words that state neither; same references. 48 cells with 21 to 24.
  - 29 to 31: rain, a federal holiday, a heat wave: the whole day changes. References are the quartiles of the
    day's demand over the usual on the days of 2018 that a rule picks from the station's records (the rule is next
    to each event). 18 cells.
- One stated, one nothing and one inexpressible event per twin, to see that a new prompt still reads those.
- Keeping rule, as for `test2`: a vague event is kept only if not knowing it costs something in at least half its
  states, judged by the perfect reader alone. It dropped 7 of 16 incubator drafts and one bike-share situation (the
  comments next to the events say which). 29 to 31 are kept whatever they cost: they test whether a reading harms.
- Not told: incubator 8.5 minutes, bike share 3.2 riders. Perfect reader: 0.50 and 0.12.

**Arms**, each scored as before (`extra`: the twin's own what-if under the mild, middle and strong reference; without
an action the twin keeps its usual choice):
- `number`: one guessed number, as at `6363443`.
- `range`: a range and the frozen robust choice, as at `6363443`.
- `checked`: the new method. It will be developed on `dev`, `test` and `test2` only, and frozen in a commit before
  any model reads a `test3` event. That commit is named in "Method" below.

**Configurations:** gpt-oss-120b at reasoning low and medium, gpt-oss-20b at low (one NVIDIA B200 each), Qwen3.5-9B
at none (a laptop); 5 runs of every arm. gpt-oss at high is left out: on `test2` it was the worst with a range and
one run of the 20b took 4.6 hours.

**Questions and what counts as yes.** Means are over vague cells and runs.
1. Does `checked` beat not telling the twin? Yes if its mean extra is below not told's on each twin in each of the
   four configurations.
2. Does `checked` beat `range`? Yes if its mean extra is lower on each twin in each configuration.
3. Does it do less harm? A cell is harmed if its extra is more than 0.5 (minutes, riders) above not told's. Yes if
   the share of harmed cells is lower for `checked` than for `range` on each twin in each configuration. The share is
   also reported for 29 to 31 alone.
4. Is one vague input easier than several, on the same twin? Yes if, for `checked` and for `range`, the mean extra on
   21 to 24 is below that on 25 to 28 in each configuration.

**Uncertainty.** For every mean difference from not told: a 95 % interval from 10,000 bootstrap draws over events
within a twin (an event's states and runs stay together). The yes or no above goes by the means; the intervals say
how far 8 or 11 events carry it.

**Sensitivity.** As before: every reference moved down and up (incubator 3 °C; bike share factors ×0.8 and ×1.25).

**What is not tested.** A choice is scored by the twin's own what-if, not by what happened. I tried the real outcome
first: on 22 holiday, bad-weather and busy days of 2018, the start that a perfect whole-day reading gives loses within
3 riders of the usual start's on 20 (it saves 4 and 13 on the other two), mostly because rentals and returns fell or
rose together. The records can ground a reference; they cannot score this decision.

## Method
`checked` is `agent.advise(..., ranges=True)` as frozen in the commit that added this section (the next one after
`41193e7`); `number` and `range` run the code as at `6363443` with today's `tools/advise_events.py`. Three things
changed, all worked out on `test2` and none on `test3`:
1. **The twin's choice** (`agent.robust_best`). The ranged inputs move together from each one's value now to the far
   end of its range, in four even steps, and the value now is always a point. Before, every range gave its low end,
   middle and high end, in every combination, and the value now was not among the points.
2. **The prompt** asks for a range the model is confident holds the input, wide enough that the true value outside
   it would be a surprise; before, for the range the input most likely lies in, as narrow as it honestly could.
3. **The bike station's inputs** say who rents (people who take a bike from this station and ride away) and who
   returns (people who arrive by bike and leave it here). Every twin's inputs now carry their value now (`default`).

On `test2`, mean extra over the vague cells (development: these numbers chose the method and test nothing):
- Change 1 alone, on the range readings saved in the earlier study (`develop.py`): incubator 1.9 to 3.7 minutes in
  the seven configurations (the earlier rule 3.2 to 9.5; not told 6.0), with 0 to 4 % of cells harmed (0 to 23 %);
  bike share 2.7 to 4.7 riders (3.1 to 4.9; not told 2.30), with 4 to 25 % harmed (8 to 35 %).
- All three, two runs per configuration, which came out the same: gpt-oss-120b at low, incubator 2.19, bike share
  1.77; at medium 1.83 and 1.75; gpt-oss-20b at low 1.92 and 2.47. Stated events 10 of 10 in each. `cannot` on the
  bike station's 12 inexpressible cells: 4, 12 and 11.
- Qwen3.5-9B was run once with change 2 only: with change 1 on those readings, incubator 5.3 (3.7 on its readings
  with the earlier prompt). It was not run with all three before the freeze.

## Results
(None yet.)
