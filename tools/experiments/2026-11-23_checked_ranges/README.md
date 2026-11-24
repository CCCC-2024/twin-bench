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
`checked` is `agent.advise(..., ranges=True)` as frozen in `ccd6ce6`, the commit that added this section (the next one
after `41193e7`); `number` and `range` run the code as at `6363443` with today's `tools/advise_events.py`. Three things
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
60 runs, all complete: 45 on the cluster (11/23, 18:59 to 19:40 EST), 15 on the laptop. Full tables: `results.md` (from
`runs.json` by `summarize.py`) and `sensitivity.md`. **The five runs of every configuration and arm came out the same**
(the score of every cell, except 2 of Qwen's 128 cells with `checked`; a few of Qwen's readings differed
without changing the choice), so each number below is one result, not a mean of five different ones: every run had
its own model server, and at temperature 0 the replies all but repeated. The uncertainty left is
over events, which the intervals show.

Mean extra over the vague cells (incubator: minutes of 60 outside the band; bike share: lost riders a day; lower is
better), with the share of cells harmed:

| Configuration | twin | not told | `number` | `range` | `checked` |
|---|---|---|---|---|---|
| gpt-oss-120b low | incubator | 8.5 | 4.7 (6 %) | 2.4 (6 %) | **1.6** (0 %) |
| gpt-oss-120b low | bike share | 3.2 | 3.0 (18 %) | 5.0 (26 %) | **2.4** (12 %) |
| gpt-oss-120b medium | incubator | 8.5 | 6.5 (9 %) | 3.3 (9 %) | **1.8** (3 %) |
| gpt-oss-120b medium | bike share | 3.2 | 2.5 (17 %) | 2.4 (20 %) | **1.7** (2 %) |
| gpt-oss-20b low | incubator | 8.5 | 10.3 (12 %) | **3.6** (3 %) | 4.4 (9 %) |
| gpt-oss-20b low | bike share | 3.2 | **1.4** (3 %) | 2.8 (12 %) | 1.5 (5 %) |
| Qwen3.5-9B none | incubator | 8.5 | 11.9 (16 %) | **4.1** (6 %) | 5.4 (0 %) |
| Qwen3.5-9B none | bike share | 3.2 | 3.6 (14 %) | 7.2 (23 %) | 6.0 (18 %) |

The perfect reader, from the registration (it is not in `runs.json`): 0.50 and 0.12.

**The registered questions.** Each asked for every one of 8 comparisons (4 configurations, 2 twins); none got all 8.
1. Does `checked` beat not telling the twin? **No: 7 of 8.** Yes for the three gpt-oss configurations at both twins.
   No for Qwen3.5-9B at the bike station (6.0 against 3.2).
2. Does `checked` beat `range`? **No: 6 of 8.** Not on the incubator for gpt-oss-20b (4.4 against 3.6) and Qwen (5.4
   against 4.1).
3. Does it harm fewer cells? **No: 7 of 8.** Not on the incubator for gpt-oss-20b (9 % against 3 %).
4. Is one vague input easier than several, in the same situations? **No: 7 of 8.** With one exception (gpt-oss-20b
   with `checked`: 2.5 against 1.5), the events that state the hours and the side cost 0.2 to 4.3 riders and the same
   situations in vague words 3.8 to 15.3 (not told 4.3). For the gpt-oss models the stated form costs 0.2 to 3.1.
   Qwen's does not beat not telling the twin (4.3 and 4.2): with `range` it acted in 3 of those 24 cells.

**Intervals** (95 %, over events) of the difference between `checked` and not told. Incubator: below zero in all four
configurations (120b low −9.9 to −3.8, medium −9.6 to −3.7, 20b −7.9 to −0.4, Qwen −4.7 to −1.5). Bike share: below
zero for 120b medium (−3.4 to −0.3) and 20b (−3.6 to −0.4); they include zero for 120b low (−3.4 to +1.1) and Qwen
(−0.1 to +7.1). With 8 and 11 events, the incubator result is firm and the bike-share result is not.

**Sensitivity (registered).** With every reference moved down or up, `checked` stays below not told in the same 7 of
8 comparisons; `range` in 6 of 8; `number` in 3 of 8.

**Other findings.**
- **`number` was not worse than not telling the twin on this set**, unlike on `test2`: it was better on average in 5
  of 8 comparisons. This set keeps only events where not knowing costs something (not told 8.5 and 3.2, against 6.0
  and 2.30 on `test2`), and there even a rough guess often helps. One guessed number is unreliable, not always
  harmful: it is below not told under every shift of the references in only 3 of 8 comparisons.
- **Whole-day events (29 to 31), where not knowing costs 0.24 riders:** `checked` harmed 0 % of cells for 120b
  medium, 20b and Qwen, and 17 % for 120b low; `range` harmed 0 to 33 %.
- **Qwen3.5-9B is the limit of the method.** In vague words it read the bike-share situations badly under every arm
  (9.2, 15.3 and 12.1 riders against 4.3 not told), and it again named its call `whatif` on stated events: 0 of 10
  right with `range`, 6 of 10 with `checked`, 8 of 10 with `number`. The gpt-oss models read all 10 in every arm.
- **Saying `cannot`** (10 inexpressible cells): 120b low 7, 7, 6 for the three arms; 120b medium 6, 10, 10; 20b 9,
  10, 9; Qwen 10, 3, 9.

## Conclusion
On events registered before the method was built, letting the twin weigh "nothing changes" along with the model's range
was better than not telling the twin for all three gpt-oss configurations at both twins, firmly at the incubator and
less firmly at the bike station. It did not meet any of the four registered criteria as written, because each asked for
every configuration: the 9B model without reasoning read the bike station's vague events too badly for any choice rule,
and on the incubator the two smaller models did better with the earlier rule (3.6 against 4.4 minutes, and 4.1 against
5.4). `checked` is three changes at once, and this test does not separate them. In the development on `test2` the choice
rule alone gave the incubator's gain, and the bike station improved further once its inputs said who rents and who
returns, which is a change to the reading. Question 4 points the same way for the gpt-oss models: in 5 of their 6
comparisons the same situations cost several times less once the words state the hours and which side changes.

**Found in review after the runs.** When the model gives both the start and the end hour as ranges that meet (from
7 to 9, to 9 to 11), the rule's last point has the same start and end hour, the twin refuses it, and the model is
asked again. In this test no vague cell ended that way; at most 3 of the 198 gpt-oss bike-share cells of a run needed
a second reply for any reason. The rule is left as tested, and `robust_best` says so.

## Run
On the cluster, from a run folder with `tasks.txt` and the two code folders (`run_array.sbatch`; its header says how);
on a laptop, the same command per run with the same folder names. Then, from this folder and the repository root:
```bash
python collect.py RAW_DIR [RAW_DIR ...]      # -> runs.json
python summarize.py > results.md             # seconds
cd ../../.. && python tools/experiments/2026-11-23_checked_ranges/sensitivity.py \
  > tools/experiments/2026-11-23_checked_ranges/sensitivity.md      # a few minutes
```

## Files
`collect.py` (run folders → `runs.json`), `summarize.py` → `results.md`, `sensitivity.py` → `sensitivity.md`,
`develop.py` (the development on `test2`), `run_array.sbatch` (the cluster job).
