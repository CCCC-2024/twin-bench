# Events in words: reading, reasoning effort, and ranges (2026-11-22)

Runs of `tools/advise_events.py` on all four twins. Frozen: the numbers below come from `runs.json` through
`summarize.py` (→ `results.md`) and `explore.py` (→ `explore.md`).

## Questions
1. How much do model size and reasoning effort change reading? Reading here means three things: reading stated
   numbers, leaving an irrelevant event alone (`no_effect`), and saying `cannot` when no input expresses the event.
2. For a vague event, is the twin better off when the model gives a range and the twin chooses robustly
   (`agent.advise(..., ranges=True)`) than when it is told nothing, or told one guessed number?

## Design
Experiment 1 answers question 1; experiment 2 was the design of the range method on the development events (no runs
here); experiment 3 answers question 2.
- **Experiment 1**, on the first test set (`test`, 9 events per twin, 90 cells per run), one number per input, at
  `ef10e9c`: gpt-oss-120b at reasoning low, medium and high, gpt-oss-20b at medium and high. Qwen3.5-9B and
  gpt-oss-20b at low ran once on a laptop; their rows are in `twins/*/results/advise_events.json`, not here.
- **Experiment 3**, on the second test set (`test2`, 144 cells per run): 80 vague cells (8 events × 4 incubator
  states, 8 × 6 bike-share nights), plus stated, nothing and inexpressible events on all four twins. The events and
  their reference conditions were committed in `b69db08` (10:21 PST, 11/22). The range method was frozen in `6363443`
  (10:58), after two development rounds on the `dev` events and the first test set. The first run on `test2` started
  at 12:25. So `test2` is held out for the range method, and the first test set is not. Each configuration was run
  with one number and with a range (`--ranges`), at `6363443` (Qwen at `216296e`, whose `agent.py` and
  `advise_events.py` are the same).
- **Registered before the runs.** The success criterion: on held-out vague events, acting on a range is never worse
  than not telling the twin, and better on average. The sensitivity check: references moved down and up.
- **Not registered.** Everything under "Exploratory" was chosen after the results were seen.
- **Scoring.** As in `tools/advise_events.py`: a choice is scored by the twin's own what-if under the event's mild,
  middle and strong reference conditions. The scale runs from a perfect reader, given the middle reference (incubator
  2.0 minutes of 60 outside the band, bike share 0.19 lost riders a day), to the twin that is not told (6.0, 2.30).
- **Machines and repeats.** gpt-oss on one NVIDIA B200 per run (HiPerGator), temperature 0, replies capped at 16,384
  tokens. Qwen3.5-9B (reasoning none) on an Apple-silicon laptop; another machine does not reproduce its choices one
  for one (bike share's README). Runs per configuration differ (1 to 5), and every table states its count.
  gpt-oss-20b at high with a range ran once, because one run took 4.6 hours.

## Run
On the cluster, one task per run (`run_array.sbatch`; its header says how), then copy the run folders back and, from
this folder:
```bash
python collect.py RAW_DIR [QWEN_DIR]     # -> runs.json (only runs whose printed summary is complete)
python summarize.py > results.md         # seconds
cd ../../.. && python tools/experiments/2026-11-22_event_reading/explore.py \
  > tools/experiments/2026-11-22_event_reading/explore.md   # about 15 minutes
```

## Results
Mean over runs, [min–max]; full tables in `results.md`.

**Reading (experiment 1, first test set, one number):**

| | gpt-oss-120b low (4 runs) | 120b medium (3) | 120b high (3) | 20b medium (3) | 20b high (2) |
|---|---|---|---|---|---|
| stated, of 30 | 30 | 30 | 29.7 | 30 | 29 |
| nothing, of 20 | 20 | 19 | 19 | 19 | 19 |
| cannot, of 20 | 10.8 | 18 | 19 | 11 | 11 |
| incubator vague, minutes (not told 8.0) | 19.0 [8.4–29.6] | 8.4 | 22.4 [8.4–29.6] | 8.0 | 8.0 |

**Reading (experiment 3, test2):** stated events are read right in 14 to 16 of 16 in every configuration except one:
Qwen with the range prompt reads 7 of 16, because on one stated event per twin it named its call `whatif`, which is
not one of the three calls. `cannot` (of 32) is 18.0 to 24.7 at low, 27.8 to 30.4 for 120b at medium and high, 20.3
to 26 for 20b at medium and high, and 32 for Qwen with one number. `nothing` (of 16) is 15.7 to 16 at low and 12.7
to 15.6 with more reasoning.

**Vague events (experiment 3, test2):** mean extra over the vague cells; lower is better.

| Configuration | runs | incubator, one number | incubator, range | bike share, one number | bike share, range |
|---|---|---|---|---|---|
| not told | | 6.0 | 6.0 | 2.30 | 2.30 |
| gpt-oss-120b low | 5 / 5 | 6.5 [4.6–9.3] | **3.2** [2.7–4.3] | 3.3 | 4.6 [3.5–5.9] |
| gpt-oss-120b medium | 5 / 5 | 9.5 | **4.8** [2.9–7.7] | 4.2 | 4.3 |
| gpt-oss-120b high | 5 / 5 | 9.4 | 9.5 [7.9–10.0] | 3.8 | 3.1 |
| gpt-oss-20b low | 3 / 3 | 6.2 [5.4–7.4] | **4.1** | 2.4 | 3.4 |
| gpt-oss-20b medium | 3 / 3 | 9.0 | 8.0 | 3.3 | 4.2 |
| gpt-oss-20b high | 3 / 1 | 9.8 | **5.7** | 2.5 | 3.3 |
| Qwen3.5-9B none | 5 / 5 | 14.6 | **3.5** | 5.2 | 4.9 |

- **One number is worse than not telling the twin on average** in every configuration, on both twins. Single runs of
  the low-effort gpt-oss models were sometimes better on the incubator (three of five 120b runs, two of three 20b runs).
- **A range helps the incubator on average in 5 of 7 configurations, and never helps the bike station.** Not even for
  the incubator is it never worse: one 120b medium run scored 7.7, and in 6 of 7 configurations some cells are worse
  than not told. The registered criterion (never worse, better on average) is **not met**.
- **Sensitivity (registered):** with every reference moved down or up (incubator 3 °C; bike share factors ×0.8 and
  ×1.25), the conclusions hold, with two exceptions, each less than 1.0 from the twin not told. gpt-oss-20b at low
  with one number is slightly better than not told on the incubator when the references move down (5.73 against
  5.87). gpt-oss-20b at high with a range (one run) is worse when they move either way (5.92 against 5.87; 6.40
  against 5.41). A range never helps the bike station, under any shift.

**Exploratory (not registered; `explore.md`, `results.md`):**
- **The bike station's range readings mostly miss the event.** A range holds the middle reference for every input in
  6 to 31 % of the bike-share cells with a reading, per configuration, against 43 to 80 % for the incubator. Typical
  misses: a conference's arriving riders read as rentals, a transit strike read as more rentals but not more returns,
  a snowstorm or a parade read as far stronger than the reference. The references are my estimates, and some of these
  readings are defensible.
- **No choice rule rescues those readings.** On the same range readings, minimax over every action, the smallest mean
  regret, the three linked ends, or the middle of the range all leave the bike station worse than not told (best 2.68
  against 2.30). For the incubator the rules agree within 0.5, except the middle of the range, which is worse than not
  told in every configuration: the gain comes from choosing across the model's range, not from the point inside it.
- **A range that says nothing does not help.** The same rule on a range that ignores the event gives the incubator
  7.18 minutes (the room anywhere from −20 to 50 °C) and the bike station 7.43 riders (both factors 0 to 4 all day),
  worse than not told (6.0, 2.30). The incubator's gain needs the model's reading of the event.

## Conclusion
Reading numbers is solved for these twins, and telling `no_effect` from `cannot` improves with reasoning effort on the
larger model. For vague events, executing the model's single guess is worse than ignoring the event. A range with a
robust choice helped on average where one input decides (the incubator's room temperature), though not in every run.
It did not help where several inputs must be read together (the station's rentals, returns and hours), and there the
readings, not the choice rule, were wrong. A twin should not act on a range of several inputs without checking it, for
example by asking a person when it differs from the twin's own expectation.

## Files
`collect.py` (raw run folders → `runs.json`: per run its model, reasoning, reply form, machine and every cell with
its reading, choice, extra and the not-told extra), `summarize.py` → `results.md`, `explore.py` → `explore.md`,
`references.json` (the vague test2 references, as committed in `b69db08`), `run_array.sbatch` (the cluster job).
