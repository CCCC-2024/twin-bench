# Pilot: reading the parameters straight from MRI summaries (2026-10-13)

Question: can a small network (MLP) read the four parameters (d, k, α_RT, α_CT) directly from summaries of the first
few MRIs (total cell count, volumes, radius), and how accurately?

- `gen.py N seed outfile` — samples parameters log-uniformly (d 0.02–0.5, k 0.02–0.1, α_RT 0.01–0.09, α_CT 0.05–0.6),
  starts from HGG_demo_001's day-0 map, and simulates virtual patients. 90 were generated (`vp_0.npz`, `vp_1.npz`;
  row 0 of `vp_0.npz` is HGG_demo_001's measured data); ~73 s each on one thread, ~58 min with 2 processes.
- `analyze.py <folder>` — 5-fold cross-validation, median relative error, inputs from the first K MRIs, noise 0 or 5 %
  → `pilot_results.json`.
- `decide.py d k aRT aCT` — minimal effective dose: the smallest total chemo dose with day-270 cell count ≤ 2.5e11
  (simulate doses 40 and 70, interpolate) → `decision_HGG_demo_001.txt`.

Run from `twins/tumor` with its environment; the scripts find `upstream/TumorTwin` themselves.
