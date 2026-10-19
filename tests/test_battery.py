"""Battery twin: what it reproduces of Thelen et al. 2023 and what its README states. Needs
twins/battery/fetch_upstream.sh."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import curve_fit

from dtcore import Recorder, run
from twins.battery import BatteryTwin, CellReplay, data, decide_paper, run_pf
from twins.battery.decision import decide_online
from twins.battery.pf import PRIOR, PROC_STD, power_law

BASELINE = Path(data.__file__).parent / "results" / "baseline.csv"     # written by scripts/baseline.py


@pytest.fixture(scope="module", autouse=True)
def upstream_data():
    assert data.UPSTREAM.exists(), "run twins/battery/fetch_upstream.sh first"


@pytest.fixture(scope="module")
def cells():
    train, test = data.split(data.load_cells())                       # 41 training cells, 82 test cells
    return SimpleNamespace(train=train, test=test)


@pytest.fixture(scope="module")
def example(cells):
    """The authors' script: seed 2, the first test cell of a random permutation, decision at 95 %."""
    np.random.seed(2)
    cell = cells.test[int(np.random.permutation(len(cells.test))[0])]
    return cell, decide_paper(cell, run_pf(cell), threshold=0.95)   # run_pf continues the same random stream


def test_the_papers_example_cell_retires_at_cycle_542(example):
    cell, decision = example
    assert (decision["x_twin"], cell["k_eol"]) == (542, 691)


def test_one_bad_reading_starts_prediction_and_the_decision_of_the_example_cell(example):
    cell, decision = example
    i = int(np.where(cell["cycles"] == 249)[0][0])
    assert round(float(cell["q"][i]), 3) == 0.890 and min(cell["q"][i - 1], cell["q"][i + 1]) > 0.98   # a dip
    assert decision["k_decide"] == cell["k_fpt"] == 249


@pytest.mark.parametrize("index", [27, 4, 24])        # the paper's example cell, a long- and a short-lived one
def test_twin_on_dtcore_is_the_original_filter_and_decision(cells, index):
    cell = cells.test[index]
    replay, twin = CellReplay(cell), BatteryTwin(cell["q0"], seed=0)
    rec = Recorder("battery")
    run(replay, twin, recorder=rec)
    states = [r["est"] for r in rec.rows if r["est"]["started"]]
    batch = run_pf(cell, rng=np.random.RandomState(0))
    for i, s in enumerate(states):
        assert (s["cycle"], s["log10_a"], s["b"], s["eol_pred"]) == \
               (batch["cycles"][i], batch["log10_a"][i], batch["b"][i], batch["eol_pred"][i])
    q = cell["q"][:cell["n_measured"]]
    k = int(np.where(q <= 0.95)[0][0])                                # the decision: first measured cycle at 95 %
    j = int(np.where(batch["cycles"] == cell["cycles"][k])[0][0])
    expected = decide_online(q[:k + 1], cell["q0"], batch["log10_a"][j], batch["b"][j], batch["eol_pred"][j])
    assert twin.decision["x_twin"] == expected["x_twin"] == replay.retire_at
    assert rec.rows[-1]["obs"]["cycle"] == min(replay.retire_at, cell["last_measured_cycle"])   # replay stopped


@pytest.fixture(scope="module")
def at95():
    baseline = pd.read_csv(BASELINE)
    return SimpleNamespace(all=baseline, decisions=baseline[baseline.threshold == 0.95])   # 82 cells x 3 seeds


def test_optimal_retirement_is_extrapolated_for_a_fifth_of_the_cells(at95):
    per_cell = at95.all.drop_duplicates("cell")
    assert round(float(np.mean(per_cell.x_global > per_cell.last_measured)), 2) == 0.21


def test_one_in_twelve_decisions_is_triggered_by_a_dip(at95):
    dip = at95.decisions.spurious_trigger
    assert (int(dip.sum()), len(dip)) == (21, 246)                    # 8.5 %


def test_dip_decisions_err_badly_twice_as_often_but_in_one_cell(at95):
    d = at95.decisions
    large = d.regret_global > 0.05                                    # utility loss > 0.05, as in plot_tradeoff.py
    dip = d.spurious_trigger
    assert (int(large[dip].sum()), int(large[~dip].sum())) == (3, 15)    # of 21 and of 225
    assert d[dip & large].cell.nunique() == 1


def test_online_and_paper_decisions_differ_in_4_of_246_by_one_cycle(at95):
    d = at95.decisions
    differ = d[d.x_online != d.x_twin]
    assert len(differ) == 4 and set(differ.x_twin - differ.x_online) == {1}


def test_the_authors_prior_is_narrow_against_fits_to_the_training_cells(cells):
    """Power law fitted to every measured cycle of each training cell, starting from the authors' prior."""
    fits = []
    with np.errstate(over="ignore", invalid="ignore"):              # the fit tries steep curves on its way
        for cell in cells.train:
            n = cell["n_measured"]
            k, q = cell["cycles"][:n].astype(float), cell["q"][:n]
            fits.append(curve_fit(power_law, k, q, p0=PRIOR, maxfev=20000)[0])
    (mean_a, mean_b), (std_a, std_b) = np.mean(fits, axis=0), np.std(fits, axis=0)
    assert (round(mean_a, 1), round(mean_b, 2), round(std_a, 1), round(std_b, 1)) == (-12.9, 4.35, 3.2, 1.2)
    assert (PRIOR, PROC_STD) == ((-15.77, 5.45), (0.05, 0.05))      # the authors' prior; its std is PROC_STD
