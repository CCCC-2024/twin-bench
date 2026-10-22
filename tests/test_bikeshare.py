"""Bike-share twin: what it reproduces of Gammelli et al. 2022. Needs twins/bikeshare/fetch_upstream.sh."""
import datetime
import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dtcore import Recorder, run
from twins.bikeshare import HistoricalRates, ReplayStation, StationTwin, data, experiment, model

JAN1 = datetime.date(2018, 1, 1)
TOL = 1e-3          # the authors integrate with Runge-Kutta: relative error up to ~1e-3


def mean_and_se(d):
    """Mean and standard error of paired daily differences, rounded as the README reports them."""
    return round(float(d.mean()), 2), round(float(d.std(ddof=1) / np.sqrt(len(d))), 2)


@pytest.fixture(scope="module")
def paper():
    assert data.DATA.exists(), "run twins/bikeshare/fetch_upstream.sh first"
    return experiment.load()


@pytest.fixture(scope="module", params=[("overall_ha", None, "ha"), ("specific_ha", 30, "ha30")],
                ids=["all-earlier-days", "last-30-days"])
def year(request, paper):
    """The twin starts knowing nothing on 1 Jan 2018 and replays the year hour by hour, deciding every night."""
    source, window, column = request.param
    twin = StationTwin(paper["C"], HistoricalRates(window_days=window))
    rec = Recorder(f"bikeshare-{source}")
    run(ReplayStation(paper["C"], data.events(), JAN1, 365), twin, recorder=rec)
    return SimpleNamespace(source=source, column=column, twin=twin, by_hour={r["obs"]["t"]: r for r in rec.rows})


def test_learned_rates_equal_the_papers_benchmarks(year):
    hours = data.hourly().dropna(subset=[f"{year.column}_pickup"])
    learned = np.array([year.twin.rates.rate(day, hour)[0] for day, hour in zip(hours["date"], hours["hour"])])
    benchmark = hours[[f"{year.column}_pickup", f"{year.column}_return"]].to_numpy()
    assert np.abs(learned - benchmark).max() < 1e-9                  # equal up to rounding
    assert len(hours) == {"ha": 8712, "ha30": 8016}[year.column]       # every hour the paper's column is defined


def test_nightly_decisions_and_shortages_equal_the_papers_experiment(year, paper):
    for day in experiment.TEST_DAYS:
        expected = experiment.run_day(paper, day, sources=(year.source,))[year.source]
        k0 = (day - JAN1).days * 24
        assert year.by_hour[k0 - 1]["action"]["set_bikes"] == expected["start"]   # decided at 23:00 the day before
        lost = sum(year.by_hour[k0 + i]["obs"]["lost_rent"] + year.by_hour[k0 + i]["obs"]["lost_return"]
                   for i in range(24))
        assert lost == expected["lost_rent"] + expected["lost_return"]


def test_decisions_equal_the_authors_code_except_one_numerical_tie(paper):
    """Against the authors' answers saved by scripts/crosscheck_upstream.py (results/ref_upstream.json)."""
    saved = json.loads((Path(data.__file__).parent / "results" / "ref_upstream.json").read_text())
    ties, differences = [], []
    for day in experiment.TEST_DAYS:
        ours_by_source = experiment.run_day(paper, day)
        for source in experiment.SOURCES + ("best_possible",):
            theirs, ours = saved[str(day)][source], ours_by_source[source]
            if source == "best_possible":
                assert theirs["start"] == ours["start"] and theirs["real"] == ours["lost_rent"] + ours["lost_return"]
                continue
            cost = model.udf(*experiment.rates(paper, source, day), paper["C"])    # expected shortages per start
            differences.append(abs(theirs["expected"] - ours["expected"]) / theirs["expected"])
            if "curve" in theirs:
                differences.append(np.max(np.abs(np.array(theirs["curve"]) - cost) / np.array(theirs["curve"])))
            if theirs["start"] == ours["start"]:
                assert theirs["real"] == ours["lost_rent"] + ours["lost_return"]
            else:                                    # allowed only when both choices are equally good
                gap = abs(cost[theirs["start"]] - cost[ours["start"]]) / cost[ours["start"]]
                ties.append((day.isoformat(), source, gap))
    assert round(float(max(differences)), 5) == 0.00092 < TOL         # the authors' Runge-Kutta error
    assert [(day, source, round(float(gap), 7)) for day, source, gap in ties] == [("2018-11-02", "vp_rnn", 3.5e-6)]



@pytest.fixture(scope="module")
def rnn_year(paper):
    """The twin with the paper's trained VP-RNN as its rate model replays 2018, deciding every night. The weather
    file ends with the year, so the last hour (whose next hour is in 2019) is not played. At the start of every
    test day it also keeps the 24 rates it has for that day."""
    pytest.importorskip("torch", reason='the RNN rate model needs torch: pip install -e ".[rnn]"')
    from twins.bikeshare import RNNRates
    assert data.RNN_MODELS.exists(), "run twins/bikeshare/fetch_upstream.sh first"
    twin, rec, day_ahead = StationTwin(paper["C"], RNNRates()), Recorder("bikeshare-rnn"), {}

    def at_midnight(k, source, twin):
        day = JAN1 + datetime.timedelta(days=k // 24)
        if k % 24 == 0 and day in experiment.TEST_DAYS:
            day_ahead[day] = np.stack(twin.rates.ahead(24), axis=1)        # (24, rentals / returns)
    run(ReplayStation(paper["C"], data.events(), JAN1, 365), twin, max_steps=365 * 24 - 1, recorder=rec,
        before_step=at_midnight)
    return SimpleNamespace(rows=rec.rows, day_ahead=day_ahead)


def test_rnn_next_hour_rates_have_the_accuracy_the_authors_recorded(rnn_year):
    """The twin's next-hour rates are the paper's one-step forecasts: with the authors' weights they have the
    accuracy the authors recorded for their model (output_st426_so_rnn.txt), all 18 numbers, on their train /
    validation / test split."""
    recorded = (data.RNN_ACCURACY / "output_st426_so_rnn.txt").read_text()
    counts = data.hourly()
    n = len(counts)
    train, test = int(n * 0.7), n - 61 * 24                       # the authors' split; test = Nov-Dec
    splits = dict(TRAIN=slice(1, train), VALIDATION=slice(train, test), TEST=slice(test, n))
    for kind, column, field in (("return", "returns", "next_return_rate"), ("pickup", "pickups", "next_rent_rate")):
        forecast = np.array([np.nan] + [r["est"][field] for r in rnn_year.rows])   # row t forecasts hour t + 1
        block = recorded.split(f"STATION 426 - {kind}")[1]
        rows = re.findall(r"\| (\w+) \|\s+RMSE: ([\d.]+), MAE: ([\d.]+), R\^2: ([\d.]+)", block)[:3]
        for split, rmse, mae, r2 in rows:
            y, f = counts[column].to_numpy(float)[splits[split]], forecast[splits[split]]
            r2_ours = 1 - np.sum((y - f) ** 2) / np.sum((y - y.mean()) ** 2)
            ours = (np.sqrt(np.mean((y - f) ** 2)), np.mean(np.abs(y - f)), r2_ours)
            assert [f"{v:.2f}" for v in ours] == [rmse, mae, r2], (kind, split)


def test_rnn_forecasts_from_what_is_known_at_midnight_are_less_accurate(rnn_year):
    """At midnight the twin has seen only the counts before the day, so its 24-hour forecast feeds the network its own
    forecasts. It is less accurate than the one-step forecasts the paper decided with (test RMSE 3.82 / 3.79, test
    above), and more accurate than the last-30-days average (the paper's better benchmark)."""
    test = data.hourly()[lambda h: h.date >= experiment.TEST_DAYS[0]]
    rmse = lambda f: tuple(np.round(np.sqrt(np.mean((f - test[["pickups", "returns"]].to_numpy()) ** 2, axis=0)), 2))
    assert rmse(np.concatenate([rnn_year.day_ahead[day] for day in experiment.TEST_DAYS])) == (4.69, 4.67)
    assert rmse(test[["ha30_pickup", "ha30_return"]].to_numpy()) == (6.17, 6.04)


@pytest.fixture(scope="module")
def compared():
    """Lost rentals + returns per test day for each rate model in the twin (scripts/compare_rates.py)."""
    saved = json.loads((Path(data.__file__).parent / "results" / "compare_rates.json").read_text())
    lost = {m: np.array([saved["days"][day][m]["lost_rent"] + saved["days"][day][m]["lost_return"]
                         for day in sorted(saved["days"])]) for m in saved["forecast_rmse"]}
    return SimpleNamespace(saved=saved, lost=lost)


def test_compared_rnn_decisions_are_what_the_code_decides(rnn_year, compared, paper):
    """The RNN twin's nightly decisions are those in results/compare_rates.json, and so are the paper's protocol's with
    the same network (each test day decided from that day's one-step forecasts)."""
    for day in experiment.TEST_DAYS:
        k0, saved = (day - JAN1).days * 24, compared.saved["days"][str(day)]
        assert rnn_year.rows[k0 - 1]["action"]["set_bikes"] == saved["rnn"]["start"]
        rows = rnn_year.rows[k0 - 1:k0 + 23]                         # row t holds the forecast of hour t + 1
        f = np.array([[r["est"]["next_rent_rate"], r["est"]["next_return_rate"]] for r in rows])
        assert model.best_start(f[:, 0], f[:, 1], paper["C"])[0] == saved["rnn_one_step"]["start"]


def test_in_the_twin_the_rnn_forecasts_better_but_does_not_decide_better(compared):
    """Knowing only what is known at 23:00, the RNN loses about as many as the last-30-days average; the paper's
    protocol with the same network, which uses each hour's previous count, loses fewer. Without any weather the
    forecast is worse but the decisions lose the same."""
    paired = lambda a, b: mean_and_se(compared.lost[a] - compared.lost[b])
    assert {m: round(float(v.mean()), 2) for m, v in compared.lost.items()} == dict(
        historical_all=8.43, historical_30=7.79, rnn=8.80, rnn_one_step=5.80, rnn_no_weather=8.80)
    assert paired("historical_30", "rnn") == (-1.02, 0.96)
    assert paired("rnn", "rnn_one_step") == (3.00, 0.64)
    assert paired("rnn_no_weather", "rnn") == (0.0, 0.16)
    assert [round(x, 2) for x in compared.saved["forecast_rmse"]["rnn_no_weather"]] == [5.13, 5.01]
    days = compared.saved["days"].values()                           # the README table's rows marked †
    start = lambda m: int(np.median([d[m]["start"] for d in days]))
    expects = lambda m: round(float(np.mean([d[m]["expected"] for d in days])), 1)
    assert (start("rnn"), expects("rnn"), start("rnn_one_step"), expects("rnn_one_step")) == (17, 16.2, 13, 13.7)


def test_two_training_runs_of_the_rnn_decide_differently(compared):
    """The authors' saved forecasts come from another training run than their saved weights. Same protocol, about the
    same accuracy (test RMSE 3.85 / 3.78 against 3.82 / 3.79), yet they rarely choose the same start."""
    saved = json.loads((Path(data.__file__).parent / "results" / "step1_decisions.json").read_text())
    days = sorted(saved)
    same = sum(saved[day]["vp_rnn"]["start"] == compared.saved["days"][day]["rnn_one_step"]["start"] for day in days)
    d = np.array([saved[day]["vp_rnn"]["lost_rent"] + saved[day]["vp_rnn"]["lost_return"] for day in days]) \
        - compared.lost["rnn_one_step"]
    assert same == 5 and len(days) == 61
    assert mean_and_se(d) == (0.97, 0.29)

@pytest.fixture(scope="module")
def nightly():
    """Lost rentals + returns per test day for each rate source (scripts/step1_model.py)."""
    saved = json.loads((Path(data.__file__).parent / "results" / "step1_decisions.json").read_text())
    lost = lambda day, source: saved[day][source]["lost_rent"] + saved[day][source]["lost_return"]
    return {source: np.array([lost(day, source) for day in sorted(saved)]) for source in experiment.SOURCES}


def test_rnn_forecasts_lose_fewer_than_the_all_history_average(nightly):
    paired = lambda a, b: mean_and_se(nightly[a] - nightly[b])
    assert paired("overall_ha", "vp_rnn") == (1.66, 0.76)
    assert paired("mo_vp_rnn", "vp_rnn") == (0.34, 0.42)             # the multi-output RNN does not decide better


def test_the_multi_output_rnn_forecasts_more_accurately():
    """RMSE of the authors' saved forecasts against the recorded counts, test period (Nov-Dec 2018)."""
    test = data.hourly()[lambda h: h.date >= experiment.TEST_DAYS[0]]
    def rmse(name):
        rates = data.rnn_rates(name)
        forecast = np.array([np.concatenate(rates(day, [hour])) for day, hour in zip(test.date, test.hour)])
        return tuple(round(float(np.sqrt(np.mean((forecast[:, i] - test[k]) ** 2))), 2)
                     for i, k in enumerate(("pickups", "returns")))
    assert rmse("mo_rnn") == (3.61, 3.58) and rmse("so_rnn") == (3.85, 3.78)


def test_demand_varies_far_more_than_poisson_and_rentals_follow_returns():
    """The busy hours: those of a day type with at least 5 rentals and 5 returns on average over 2018."""
    by_hour = data.hourly().groupby(["weekday", "hour"])
    s = by_hour.agg(mp=("pickups", "mean"), vp=("pickups", "var"), mr=("returns", "mean"), vr=("returns", "var"))
    busy = s[(s.mp >= 5) & (s.mr >= 5)]
    corr = by_hour.apply(lambda g: g.pickups.corr(g.returns))[busy.index]
    dispersion = np.concatenate([busy.vp / busy.mp, busy.vr / busy.mr])
    assert len(busy) == 29 and (round(dispersion.min()), round(dispersion.max())) == (3, 26)
    assert (round(float(corr.min()), 2), round(float(corr.max()), 2)) == (0.78, 0.96)


def test_rates_learned_by_october_are_twice_the_test_period_demand():
    h = data.hourly()
    weekdays = h[h.weekday]
    start = experiment.TEST_DAYS[0]
    before, after = weekdays[weekdays.date < start], weekdays[weekdays.date >= start]
    per_day = lambda d: round(float(d.groupby("date").pickups.sum().mean()))
    assert (per_day(before), per_day(after)) == (292, 139)
    assert before.date.nunique() == 218                               # each new weekday moves a rate by about 1/220


class FixedModel:
    """A fake language model with fixed answers: its i-th reply in every conversation is answers[i]. It keeps every
    conversation it was sent."""

    def __init__(self, *answers):
        self.answers, self.sent = answers, []

    def __call__(self, messages):
        self.sent.append(list(messages))
        return self.answers[len(messages) // 2 - 1]                  # system + task, then one exchange per reply


def test_the_agent_asks_the_whatif_then_chooses():
    from twins.bikeshare import agent
    llm = FixedModel('{"whatif": 3}', '{"choose": 3, "reason": "fewest lost"}')
    answer = agent.ask(llm, "Choose a number.", range(5), lambda n: {"lost": abs(n - 3)})
    assert answer == dict(choice=3, reason="fewest lost", whatifs=[(3, {"lost": 0})])
    assert '"lost": 0' in llm.sent[1][-1]["content"]                   # the model was told the what-if's answer


def test_the_agent_is_told_what_was_wrong_with_a_reply_and_tries_again():
    from twins.bikeshare import agent
    llm = FixedModel("Three looks best.", '{"choose": 7}', '```json\n{"choose": 3}\n```')
    answer = agent.ask(llm, "Choose a number.", range(5), lambda n: {})
    assert answer["choice"] == 3 and answer["whatifs"] == []            # a model may wrap its JSON in a code block
    told = [conversation[-1]["content"] for conversation in llm.sent[1:]]
    assert "not a JSON object" in told[0] and "7 is not one of the choices" in told[1]


def test_the_agent_makes_no_choice_when_its_turns_run_out():
    from twins.bikeshare import agent
    llm = FixedModel(*['{"whatif": 1}'] * 4)
    answer = agent.ask(llm, "Choose a number.", range(5), lambda n: {}, turns=4)
    assert answer["choice"] is None and answer["reason"] == "no valid choice in 4 replies"
    assert len(llm.sent) == 4 and len(answer["whatifs"]) == 4


def test_in_the_twin_the_agent_decides_each_night_with_the_twins_whatif(paper):
    """The twin asks its policy at 23:00 once it can forecast the next day: not on Friday 5 Jan, before any Saturday."""
    from twins.bikeshare import AgentPolicy
    llm = FixedModel('{"whatif": 10}', '{"choose": 10, "reason": "tried 10"}')
    rec = Recorder("bikeshare-agent")
    twin = StationTwin(paper["C"], policy=AgentPolicy(llm))
    run(ReplayStation(paper["C"], data.events(), JAN1, 7), twin, recorder=rec)
    decided = {r["obs"]["date"]: r["action"] for r in rec.rows if r["action"] is not None}
    assert {r["obs"]["hour"] for r in rec.rows if r["action"] is not None} == {23}
    assert sorted(decided) == ["2018-01-01", "2018-01-02", "2018-01-03", "2018-01-04", "2018-01-06", "2018-01-07"]
    assert len(llm.sent) == 2 * 6
    jan1 = data.hourly()[lambda h: h.date == JAN1]                     # Tuesday's rates: Monday's counts
    expected = model.udf(jan1.pickups.to_numpy(float), jan1.returns.to_numpy(float), paper["C"])[10]
    told = json.loads(llm.sent[1][-1]["content"])                       # what the what-if answered on the first night
    assert told["whatif"] == 10 and told["expected_lost"] == round(expected, 2)
    assert decided["2018-01-01"] == dict(set_bikes=10, expected_lost=pytest.approx(expected), asked=[10],
                                         reason="tried 10")


def test_when_the_model_gives_no_valid_choice_no_truck_goes_out(paper):
    from twins.bikeshare import AgentPolicy
    station, rec = ReplayStation(paper["C"], data.events(), JAN1, 2), Recorder("bikeshare-agent")
    run(station, StationTwin(paper["C"], policy=AgentPolicy(FixedModel(*["About 12."] * 3), turns=3)), recorder=rec)
    assert rec.rows[23]["action"] == dict(asked=[], reason="no valid choice in 3 replies")
    assert station.moved == 0                                          # bikes added or removed by actions
