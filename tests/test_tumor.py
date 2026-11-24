"""Tumor twin on dtcore. Needs twins/tumor/fetch_upstream.sh and the authors' package (twins/tumor/README.md, Setup);
skipped without them. About a minute: each solve of the 270 days takes 17 s on a laptop CPU."""
import json
from pathlib import Path

import pytest

pytest.importorskip("tumortwin", reason="the tumor twin needs the authors' package: twins/tumor/README.md, Setup")
pytest.importorskip("twins.tumor", reason="the tumor twin needs twins/tumor/fetch_upstream.sh")
from dtcore import Recorder, run                                        # noqa: E402
from twins.tumor import Glioma, PatientReplay, TumorTwin                # noqa: E402

SAVED = Path(__file__).resolve().parents[1] / "twins" / "tumor" / "results" / "HGG_demo_001"


@pytest.fixture(scope="module")
def glioma():
    return Glioma()


def test_the_model_gives_the_counts_and_the_dose_what_if_of_the_reproduction(glioma):
    saved = json.loads((SAVED / "repro.json").read_text())             # scripts/repro_hgg.py, the authors' tutorial
    days, plan = [glioma.day(t) for t in glioma.dates], list(glioma.chemo.doses)
    counts = [glioma.cell_count(m) for m in glioma.solve(saved["calibrated_params"], days)]
    assert counts == pytest.approx(saved["pred_tcc_calibrated"], rel=1e-3)
    more = glioma.cell_count(glioma.solve(saved["calibrated_params"], [0.0, days[-1]], chemo_total=70)[-1])
    assert more == pytest.approx(saved["whatif_tcc"]["70"][-1], rel=1e-3) and more < counts[-1]
    assert list(glioma.chemo.doses) == plan                             # the plan is put back after a what-if


def test_at_each_visit_the_twin_says_what_it_expected_and_calibrates_again(glioma):
    twin, recorder = TumorTwin(glioma, steps=1), Recorder("tumor")
    assert run(PatientReplay(glioma), twin, max_steps=2, recorder=recorder) == 2
    first, second = (r["est"] for r in recorder.rows)
    assert first["expected_cell_count"] is None and first["d"] == 0.025     # nothing to calibrate on yet
    assert second["expected_cell_count"] == pytest.approx(recorder.rows[1]["obs"]["cell_count"], rel=0.15)
    assert second["d"] != first["d"] and twin.day == 30.0
    ahead = twin.whatif(horizon=30, chemo_total=60)
    assert ahead["cell_count"] > 0
    with pytest.raises(ValueError):
        twin.whatif(horizon=0)
    assert twin.predict(12)["day"] == [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 42.0]
