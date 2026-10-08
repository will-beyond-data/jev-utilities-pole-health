import json
import shutil
from pathlib import Path

from click.testing import CliRunner
from PIL import Image

from polehealth import demo_data, evaluate
from polehealth.cli import main
from polehealth.jev import MockBackend
from polehealth.register import build_register, load_wrc
from tests.conftest import FIXTURES, write_wrc_xlsx


def build(register, raw_dir, tmp_path, inspect_n=10):
    run_dir = tmp_path / "run"
    run_dir.mkdir(exist_ok=True)
    backend = MockBackend(seed=7)
    evaluate.write_run_meta(run_dir, backend, [], "none (mock backend)", None, 7, None)
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    built = demo_data.build_demo_data(
        register, raw_dir, out, run_dir, backend, "none (mock backend)", inspect_n=inspect_n, log=lambda _: None
    )
    return out, run_dir, built


def test_poles_json_shape(register, raw_dir, tmp_path):
    out, _, _ = build(register, raw_dir, tmp_path)
    poles = json.loads((out / "poles.json").read_text())
    assert {"generated_at", "town", "attribution", "poles", "backend"} <= set(poles)
    assert poles["town"] == "Toowoomba, QLD" and poles["backend"] == "mock" and poles["mock"] is True
    assert len(poles["poles"]) == 12
    inspected = [p for p in poles["poles"] if p["inspected"]]
    assert len(inspected) == 10
    assert len({p["photo"] for p in inspected}) == 10  # no photo reused
    for p in inspected:
        assert set(p["photo_source"]) >= {"dataset", "label", "credit"}
        assert (out / p["photo"]).is_file()
        assert max(Image.open(out / p["photo"]).size) <= 960
    for p in poles["poles"]:
        assert {"id", "lat", "lon", "inspected", "photo", "photo_source", "record"} <= set(p)
        assert set(p["record"]) >= {
            "material",
            "age_years",
            "install_year",
            "inspections",
            "shell_thickness_change_10y",
            "transformer",
            "last_maintained_year",
            "years_since_maintenance",
            "bushfire_zone",
            "customers_served",
            "critical_customer",
            "critical_customer_type",
            "source_history_id",
        }
        if not p["inspected"]:
            assert p["photo"] is None


def test_decisions_json_shape(register, raw_dir, tmp_path):
    out, _, _ = build(register, raw_dir, tmp_path)
    doc = json.loads((out / "decisions.json").read_text())
    poles = json.loads((out / "poles.json").read_text())["poles"]
    assert doc["backend"] == "mock" and doc["mock"] is True
    assert {"model", "hardware", "recorded_at", "engineer_threshold", "decisions"} <= set(doc)
    assert doc["engineer_threshold"] == 0.6
    assert set(doc["decisions"]) == {p["id"] for p in poles if p["inspected"]}
    for entry in doc["decisions"].values():
        assert set(entry["v1"]["answers"]) == {"lean", "crossarm", "vegetation", "health"}
        assert set(entry["v2"]["answers"]) == {"action", "safety_risk_now"}
        assert {"latency_ms", "model_ms"} <= set(entry["v1"]) | set(entry["v2"])
        assert entry["action"] in {"replace", "maintain", "defer", "reinspect"}
        assert entry["action"] == entry["v2"]["answers"]["action"]["choice"]
        assert abs(entry["confidence"] - entry["v2"]["answers"]["action"]["probabilities"][entry["action"]]) < 0.001
        lean = entry["v1"]["answers"]["lean"]["choice"]
        weak = entry["confidence"] < 0.6
        assert entry["route"] == ("engineer" if weak or lean == "cannot_assess" else "auto")
        assert entry["v1"]["model_ms"] is None  # mock never claims a model time


def test_flip_json_has_two_contrasting_sides(register, raw_dir, tmp_path):
    out, _, _ = build(register, raw_dir, tmp_path)
    flip = json.loads((out / "flip.json").read_text())
    assert flip["backend"] == "mock" and (out / flip["photo"]).is_file()
    a, b = flip["sides"]
    assert a["v1"] == b["v1"]  # one photo, one set of findings
    assert b["record"]["critical_customer"] is True
    assert b["record"]["bushfire_zone"] == "high"
    assert demo_data.badness(a["record"]) < demo_data.badness(b["record"])
    for side in (a, b):
        assert {"label", "record", "v1", "v2", "action", "confidence", "route"} <= set(side)


def test_eval_summary_is_copied_and_marked_mock(register, raw_dir, tmp_path):
    run_dir = tmp_path / "run"
    evaluate.run_eval(MockBackend(seed=7), ["lean"], raw_dir, run_dir, 4, 1, "none (mock backend)", log=lambda _: None)
    out = tmp_path / "out"
    out.mkdir()
    demo_data.build_demo_data(
        register, raw_dir, out, run_dir, MockBackend(seed=7), "none (mock backend)", inspect_n=5, log=lambda _: None
    )
    summary = json.loads((out / "eval_summary.json").read_text())
    assert summary["backend"] == "mock" and summary["mock"] is True
    assert summary["tasks"]["lean"]["n"] == 4


def test_decisions_are_cached_and_policy_change_reruns_v2(register, raw_dir, tmp_path):
    out, run_dir, _ = build(register, raw_dir, tmp_path, inspect_n=6)
    cache = run_dir / "demo_decisions.jsonl"
    assert len(cache.read_text().splitlines()) == 6
    build(register, raw_dir, tmp_path, inspect_n=6)
    assert len(cache.read_text().splitlines()) == 6  # nothing re-run
    backend = MockBackend(seed=7)
    demo_data.build_demo_data(
        register, raw_dir, out, run_dir, backend, "x", inspect_n=6, policy="Replace nothing.", log=lambda _: None
    )
    assert len(cache.read_text().splitlines()) == 12
    assert json.loads((out / "decisions.json").read_text())["policy_is_default"] is False


def test_stale_photos_are_removed(register, raw_dir, tmp_path):
    out, _, _ = build(register, raw_dir, tmp_path)
    (out / "photos" / "stale.jpg").write_bytes(b"x")
    build(register, raw_dir, tmp_path)
    assert not (out / "photos" / "stale.jpg").exists()


def test_cli_end_to_end_with_mock(raw_dir, tmp_path, monkeypatch):
    root = tmp_path / "repo"
    (root / "data").mkdir(parents=True)
    (root / "pyproject.toml").write_text('[project]\nname = "polehealth"\n')
    shutil.copytree(raw_dir, root / "data" / "raw")
    shutil.copy(FIXTURES / "osm_tiny.json", root / "data" / "raw" / "osm_toowoomba.json")
    monkeypatch.setenv("POLEHEALTH_ROOT", str(root))
    runner = CliRunner()

    result = runner.invoke(main, ["build-register"])
    assert result.exit_code == 0, result.output
    assert len(json.loads((root / "data" / "register.json").read_text())["poles"]) == 12

    result = runner.invoke(main, ["eval", "--backend", "mock", "--task", "all", "--limit", "3"])
    assert result.exit_code == 0, result.output
    (run_dir,) = list((root / "runs").iterdir())
    assert run_dir.name.endswith("-mock")
    assert (run_dir / "results.jsonl").is_file() and (run_dir / "eval_summary.json").is_file()

    result = runner.invoke(main, ["demo-data", "--backend", "mock", "--inspect", "8"])
    assert result.exit_code == 0, result.output
    assert (root / "demo" / "data" / "poles.json").is_file()
    assert json.loads((root / "demo" / "data" / "decisions.json").read_text())["backend"] == "mock"

    result = runner.invoke(main, ["cost", "--gpu-hourly-usd", "2", "--run", str(run_dir)])
    assert result.exit_code != 0 and "mock" in result.output

    photo = next((root / "demo" / "data" / "photos").glob("*.jpg"))
    result = runner.invoke(main, ["ask", "--backend", "mock", "--image", str(photo), "--record-id", "848000651"])
    assert result.exit_code != 0  # id is not in this tiny register
    first = json.loads((root / "data" / "register.json").read_text())["poles"][0]["id"]
    result = runner.invoke(main, ["ask", "--backend", "mock", "--image", str(photo), "--record-id", first])
    assert result.exit_code == 0 and "action:" in result.output


def test_flip_records_pick_extremes(register):
    young, old = demo_data.pick_flip_records(register["poles"])
    assert demo_data.badness(young["record"]) < demo_data.badness(old["record"])
    assert old["record"]["critical_customer"] is True


def test_register_fixture_matches_helper(osm_payload, tmp_path):
    xlsx = write_wrc_xlsx(Path(tmp_path) / "w.xlsx")
    assert build_register(osm_payload, load_wrc(xlsx))["year_shift"] == 7
