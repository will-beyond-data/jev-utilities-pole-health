import json

import pytest

from polehealth import evaluate
from polehealth.jev import MockBackend


def run(raw_dir, tmp_path, tasks=None, limit=4, name="run"):
    run_dir = tmp_path / name
    summary = evaluate.run_eval(
        MockBackend(seed=1),
        tasks or list(evaluate.TASKS),
        raw_dir,
        run_dir,
        limit,
        1,
        "none (mock backend)",
        log=lambda _: None,
    )
    return run_dir, summary


def test_eval_writes_results_and_summary(raw_dir, tmp_path):
    run_dir, summary = run(raw_dir, tmp_path)
    rows = [json.loads(line) for line in (run_dir / "results.jsonl").read_text().splitlines()]
    assert len(rows) == 16 and all(r["error"] is None for r in rows)
    row = next(r for r in rows if r["task"] == "lean")
    assert {"id", "truth", "pred", "confidence", "latency_ms", "model_ms", "answers"} <= set(row)
    assert row["model_ms"] is None
    assert json.loads((run_dir / "eval_summary.json").read_text()) == summary
    assert set(summary["tasks"]) == set(evaluate.TASKS)
    assert summary["backend"] == "mock" and summary["mock"] is True
    for entry in summary["tasks"].values():
        assert {
            "n",
            "accuracy",
            "macro_f1",
            "confusion",
            "ece",
            "reliability",
            "latency_ms",
            "model_ms",
            "backend",
            "hardware",
            "recorded_at",
        } <= set(entry)
        assert entry["n"] == 4


def test_resume_skips_finished_items(raw_dir, tmp_path):
    run_dir, _ = run(raw_dir, tmp_path, tasks=["lean"], limit=4)
    before = (run_dir / "results.jsonl").read_text()
    summary = evaluate.run_eval(MockBackend(seed=1), ["lean"], raw_dir, run_dir, 4, 1, "x", log=lambda _: None)
    assert (run_dir / "results.jsonl").read_text() == before
    assert summary["tasks"]["lean"]["n"] == 4
    evaluate.run_eval(MockBackend(seed=1), ["lean"], raw_dir, run_dir, 6, 1, "x", log=lambda _: None)
    assert len((run_dir / "results.jsonl").read_text().splitlines()) >= 6


def test_crossarm_truth_is_two_class_and_record_labels_are_1_to_5(raw_dir, tmp_path):
    _, summary = run(raw_dir, tmp_path, limit=None)
    assert set(summary["tasks"]["crossarm"]["support"]) <= {"straight", "tilted"}
    assert set(summary["tasks"]["record_health_index"]["support"]) <= {"1", "2", "3", "4", "5"}
    assert summary["tasks"]["record_health_index"]["n"] == 16


def test_missing_data_is_skipped_not_fatal(tmp_path):
    summary = evaluate.run_eval(
        MockBackend(), ["lean"], tmp_path / "empty", tmp_path / "r", 5, 1, "x", log=lambda _: None
    )
    assert summary["tasks"] == {}


def test_cost_needs_http_and_measured_time(raw_dir, tmp_path):
    run_dir, _ = run(raw_dir, tmp_path, tasks=["lean"])
    with pytest.raises(ValueError, match="mock backend"):
        evaluate.compute_cost(run_dir, 2.0)


def test_cost_maths_from_measured_model_ms(tmp_path):
    run_dir = tmp_path / "r"
    run_dir.mkdir()
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "backend": "http",
                "model": "m",
                "hardware": "1x GPU",
                "started_at": "t",
                "limit": None,
                "seed": 7,
            }
        )
    )
    rows = [
        {
            "task": "lean",
            "id": str(i),
            "truth": "straight",
            "pred": "straight",
            "confidence": 0.9,
            "latency_ms": 60.0,
            "model_ms": 50.0,
            "error": None,
            "recorded_at": "2026-10-08T00:00:00+00:00",
        }
        for i in range(4)
    ]
    (run_dir / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    cost = evaluate.compute_cost(run_dir, gpu_hourly_usd=3.6, aud_per_usd=1.5)
    # 50 ms per pole -> 72,000 per hour -> USD 3.6 / 72,000 * 1,000 = 0.05
    assert cost["decisions_per_hour"] == 72000
    assert cost["usd_per_1000_poles"] == pytest.approx(0.05)
    assert cost["aud_per_1000_poles"] == pytest.approx(0.075)
    assert "v1 only" in cost["basis"]
    saved = json.loads((run_dir / "eval_summary.json").read_text())
    assert saved["cost"]["gpu_hourly_usd"] == 3.6
    assert "cost" in evaluate.summarise_run(run_dir)  # a re-summary keeps the cost block
