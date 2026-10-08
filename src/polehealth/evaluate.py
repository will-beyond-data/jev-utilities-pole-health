"""Run the benchmark tasks against a backend and summarise them. Runs are resumable."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import __version__, images
from .jev import Backend, BackendError
from .metrics import task_summary
from .questions import (
    CROSSARM_OPTIONS,
    LEAN_OPTIONS,
    VEGETATION_OPTIONS,
    argmax_key,
    record_request,
    top_probability,
    v1_request,
)
from .register import benchmark_rows
from .sources import (
    Sample,
    crossarm_samples,
    lean_samples,
    take_balanced,
    vegetation_samples,
    wrc_path,
)

IMAGE_TASKS = ("lean", "crossarm", "vegetation")
TASKS = (*IMAGE_TASKS, "record_health_index")
TASK_LABELS: dict[str, tuple[str, ...]] = {
    "lean": LEAN_OPTIONS,
    "crossarm": CROSSARM_OPTIONS,
    "vegetation": VEGETATION_OPTIONS,
    "record_health_index": ("1", "2", "3", "4", "5"),
}
SAMPLE_LOADERS: dict[str, Callable[[Path], list[Sample]]] = {
    "lean": lean_samples,
    "crossarm": crossarm_samples,
    "vegetation": vegetation_samples,
}

Log = Callable[[str], None]


@dataclass(frozen=True)
class Item:
    task: str
    id: str
    truth: str
    sample: Sample | None = None  # image tasks
    record: Mapping[str, Any] | None = None  # record_health_index
    health_index: int | None = None


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def new_run_dir(runs: Path, backend_name: str) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = runs / f"{stamp}-{backend_name}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def collect_items(task: str, raw: Path, limit: int | None, seed: int = 7) -> list[Item]:
    if task in IMAGE_TASKS:
        samples = SAMPLE_LOADERS[task](raw)
        items = [Item(task, s.id, s.label, sample=s) for s in samples]
    else:
        path = wrc_path(raw, "wrc50")
        if not path.is_file():
            return []
        items = [
            Item(task, r["id"], str(r["health_index"]), record=r["fields"], health_index=r["health_index"])
            for r in benchmark_rows(path)
        ]
    return take_balanced(items, lambda i: i.truth, limit, seed)


def build_request(item: Item, model: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """(request body, mock hints). Hints carry the ground truth so the mock backend can look plausible."""
    if item.sample is not None:
        return v1_request(images.data_url(item.sample.path), model), {item.task: item.truth}
    assert item.record is not None and item.health_index is not None
    return record_request(item.record, model), {"health": item.health_index - 1}


def predict(item: Item, answers: Mapping[str, Any]) -> tuple[str, float]:
    """(predicted label, confidence) for the item's task. Confidence is the top option probability."""
    if item.task in IMAGE_TASKS:
        answer = answers[item.task]
        return argmax_key(answer), top_probability(answer)
    answer = answers["health"]
    return str(int(argmax_key(answer)) + 1), top_probability(answer)


def run_item(backend: Backend, item: Item) -> dict[str, Any]:
    base = {"task": item.task, "id": item.id, "truth": item.truth}
    try:
        body, hints = build_request(item, backend.model)
        result = backend.decide(body, hints)
        pred, confidence = predict(item, result.response["answers"])
    except (BackendError, KeyError, ValueError, OSError) as error:
        return {**base, "error": f"{type(error).__name__}: {error}", "recorded_at": now_iso()}
    return {
        **base,
        "pred": pred,
        "confidence": confidence,
        "correct": pred == item.truth,
        "answers": result.response["answers"],
        "latency_ms": result.latency_ms,
        "model_ms": result.model_ms,
        "attempts": result.attempts,
        "error": None,
        "recorded_at": now_iso(),
    }


def read_results(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def finished_ids(results: list[dict[str, Any]]) -> set[tuple[str, str]]:
    return {(r["task"], r["id"]) for r in results if not r.get("error")}


def _run_pending(
    backend: Backend, pending: list[Item], handle: Callable[[dict[str, Any]], None], concurrency: int, log: Log
) -> None:
    started = time.perf_counter()
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = [pool.submit(run_item, backend, item) for item in pending]
        for future in as_completed(futures):
            handle(future.result())
            done += 1
            if done % 10 == 0 or done == len(pending):
                log(f"  {done}/{len(pending)} done ({done / (time.perf_counter() - started):.1f}/s)")


def run_eval(
    backend: Backend,
    tasks: list[str],
    raw: Path,
    run_dir: Path,
    limit: int | None,
    concurrency: int,
    hardware: str,
    seed: int = 7,
    base_url: str | None = None,
    log: Log = print,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "results.jsonl"
    already = finished_ids(read_results(results_path))
    write_run_meta(run_dir, backend, tasks, hardware, limit, seed, base_url)
    with results_path.open("a", encoding="utf-8") as handle:

        def write(row: dict[str, Any]) -> None:
            handle.write(json.dumps(row) + "\n")
            handle.flush()

        for task in tasks:
            items = collect_items(task, raw, limit, seed)
            if not items:
                log(f"{task}: no local data, run `polehealth fetch` first")
                continue
            pending = [i for i in items if (i.task, i.id) not in already]
            log(f"{task}: {len(items)} items, {len(items) - len(pending)} already done, {len(pending)} to run")
            _run_pending(backend, pending, write, concurrency, log)
    return summarise_run(run_dir)


def write_run_meta(
    run_dir: Path, backend: Backend, tasks: list[str], hardware: str, limit: int | None, seed: int, base_url: str | None
) -> None:
    path = run_dir / "run.json"
    existing = json.loads(path.read_text()) if path.is_file() else {}
    meta = {
        "backend": backend.name,
        "model": backend.model,
        "hardware": hardware,
        "base_url": base_url,
        "tasks": sorted(set(existing.get("tasks", [])) | set(tasks)),
        "limit": limit,
        "seed": seed,
        "started_at": existing.get("started_at") or now_iso(),
        "polehealth_version": __version__,
    }
    path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def round_floats(value: Any, digits: int = 4) -> Any:
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, dict):
        return {k: round_floats(v, digits) for k, v in value.items()}
    if isinstance(value, list):
        return [round_floats(v, digits) for v in value]
    return value


def summarise_run(run_dir: Path) -> dict[str, Any]:
    """Compute eval_summary.json from results.jsonl. Keeps an existing `cost` block."""
    meta = json.loads((run_dir / "run.json").read_text())
    rows = read_results(run_dir / "results.jsonl")
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (row["task"], row["id"])
        if not row.get("error") or key not in latest:
            latest[key] = row
    tasks: dict[str, Any] = {}
    for task in TASKS:
        good = [r for (t, _), r in latest.items() if t == task and not r.get("error")]
        failed = sum(1 for (t, _), r in latest.items() if t == task and r.get("error"))
        if not good:
            continue
        entry = task_summary(
            [r["truth"] for r in good],
            [r["pred"] for r in good],
            [r["confidence"] for r in good],
            TASK_LABELS[task],
            [r["latency_ms"] for r in good],
            [r["model_ms"] for r in good],
        )
        entry.update(
            {
                "errors": failed,
                "backend": meta["backend"],
                "hardware": meta["hardware"],
                "recorded_at": max(r["recorded_at"] for r in good),
            }
        )
        tasks[task] = entry
    summary: dict[str, Any] = {
        "backend": meta["backend"],
        "mock": meta["backend"] == "mock",
        "model": meta["model"],
        "hardware": meta["hardware"],
        "recorded_at": max((t["recorded_at"] for t in tasks.values()), default=None),
        "started_at": meta["started_at"],
        "limit": meta["limit"],
        "seed": meta["seed"],
        "tasks": tasks,
    }
    path = run_dir / "eval_summary.json"
    if path.is_file():
        previous = json.loads(path.read_text())
        if "cost" in previous:
            summary["cost"] = previous["cost"]
    summary = round_floats(summary)
    path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def latest_run(runs: Path, backend_name: str | None = None) -> Path | None:
    candidates = sorted(p for p in runs.glob("*") if p.is_dir() and (p / "run.json").is_file())
    if backend_name:
        candidates = [p for p in candidates if json.loads((p / "run.json").read_text())["backend"] == backend_name]
    return candidates[-1] if candidates else None


def compute_cost(run_dir: Path, gpu_hourly_usd: float, aud_per_usd: float = 1.52) -> dict[str, Any]:
    """Cost per 1,000 poles from measured server-side model time. Only meaningful for the http backend."""
    meta = json.loads((run_dir / "run.json").read_text())
    if meta["backend"] != "http":
        raise ValueError(f"cost needs measured GPU time; this run used the {meta['backend']} backend")
    per_pole: list[float] = []
    basis = "v1 + v2 per pole (demo decisions)"
    for row in read_results(run_dir / "demo_decisions.jsonl"):
        entry = row["entry"]
        if entry["v1"]["model_ms"] is not None and entry["v2"]["model_ms"] is not None:
            per_pole.append(entry["v1"]["model_ms"] + entry["v2"]["model_ms"])
    if not per_pole:
        basis = "v1 only (eval image tasks); a v2 request is not included, so this understates the cost"
        per_pole = [
            r["model_ms"]
            for r in read_results(run_dir / "results.jsonl")
            if r["task"] in IMAGE_TASKS and not r.get("error") and r.get("model_ms") is not None
        ]
    if not per_pole:
        raise ValueError("no measured model_ms in this run; run `eval` or `demo-data` against the http backend first")
    mean_ms = sum(per_pole) / len(per_pole)
    per_hour = 3_600_000 / mean_ms
    usd = gpu_hourly_usd / per_hour * 1000
    cost = {
        "gpu_hourly_usd": gpu_hourly_usd,
        "aud_per_usd": aud_per_usd,
        "basis": basis,
        "n_poles": len(per_pole),
        "mean_model_ms_per_pole": round(mean_ms, 2),
        "decisions_per_hour": round(per_hour),
        "usd_per_1000_poles": round(usd, 4),
        "aud_per_1000_poles": round(usd * aud_per_usd, 4),
        "note": "Assumes the GPU is busy with back-to-back requests. "
        "Idle time, model load and image upload are not counted.",
    }
    summary_path = run_dir / "eval_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else summarise_run(run_dir)
    summary["cost"] = cost
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return cost
