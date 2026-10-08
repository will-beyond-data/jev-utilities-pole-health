"""Build the JSON and photos the static demo reads (demo/data/)."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import images
from .evaluate import round_floats, summarise_run
from .jev import Backend, Result
from .questions import (
    DEFAULT_POLICY,
    ENGINEER_THRESHOLD,
    chosen_action,
    photo_findings,
    route,
    v1_request,
    v2_request,
)
from .sources import Sample, crossarm_samples, lean_samples, vegetation_samples

TOWN = "Toowoomba, QLD"
PHOTO_CREDIT = "PD-Defect, APEPDCL, CC BY 4.0"
DATASET_WEIGHTS = (("pd_lean", 0.5), ("pd_crossarm", 0.3), ("pd_vegetation", 0.2))
DATASET_QUESTION = {"pd_lean": "lean", "pd_crossarm": "crossarm", "pd_vegetation": "vegetation"}
ATTRIBUTION = [
    "Pole photos: PD-Defect dataset (Masabathula et al., APEPDCL, Eastern Andhra Pradesh, India), CC BY 4.0, "
    "doi:10.5281/zenodo.18074045. Photos resized.",
    "Pole locations: © OpenStreetMap contributors, ODbL.",
    "Inspection histories: Western Red Cedar 50-ft pole dataset, Kaggle (utilityanalytics). Inspection years shifted "
    "forward by 7.",
]
PAIRING_NOTE = (
    "Photos, locations and asset records come from three unrelated public sources. The pairing is synthetic: "
    "no photo shows the pole its record describes."
)

Log = Callable[[str], None]


def policy_sha(policy: str) -> str:
    return hashlib.sha256(policy.encode()).hexdigest()[:12]


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 1)


def side_result(result: Result) -> dict[str, Any]:
    return {
        "answers": round_floats(result.response["answers"]),
        "latency_ms": _round(result.latency_ms),
        "model_ms": _round(result.model_ms),
    }


def decide_pole(
    backend: Backend,
    image_url: str,
    record: Mapping[str, Any] | None,
    policy: str,
    threshold: float,
    hints: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """v1 (photo to health), then v2 (photo plus record to action) when a record is given."""
    first = backend.decide(v1_request(image_url, backend.model), hints)
    entry: dict[str, Any] = {"v1": side_result(first)}
    if record is None:
        lean = first.response["answers"]["lean"]["choice"]
        return {**entry, "v2": None, "action": None, "confidence": None, "route": route(1.0, lean, threshold)}
    entry.update(decide_v2(backend, first, image_url, record, policy, threshold))
    return entry


def decide_v2(
    backend: Backend, first: Result, image_url: str, record: Mapping[str, Any], policy: str, threshold: float
) -> dict[str, Any]:
    answers = first.response["answers"]
    second = backend.decide(v2_request(record, photo_findings(answers), policy, image_url, backend.model))
    action, probability = chosen_action(second.response["answers"])
    return {
        "v2": side_result(second),
        "action": action,
        "confidence": round(probability, 3),
        "route": route(probability, answers["lean"]["choice"], threshold),
    }


# ---------------------------------------------------------------- selection


def available_photos(raw: Path) -> dict[str, list[Sample]]:
    return {
        "pd_lean": lean_samples(raw),
        "pd_crossarm": crossarm_samples(raw),
        "pd_vegetation": vegetation_samples(raw),
    }


def pick_photos(pool: Mapping[str, Sequence[Sample]], n: int, seed: int) -> list[Sample]:
    """n distinct photos in roughly 50/30/20 lean/crossarm/vegetation mix, topped up from what is available."""
    rng = random.Random(f"{seed}|photos")
    shuffled = {name: rng.sample(list(items), len(items)) for name, items in pool.items()}
    n = min(n, sum(len(v) for v in shuffled.values()))
    quota = {name: min(round(n * weight), len(shuffled[name])) for name, weight in DATASET_WEIGHTS}
    shortfall = n - sum(quota.values())
    for name, _ in DATASET_WEIGHTS:  # top up, or trim if rounding overshot
        extra = min(shortfall, len(shuffled[name]) - quota[name])
        quota[name] += extra
        shortfall -= extra
    while shortfall < 0:
        biggest = max(quota, key=quota.__getitem__)
        quota[biggest] -= 1
        shortfall += 1
    chosen = [s for name, _ in DATASET_WEIGHTS for s in shuffled[name][: quota[name]]]
    rng.shuffle(chosen)
    return chosen


def photo_source(sample: Sample) -> dict[str, Any]:
    return {
        "dataset": sample.dataset,
        "label": sample.native_label,
        "credit": PHOTO_CREDIT,
        "split": sample.split,
        "file": sample.path.name,
    }


def badness(record: Mapping[str, Any]) -> float:
    """Higher means a worse asset record. Only used to pick the two contrasting flip records."""
    inspections = record["inspections"]
    health = inspections[-1].get("health_index", 3)
    since = record["years_since_maintenance"]
    score = record["age_years"] / 100 + (5 - health) * 0.5 - (record["shell_thickness_change_10y"] or 0) * 3
    score += {"high": 1.0, "medium": 0.5, "low": 0.0}[record["bushfire_zone"]]
    score += 1.0 if since is None else since / 10
    return score


def pick_flip_records(poles: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    """A young, low-risk, recently maintained record and an old, high-bushfire one with a critical customer."""
    calm = [
        p
        for p in poles
        if p["record"]["bushfire_zone"] == "low"
        and not p["record"]["critical_customer"]
        and p["record"]["years_since_maintenance"] is not None
    ] or list(poles)
    harsh = [p for p in poles if p["record"]["bushfire_zone"] == "high"] or list(poles)
    young = min(calm, key=lambda p: (badness(p["record"]), p["id"]))
    old = max(harsh, key=lambda p: (badness(p["record"]), p["id"]))
    old_record = json.loads(json.dumps(old["record"]))
    overrides = []
    if not old_record["critical_customer"]:
        old_record.update({"critical_customer": True, "critical_customer_type": "aged_care"})
        old_record["customers_served"] = max(old_record["customers_served"], 120)
        overrides = ["critical_customer", "critical_customer_type", "customers_served"]
    return (
        {"from_pole": young["id"], "overrides": [], "record": young["record"]},
        {"from_pole": old["id"], "overrides": overrides, "record": old_record},
    )


# ---------------------------------------------------------------- build


def _load_cache(path: Path, sha: str) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    cache = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row["policy_sha"] == sha:
                cache[row["pole_id"]] = row
    return cache


def build_demo_data(
    register: Mapping[str, Any],
    raw: Path,
    out_dir: Path,
    run_dir: Path,
    backend: Backend,
    hardware: str,
    inspect_n: int = 400,
    policy: str = DEFAULT_POLICY,
    threshold: float = ENGINEER_THRESHOLD,
    seed: int = 7,
    concurrency: int = 1,
    log: Log = print,
) -> dict[str, Any]:
    mock = backend.name == "mock"
    pool = available_photos(raw)
    photos = pick_photos(pool, inspect_n + 1, seed)  # one spare for the flip view
    if not photos:
        raise RuntimeError("no local photos; run `polehealth fetch` first")
    flip_photo = next((s for s in photos if s.dataset == "pd_lean" and s.label == "straight"), photos[-1])
    inspect_photos = [s for s in photos if s is not flip_photo][:inspect_n]
    poles = list(register["poles"])
    rng = random.Random(f"{seed}|inspected")
    chosen_poles = rng.sample(poles, min(len(inspect_photos), len(poles)))
    assignment = dict(zip((p["id"] for p in chosen_poles), inspect_photos, strict=False))
    log(f"inspecting {len(assignment)} of {len(poles)} poles, one photo each, none reused")

    photo_dir = out_dir / "photos"
    photo_dir.mkdir(parents=True, exist_ok=True)
    wanted = {s.photo_name for s in [*assignment.values(), flip_photo]}
    for stale in photo_dir.glob("*.jpg"):
        if stale.name not in wanted:
            stale.unlink()
    for sample in [*assignment.values(), flip_photo]:
        images.save_demo_photo(sample.path, photo_dir / sample.photo_name)

    sha = policy_sha(policy)
    cache_path = run_dir / "demo_decisions.jsonl"
    cache = _load_cache(cache_path, sha)
    by_id = {p["id"]: p for p in poles}
    todo = [pid for pid in assignment if pid not in cache or cache[pid]["photo"] != assignment[pid].photo_name]
    log(f"decisions: {len(assignment) - len(todo)} cached, {len(todo)} to run on {backend.name}")

    def work(pole_id: str) -> dict[str, Any]:
        sample = assignment[pole_id]
        hints = {DATASET_QUESTION[sample.dataset]: sample.label}
        entry = decide_pole(backend, images.data_url(sample.path), by_id[pole_id]["record"], policy, threshold, hints)
        return {"pole_id": pole_id, "photo": sample.photo_name, "policy_sha": sha, "entry": entry}

    with cache_path.open("a", encoding="utf-8") as handle, ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool_:
        for done, row in enumerate(pool_.map(work, todo), start=1):
            cache[row["pole_id"]] = row
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            if done % 25 == 0 or done == len(todo):
                log(f"  {done}/{len(todo)} poles decided")

    recorded_at = datetime.now(UTC).isoformat(timespec="seconds")
    header = {"backend": backend.name, "mock": mock, "model": backend.model, "hardware": hardware}
    decisions_doc = {
        **header,
        "recorded_at": recorded_at,
        "engineer_threshold": threshold,
        "policy": policy,
        "policy_is_default": policy == DEFAULT_POLICY,
        "decisions": {pid: cache[pid]["entry"] for pid in assignment},
    }
    poles_doc = {
        "generated_at": recorded_at,
        "backend": backend.name,
        "mock": mock,
        "town": TOWN,
        "attribution": ATTRIBUTION,
        "pairing_note": PAIRING_NOTE,
        "poles": [_pole_entry(p, assignment.get(p["id"])) for p in poles],
    }
    flip_doc = _build_flip(backend, header, flip_photo, poles, policy, threshold, recorded_at)
    summary = summarise_run(run_dir)

    for name, doc in (
        ("poles.json", poles_doc),
        ("decisions.json", decisions_doc),
        ("flip.json", flip_doc),
        ("eval_summary.json", summary),
    ):
        indent = None if name == "poles.json" else 1
        (out_dir / name).write_text(json.dumps(doc, indent=indent, allow_nan=False) + "\n", encoding="utf-8")
    return {"poles": poles_doc, "decisions": decisions_doc, "flip": flip_doc, "eval_summary": summary}


def _pole_entry(pole: Mapping[str, Any], sample: Sample | None) -> dict[str, Any]:
    return {
        "id": pole["id"],
        "lat": pole["lat"],
        "lon": pole["lon"],
        "inspected": sample is not None,
        "photo": f"photos/{sample.photo_name}" if sample else None,
        "photo_source": photo_source(sample) if sample else None,
        "record": pole["record"],
    }


def _build_flip(
    backend: Backend,
    header: Mapping[str, Any],
    photo: Sample,
    poles: Sequence[Mapping[str, Any]],
    policy: str,
    threshold: float,
    recorded_at: str,
) -> dict[str, Any]:
    url = images.data_url(photo.path)
    hints = {DATASET_QUESTION[photo.dataset]: photo.label}
    first = backend.decide(v1_request(url, backend.model), hints)
    young, old = pick_flip_records(poles)
    doc: dict[str, Any] = {
        **header,
        "recorded_at": recorded_at,
        "engineer_threshold": threshold,
        "photo": f"photos/{photo.photo_name}",
        "photo_source": photo_source(photo),
    }
    sides = []
    for key, label, side in (
        ("a", "Young, low risk, recently maintained", young),
        ("b", "Old, bushfire zone, critical customer", old),
    ):
        decided = {"v1": side_result(first), **decide_v2(backend, first, url, side["record"], policy, threshold)}
        sides.append(
            {
                "key": key,
                "label": label,
                "from_pole": side["from_pole"],
                "overrides": side["overrides"],
                "record": side["record"],
                **decided,
            }
        )
    doc["sides"] = sides
    return doc
