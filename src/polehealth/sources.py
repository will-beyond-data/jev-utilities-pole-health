"""Download and discover the source data. Everything lands in data/raw/ (gitignored) and downloads are idempotent."""

from __future__ import annotations

import ast
import csv
import io
import json
import os
import random
import re
import time
import zipfile
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

HF_BASE = "https://huggingface.co/datasets/EPDCL/pd-defect"
KAGGLE_URL = "https://www.kaggle.com/api/v1/datasets/download/utilityanalytics/{slug}"
KAGGLE_SLUGS = {"wrc50": "western-red-cedar-50ft-pole", "wrc45": "utility-power-pole-condition-dataset1"}
OVERPASS_URLS = ("https://overpass.private.coffee/api/interpreter", "https://overpass.kumi.systems/api/interpreter")
TOOWOOMBA_BBOX = (-27.62, 151.88, -27.50, 152.02)  # south, west, north, east

LEAN_DIR = "pole_lean_assessment/classification"
CROSSARM_DIR = "crossarm_top_cleat_tilt_assessment/object_detection"
VEGETATION_DIR = "vegetation_conductor"
LEAN_SPLITS = ("test", "val")
SOURCE_KEYS = ("pd_lean", "pd_crossarm", "pd_vegetation", "wrc50", "wrc45", "osm")

LEAN_LABELS = {"Leaned": "leaning", "Straight": "straight", "Rejected": "cannot_assess"}
VEGETATION_LABELS = {"Risky": "encroaching", "Safe": "clear"}

Log = Callable[[str], None]


def repo_root() -> Path:
    """The repo checkout: $POLEHEALTH_ROOT, else the nearest parent of the cwd that holds this project's pyproject."""
    override = os.environ.get("POLEHEALTH_ROOT")
    if override:
        return Path(override).resolve()
    here = Path.cwd().resolve()
    for candidate in (here, *here.parents):
        pyproject = candidate / "pyproject.toml"
        if pyproject.is_file() and 'name = "polehealth"' in pyproject.read_text(encoding="utf-8"):
            return candidate
    return here


def raw_dir(root: Path | None = None) -> Path:
    return (root or repo_root()) / "data" / "raw"


def hf_dir(raw: Path) -> Path:
    return raw / "pd-defect"


def wrc_path(raw: Path, key: str) -> Path:
    return raw / f"{key}.xlsx"


def osm_path(raw: Path) -> Path:
    return raw / "osm_toowoomba.json"


# ---------------------------------------------------------------- HTTP helpers


def _client() -> httpx.Client:
    headers = {"User-Agent": "polehealth-demo/0.1"}
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return httpx.Client(headers=headers, timeout=httpx.Timeout(60.0, connect=20.0), follow_redirects=True)


def _retrying_get(client: httpx.Client, url: str, attempts: int = 5, **kwargs: Any) -> httpx.Response:
    last = "no attempt made"
    for attempt in range(attempts):
        try:
            response = client.get(url, **kwargs)
        except httpx.TransportError as error:
            last = repr(error)
        else:
            if response.status_code == 200:
                return response
            last = f"HTTP {response.status_code}"
            if response.status_code not in {429, 500, 502, 503, 504}:
                break
        time.sleep(min(2.0 * 2**attempt, 30.0))
    raise RuntimeError(f"GET {url} failed: {last}")


def hf_list(client: httpx.Client, folder: str) -> list[dict[str, Any]]:
    """Files directly inside a dataset folder, following the paginated `Link: <...>; rel="next"` header."""
    url: str | None = f"https://huggingface.co/api/datasets/EPDCL/pd-defect/tree/main/{folder}"
    entries: list[dict[str, Any]] = []
    while url:
        response = _retrying_get(client, url)
        entries.extend(response.json())
        url = response.links.get("next", {}).get("url")
    return [e for e in entries if e["type"] == "file"]


def hf_fetch(client: httpx.Client, path: str, dest: Path, size: int | None = None) -> bool:
    """Download one dataset file. Returns False when it already exists (and matches `size` if known)."""
    if dest.is_file() and dest.stat().st_size > 0 and (size is None or dest.stat().st_size == size):
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".part")
    last = "no attempt made"
    for attempt in range(5):
        try:
            with client.stream("GET", f"{HF_BASE}/resolve/main/{path}") as response:
                if response.status_code != 200:
                    last = f"HTTP {response.status_code}"
                    if response.status_code not in {429, 500, 502, 503, 504}:
                        break
                else:
                    with partial.open("wb") as handle:
                        for chunk in response.iter_bytes(1 << 16):
                            handle.write(chunk)
                    partial.replace(dest)
                    return True
        except httpx.TransportError as error:
            last = repr(error)
        time.sleep(min(2.0 * 2**attempt, 30.0))
    raise RuntimeError(f"download of {path} failed: {last}")


def _fetch_many(client: httpx.Client, jobs: Sequence[tuple[str, int | None]], raw: Path, log: Log, label: str) -> int:
    """Download (path, size) jobs with a few threads. Returns how many files were newly downloaded."""
    done = fresh = 0

    def work(job: tuple[str, int | None]) -> bool:
        path, size = job
        return hf_fetch(client, path, hf_dir(raw) / path, size)

    with ThreadPoolExecutor(max_workers=6) as pool:
        for downloaded in pool.map(work, jobs):
            done += 1
            fresh += downloaded
            if done % 25 == 0 or done == len(jobs):
                log(f"  {label}: {done}/{len(jobs)} files ({fresh} new)")
    return fresh


def take_balanced(items: Sequence[Any], key: Callable[[Any], str], n: int | None, seed: int = 7) -> list[Any]:
    """Pick n items spread evenly over the groups given by `key`, deterministically. n=None keeps everything."""
    if n is None or n >= len(items):
        return list(items)
    rng = random.Random(seed)
    groups: dict[str, list[Any]] = defaultdict(list)
    for item in items:
        groups[key(item)].append(item)
    for group in groups.values():
        rng.shuffle(group)
    picked: list[Any] = []
    names = sorted(groups)
    while len(picked) < n and any(groups.values()):
        for name in names:
            if groups[name] and len(picked) < n:
                picked.append(groups[name].pop())
    return picked


# ---------------------------------------------------------------- pd-defect


def fetch_pd_lean(raw: Path, limit: int | None, log: Log) -> None:
    with _client() as client:
        jobs: list[tuple[str, int | None]] = []
        candidates: list[tuple[str, str, int]] = []  # (label, path, size)
        for split in LEAN_SPLITS:
            jobs.append((f"{LEAN_DIR}/{split}/{split}.csv", None))
            for label in LEAN_LABELS:
                for entry in hf_list(client, f"{LEAN_DIR}/{split}/{label}"):
                    candidates.append((label, entry["path"], entry["size"]))
        jobs.append((f"{LEAN_DIR}/labels.csv", None))
        chosen = take_balanced(candidates, lambda c: c[0], limit)
        jobs += [(path, size) for _, path, size in chosen]
        log(f"pd_lean: {len(chosen)} of {len(candidates)} test/val images")
        _fetch_many(client, jobs, raw, log, "pd_lean")


def fetch_pd_crossarm(raw: Path, limit: int | None, log: Log) -> None:
    base = f"{CROSSARM_DIR}/test"
    with _client() as client:
        images = hf_list(client, f"{base}/images")
        labels = hf_list(client, f"{base}/labels")
        _fetch_many(
            client,
            [(f"{CROSSARM_DIR}/data.yaml", None)] + [(e["path"], e["size"]) for e in labels],
            raw,
            log,
            "pd_crossarm labels",
        )
        classes = read_crossarm_classes(raw)
        by_stem = {Path(e["path"]).stem: e for e in images}
        usable = []
        for label_file in labels:
            stem = Path(label_file["path"]).stem
            label = crossarm_image_label(hf_dir(raw) / label_file["path"], classes)
            if label and stem in by_stem:
                usable.append((label, by_stem[stem]))
        chosen = take_balanced(usable, lambda u: u[0], limit)
        log(f"pd_crossarm: {len(chosen)} of {len(usable)} labelled test images")
        _fetch_many(client, [(e["path"], e["size"]) for _, e in chosen], raw, log, "pd_crossarm")


def fetch_pd_vegetation(raw: Path, limit: int | None, log: Log) -> None:
    with _client() as client:
        gt_path = f"{VEGETATION_DIR}/GT_ValidationSet.csv"
        hf_fetch(client, gt_path, hf_dir(raw) / gt_path)
        truth = read_vegetation_truth(raw)
        listed = {Path(e["path"]).name: e for e in hf_list(client, f"{VEGETATION_DIR}/valid")}
        usable = [(label, listed[name]) for name, label in truth.items() if name in listed]
        chosen = take_balanced(usable, lambda u: u[0], limit)
        log(f"pd_vegetation: {len(chosen)} of {len(usable)} labelled validation images")
        _fetch_many(client, [(e["path"], e["size"]) for _, e in chosen], raw, log, "pd_vegetation")


def read_crossarm_classes(raw: Path) -> list[str]:
    text = (hf_dir(raw) / CROSSARM_DIR / "data.yaml").read_text(encoding="utf-8")
    match = re.search(r"^names:\s*(\[.*\])\s*$", text, re.MULTILINE)
    if not match:
        raise ValueError("data.yaml has no `names: [...]` list")
    return list(ast.literal_eval(match.group(1)))


def crossarm_image_label(label_file: Path, classes: Sequence[str]) -> str | None:
    """Image-level label from YOLO boxes: tilted if any crossarm or top-cleat box is tilted, else straight."""
    if not label_file.is_file():
        return None
    ids = [int(line.split()[0]) for line in label_file.read_text().splitlines() if line.strip()]
    if not ids:
        return None
    return "tilted" if any(classes[i].endswith("_tilted") for i in ids) else "straight"


def read_vegetation_truth(raw: Path) -> dict[str, str]:
    """image file name -> encroaching | clear, from GT_ValidationSet.csv (header: ID,Ground_Truth)."""
    path = hf_dir(raw) / VEGETATION_DIR / "GT_ValidationSet.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["ID"].strip(): VEGETATION_LABELS[row["Ground_Truth"].strip()] for row in csv.DictReader(handle)}


# ---------------------------------------------------------------- local sample discovery


@dataclass(frozen=True)
class Sample:
    id: str  # unique and stable, e.g. pd_lean_test_100
    path: Path
    label: str  # canonical task label (straight, tilted, encroaching, ...)
    dataset: str  # pd_lean | pd_crossarm | pd_vegetation
    native_label: str  # the dataset's own label
    split: str

    @property
    def photo_name(self) -> str:
        return f"{self.id}.jpg"


def lean_samples(raw: Path) -> list[Sample]:
    samples = []
    for split in LEAN_SPLITS:
        for native, label in LEAN_LABELS.items():
            folder = hf_dir(raw) / LEAN_DIR / split / native
            for image in sorted(folder.glob("*.jpg")):
                samples.append(Sample(f"pd_lean_{split}_{image.stem}", image, label, "pd_lean", native, split))
    return samples


def crossarm_samples(raw: Path) -> list[Sample]:
    base = hf_dir(raw) / CROSSARM_DIR
    if not (base / "data.yaml").is_file():
        return []
    classes = read_crossarm_classes(raw)
    samples = []
    for image in sorted((base / "test" / "images").glob("*.jpg")):
        label = crossarm_image_label(base / "test" / "labels" / f"{image.stem}.txt", classes)
        if label:
            samples.append(Sample(f"pd_crossarm_test_{image.stem}", image, label, "pd_crossarm", label, "test"))
    return samples


def vegetation_samples(raw: Path) -> list[Sample]:
    gt = hf_dir(raw) / VEGETATION_DIR / "GT_ValidationSet.csv"
    if not gt.is_file():
        return []
    natives = {v: k for k, v in VEGETATION_LABELS.items()}
    samples = []
    for name, label in sorted(read_vegetation_truth(raw).items()):
        image = hf_dir(raw) / VEGETATION_DIR / "valid" / name
        if image.is_file():
            samples.append(
                Sample(f"pd_vegetation_valid_{Path(name).stem}", image, label, "pd_vegetation", natives[label], "valid")
            )
    return samples


# ---------------------------------------------------------------- Kaggle and Overpass


def fetch_wrc(raw: Path, key: str, log: Log) -> None:
    target = wrc_path(raw, key)
    if target.is_file() and target.stat().st_size > 0:
        log(f"{key}: already downloaded")
        return
    with _client() as client:
        response = _retrying_get(client, KAGGLE_URL.format(slug=KAGGLE_SLUGS[key]), timeout=180.0)
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        names = [n for n in archive.namelist() if n.lower().endswith(".xlsx")]
        if len(names) != 1:
            raise RuntimeError(f"expected one xlsx in the {key} zip, found {archive.namelist()}")
        raw.mkdir(parents=True, exist_ok=True)
        target.write_bytes(archive.read(names[0]))
    log(f"{key}: saved {target.name} ({target.stat().st_size // 1024} KiB)")


def overpass_query(bbox: tuple[float, float, float, float] = TOOWOOMBA_BBOX) -> str:
    south, west, north, east = bbox
    return f"[out:json][timeout:90];node[power=pole]({south},{west},{north},{east});out body;"


def fetch_osm(raw: Path, log: Log, attempts: int = 8, sleep: Callable[[float], None] = time.sleep) -> None:
    target = osm_path(raw)
    if target.is_file() and target.stat().st_size > 0:
        log("osm: already downloaded")
        return
    override = os.environ.get("OVERPASS_URL")
    urls = (override,) if override else OVERPASS_URLS
    last = "no attempt made"
    with _client() as client:
        for attempt in range(attempts):
            url = urls[attempt % len(urls)]
            try:
                response = client.post(url, data={"data": overpass_query()}, timeout=120.0)
                payload = response.json() if response.status_code == 200 else None
            except (httpx.TransportError, ValueError) as error:
                last, payload = repr(error), None
            else:
                last = f"HTTP {response.status_code} from {url}"
            if payload and isinstance(payload.get("elements"), list):
                raw.mkdir(parents=True, exist_ok=True)
                target.write_text(json.dumps(payload), encoding="utf-8")
                log(f"osm: {len(payload['elements'])} poles from {url}")
                return
            log(f"osm: attempt {attempt + 1}/{attempts} failed ({last})")
            sleep(min(5.0 * 2**attempt, 45.0))
    raise RuntimeError(f"Overpass failed after {attempts} attempts ({last}). Retry later, or set OVERPASS_URL.")


def fetch(only: Iterable[str] | None, limit: int | None, raw: Path, log: Log) -> list[str]:
    """Run the requested downloads. Returns the keys that failed (the others completed)."""
    keys = list(only) if only else list(SOURCE_KEYS)
    unknown = sorted(set(keys) - set(SOURCE_KEYS))
    if unknown:
        raise ValueError(f"unknown source keys {unknown}; choose from {list(SOURCE_KEYS)}")
    steps: dict[str, Callable[[], None]] = {
        "pd_lean": lambda: fetch_pd_lean(raw, limit, log),
        "pd_crossarm": lambda: fetch_pd_crossarm(raw, limit, log),
        "pd_vegetation": lambda: fetch_pd_vegetation(raw, limit, log),
        "wrc50": lambda: fetch_wrc(raw, "wrc50", log),
        "wrc45": lambda: fetch_wrc(raw, "wrc45", log),
        "osm": lambda: fetch_osm(raw, log),
    }
    failed = []
    for key in keys:
        log(f"== {key}")
        try:
            steps[key]()
        except Exception as error:
            log(f"!! {key} failed: {error}")
            failed.append(key)
    return failed
