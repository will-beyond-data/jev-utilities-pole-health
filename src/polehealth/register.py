"""Build the synthetic asset register: real OSM pole positions paired with real wrc50 inspection histories."""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from .sources import TOOWOOMBA_BBOX

YEAR_SHIFT = 7  # wrc50 inspections 1999/2009/2019 read as 2006/2016/2026
REFERENCE_YEAR = 2026
DEFAULT_SEED = 7

BUSHFIRE_SHARES = (("high", 0.15), ("medium", 0.30))  # the remainder is "low"
NEVER_MAINTAINED = 0.30
CRITICAL_SHARE = 0.03
CRITICAL_TYPES = (("hospital", 0.2), ("aged_care", 0.5), ("water_pumping", 0.3))
TRANSFORMERS = {"no": "none", "single-phase": "single_phase", "three-phase": "three_phase"}


# ---------------------------------------------------------------- wrc workbooks


def load_wrc(path: Path) -> dict[int, pd.DataFrame]:
    """Sheets of a Kaggle wrc workbook keyed by (unshifted) inspection year, indexed by pole ID."""
    sheets = pd.read_excel(path, sheet_name=None)
    result = {}
    for name, frame in sheets.items():
        frame = frame.rename(columns=lambda c: str(c).strip())
        result[int(str(name).strip())] = frame.set_index("ID")
    return dict(sorted(result.items()))


def _flag(value: Any) -> bool:
    return str(value).strip().lower() in {"yes", "y", "true", "1"}


def _inspection(row: Mapping[str, Any], year: int) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "year": year,
        "shell_thickness": [round(float(row[c]), 2) for c in ("ST1", "ST2", "ST3")],
        "groundline": round(float(row["GL"]), 2),
        "surface": str(row["Surface conditions"]).strip(),
        "woodpecker_holes": _flag(row["Wood pecker holes"]),
    }
    if "Health Index" in row and pd.notna(row["Health Index"]):
        entry["health_index"] = int(row["Health Index"])
    return entry


def pole_history(sheets: Mapping[int, pd.DataFrame], pole_id: int, shift: int = YEAR_SHIFT) -> dict[str, Any]:
    """One pole's inspections across all sheets, years and age shifted forward by `shift`."""
    years = list(sheets)
    inspections = [_inspection(sheets[y].loc[pole_id].to_dict(), y + shift) for y in years]
    newest = sheets[years[-1]].loc[pole_id]
    age = int(newest["Age"]) + shift
    transformer = TRANSFORMERS.get(str(newest["Carrying transformer"]).strip().lower(), "none")
    change = None
    if len(inspections) >= 2:
        change = round(_mean(inspections[-1]["shell_thickness"]) - _mean(inspections[-2]["shell_thickness"]), 2)
    return {
        "material": "timber",
        "age_years": age,
        "install_year": years[-1] + shift - age,
        "inspections": inspections,
        "shell_thickness_change_10y": change,
        "transformer": transformer,
    }


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def benchmark_rows(path: Path) -> list[dict[str, Any]]:
    """Rows for the record-only benchmark: the newest inspection with its real Health Index (1-5), unshifted."""
    sheets = load_wrc(path)
    newest_year = list(sheets)[-1]
    frame = sheets[newest_year]
    rows = []
    for pole_id, row in frame.iterrows():
        if pd.isna(row.get("Health Index")):
            continue
        entry = _inspection(row.to_dict(), newest_year)
        rows.append(
            {
                "id": f"wrc50-{pole_id}",
                "health_index": int(row["Health Index"]),
                "fields": {
                    "age_years": int(row["Age"]),
                    "shell_thickness": entry["shell_thickness"],
                    "groundline": entry["groundline"],
                    "surface": entry["surface"],
                    "woodpecker_holes": entry["woodpecker_holes"],
                    "transformer": TRANSFORMERS.get(str(row["Carrying transformer"]).strip().lower(), "none"),
                },
            }
        )
    return rows


# ---------------------------------------------------------------- register


def osm_poles(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Unique OSM nodes with coordinates, sorted by OSM id."""
    nodes = {
        int(e["id"]): e for e in payload.get("elements", []) if e.get("type") == "node" and "lat" in e and "lon" in e
    }
    return [nodes[i] for i in sorted(nodes)]


def _weighted(rng: random.Random, options: Iterable[tuple[str, float]]) -> str:
    names, weights = zip(*options, strict=True)
    return rng.choices(names, weights=weights)[0]


def _bushfire_zones(
    poles: Sequence[Mapping[str, Any]], seed: int, bbox: tuple[float, float, float, float]
) -> dict[int, str]:
    """Rank poles by how far out they sit in the bbox (plus noise) and cut at 15 / 30 / 55 percent."""
    south, west, north, east = bbox
    lat_c, lon_c = (south + north) / 2, (west + east) / 2
    half_lat, half_lon = (north - south) / 2, (east - west) / 2
    scored = []
    for pole in poles:
        rng = random.Random(f"{seed}|{pole['id']}|bushfire")
        edge = max(abs(pole["lat"] - lat_c) / half_lat, abs(pole["lon"] - lon_c) / half_lon)
        scored.append((-(edge + rng.gauss(0, 0.12)), int(pole["id"])))
    scored.sort()
    n = len(scored)
    n_high = round(BUSHFIRE_SHARES[0][1] * n)
    n_medium = round(BUSHFIRE_SHARES[1][1] * n)
    zones = {}
    for rank, (_, pole_id) in enumerate(scored):
        zones[pole_id] = "high" if rank < n_high else "medium" if rank < n_high + n_medium else "low"
    return zones


def build_register(
    osm_payload: Mapping[str, Any],
    sheets: Mapping[int, pd.DataFrame],
    seed: int = DEFAULT_SEED,
    bbox: tuple[float, float, float, float] = TOOWOOMBA_BBOX,
) -> dict[str, Any]:
    poles = osm_poles(osm_payload)
    history_ids = sorted(int(i) for i in sheets[list(sheets)[-1]].index)
    picker = random.Random(f"{seed}|histories")
    if len(poles) <= len(history_ids):
        chosen = picker.sample(history_ids, len(poles))
    else:
        chosen = [picker.choice(history_ids) for _ in poles]
    zones = _bushfire_zones(poles, seed, bbox)
    entries = []
    for pole, history_id in zip(poles, chosen, strict=True):
        rng = random.Random(f"{seed}|{pole['id']}|attrs")
        record = pole_history(sheets, history_id)
        years = [i["year"] for i in record["inspections"]]
        maintained = None if rng.random() < NEVER_MAINTAINED else rng.randint(years[-2], years[-1])
        critical = rng.random() < CRITICAL_SHARE
        record.update(
            {
                "last_maintained_year": maintained,
                "years_since_maintenance": None if maintained is None else REFERENCE_YEAR - maintained,
                "bushfire_zone": zones[int(pole["id"])],
                "customers_served": min(400, max(1, round(rng.lognormvariate(3.0, 1.1)))),
                "critical_customer": critical,
                "critical_customer_type": _weighted(rng, CRITICAL_TYPES) if critical else None,
                "source_history_id": f"wrc50-{history_id}",
            }
        )
        entries.append({"id": f"osm-{pole['id']}", "lat": pole["lat"], "lon": pole["lon"], "record": record})
    return {
        "seed": seed,
        "year_shift": YEAR_SHIFT,
        "reference_year": REFERENCE_YEAR,
        "bbox": list(bbox),
        "synthetic_pairing": True,
        "sources": {"positions": "OpenStreetMap node[power=pole]", "histories": "Kaggle wrc50 (utilityanalytics)"},
        "poles": entries,
    }


def write_register(register: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(register, indent=1, sort_keys=False) + "\n", encoding="utf-8")


def load_register(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
