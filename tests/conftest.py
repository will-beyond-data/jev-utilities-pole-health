from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from polehealth import images
from polehealth.register import build_register, load_wrc
from polehealth.sources import CROSSARM_DIR, LEAN_DIR, VEGETATION_DIR

FIXTURES = Path(__file__).parent / "fixtures"


def make_jpeg(path: Path, colour: tuple[int, int, int], size: tuple[int, int] = (96, 64)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, colour).save(path, "JPEG")
    return path


@pytest.fixture
def jpeg_path(tmp_path: Path) -> Path:
    return make_jpeg(tmp_path / "pole.jpg", (120, 90, 60))


@pytest.fixture
def jpeg_url(jpeg_path: Path) -> str:
    return images.data_url(jpeg_path)


@pytest.fixture
def osm_payload() -> dict:
    return json.loads((FIXTURES / "osm_tiny.json").read_text())


def write_wrc_xlsx(path: Path) -> Path:
    sheets = json.loads((FIXTURES / "wrc50_tiny.json").read_text())
    with pd.ExcelWriter(path) as writer:
        for name in ("2019", "2009", "1999"):  # newest first, like the real workbook
            pd.DataFrame(sheets[name]).to_excel(writer, sheet_name=name, index=False)
    return path


@pytest.fixture
def wrc_xlsx(tmp_path: Path) -> Path:
    return write_wrc_xlsx(tmp_path / "wrc50.xlsx")


@pytest.fixture
def sheets(wrc_xlsx: Path):
    return load_wrc(wrc_xlsx)


@pytest.fixture
def register(osm_payload, sheets) -> dict:
    return build_register(osm_payload, sheets, seed=7)


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    """A miniature data/raw/ tree with the same layout `polehealth fetch` produces."""
    raw = tmp_path / "raw"
    hf = raw / "pd-defect"
    n = 0
    for split in ("test", "val"):
        rows = ["image_id,label"]
        for label in ("Leaned", "Straight", "Rejected"):
            for _ in range(3):
                n += 1
                make_jpeg(hf / LEAN_DIR / split / label / f"{n}.jpg", (n * 5 % 255, 60, 90))
                rows.append(f"{n}.jpg,{label}")
        (hf / LEAN_DIR / split).mkdir(parents=True, exist_ok=True)
        (hf / LEAN_DIR / split / f"{split}.csv").write_text("\n".join(rows) + "\n")
    base = hf / CROSSARM_DIR
    base.mkdir(parents=True, exist_ok=True)
    (base / "data.yaml").write_text(
        "train: train/images\nnc: 6\n"
        "names: ['crossarm_straight', 'crossarm_tilted', 'topcleat_straight', 'topcleat_tilted', "
        "'v-crossarm_straight', 'v-crossarm_tilted']\n"
    )
    for i, boxes in enumerate(["0", "0\n2", "1", "2\n3", "4", "5\n0", ""]):
        make_jpeg(base / "test" / "images" / f"{i}_catc.jpg", (30 * i, 100, 40))
        (base / "test" / "labels").mkdir(parents=True, exist_ok=True)
        lines = "\n".join(f"{c} 0.5 0.5 0.1 0.1" for c in boxes.split("\n") if c)
        (base / "test" / "labels" / f"{i}_catc.txt").write_text(lines)
    veg = hf / VEGETATION_DIR
    rows = ["ID,Ground_Truth"]
    for i in range(8):
        make_jpeg(veg / "valid" / f"{i}_vc.jpg", (10, 30 * i, 200))
        rows.append(f"{i}_vc.jpg,{'Risky' if i % 2 else 'Safe'}")
    (veg / "GT_ValidationSet.csv").write_text("\n".join(rows) + "\n")
    write_wrc_xlsx(raw / "wrc50.xlsx")
    return raw
