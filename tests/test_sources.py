import httpx

from polehealth import sources


def test_crossarm_classes_and_image_label(raw_dir):
    classes = sources.read_crossarm_classes(raw_dir)
    assert classes[1] == "crossarm_tilted" and len(classes) == 6
    labels = sources.hf_dir(raw_dir) / sources.CROSSARM_DIR / "test" / "labels"
    got = [sources.crossarm_image_label(labels / f"{i}_catc.txt", classes) for i in range(7)]
    # boxes: [0] [0,2] [1] [2,3] [4] [5,0] []
    assert got == ["straight", "straight", "tilted", "tilted", "straight", "tilted", None]


def test_sample_discovery_maps_labels(raw_dir):
    lean = sources.lean_samples(raw_dir)
    assert len(lean) == 18
    assert {s.label for s in lean} == {"straight", "leaning", "cannot_assess"}
    assert {s.native_label for s in lean} == {"Straight", "Leaned", "Rejected"}
    assert lean[0].id.startswith("pd_lean_test_")
    cross = sources.crossarm_samples(raw_dir)
    assert len(cross) == 6 and {s.label for s in cross} == {"straight", "tilted"}
    veg = sources.vegetation_samples(raw_dir)
    assert {s.label for s in veg} == {"clear", "encroaching"}
    assert {s.native_label for s in veg} == {"Safe", "Risky"}


def test_take_balanced_is_deterministic_and_spread():
    items = [("a", i) for i in range(20)] + [("b", i) for i in range(5)]
    first = sources.take_balanced(items, lambda x: x[0], 6)
    assert first == sources.take_balanced(items, lambda x: x[0], 6)
    assert sorted(k for k, _ in first) == ["a", "a", "a", "b", "b", "b"]
    assert sources.take_balanced(items, lambda x: x[0], None) == items


def test_hf_list_follows_the_link_header_and_keeps_files_only():
    pages = {
        "https://huggingface.co/api/datasets/EPDCL/pd-defect/tree/main/f": (
            [{"type": "file", "path": "f/1.jpg", "size": 1}, {"type": "directory", "path": "f/d"}],
            '<https://huggingface.co/page2>; rel="next"',
        ),
        "https://huggingface.co/page2": ([{"type": "file", "path": "f/2.jpg", "size": 2}], None),
    }

    def handler(request: httpx.Request) -> httpx.Response:
        body, link = pages[str(request.url)]
        return httpx.Response(200, json=body, headers={"link": link} if link else {})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert [e["path"] for e in sources.hf_list(client, "f")] == ["f/1.jpg", "f/2.jpg"]


def test_hf_fetch_is_idempotent(tmp_path):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(200, content=b"abc")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    dest = tmp_path / "x" / "a.bin"
    assert sources.hf_fetch(client, "a.bin", dest, size=3) is True
    assert sources.hf_fetch(client, "a.bin", dest, size=3) is False
    assert len(calls) == 1 and dest.read_bytes() == b"abc"


def test_fetch_osm_retries_html_errors(tmp_path):
    attempts = []

    def fake_post(self, url, **kwargs):
        attempts.append(url)
        if len(attempts) < 3:
            return httpx.Response(500, text="<html>500</html>")
        return httpx.Response(200, json={"elements": [{"type": "node", "id": 1, "lat": 0, "lon": 0}]})

    original = httpx.Client.post
    httpx.Client.post = fake_post
    try:
        logs = []
        sources.fetch_osm(tmp_path, logs.append, sleep=lambda _: None)
    finally:
        httpx.Client.post = original
    assert len(attempts) == 3
    assert sources.osm_path(tmp_path).is_file()
    assert any("attempt 1" in line for line in logs)


def test_unknown_source_key_is_rejected(tmp_path):
    import pytest

    with pytest.raises(ValueError, match="unknown source keys"):
        sources.fetch(["nope"], None, tmp_path, lambda _: None)
