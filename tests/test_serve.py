import json
import threading
import urllib.error
import urllib.request

import pytest

from polehealth import serve
from polehealth.jev import MockBackend


@pytest.fixture
def server(tmp_path):
    demo = tmp_path / "demo"
    demo.mkdir()
    (demo / "index.html").write_text("<h1>demo</h1>")
    httpd = serve.make_server(serve.ServeState(MockBackend(seed=1)), demo, "127.0.0.1", 0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def call(url, body=None, raw=None):
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def test_serves_static_demo_and_health(server):
    status, body = call(server + "/")
    assert status == 200 and b"<h1>demo</h1>" in body
    status, body = call(server + "/api/health")
    health = json.loads(body)
    assert status == 200 and health["status"] == "ready" and health["backend"] == "mock" and health["mock"] is True


def test_health_is_503_until_the_backend_is_ready(tmp_path):
    state = serve.ServeState(MockBackend())
    state.ready = lambda: False  # type: ignore[method-assign]
    httpd = serve.make_server(state, tmp_path, "127.0.0.1", 0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        status, body = call(f"http://127.0.0.1:{httpd.server_address[1]}/api/health")
        assert status == 503 and json.loads(body)["status"] == "unavailable"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_decide_with_a_record_returns_a_decisions_entry(server, jpeg_url, register):
    record = register["poles"][0]["record"]
    status, body = call(server + "/api/decide", {"image": jpeg_url, "record": record})
    out = json.loads(body)
    assert status == 200
    assert set(out["v1"]["answers"]) == {"lean", "crossarm", "vegetation", "health"}
    assert set(out["v2"]["answers"]) == {"action", "safety_risk_now"}
    assert out["action"] in {"replace", "maintain", "defer", "reinspect"}
    assert out["route"] in {"auto", "engineer"} and out["mock"] is True
    assert {"latency_ms", "model_ms"} <= set(out["v1"])


def test_decide_without_a_record_is_v1_only(server, jpeg_url):
    status, body = call(server + "/api/decide", {"image": jpeg_url, "record": None})
    out = json.loads(body)
    assert status == 200 and out["v2"] is None and out["action"] is None and out["confidence"] is None


def test_policy_is_optional_and_passed_through(server, jpeg_url, register):
    record = register["poles"][0]["record"]
    status, _ = call(server + "/api/decide", {"image": jpeg_url, "record": record, "policy": "Never replace."})
    assert status == 200


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"image": 5},
        {"image": "data:image/gif;base64,AAAA"},
        {"image": "data:image/jpeg;base64,AAAA"},
        {"image": "http://example.com/x.jpg"},
        [],
    ],
)
def test_bad_images_get_400(server, body):
    status, out = call(server + "/api/decide", body)
    assert status == 400 and "error" in json.loads(out)


def test_bad_record_policy_and_json_get_400(server, jpeg_url):
    assert call(server + "/api/decide", {"image": jpeg_url, "record": "nope"})[0] == 400
    assert call(server + "/api/decide", {"image": jpeg_url, "policy": 3})[0] == 400
    assert call(server + "/api/decide", raw=b"{not json")[0] == 400
    assert call(server + "/api/decide", raw=b"")[0] == 400


def test_unknown_api_paths_are_404(server):
    assert call(server + "/api/nope")[0] == 404
    assert call(server + "/api/nope", {"a": 1})[0] == 404


def test_secrets_never_appear_in_responses(tmp_path, jpeg_url, monkeypatch):
    monkeypatch.setenv("MJ_API_KEY", "super-secret-key")
    monkeypatch.setenv("HF_TOKEN", "hf_secret_token")
    httpd = serve.make_server(serve.ServeState(MockBackend()), tmp_path, "127.0.0.1", 0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        texts = [call(base + "/api/health")[1], call(base + "/api/decide", {"image": jpeg_url})[1]]
        assert all(b"secret" not in t for t in texts)
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_normalised_upload_is_resized(tmp_path):
    from PIL import Image

    from polehealth import images

    path = tmp_path / "big.png"
    Image.new("RGB", (3000, 2000)).save(path)
    big = "data:image/png;base64," + __import__("base64").b64encode(path.read_bytes()).decode()
    assert images.normalise_data_url(big).startswith("data:image/jpeg;base64,")
