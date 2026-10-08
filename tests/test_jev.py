import json

import httpx
import pytest

from polehealth import questions as q
from polehealth.jev import (
    BackendError,
    HttpBackend,
    MockBackend,
    SpaceBackend,
    build_answer,
    parse_server_timing,
)


def test_parse_server_timing():
    assert parse_server_timing("total;dur=48.3") == 48.3
    assert parse_server_timing("cache;dur=1, total; dur=7.5") == 7.5
    assert parse_server_timing(None) is None
    assert parse_server_timing("other;dur=1") is None


def test_build_answer_matches_the_documented_shapes():
    choice = build_answer({"type": "choice", "criteria": {"a": None, "b": None}}, [0.75, 0.25])
    assert choice["choice"] == "a" and choice["probabilities"] == {"a": 0.75, "b": 0.25}
    assert choice["confidence"] == pytest.approx(0.5)  # (0.75 - 1/2) / (1 - 1/2)
    noul = build_answer({"type": "noul"}, [0.2, 0.8])
    assert noul == {"type": "noul", "noul": 0.8}
    score = build_answer({"type": "score", "criteria": ["lo", "mid", "hi"]}, [0.0, 0.5, 0.5])
    assert score["score"] == pytest.approx(1.5)
    assert score["legend"] == {"0": "lo", "1": "mid", "2": "hi"}
    assert set(score["probabilities"]) == {"0", "1", "2"}


class TestMock:
    def test_is_deterministic(self, jpeg_url):
        body = q.v1_request(jpeg_url)
        a = MockBackend(seed=1).decide(body).response
        b = MockBackend(seed=1).decide(body).response
        assert a == b
        assert MockBackend(seed=2).decide(body).response != a

    def test_answers_have_the_server_shape(self, jpeg_url):
        result = MockBackend().decide(q.v1_request(jpeg_url))
        answers = result.response["answers"]
        for key in ("lean", "crossarm", "vegetation"):
            answer = answers[key]
            assert answer["type"] == "choice" and answer["choice"] in answer["probabilities"]
            assert sum(answer["probabilities"].values()) == pytest.approx(1.0)
        assert answers["health"]["type"] == "score"
        assert list(answers["health"]["probabilities"]) == ["0", "1", "2", "3", "4"]
        assert result.model_ms is None
        assert result.latency_ms >= 0

    def test_follows_the_label_hint_most_of_the_time(self, jpeg_url):
        backend = MockBackend(seed=3)
        hits = 0
        for i in range(60):
            body = q.v1_request(jpeg_url)
            body["state"] = {**body["state"], "n": i}
            hits += backend.decide(body, {"lean": "leaning"}).response["answers"]["lean"]["choice"] == "leaning"
        assert 35 <= hits <= 58

    def test_rejects_requests_the_real_server_would_reject(self, jpeg_url):
        body = q.v1_request(jpeg_url)
        body["questions"]["health"]["criteria"] = ["only one"]
        with pytest.raises(BackendError, match="422"):
            MockBackend().decide(body)

    def test_v2_actions_follow_the_record(self, jpeg_url, register):
        backend = MockBackend()
        findings = {
            "lean": {"chosen": "leaning", "probability": 0.9},
            "crossarm": {"chosen": "tilted", "probability": 0.9},
            "vegetation": {"chosen": "encroaching", "probability": 0.9},
            "health": {"score": 0.5},
        }
        bad = json.loads(json.dumps(register["poles"][0]["record"]))
        bad.update(bushfire_zone="high", critical_customer=True, shell_thickness_change_10y=-0.3)
        bad["inspections"][-1]["health_index"] = 1
        body = q.v2_request(bad, findings, q.DEFAULT_POLICY, jpeg_url)
        actions = {
            backend.decide({**body, "state": {**body["state"], "n": i}}).response["answers"]["action"]["choice"]
            for i in range(30)
        }
        assert "replace" in actions
        assert actions <= set(q.ACTION_OPTIONS)


def http_backend(handler, **kwargs):
    client = httpx.Client(base_url="http://test", transport=httpx.MockTransport(handler))
    return HttpBackend(base_url="http://test", client=client, sleep=lambda _: None, **kwargs)


OK = {"model": "matilda-jev-v1", "answers": {"x": {"type": "noul", "noul": 0.5}}, "usage": {}}


def test_http_reads_server_timing_and_sends_the_body(jpeg_url):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=OK, headers={"server-timing": "total;dur=42.5"})

    result = http_backend(handler).decide(q.v1_request(jpeg_url))
    assert seen["url"] == "http://test/v1/systemone"
    assert q.validate_request(seen["body"]) == []
    assert result.model_ms == 42.5 and result.latency_ms > 0 and result.attempts == 1


def test_http_retries_503_and_529_then_succeeds():
    codes = iter([503, 529, 200])

    def handler(request: httpx.Request) -> httpx.Response:
        code = next(codes)
        return httpx.Response(code, json=OK if code == 200 else {"detail": "busy"})

    result = http_backend(handler).decide({"model": "m"})
    assert result.attempts == 3


def test_http_retries_network_errors():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) < 3:
            raise httpx.ConnectError("refused")
        return httpx.Response(200, json=OK)

    assert http_backend(handler).decide({}).attempts == 3


def test_http_does_not_retry_validation_errors():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(422, json={"detail": []})

    with pytest.raises(BackendError, match="422"):
        http_backend(handler).decide({})
    assert len(calls) == 1


def test_http_gives_up_after_max_attempts():
    backend = http_backend(lambda r: httpx.Response(503), max_attempts=3)
    with pytest.raises(BackendError, match="gave up after 3"):
        backend.decide({})


def test_http_hardware_comes_from_health():
    backend = http_backend(lambda r: httpx.Response(200, json={"status": "ready", "device": "NVIDIA A100 80GB"}))
    assert backend.describe_hardware() == "1x NVIDIA A100 80GB"


class FakeGradio:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def predict(self, text, api_name):
        self.calls.append((json.loads(text), api_name))
        return self.payload


def test_space_converts_latency_s_to_model_ms():
    payload = json.dumps({**OK, "latency_s": 0.25})
    fake = FakeGradio(payload)
    result = SpaceBackend(client=fake).decide({"model": "m"})
    assert result.model_ms == 250.0
    assert "latency_s" not in result.response
    assert fake.calls == [({"model": "m"}, "/decide_request")]


def test_space_accepts_dict_payload_without_latency():
    result = SpaceBackend(client=FakeGradio(dict(OK))).decide({})
    assert result.model_ms is None
