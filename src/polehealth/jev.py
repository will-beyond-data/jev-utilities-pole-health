"""Backends that answer Matilda Jev decision requests: self-hosted HTTP, the Hugging Face Space, and a mock."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from .questions import MODEL_NAME, validate_request

DEFAULT_BASE_URL = "http://127.0.0.1:8083"
DEFAULT_SPACE = "hugging-apps/matilda-jev"
RETRY_STATUSES = {502, 503, 504, 529}


class BackendError(RuntimeError):
    """A request failed for good (non-retryable status, bad payload, or retries exhausted)."""


@dataclass
class Result:
    response: dict[str, Any]
    latency_ms: float  # client wall time of the successful attempt
    model_ms: float | None  # server-side compute time when the backend reports it
    attempts: int = 1


class Backend(Protocol):
    name: str
    model: str

    def decide(self, body: dict[str, Any], hints: Mapping[str, Any] | None = None) -> Result:
        """Send one decision request. `hints` (question key to expected answer) is read by the mock only."""
        ...

    def describe_hardware(self) -> str: ...

    def close(self) -> None: ...


def backoff_delay(attempt: int, retry_after: float | None = None, rng: random.Random | None = None) -> float:
    if retry_after is not None:
        return min(retry_after, 30.0)
    jitter = (rng or random).random() * 0.25
    return min(0.5 * 2**attempt, 20.0) + jitter


_SERVER_TIMING = re.compile(r"total;\s*dur=([0-9.]+)")


def parse_server_timing(header: str | None) -> float | None:
    if not header:
        return None
    match = _SERVER_TIMING.search(header)
    return float(match.group(1)) if match else None


class HttpBackend:
    name = "http"

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str = MODEL_NAME,
        timeout: float = 120.0,
        max_attempts: int = 6,
        hardware: str | None = None,
        sleep: Callable[[float], None] = time.sleep,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = (base_url or os.environ.get("MJ_URL") or DEFAULT_BASE_URL).rstrip("/")
        key = api_key if api_key is not None else os.environ.get("MJ_API_KEY")
        self.model = model
        self.max_attempts = max_attempts
        self._hardware = hardware
        self._sleep = sleep
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        self._client = client or httpx.Client(base_url=self.base_url, headers=headers, timeout=timeout)

    def health(self) -> dict[str, Any]:
        response = self._client.get("/health")
        response.raise_for_status()
        return response.json()

    def describe_hardware(self) -> str:
        if self._hardware:
            return self._hardware
        try:
            device = self.health().get("device")
        except (httpx.HTTPError, ValueError):
            device = None
        return f"1x {device}" if device else "unknown"

    def decide(self, body: dict[str, Any], hints: Mapping[str, Any] | None = None) -> Result:
        last_error = "no attempt made"
        for attempt in range(self.max_attempts):
            started = time.perf_counter()
            try:
                response = self._client.post("/v1/systemone", json=body)
            except httpx.TransportError as error:
                last_error = f"network error: {error!r}"
                self._sleep(backoff_delay(attempt))
                continue
            latency_ms = (time.perf_counter() - started) * 1000
            if response.status_code in RETRY_STATUSES:
                last_error = f"HTTP {response.status_code}"
                retry_after = response.headers.get("retry-after")
                self._sleep(backoff_delay(attempt, float(retry_after) if retry_after else None))
                continue
            if response.status_code != 200:
                raise BackendError(f"HTTP {response.status_code}: {response.text[:300]}")
            payload = response.json()
            if "answers" not in payload:
                raise BackendError(f"response has no answers: {str(payload)[:300]}")
            model_ms = parse_server_timing(response.headers.get("server-timing"))
            return Result(payload, latency_ms, model_ms, attempts=attempt + 1)
        raise BackendError(f"gave up after {self.max_attempts} attempts ({last_error})")

    def close(self) -> None:
        self._client.close()


class SpaceBackend:
    """The public ZeroGPU Space. Wall time includes queueing and GPU attach, so never present it as model speed."""

    name = "space"

    def __init__(
        self,
        space: str = DEFAULT_SPACE,
        token: str | None = None,
        model: str = MODEL_NAME,
        max_attempts: int = 4,
        sleep: Callable[[float], None] = time.sleep,
        client: Any = None,
    ) -> None:
        self.space = space
        self.model = model
        self.max_attempts = max_attempts
        self._sleep = sleep
        if client is None:
            try:
                from gradio_client import Client
            except ImportError as error:
                raise BackendError("the space backend needs gradio_client: uv sync --extra space") from error
            client = Client(space, token=token or os.environ.get("HF_TOKEN"))
        self._client = client

    def describe_hardware(self) -> str:
        return f"Hugging Face ZeroGPU Space ({self.space})"

    def decide(self, body: dict[str, Any], hints: Mapping[str, Any] | None = None) -> Result:
        last_error = "no attempt made"
        for attempt in range(self.max_attempts):
            started = time.perf_counter()
            try:
                raw = self._client.predict(json.dumps(body), api_name="/decide_request")
            except Exception as error:  # gradio_client raises a variety of types for queue and quota problems
                last_error = f"{type(error).__name__}: {error}"
                if "quota" in last_error.lower():
                    raise BackendError(f"ZeroGPU quota exhausted (set HF_TOKEN): {last_error}") from error
                self._sleep(backoff_delay(attempt + 2))
                continue
            latency_ms = (time.perf_counter() - started) * 1000
            payload = _space_payload(raw)
            latency_s = payload.pop("latency_s", None)
            model_ms = float(latency_s) * 1000 if latency_s is not None else None
            return Result(payload, latency_ms, model_ms, attempts=attempt + 1)
        raise BackendError(f"gave up after {self.max_attempts} attempts ({last_error})")

    def close(self) -> None:
        pass


def _space_payload(raw: Any) -> dict[str, Any]:
    if isinstance(raw, tuple | list):
        raw = raw[0]
    payload = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(payload, dict) or "answers" not in payload:
        raise BackendError(f"unexpected Space response: {str(payload)[:300]}")
    return dict(payload)


def build_answer(question: Mapping[str, Any], probabilities: Sequence[float]) -> dict[str, Any]:
    """Port of the runtime's `model.answer()`: the exact answer shape the real server returns."""
    kind = question["type"]
    if kind == "choice":
        keys = list(question["criteria"])
    elif kind == "score":
        keys = [str(i) for i in range(len(question["criteria"]))]
    else:
        keys = ["false", "true"]
    values = [float(p) for p in probabilities]
    if len(values) != len(keys) or sum(values) <= 0:
        raise ValueError("each option needs a probability")
    total = sum(values)
    values = [v / total for v in values]
    if kind == "noul":
        return {"type": "noul", "noul": values[1]}
    best = max(range(len(values)), key=values.__getitem__)
    distribution = dict(zip(keys, values, strict=True))
    if kind == "choice":
        confidence = 1.0 if len(values) == 1 else (values[best] - 1 / len(values)) / (1 - 1 / len(values))
        return {
            "type": "choice",
            "probabilities": distribution,
            "choice": keys[best],
            "confidence": max(0.0, min(1.0, confidence)),
        }
    distance = sum(p * abs(i - best) for i, p in enumerate(values))
    midpoint = (len(values) - 1) / 2
    baseline = sum(abs(i - midpoint) for i in range(len(values))) / len(values)
    return {
        "type": "score",
        "probabilities": distribution,
        "legend": dict(zip(keys, question["criteria"], strict=True)),
        "score": sum(i * p for i, p in enumerate(values)),
        "confidence": max(0.0, 1.0 - distance / baseline),
    }


# Rough option priors per question key so unlabelled mock answers look like a real mix.
MOCK_PRIORS: dict[str, dict[str, float]] = {
    "lean": {"straight": 0.55, "leaning": 0.25, "cannot_assess": 0.20},
    "crossarm": {"straight": 0.40, "tilted": 0.20, "not_visible": 0.40},
    "vegetation": {"clear": 0.50, "encroaching": 0.25, "not_visible": 0.25},
    "health": {"0": 0.08, "1": 0.22, "2": 0.30, "3": 0.28, "4": 0.12},
}


@dataclass
class MockBackend:
    """Deterministic stand-in so the UI can be built without a GPU. Never used for any published number."""

    seed: int = 7
    model: str = MODEL_NAME
    name: str = field(default="mock", init=False)
    hint_agreement: float = 0.8

    def describe_hardware(self) -> str:
        return "none (mock backend)"

    def close(self) -> None:
        pass

    def decide(self, body: dict[str, Any], hints: Mapping[str, Any] | None = None) -> Result:
        problems = validate_request(body)
        if problems:
            raise BackendError("HTTP 422: " + "; ".join(problems))
        started = time.perf_counter()
        fingerprint = hashlib.sha256(
            json.dumps([body["state"], body.get("images", [])], sort_keys=True, default=str).encode()
        ).hexdigest()
        answers = {}
        for key, question in body["questions"].items():
            rng = random.Random(f"{self.seed}|{key}|{fingerprint}")
            answers[key] = self._answer(key, question, body["state"], (hints or {}).get(key), rng)
        response = {"model": body["model"], "answers": answers, "usage": {"input_tokens": 0, "output_tokens": 0}}
        return Result(response, (time.perf_counter() - started) * 1000, None)

    def _answer(
        self, key: str, question: Mapping[str, Any], state: Any, hint: Any, rng: random.Random
    ) -> dict[str, Any]:
        kind = question["type"]
        if kind == "noul":
            return build_answer(question, _noul_probabilities(key, state, hint, rng))
        keys = list(question["criteria"]) if kind == "choice" else [str(i) for i in range(len(question["criteria"]))]
        target = self._target(key, keys, state, hint, rng)
        if kind == "choice":
            logits = [(rng.uniform(1.0, 3.5) if k == target else 0.0) + rng.gauss(0, 0.7) for k in keys]
        else:
            centre = min(max(int(target) + rng.gauss(0, 0.35), 0), len(keys) - 1)
            spread = rng.uniform(0.9, 2.0)
            logits = [-spread * abs(i - centre) + rng.gauss(0, 0.3) for i in range(len(keys))]
        return build_answer(question, _softmax(logits))

    def _target(self, key: str, keys: list[str], state: Any, hint: Any, rng: random.Random) -> str:
        if hint is not None:
            hinted = str(hint)
            if hinted in keys and rng.random() < self.hint_agreement:
                return hinted
        elif key == "action" and isinstance(state, dict) and "pole" in state:
            heuristic = _heuristic_action(state)
            if heuristic in keys and rng.random() < 0.85:
                return heuristic
        priors = MOCK_PRIORS.get(key, {})
        weights = [priors.get(k, 1.0) for k in keys]
        return rng.choices(keys, weights=weights)[0]


def _softmax(logits: Sequence[float]) -> list[float]:
    top = max(logits)
    exps = [math.exp(x - top) for x in logits]
    total = sum(exps)
    return [e / total for e in exps]


def _risk_score(state: Mapping[str, Any]) -> float:
    """Crude additive risk used only to make mock v2 answers look coherent with their inputs."""
    pole = state.get("pole", {})
    findings = state.get("photo_findings", {})
    score = 0.0
    if findings.get("lean", {}).get("chosen") == "leaning":
        score += 2.0
    if findings.get("crossarm", {}).get("chosen") == "tilted":
        score += 2.0
    if findings.get("vegetation", {}).get("chosen") == "encroaching":
        score += 1.0
    health = findings.get("health", {}).get("score")
    if health is not None:
        score += 3.0 if health < 1.5 else 1.5 if health < 2.5 else 0.0
    if pole.get("shell_thickness_change_10y", 0) < -0.15:
        score += 1.5
    inspections = pole.get("inspections") or [{}]
    if inspections[-1].get("health_index", 5) <= 2:
        score += 2.0
    if pole.get("bushfire_zone") == "high":
        score += 1.0
    if pole.get("critical_customer"):
        score += 1.5
    since = pole.get("years_since_maintenance")
    if since is None or since >= 7:
        score += 0.5
    return score


def _heuristic_action(state: Mapping[str, Any]) -> str:
    if state.get("photo_findings", {}).get("lean", {}).get("chosen") == "cannot_assess":
        return "reinspect"
    score = _risk_score(state)
    return "replace" if score >= 7 else "maintain" if score >= 3.5 else "defer"


def _noul_probabilities(key: str, state: Any, hint: Any, rng: random.Random) -> list[float]:
    if isinstance(state, dict) and "pole" in state:
        p_true = 1 / (1 + math.exp(-(_risk_score(state) - 3.5)))
    else:
        p_true = rng.random()
    p_true = min(max(p_true + rng.gauss(0, 0.05), 0.02), 0.98)
    return [1 - p_true, p_true]


def make_backend(
    kind: str,
    base_url: str | None = None,
    api_key: str | None = None,
    hf_token: str | None = None,
    seed: int = 7,
    hardware: str | None = None,
) -> Backend:
    if kind == "http":
        return HttpBackend(base_url=base_url, api_key=api_key, hardware=hardware)
    if kind == "space":
        return SpaceBackend(token=hf_token)
    if kind == "mock":
        return MockBackend(seed=seed)
    raise ValueError(f"unknown backend {kind!r}")
