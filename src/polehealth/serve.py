"""Local web server for the live "drop in a photo" moment: serves demo/ and a small decision API.

The browser only ever talks to this process. MJ_API_KEY and HF_TOKEN stay here, inside the backend object.
"""

from __future__ import annotations

import json
import threading
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from . import images
from .demo_data import decide_pole
from .jev import Backend, BackendError, HttpBackend
from .questions import DEFAULT_POLICY, ENGINEER_THRESHOLD, validate_image_url

MAX_BODY_BYTES = 13_000_000  # an 8 MB image is about 10.7 MB as base64, plus the JSON around it
MAX_POLICY_CHARS = 4000
MAX_RECORD_BYTES = 20_000


class BadRequest(ValueError):
    pass


class ServeState:
    """What the request handlers share: the backend, the policy, and a cached readiness check."""

    def __init__(self, backend: Backend, policy: str = DEFAULT_POLICY, threshold: float = ENGINEER_THRESHOLD) -> None:
        self.backend = backend
        self.policy = policy
        self.threshold = threshold
        self._hardware: str | None = None
        self._lock = threading.Lock()

    def ready(self) -> bool:
        if isinstance(self.backend, HttpBackend):
            try:
                return self.backend.health().get("status") == "ready"
            except Exception:  # any failure means "not ready"; the detail stays in the server log
                return False
        return True

    def hardware(self) -> str:
        with self._lock:
            if self._hardware is None:
                self._hardware = self.backend.describe_hardware()
            return self._hardware


def parse_decide_request(payload: Any) -> tuple[str, dict[str, Any] | None, str | None]:
    """Validate a /api/decide body. Raises BadRequest with a message that is safe to show the caller."""
    if not isinstance(payload, dict):
        raise BadRequest("body must be a JSON object")
    image = payload.get("image")
    if not isinstance(image, str):
        raise BadRequest("image must be a data URL string")
    problems = validate_image_url(image)
    if problems:
        raise BadRequest(f"image: {problems[0]}")
    record = payload.get("record")
    if record is not None and (not isinstance(record, dict) or len(json.dumps(record)) > MAX_RECORD_BYTES):
        raise BadRequest("record must be null or a pole record object")
    policy = payload.get("policy")
    if policy is not None and (not isinstance(policy, str) or len(policy) > MAX_POLICY_CHARS):
        raise BadRequest(f"policy must be a string of at most {MAX_POLICY_CHARS} characters")
    return image, record, policy


def make_handler(state: ServeState, demo_dir: Path) -> type[SimpleHTTPRequestHandler]:
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, directory=str(demo_dir), **kwargs)

        def _json(self, status: int, body: dict[str, Any]) -> None:
            payload = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:
            if self.path.split("?")[0] == "/api/health":
                if state.ready():
                    self._json(
                        HTTPStatus.OK,
                        {
                            "status": "ready",
                            "backend": state.backend.name,
                            "mock": state.backend.name == "mock",
                            "model": state.backend.model,
                            "hardware": state.hardware(),
                        },
                    )
                else:
                    self._json(HTTPStatus.SERVICE_UNAVAILABLE, {"status": "unavailable", "backend": state.backend.name})
                return
            if self.path.startswith("/api/"):
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            super().do_GET()

        def do_POST(self) -> None:
            if self.path.split("?")[0] != "/api/decide":
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > MAX_BODY_BYTES:
                    raise BadRequest(f"body must be 1 to {MAX_BODY_BYTES} bytes")
                try:
                    payload = json.loads(self.rfile.read(length))
                except ValueError as error:
                    raise BadRequest("body is not valid JSON") from error
                image, record, policy = parse_decide_request(payload)
                url = images.normalise_data_url(image)
                entry = decide_pole(state.backend, url, record, policy or state.policy, state.threshold)
            except BadRequest as error:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            except BackendError as error:
                self._json(HTTPStatus.BAD_GATEWAY, {"error": f"model backend failed: {error}"})
            else:
                self._json(
                    HTTPStatus.OK, {**entry, "backend": state.backend.name, "mock": state.backend.name == "mock"}
                )

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            if not self.path.startswith("/photos/"):
                super().log_message(format, *args)

    return Handler


def make_server(state: ServeState, demo_dir: Path, host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), make_handler(state, demo_dir))
