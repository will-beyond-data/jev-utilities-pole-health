"""Single source of truth for the questions we ask Matilda Jev, plus request validation and routing."""

from __future__ import annotations

import base64
import binascii
import copy
import io
from collections.abc import Mapping
from typing import Any

from PIL import Image, UnidentifiedImageError

MODEL_NAME = "matilda-jev-v1"
MODEL_ALIAS = "maincode-jev-latest"
ENGINEER_THRESHOLD = 0.6

LEAN_OPTIONS = ("straight", "leaning", "cannot_assess")
CROSSARM_OPTIONS = ("straight", "tilted", "not_visible")
VEGETATION_OPTIONS = ("clear", "encroaching", "not_visible")
ACTION_OPTIONS = ("replace", "maintain", "defer", "reinspect")
HEALTH_LEVELS = (
    "critical: imminent failure risk",
    "poor: significant defects",
    "fair: some defects",
    "good: minor wear",
    "as new",
)
HEALTH_NAMES = tuple(level.split(":")[0] for level in HEALTH_LEVELS)
# The record benchmark is graded on the wrc50 inspection scale, so the record question carries that rubric.
RECORD_HEALTH_LEVELS = (
    "1: serious defects and most of the wood lost: replace",
    "2: serious defects",
    "3: significant defects",
    "4: moderate defects",
    "5: minor defects only",
)

DEFAULT_POLICY = (
    "Public safety comes first. Poles in high bushfire zones and poles that feed a critical customer "
    "(hospital, aged care, water pumping) get a lower threshold to act. Prefer maintain over replace "
    "when the structure is sound and the defect is repairable. Defer only when the pole is safe to leave "
    "until the next scheduled inspection. Choose reinspect when the photo and the record do not give enough "
    "evidence to decide."
)

V1_STATE = {"asset": "electricity distribution pole", "image": "ground-level inspection photo"}


def v1_questions() -> dict[str, dict[str, Any]]:
    return {
        "lean": {
            "type": "choice",
            "instructions": "Look at the main pole in the photo. Is it standing vertical?",
            "criteria": {
                "straight": "the pole is vertical",
                "leaning": "the pole visibly deviates from vertical",
                "cannot_assess": "the top or base of the pole is hidden, or the camera angle is too distorted to judge",
            },
        },
        "crossarm": {
            "type": "choice",
            "instructions": "Look at the crossarm and top cleat at the top of the pole. Are they level?",
            "criteria": {
                "straight": "the crossarm and top cleat are level and square to the pole",
                "tilted": "a crossarm or top cleat visibly tilts away from level",
                "not_visible": "no crossarm is visible, or it is too small or hidden to judge",
            },
        },
        "vegetation": {
            "type": "choice",
            "instructions": "Look at the conductors (overhead wires) and any nearby vegetation.",
            "criteria": {
                "clear": "no vegetation near the conductors",
                "encroaching": "vegetation is touching or within reach of the conductors",
                "not_visible": "the conductors are not visible in the photo",
            },
        },
        "health": {
            "type": "score",
            "instructions": "Rate the overall visible condition of the pole, crossarm and fittings.",
            "criteria": list(HEALTH_LEVELS),
        },
    }


def v2_questions() -> dict[str, dict[str, Any]]:
    return {
        "action": {
            "type": "choice",
            "instructions": (
                "Given the photo, the photo findings, the asset record and the maintenance policy, "
                "what should the network do with this pole?"
            ),
            "criteria": {
                "replace": "structural failure risk: replace this cycle",
                "maintain": "repair or treat now, for example reinforce, re-tension or clear vegetation",
                "defer": "safe to leave until the next inspection cycle",
                "reinspect": "the photo and record are not enough, send a crew to look",
            },
        },
        "safety_risk_now": {
            "type": "noul",
            "instructions": "Is there a risk to public safety before the next scheduled inspection?",
            "criteria": {"true": "yes, there is a risk to public safety", "false": "no meaningful risk"},
        },
    }


def record_questions() -> dict[str, dict[str, Any]]:
    return {
        "health": {
            "type": "score",
            "instructions": (
                "Using only the inspection record, rate the structural condition of this timber pole. "
                "shell_thickness and groundline are measured remaining wood: values near 1.0 are as new, "
                "lower values mean more wood has been lost."
            ),
            "criteria": list(RECORD_HEALTH_LEVELS),
        }
    }


# Fields that are noise to the model; lat/lon are dropped because the model has no use for coordinates.
_RECORD_DROP = {"lat", "lon", "source_history_id"}


def record_for_model(record: Mapping[str, Any]) -> dict[str, Any]:
    return {key: copy.deepcopy(value) for key, value in record.items() if key not in _RECORD_DROP}


def v1_request(image_url: str, model: str = MODEL_NAME) -> dict[str, Any]:
    return {"model": model, "state": dict(V1_STATE), "questions": v1_questions(), "images": [image_url]}


def photo_findings(v1_answers: Mapping[str, Any]) -> dict[str, Any]:
    """Compress v1 answers into the findings the v2 request carries (choice plus its probability)."""
    findings: dict[str, Any] = {}
    for key in ("lean", "crossarm", "vegetation"):
        answer = v1_answers[key]
        choice = answer["choice"]
        findings[key] = {"chosen": choice, "probability": round(float(answer["probabilities"][choice]), 3)}
    health = v1_answers["health"]
    level = max(health["probabilities"], key=lambda k: health["probabilities"][k])
    findings["health"] = {
        "score": round(float(health["score"]), 2),
        "scale": "0 = critical, 4 = as new",
        "most_likely_level": HEALTH_NAMES[int(level)],
    }
    return findings


def v2_request(
    record: Mapping[str, Any],
    findings: Mapping[str, Any],
    policy: str,
    image_url: str,
    model: str = MODEL_NAME,
) -> dict[str, Any]:
    state = {"pole": record_for_model(record), "photo_findings": dict(findings), "policy": policy}
    return {"model": model, "state": state, "questions": v2_questions(), "images": [image_url]}


def record_request(record_fields: Mapping[str, Any], model: str = MODEL_NAME) -> dict[str, Any]:
    """Text-only request for the record-only benchmark (no image)."""
    state = {"asset": "timber distribution pole", "inspection_record": dict(record_fields)}
    return {"model": model, "state": state, "questions": record_questions(), "images": []}


def chosen_action(v2_answers: Mapping[str, Any]) -> tuple[str, float]:
    answer = v2_answers["action"]
    choice = answer["choice"]
    return choice, float(answer["probabilities"][choice])


def route(action_probability: float, lean_choice: str | None, threshold: float = ENGINEER_THRESHOLD) -> str:
    """Engineer review when the action is low-confidence or the lean could not be assessed."""
    if lean_choice == "cannot_assess" or action_probability < threshold:
        return "engineer"
    return "auto"


def top_probability(answer: Mapping[str, Any]) -> float:
    """Max option probability; the confidence used for routing and calibration."""
    probabilities = answer.get("probabilities")
    if probabilities:
        return max(float(p) for p in probabilities.values())
    if "noul" in answer:
        return max(float(answer["noul"]), 1.0 - float(answer["noul"]))
    return float(answer.get("confidence", 0.0))


def argmax_key(answer: Mapping[str, Any]) -> str:
    probabilities = answer["probabilities"]
    return max(probabilities, key=lambda k: probabilities[k])


_ALLOWED_HEADERS = {"data:image/png;base64", "data:image/jpeg;base64", "data:image/webp;base64"}
_QUESTION_FIELDS = {"type", "instructions", "criteria"}
_TOP_FIELDS = {"model", "state", "questions", "images"}


def validate_request(body: Mapping[str, Any], model_names: tuple[str, ...] = (MODEL_NAME, MODEL_ALIAS)) -> list[str]:
    """Mirror the server's request rules (runtime server.py). Returns a list of problems, empty if valid."""
    errors: list[str] = []
    extra = set(body) - _TOP_FIELDS
    if extra:
        errors.append(f"unexpected fields: {sorted(extra)}")
    if body.get("model") not in model_names:
        errors.append(f"unknown model {body.get('model')!r}")
    if "state" not in body or not isinstance(body["state"], str | dict | list):
        errors.append("state must be a string, object or array")
    questions = body.get("questions")
    if not isinstance(questions, dict) or not questions:
        errors.append("questions must be a non-empty object")
    else:
        for key, question in questions.items():
            errors.extend(f"{key}: {problem}" for problem in _validate_question(question))
    images = body.get("images", [])
    if not isinstance(images, list) or len(images) > 4:
        errors.append("images must be a list of at most 4")
    else:
        for index, value in enumerate(images):
            errors.extend(f"images[{index}]: {problem}" for problem in validate_image_url(value))
    return errors


def _validate_question(question: Any) -> list[str]:
    if not isinstance(question, dict):
        return ["question must be an object"]
    problems: list[str] = []
    extra = set(question) - _QUESTION_FIELDS
    if extra:
        problems.append(f"unexpected fields: {sorted(extra)}")
    kind, criteria = question.get("type"), question.get("criteria")
    if kind == "choice":
        if not isinstance(criteria, dict) or not 1 <= len(criteria) <= 255:
            problems.append("choice needs 1 to 255 criteria options")
    elif kind == "score":
        if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
            problems.append("score needs 2 to 10 criteria levels")
    elif kind == "noul":
        if criteria is not None and (not isinstance(criteria, dict) or not set(criteria) <= {"true", "false"}):
            problems.append("noul criteria may only have true/false keys")
    else:
        problems.append(f"unknown question type {kind!r}")
    return problems


def validate_image_url(value: Any) -> list[str]:
    if not isinstance(value, str) or len(value) > 12_000_000:
        return ["image must be a string of at most 12,000,000 characters"]
    header, separator, encoded = value.partition(",")
    if not separator or header not in _ALLOWED_HEADERS:
        return ["images must be base64 PNG, JPEG or WebP data URLs"]
    try:
        content = base64.b64decode(encoded, validate=True)
        if len(content) > 8_000_000:
            return ["image must be at most 8 MB"]
        with Image.open(io.BytesIO(content)) as image:
            if image.width * image.height > 16_000_000:
                return ["image must have at most 16 million pixels"]
            if image.format not in {"PNG", "JPEG", "WEBP"}:
                return ["unsupported image format"]
            image.verify()
    except (binascii.Error, OSError, SyntaxError, UnidentifiedImageError):
        return ["invalid image data"]
    return []
