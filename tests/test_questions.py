import copy

import pytest

from polehealth import questions as q


def test_v1_request_is_valid(jpeg_url):
    body = q.v1_request(jpeg_url)
    assert q.validate_request(body) == []
    assert body["model"] == "matilda-jev-v1"
    assert set(body["questions"]) == {"lean", "crossarm", "vegetation", "health"}
    assert body["questions"]["lean"]["type"] == "choice"
    assert list(body["questions"]["lean"]["criteria"]) == ["straight", "leaning", "cannot_assess"]


def test_image_is_a_jpeg_data_url(jpeg_url):
    assert jpeg_url.startswith("data:image/jpeg;base64,")


def test_health_score_has_five_levels_lowest_first():
    levels = q.v1_questions()["health"]["criteria"]
    assert len(levels) == 5
    assert levels[0].startswith("critical")
    assert levels[-1] == "as new"
    assert 2 <= len(levels) <= 10


def test_v2_request_is_valid_and_has_no_coordinates(jpeg_url, register):
    record = register["poles"][0]["record"]
    findings = {"lean": {"chosen": "straight", "probability": 0.9}}
    body = q.v2_request({**record, "lat": -27.5, "lon": 151.9}, findings, q.DEFAULT_POLICY, jpeg_url)
    assert q.validate_request(body) == []
    pole = body["state"]["pole"]
    assert "lat" not in pole and "lon" not in pole
    assert "source_history_id" not in pole
    assert body["state"]["policy"] == q.DEFAULT_POLICY
    assert set(body["questions"]) == {"action", "safety_risk_now"}
    assert list(body["questions"]["action"]["criteria"]) == ["replace", "maintain", "defer", "reinspect"]
    assert body["questions"]["safety_risk_now"]["type"] == "noul"


def test_record_request_is_text_only_and_valid():
    body = q.record_request({"age_years": 40, "shell_thickness": [0.9, 0.9, 0.9]})
    assert body["images"] == []
    assert q.validate_request(body) == []


def test_validator_rejects_what_the_server_rejects(jpeg_url):
    good = q.v1_request(jpeg_url)

    def broken(mutate):
        body = copy.deepcopy(good)
        mutate(body)
        return q.validate_request(body)

    assert broken(lambda b: b.update(model="other"))
    assert broken(lambda b: b.update(extra=1))
    assert broken(lambda b: b.update(questions={}))
    assert broken(lambda b: b["questions"]["health"].update(criteria=["only one"]))
    assert broken(lambda b: b["questions"]["health"].update(criteria=[str(i) for i in range(11)]))
    assert not broken(lambda b: b["questions"]["health"].update(criteria=["a", "b"]))
    assert broken(lambda b: b["questions"]["lean"].update(criteria={}))
    assert broken(lambda b: b["questions"]["lean"].update(unknown="x"))
    assert broken(lambda b: b["questions"]["lean"].update(type="rank"))
    assert broken(lambda b: b.update(images=[jpeg_url] * 5))
    assert broken(lambda b: b.update(images=["not a data url"]))
    assert broken(lambda b: b.update(images=["data:image/gif;base64,AAAA"]))
    assert broken(lambda b: b.update(images=["data:image/jpeg;base64,@@@@"]))
    assert broken(lambda b: b.update(images=["data:image/jpeg;base64,AAAA"]))


def test_model_alias_is_accepted(jpeg_url):
    assert q.validate_request(q.v1_request(jpeg_url, model="maincode-jev-latest")) == []


def test_choice_keys_are_the_documented_options():
    assert q.LEAN_OPTIONS == ("straight", "leaning", "cannot_assess")
    assert q.CROSSARM_OPTIONS == ("straight", "tilted", "not_visible")
    assert q.VEGETATION_OPTIONS == ("clear", "encroaching", "not_visible")
    assert q.ACTION_OPTIONS == ("replace", "maintain", "defer", "reinspect")


@pytest.mark.parametrize(
    ("probability", "lean", "threshold", "expected"),
    [
        (0.59, "straight", 0.6, "engineer"),
        (0.60, "straight", 0.6, "auto"),
        (0.95, "cannot_assess", 0.6, "engineer"),
        (0.95, "leaning", 0.6, "auto"),
        (0.70, "straight", 0.8, "engineer"),
    ],
)
def test_routing_rule(probability, lean, threshold, expected):
    assert q.route(probability, lean, threshold) == expected


def test_chosen_action_returns_the_probability_of_the_choice():
    answers = {"action": {"choice": "defer", "probabilities": {"replace": 0.1, "defer": 0.7, "maintain": 0.2}}}
    assert q.chosen_action(answers) == ("defer", 0.7)


def test_top_probability_for_each_answer_kind():
    assert q.top_probability({"probabilities": {"a": 0.2, "b": 0.8}}) == 0.8
    assert q.top_probability({"noul": 0.3}) == 0.7
    assert q.top_probability({"confidence": 0.4}) == 0.4
