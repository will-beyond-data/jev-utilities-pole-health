import math

from polehealth import metrics as m


def test_accuracy():
    assert m.accuracy(["a", "b", "a", "b"], ["a", "b", "b", "b"]) == 0.75


def test_confusion_has_every_label_row_and_column():
    table = m.confusion(["a", "a", "b"], ["a", "b", "b"], ["a", "b", "c"])
    assert table == {"a": {"a": 1, "b": 1, "c": 0}, "b": {"a": 0, "b": 1, "c": 0}, "c": {"a": 0, "b": 0, "c": 0}}


def test_macro_f1_hand_computed():
    truth = ["a", "a", "a", "b", "b", "c"]
    pred = ["a", "a", "b", "b", "b", "a"]
    # a: tp2 fp1 fn1 -> 4/6 ; b: tp2 fp1 fn0 -> 4/5 ; c: tp0 fn1 fp0 -> 0
    expected = (4 / 6 + 4 / 5 + 0) / 3
    assert math.isclose(m.macro_f1(truth, pred, ["a", "b", "c"]), expected)


def test_macro_f1_ignores_classes_absent_from_the_truth():
    truth, pred = ["a", "a", "b", "b"], ["a", "a", "b", "c"]
    # c only appears as a wrong prediction, so it is not averaged in: a=1.0, b=2/3
    assert math.isclose(m.macro_f1(truth, pred, ["a", "b", "c"]), (1.0 + 2 / 3) / 2)


def test_ece_hand_computed():
    confidences = [0.9, 0.9, 0.9, 0.9, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6]
    correct = [True, True, True, False] + [True, False, False, False, False, False]
    # bin 0.9-1.0: n=4 acc .75 conf .9 gap .15 ; bin 0.6-0.7: n=6 acc 1/6 conf .6 gap .4333
    expected = 4 / 10 * 0.15 + 6 / 10 * abs(1 / 6 - 0.6)
    assert math.isclose(m.expected_calibration_error(confidences, correct), expected)


def test_ece_is_zero_when_perfectly_calibrated():
    assert m.expected_calibration_error([0.5, 0.5], [True, False]) == 0.0


def test_confidence_of_exactly_one_lands_in_the_top_bin():
    bins = m.reliability_bins([1.0, 0.05], [True, False])
    assert [(b["lo"], b["hi"], b["count"]) for b in bins] == [(0.0, 0.1, 1), (0.9, 1.0, 1)]
    assert set(bins[0]) == {"lo", "hi", "count", "mean_confidence", "accuracy"}


def test_percentiles_linear_interpolation():
    values = [10, 20, 30, 40, 50]
    assert m.percentile(values, 50) == 30
    assert m.percentile(values, 90) == 46
    assert math.isclose(m.percentile(values, 99), 49.6)
    assert m.percentile([7], 99) == 7
    assert m.percentile([], 50) is None


def test_percentiles_skip_missing_values():
    assert m.percentiles([None, None]) == {"p50": None, "p90": None, "p99": None}
    assert m.percentiles([1.0, None, 3.0])["p50"] == 2.0


def test_task_summary_shape():
    out = m.task_summary(["a", "b"], ["a", "a"], [0.9, 0.6], ["a", "b"], [10.0, 20.0], [None, None])
    assert out["n"] == 2 and out["accuracy"] == 0.5
    assert set(out) >= {"accuracy", "macro_f1", "confusion", "ece", "reliability", "latency_ms", "model_ms"}
    assert out["model_ms"]["p50"] is None
