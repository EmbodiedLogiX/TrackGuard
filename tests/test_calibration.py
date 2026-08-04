import numpy as np

from trackguard.calibration import apply_prior_shift, prior_logit_shift, sweep_thresholds


def test_prior_shift_zero_when_equal():
    assert abs(prior_logit_shift(0.5, 0.5)) < 1e-9


def test_prior_shift_negative_when_target_lower():
    assert prior_logit_shift(0.5, 0.01) < 0.0


def test_apply_prior_shift_monotone():
    margin = np.array([-2.0, 0.0, 2.0])
    scores = apply_prior_shift(margin, 0.0)
    assert scores[0] < scores[1] < scores[2]


def test_sweep_returns_recommendation():
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    y = np.array([0, 0, 1, 1])
    sweep = sweep_thresholds(scores, y)
    assert sweep["recommended"] == "precision_target"
    assert 0.0 <= sweep["f1_max"]["threshold"] <= 1.0 + 1e-6
