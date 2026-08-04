import random

from trackguard.config import RecoveryConfig
from trackguard.recovery import Candidate, build_option_set, coerce_letter


def test_option_set_keeps_anchor():
    cfg = RecoveryConfig(max_options=3)
    rng = random.Random(0)
    records = [
        Candidate(1, [100, 100, 40, 60]),
        Candidate(2, [120, 110, 40, 60]),
        Candidate(3, [140, 120, 40, 60]),
    ]
    result = build_option_set(records, [100, 100, 40, 60], anchor_id=1, config=cfg, rng=rng)
    assert result is not None
    assert 1 in result.mapping.values()
    assert len(result.options) <= 3


def test_option_set_missing_anchor():
    cfg = RecoveryConfig()
    rng = random.Random(0)
    records = [Candidate(2, [120, 110, 40, 60])]
    assert build_option_set(records, [100, 100, 40, 60], anchor_id=1, config=cfg, rng=rng) is None


def test_distance_filter_drops_far_candidate():
    cfg = RecoveryConfig(max_options=4, cand_center_dist_max=50.0)
    rng = random.Random(1)
    records = [
        Candidate(1, [100, 100, 40, 60]),
        Candidate(9, [900, 900, 40, 60]),
    ]
    result = build_option_set(records, [100, 100, 40, 60], anchor_id=1, config=cfg, rng=rng)
    assert 9 not in result.mapping.values()


def test_coerce_letter():
    assert coerce_letter("The answer is B.", ["A", "B", "C"]) == "B"
    assert coerce_letter("z", ["A", "B"]) is None
