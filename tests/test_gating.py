from trackguard.config import GateConfig
from trackguard.gating import DriftGate, GateDecision, iou, aspect_extreme


def test_iou_identity():
    box = [10.0, 10.0, 20.0, 20.0]
    assert iou(box, box) == 1.0


def test_iou_disjoint():
    assert iou([0, 0, 10, 10], [100, 100, 10, 10]) == 0.0


def test_aspect_extreme():
    assert aspect_extreme([0, 0, 40, 10]) == 4.0


def test_gate_low_confidence():
    gate = DriftGate(GateConfig())
    outcome = gate.evaluate([0, 0, 40, 40], confidence=0.1)
    assert outcome.decision == GateDecision.LOW_CONFIDENCE
    assert not outcome.escalate


def test_gate_stable_high_iou():
    gate = DriftGate(GateConfig())
    outcome = gate.evaluate([0, 0, 40, 40], confidence=0.9, prev_box=[0, 0, 40, 40])
    assert outcome.decision == GateDecision.STABLE
    assert not outcome.escalate


def test_gate_escalates_on_low_iou():
    gate = DriftGate(GateConfig())
    outcome = gate.evaluate([0, 0, 40, 40], confidence=0.9, prev_box=[200, 200, 40, 40])
    assert outcome.decision == GateDecision.ESCALATE
    assert outcome.escalate


def test_gate_small_box():
    gate = DriftGate(GateConfig())
    outcome = gate.evaluate([0, 0, 5, 5], confidence=0.9)
    assert outcome.decision == GateDecision.SMALL_BOX
