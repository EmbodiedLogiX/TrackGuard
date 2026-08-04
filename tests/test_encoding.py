import numpy as np

from trackguard.config import EncodingConfig
from trackguard.encoding import (
    TimePositionEncoding,
    build_arm_tokens,
    build_trajectory_features,
    relative_time,
)
import torch


def test_relative_time_ends_at_zero():
    rel = relative_time(5)
    assert rel[-1] == 0.0
    assert rel[0] < 0.0


def test_trajectory_features_target_flag():
    cfg = EncodingConfig()
    traj = np.ones((3, 6, 4), dtype=np.float32)
    valid = np.ones((3, 6), dtype=bool)
    feat, feat_valid = build_trajectory_features(traj, valid, cfg)
    assert feat.shape == (3, 6, cfg.feat_dim)
    assert feat[0, 0, 5] == 1.0
    assert feat[1, 0, 5] == 0.0
    assert feat_valid.all()


def test_arm_tokens_shape_and_current():
    cfg = EncodingConfig()
    hist_bbox = np.zeros((6, 4), dtype=np.float32)
    hist_present = np.zeros((6,), dtype=bool)
    cur_bbox = np.array([100.0, 50.0, 80.0, 120.0], dtype=np.float32)
    feat, valid = build_arm_tokens(hist_bbox, hist_present, cur_bbox, True, 6, cfg)
    assert feat.shape == (7, cfg.arm_feat_dim)
    assert valid[-1]
    assert feat[-1, 5] == 1.0


def test_time_position_encoding_sinusoidal():
    enc = TimePositionEncoding(16, 32, mode="sinusoidal")
    out = enc(torch.arange(4))
    assert out.shape == (4, 16)
