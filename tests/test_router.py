import torch

from trackguard.config import RouterConfig
from trackguard.router import SwitchClassifier, SwitchSpec


def _spec():
    return SwitchSpec.from_config(
        RouterConfig(hidden=32, n_heads=4, n_layers=2, hidden_cls=32),
        max_hist_len=6, max_tracks=4, feat_dim=6, arm_feat_dim=6,
    )


def _batch(b=2, n=4, t=7, feat=6, arm=6):
    x_in = torch.randn(b, n, t, feat)
    token_valid = torch.ones(b, n, t, dtype=torch.bool)
    arm_in = torch.randn(b, t, arm)
    arm_valid = torch.ones(b, t, dtype=torch.bool)
    return x_in, token_valid, arm_in, arm_valid


def test_forward_logits_shape():
    model = SwitchClassifier(_spec())
    logits = model(*_batch())
    assert logits.shape == (2, 2)


def test_predict_binary():
    model = SwitchClassifier(_spec())
    pred = model.predict(*_batch())
    assert pred.shape == (2,)
    assert set(pred.tolist()).issubset({0, 1})


def test_arm_ablation_runs():
    spec = _spec()
    spec.use_arm = False
    model = SwitchClassifier(spec)
    logits = model(*_batch())
    assert logits.shape == (2, 2)


def test_cross_attention_variant():
    spec = _spec()
    spec.use_arm_cross_attn = True
    model = SwitchClassifier(spec)
    logits = model(*_batch())
    assert logits.shape == (2, 2)
