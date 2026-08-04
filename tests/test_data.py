from trackguard.config import EncodingConfig
from trackguard.data.dataset import SwitchDataset, collate_switch
from trackguard.data.synthetic import generate_samples, write_dataset


def test_generate_balanced(tmp_path):
    cfg = EncodingConfig()
    samples = generate_samples(100, 0.5, 12, 3, 6, 0.4, cfg, seed=42)
    assert len(samples) == 100
    positives = sum(s.label for s in samples)
    assert positives == 50


def test_write_and_load(tmp_path):
    cfg = EncodingConfig()
    samples = generate_samples(40, 0.5, 10, 3, 5, 0.4, cfg, seed=7)
    counts = write_dataset(str(tmp_path), samples, seed=7)
    assert counts["train"]["total"] > 0
    ds = SwitchDataset(str(tmp_path), "train")
    batch = collate_switch([ds[i] for i in range(min(4, len(ds)))])
    assert batch["x_in"].shape[0] == min(4, len(ds))
    assert batch["arm_in"].shape[0] == batch["x_in"].shape[0]
