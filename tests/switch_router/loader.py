from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from tests.switch_router import paths


def build_model(checkpoint: str, device: torch.device):
    engine = paths.load_engine()
    model, ckpt = engine.load_model_from_checkpoint(checkpoint, device)
    return model, ckpt


def build_loader(data_dir: str, split: str, batch_size: int, device: torch.device):
    engine = paths.load_engine()
    dataset = engine.NpyJointSwitchDataset(data_dir, split)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=engine.collate_mae_batch,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    return dataset, loader


def resolve_device(device: str) -> torch.device:
    if device.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)
