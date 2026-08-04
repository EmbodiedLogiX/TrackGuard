from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from trackguard.calibration.threshold import (
    apply_prior_shift,
    prior_logit_shift,
    sweep_thresholds,
)
from trackguard.config import load_config
from trackguard.data.dataset import SwitchDataset, collate_switch
from trackguard.router.classifier import load_checkpoint
from trackguard.router.trainer import evaluate


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained router on a test split")
    parser.add_argument("--checkpoint", default="runs/router/router.pt")
    parser.add_argument("--data_dir", default="track_data/switch_test")
    parser.add_argument("--split", default="test")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out_json", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    model = load_checkpoint(args.checkpoint, device)
    dataset = SwitchDataset(args.data_dir, args.split)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate_switch)

    metrics = evaluate(model, loader, device)
    labels = metrics["y_true"]
    shift = prior_logit_shift(config.calibration.train_pos_ratio,
                              float((labels == 1).mean()) if len(labels) else 0.5)
    scores = (apply_prior_shift(metrics["y_margin"], shift)
              if config.calibration.prior_correction else metrics["y_margin"])
    sweep = sweep_thresholds(scores, labels,
                             config.calibration.target_precision,
                             config.calibration.target_recall)

    report = {
        "checkpoint": args.checkpoint,
        "data_dir": args.data_dir,
        "split": args.split,
        "n_samples": len(dataset),
        "n_positive": int((labels == 1).sum()),
        "n_negative": int((labels == 0).sum()),
        "metrics": {k: metrics[k] for k in
                    ("accuracy", "precision", "recall", "f1", "confusion_matrix")},
        "calibration": {"prior_logit_shift": shift,
                        "recommended": sweep["recommended"],
                        "chosen": sweep[sweep["recommended"]]},
    }
    print(json.dumps(report, indent=2, default=float))
    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, default=float)


if __name__ == "__main__":
    main()
