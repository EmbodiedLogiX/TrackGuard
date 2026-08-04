from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from trackguard.calibration.threshold import (
    apply_prior_shift,
    prior_logit_shift,
    sweep_thresholds,
)
from trackguard.config import load_config
from trackguard.data.dataset import SwitchDataset, collate_switch
from trackguard.router.classifier import SwitchClassifier, SwitchSpec, save_checkpoint
from trackguard.router.trainer import evaluate, fit


def main():
    parser = argparse.ArgumentParser(description="Train the switch router on a synthetic dataset")
    parser.add_argument("--data_dir", default="data/toy")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out_dir", default="runs/router")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no_arm", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    config = load_config(args.config)
    if args.no_arm:
        config.router.use_arm = False

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    train_ds = SwitchDataset(args.data_dir, "train")
    val_ds = SwitchDataset(args.data_dir, "val")
    test_ds = SwitchDataset(args.data_dir, "test")

    max_hist = max(train_ds.max_hist_len, val_ds.max_hist_len, test_ds.max_hist_len)
    max_tracks = max(train_ds.max_tracks, val_ds.max_tracks, test_ds.max_tracks)

    spec = SwitchSpec.from_config(
        config.router,
        max_hist_len=max_hist,
        max_tracks=max_tracks,
        feat_dim=config.encoding.feat_dim,
        arm_feat_dim=config.encoding.arm_feat_dim,
    )
    model = SwitchClassifier(spec).to(device)

    summary = fit(model, train_ds, val_ds, device,
                  epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    for row in summary["history"]:
        print(f"epoch {row['epoch']:>2} loss={row['loss']:.4f} "
              f"val_f1={row['f1']:.4f} val_recall={row['recall']:.4f}")

    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=max(1, args.batch_size // 2), shuffle=False,
        collate_fn=collate_switch)
    metrics = evaluate(model, test_loader, device)

    shift = prior_logit_shift(config.calibration.train_pos_ratio,
                              float((metrics["y_true"] == 1).mean()))
    scores = (apply_prior_shift(metrics["y_margin"], shift)
              if config.calibration.prior_correction else metrics["y_margin"])
    sweep = sweep_thresholds(scores, metrics["y_true"],
                             config.calibration.target_precision,
                             config.calibration.target_recall)

    ckpt_path = os.path.join(args.out_dir, "router.pt")
    save_checkpoint(model, ckpt_path, extra={"best_val_f1": summary["best_f1"]})

    report = {
        "best_val_f1": summary["best_f1"],
        "test_metrics": {k: metrics[k] for k in
                         ("accuracy", "precision", "recall", "f1", "confusion_matrix")},
        "calibration": {"prior_logit_shift": shift,
                        "recommended": sweep["recommended"],
                        "chosen": sweep[sweep["recommended"]]},
        "checkpoint": ckpt_path,
    }
    with open(os.path.join(args.out_dir, "report.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=float)
    print(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()
