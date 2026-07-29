from __future__ import annotations

from tests.switch_router import paths


def compute_metrics(model, loader, device) -> dict:
    engine = paths.load_engine()
    return engine.evaluate_classifier(model, loader, device)


def format_report(metrics: dict, checkpoint: str, data_dir: str,
                  split: str, device, n_samples: int) -> str:
    cm = metrics["confusion_matrix"]
    lines = [
        "========== Router test (real weights + real data) ==========",
        f"  checkpoint: {checkpoint}",
        f"  data_dir:   {data_dir}/{split}",
        f"  device:     {device}",
        f"  samples:    {n_samples}",
        "  confusion matrix  label0=stable 1=switch",
        "                pred_0   pred_1",
        f"    true_0      {cm['tn']:6d}   {cm['fp']:6d}",
        f"    true_1      {cm['fn']:6d}   {cm['tp']:6d}",
        f"  Accuracy:  {metrics['accuracy']:.4f}",
        f"  Precision: {metrics['precision']:.4f}",
        f"  Recall:    {metrics['recall']:.4f}",
        f"  F1:        {metrics['f1']:.4f}",
    ]
    return "\n".join(lines)
