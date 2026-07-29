from __future__ import annotations

import argparse
import os

from tests.switch_router import loader, metrics, paths


def run(checkpoint: str = paths.DEFAULT_CHECKPOINT,
        data_dir: str = paths.DEFAULT_DATA_DIR,
        split: str = paths.DEFAULT_SPLIT,
        device: str = paths.DEFAULT_DEVICE,
        batch_size: int = 8,
        verbose: bool = True) -> dict:
    ckpt = paths.checkpoint_path(checkpoint)
    data = paths.data_path(data_dir)
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(ckpt)

    device_t = loader.resolve_device(device)
    model, _ = loader.build_model(ckpt, device_t)
    dataset, dl = loader.build_loader(data, split, batch_size, device_t)
    result = metrics.compute_metrics(model, dl, device_t)
    if verbose:
        print(metrics.format_report(result, ckpt, data, split, device_t, len(dataset)))
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run the trajectory-aware router on real switch samples")
    parser.add_argument("--checkpoint", default=paths.DEFAULT_CHECKPOINT)
    parser.add_argument("--data_dir", default=paths.DEFAULT_DATA_DIR)
    parser.add_argument("--split", default=paths.DEFAULT_SPLIT)
    parser.add_argument("--device", default=paths.DEFAULT_DEVICE)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()
    run(args.checkpoint, args.data_dir, args.split, args.device, args.batch_size)


if __name__ == "__main__":
    main()
