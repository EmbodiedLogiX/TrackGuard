from __future__ import annotations

import argparse
import json
import os

import yaml

from trackguard.config import load_config
from trackguard.data.synthetic import generate_samples, write_dataset


def _read_toy_config(path):
    if path and os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    return {}


def main():
    parser = argparse.ArgumentParser(description="Generate a balanced synthetic switch dataset")
    parser.add_argument("--out_dir", default="data/toy")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--toy_config", default="configs/toy_data.yaml")
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--positive_ratio", type=float, default=None)
    parser.add_argument("--window", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    toy = _read_toy_config(args.toy_config)
    window_cfg = toy.get("window", {})
    synth_cfg = toy.get("synthetic", {})

    n_samples = args.n_samples if args.n_samples is not None else int(synth_cfg.get("n_samples", 100))
    positive_ratio = (args.positive_ratio if args.positive_ratio is not None
                      else float(synth_cfg.get("positive_ratio", 0.5)))
    window = args.window if args.window is not None else int(window_cfg.get("span", 100))
    seed = args.seed if args.seed is not None else int(synth_cfg.get("seed", 42))

    samples = generate_samples(
        n_samples=n_samples,
        positive_ratio=positive_ratio,
        window=window,
        min_tracks=int(synth_cfg.get("min_tracks", 3)),
        max_tracks=int(synth_cfg.get("max_tracks", 8)),
        arm_present_prob=float(synth_cfg.get("arm_present_prob", 0.4)),
        config=config.encoding,
        seed=seed,
    )
    counts = write_dataset(args.out_dir, samples, seed)
    print(json.dumps({"out_dir": args.out_dir, "total": len(samples),
                      "splits": counts}, indent=2))


if __name__ == "__main__":
    main()
