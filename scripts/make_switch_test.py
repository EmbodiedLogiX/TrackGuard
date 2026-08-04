from __future__ import annotations

import argparse
import json
import os

import yaml

from trackguard.config import load_config
from trackguard.data.synthetic import generate_samples, write_split


def _read_yaml(path):
    if path and os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    return {}


def main():
    parser = argparse.ArgumentParser(
        description="Emit a balanced switch test split (N positive + N negative)")
    parser.add_argument("--out_dir", default="track_data/switch_test")
    parser.add_argument("--split", default="test")
    parser.add_argument("--n_per_class", type=int, default=100)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--toy_config", default="configs/toy_data.yaml")
    parser.add_argument("--window", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    toy = _read_yaml(args.toy_config)
    window_cfg = toy.get("window", {})
    synth_cfg = toy.get("synthetic", {})

    window = args.window if args.window is not None else int(window_cfg.get("span", 100))
    seed = args.seed if args.seed is not None else int(synth_cfg.get("seed", 42))
    total = args.n_per_class * 2

    samples = generate_samples(
        n_samples=total,
        positive_ratio=0.5,
        window=window,
        min_tracks=int(synth_cfg.get("min_tracks", 3)),
        max_tracks=int(synth_cfg.get("max_tracks", 8)),
        arm_present_prob=float(synth_cfg.get("arm_present_prob", 0.4)),
        config=config.encoding,
        seed=seed,
    )
    counts = write_split(args.out_dir, samples, args.split)
    print(json.dumps({"out_dir": os.path.join(args.out_dir, args.split),
                      "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
