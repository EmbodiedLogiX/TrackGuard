from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
LF_DATA_DIR = REPO_DIR / "data"
DATASET_INFO_PATH = LF_DATA_DIR / "dataset_info.json"

SHAREGPT_TAGS = {
    "role_tag": "role",
    "content_tag": "content",
    "user_tag": "user",
    "assistant_tag": "assistant",
}


def build_entry(json_path: Path) -> dict:
    return {
        "file_name": str(json_path.resolve()),
        "formatting": "sharegpt",
        "columns": {"messages": "messages", "images": "images"},
        "tags": SHAREGPT_TAGS,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=str(REPO_DIR / "data" / "recovery_mcq_dataset"),
    )
    parser.add_argument(
        "--name_prefix",
        type=str,
        default="recovery_mcq",
    )
    args = parser.parse_args()

    prefix = args.name_prefix
    train_name = f"{prefix}_train"
    val_name = f"{prefix}_val"
    test_name = f"{prefix}_test"

    lf_json_dir = Path(args.dataset_dir) / "llama_factory"
    train_json = lf_json_dir / "train.json"
    val_json = lf_json_dir / "val.json"
    test_json = lf_json_dir / "test.json"
    if not train_json.is_file():
        raise FileNotFoundError(
            f"{train_json} not found. The recovery MCQ dataset is private and must "
            "be built locally; see data/README.md for the required format."
        )

    entries: dict = {}
    if DATASET_INFO_PATH.is_file():
        try:
            with open(DATASET_INFO_PATH, "r", encoding="utf-8") as f:
                entries = json.load(f)
        except Exception:
            entries = {}

    entries[train_name] = build_entry(train_json)
    if val_json.is_file():
        entries[val_name] = build_entry(val_json)
    if test_json.is_file():
        entries[test_name] = build_entry(test_json)

    LF_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(DATASET_INFO_PATH, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)

    print(f"dataset_info.json -> {DATASET_INFO_PATH}")
    print(f"   set dataset_dir at train time to: {LF_DATA_DIR.resolve()}")
    for name in (train_name, val_name, test_name):
        if name in entries:
            print(f"   {name} -> {entries[name]['file_name']}")


if __name__ == "__main__":
    main()
