from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ENGINE_MODULE = "switch_router_engine"
DEFAULT_CHECKPOINT = "mae_switch_joint_bbox_arm_npy.pt"
DEFAULT_DATA_DIR = os.path.join("track_data", "switch_real")
DEFAULT_SPLIT = "test"
DEFAULT_DEVICE = "cuda:1"


def _ensure_repo_on_path() -> None:
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)


def load_engine():
    _ensure_repo_on_path()
    import switch_router_engine as engine
    return engine


def resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


def checkpoint_path(checkpoint: str = DEFAULT_CHECKPOINT) -> str:
    return resolve(checkpoint)


def data_path(data_dir: str = DEFAULT_DATA_DIR) -> str:
    return resolve(data_dir)
