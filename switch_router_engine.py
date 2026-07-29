"""
mae_switch_classifier_joint_bbox_arm_npy.py — 在 joint bbox 分类器基础上引入机械臂运动

在 mae_switch_classifier_joint_bbox_npy.py 的基础上，额外读取 generate_mixed_switch_npy.py
写入 .npy 的机械臂字段（arm_hist_bbox / arm_hist_present / arm_cur_bbox / arm_cur_present），
将机械臂视为一条“额外轨迹”与包裹坐标 token 一起送入同一个 Transformer：
  - 机械臂在历史窗口 + 当前帧上各有一个 token（与包裹 token 时间对齐）；
  - 机械臂 token 使用独立的特征投影 arm_proj、独立身份向量 arm_identity、
    独立 token 类型 (=2)，其余时间位置/时序衰减编码与包裹 token 共用；
  - 拼接后过 Transformer，包裹 query 轨迹通过自注意力感知机械臂运动；
  - 编码结束后仅保留 query 轨迹历史/当前表征做二分类，**输入(样本)与输出(2 类 logits)保持不变**。

数据集目录结构（同 generate_mixed_switch_npy.py 输出）:
  {data_dir}/train/Positive|Negative/*.npy
  {data_dir}/val/Positive|Negative/*.npy
  {data_dir}/test/Positive|Negative/*.npy

用法:
  python mae_switch_classifier_joint_bbox_arm_npy.py train --data_dir ./switch_bbox_npy_mixed
  python mae_switch_classifier_joint_bbox_arm_npy.py test --checkpoint mae_switch_joint_bbox_arm_npy.pt --data_dir ./switch_bbox_npy_mixed

依赖与 py310 conda 环境一致（numpy/pandas/torch/scikit-learn），无需额外安装。
"""

# =============================================================================
# 导入
# =============================================================================

from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# =============================================================================
# 常量
# =============================================================================

FEAT_DIM = 6
# 包裹 token: [norm_x, norm_y, norm_w, norm_h, rel_t, is_target]

ARM_FEAT_DIM = 6
# 机械臂 token: [norm_x, norm_y, norm_w, norm_h, rel_t, present]

# 与 export_switch_bbox_npy.py 保持一致的坐标归一化尺度（机械臂框亦为原始像素）
COORD_NORM_X = 1260.0
COORD_NORM_Y = 720.0

SPLIT_NAMES = ("train", "val", "test")
LABEL_DIR_TO_ID = {"Negative": 0, "Positive": 1}
LABEL_ID_TO_DIR = {0: "Negative", 1: "Positive"}


# =============================================================================
# 预导出 .npy 数据集加载
# =============================================================================

class SampleRef(NamedTuple):
    path: str
    label: int
    sequence: str
    frame: int
    track_id: int


def load_dataset_meta(data_dir: str) -> dict:
    meta_path = os.path.join(data_dir, "dataset_meta.json")
    if not os.path.isfile(meta_path):
        return {}
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def scan_max_dims(refs: List[SampleRef]) -> Tuple[int, int]:
    """扫描 split 内样本，得到 max_hist_len 与 max_tracks（数组第 0 维）。"""
    if not refs:
        return 1, 32
    max_hist, max_tracks = 0, 0
    for ref in refs:
        rec = np.load(ref.path, allow_pickle=True).item()
        seq_len = int(rec["x_in"].shape[1])
        hist_len = int(rec.get("window", seq_len - 1))
        max_hist = max(max_hist, hist_len)
        max_tracks = max(max_tracks, int(rec["x_in"].shape[0]))
    return max(max_hist, 1), max(max_tracks, 1)


def infer_model_dims(
    data_dir: str,
    splits: Tuple[str, ...] = ("train", "val", "test"),
    max_hist_len: Optional[int] = None,
    max_tracks: Optional[int] = None,
) -> Tuple[int, int]:
    """
    从 npy 样本自动推断模型所需 max_hist_len / max_tracks。
    若手动传入 max_hist_len / max_tracks，则取 manual 与扫描值的较大者（保证能容纳数据）。
    """
    scanned_hist, scanned_tracks = 0, 0
    for split in splits:
        refs = collect_split_samples(data_dir, split)
        h, t = scan_max_dims(refs)
        scanned_hist = max(scanned_hist, h)
        scanned_tracks = max(scanned_tracks, t)

    if scanned_hist == 0:
        meta = load_dataset_meta(data_dir)
        exp = meta.get("export_config", {})
        scanned_hist = int(exp.get("window", 50))
        scanned_tracks = int(exp.get("max_tracks", 32))

    out_hist = max(scanned_hist, int(max_hist_len or 0))
    out_tracks = max(scanned_tracks, int(max_tracks or 0))
    return out_hist, out_tracks


def _scan_split_dir(data_dir: str, split: str) -> List[SampleRef]:
    refs: List[SampleRef] = []
    for label_name, label_id in LABEL_DIR_TO_ID.items():
        pattern = os.path.join(data_dir, split, label_name, "*.npy")
        for path in sorted(glob.glob(pattern)):
            refs.append(SampleRef(path=path, label=label_id, sequence="", frame=0, track_id=0))
    return refs


def collect_split_samples(
    data_dir: str,
    split: str,
    max_per_class: Optional[int] = None,
    balance: bool = False,
    random_state: int = 42,
) -> List[SampleRef]:
    manifest_path = os.path.join(data_dir, "manifest.csv")
    refs: List[SampleRef] = []

    if os.path.isfile(manifest_path):
        df = pd.read_csv(manifest_path)
        df = df[(df["split"] == split) & df["status"].isin(["saved", "exists"])]
        for _, row in df.iterrows():
            refs.append(SampleRef(
                path=str(row["path"]),
                label=int(row["label"]),
                sequence=str(row["sequence"]),
                frame=int(row["frame"]),
                track_id=int(row["id"]),
            ))
    else:
        raw_refs = _scan_split_dir(data_dir, split)
        refs = []
        for ref in raw_refs:
            if ref.sequence:
                refs.append(ref)
                continue
            base = os.path.basename(ref.path)
            stem = os.path.splitext(base)[0]
            parts = stem.rsplit("_id", 1)
            if len(parts) != 2:
                refs.append(ref)
                continue
            head, tid = parts[0], parts[1]
            seq, frame_str = head.rsplit("_f", 1)
            refs.append(SampleRef(
                path=ref.path,
                label=ref.label,
                sequence=seq,
                frame=int(frame_str),
                track_id=int(tid),
            ))

    if balance and split == "train":
        pos = [r for r in refs if r.label == 1]
        neg = [r for r in refs if r.label == 0]
        if pos and neg:
            n = min(len(pos), len(neg))
            rng = np.random.default_rng(random_state)
            pos_idx = rng.choice(len(pos), size=n, replace=False)
            neg_idx = rng.choice(len(neg), size=n, replace=False)
            refs = [pos[i] for i in pos_idx] + [neg[i] for i in neg_idx]
            rng.shuffle(refs)

    if max_per_class is not None and max_per_class > 0:
        out: List[SampleRef] = []
        rng = np.random.default_rng(random_state)
        for label_id in (0, 1):
            sub = [r for r in refs if r.label == label_id]
            if len(sub) > max_per_class:
                idx = rng.choice(len(sub), size=max_per_class, replace=False)
                sub = [sub[i] for i in idx]
            out.extend(sub)
        rng.shuffle(out)
        refs = out

    return refs


def print_sample_distribution(refs: List[SampleRef], name: str):
    labels = np.array([r.label for r in refs], dtype=np.int64)
    if len(labels) == 0:
        print(f"  {name}: total=0")
        return
    vc = np.bincount(labels, minlength=2)
    total = len(labels)
    print(
        f"  {name}: total={total}  "
        f"label_0={vc[0]} ({100 * vc[0] / total:.1f}%)  "
        f"label_1={vc[1]} ({100 * vc[1] / total:.1f}%)"
    )


def _rel_t_for_window(window_len: int) -> np.ndarray:
    """与 export_switch_bbox_npy 一致的历史帧相对时间：末位=0，越早越负。"""
    denom = max(window_len - 1, 1)
    return (np.arange(window_len, dtype=np.float32) - (window_len - 1)) / denom


def build_arm_tokens(rec: dict, hist_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    从 npy record 的机械臂字段构造 (T_all=hist_len+1, ARM_FEAT_DIM) 特征与 (T_all,) 有效位。
    机械臂坐标与包裹相同尺度归一化；present=False 的帧标记为无效(被注意力屏蔽)。
    缺少机械臂字段的旧样本 -> 全 0 / 全无效，等价于“无机械臂信息”。
    """
    t_all = hist_len + 1
    arm_feat = np.zeros((t_all, ARM_FEAT_DIM), dtype=np.float32)
    arm_valid = np.zeros((t_all,), dtype=bool)

    arm_hist_bbox = rec.get("arm_hist_bbox")
    arm_hist_present = rec.get("arm_hist_present")
    if arm_hist_bbox is not None and arm_hist_present is not None:
        h = min(hist_len, int(arm_hist_bbox.shape[0]))
        rel_t = _rel_t_for_window(hist_len)
        for ti in range(h):
            if not bool(arm_hist_present[ti]):
                continue
            x, y, w, hgt = arm_hist_bbox[ti]
            arm_feat[ti, 0] = float(x) / COORD_NORM_X
            arm_feat[ti, 1] = float(y) / COORD_NORM_Y
            arm_feat[ti, 2] = float(w) / COORD_NORM_X
            arm_feat[ti, 3] = float(hgt) / COORD_NORM_Y
            arm_feat[ti, 4] = rel_t[ti]
            arm_feat[ti, 5] = 1.0
            arm_valid[ti] = True

    arm_cur_bbox = rec.get("arm_cur_bbox")
    arm_cur_present = rec.get("arm_cur_present")
    if arm_cur_bbox is not None and arm_cur_present is not None and bool(arm_cur_present):
        x, y, w, hgt = arm_cur_bbox
        arm_feat[hist_len, 0] = float(x) / COORD_NORM_X
        arm_feat[hist_len, 1] = float(y) / COORD_NORM_Y
        arm_feat[hist_len, 2] = float(w) / COORD_NORM_X
        arm_feat[hist_len, 3] = float(hgt) / COORD_NORM_Y
        arm_feat[hist_len, 4] = 0.0
        arm_feat[hist_len, 5] = 1.0
        arm_valid[hist_len] = True

    return arm_feat, arm_valid


class NpyJointSwitchDataset(Dataset):
    """从 generate_mixed_switch_npy.py 导出的 .npy 加载样本（含机械臂字段，历史长度可变）。"""

    def __init__(
        self,
        data_dir: str,
        split: str,
        max_per_class: Optional[int] = None,
        balance: bool = False,
        random_state: int = 42,
        use_hist_len: Optional[int] = None,
    ):
        if split not in SPLIT_NAMES:
            raise ValueError(f"split 必须是 {SPLIT_NAMES} 之一，收到: {split}")
        self.data_dir = data_dir
        self.split = split
        # 运行时历史截断：只保留最近 use_hist_len 帧历史 + 当前帧；None=用数据集完整 window
        self.use_hist_len = int(use_hist_len) if use_hist_len and use_hist_len > 0 else None
        self.refs = collect_split_samples(
            data_dir, split,
            max_per_class=max_per_class,
            balance=balance,
            random_state=random_state,
        )
        if not self.refs:
            raise RuntimeError(f"未找到样本: {data_dir}/{split}/{{Positive,Negative}}/*.npy")
        self._labels = np.array([r.label for r in self.refs], dtype=np.int64)
        self.max_hist_len, self.max_tracks_dim = scan_max_dims(self.refs)
        if self.use_hist_len is not None:
            self.max_hist_len = min(self.max_hist_len, self.use_hist_len)

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, idx: int) -> dict:
        ref = self.refs[idx]
        rec = np.load(ref.path, allow_pickle=True).item()

        x_in = rec["x_in"].astype(np.float32, copy=False)
        token_valid = rec["token_valid"].astype(bool, copy=False)
        n_tracks = int(rec["n_tracks"])
        hist_len = int(rec.get("window", x_in.shape[1] - 1))

        arm_feat, arm_valid = build_arm_tokens(rec, x_in.shape[1] - 1)

        # 运行时历史截断：保留最近 K 帧历史 + 当前帧(末位)，arm token 同步尾部截断。
        # rel_t 等特征保留原值（末位=0，越早越负），语义不变。
        if self.use_hist_len is not None and hist_len > self.use_hist_len:
            keep = self.use_hist_len + 1  # +1 = 当前帧
            x_in = x_in[:, -keep:, :]
            token_valid = token_valid[:, -keep:]
            arm_feat = arm_feat[-keep:, :]
            arm_valid = arm_valid[-keep:]
            hist_len = self.use_hist_len

        return {
            "x_in": torch.from_numpy(np.ascontiguousarray(x_in)),
            "hist_valid": torch.from_numpy(np.ascontiguousarray(token_valid)),
            "arm_in": torch.from_numpy(np.ascontiguousarray(arm_feat)),
            "arm_valid": torch.from_numpy(np.ascontiguousarray(arm_valid)),
            "n_tracks": n_tracks,
            "hist_len": hist_len,
            "label": torch.tensor(int(rec["label"]), dtype=torch.long),
            "meta": {
                "path": ref.path,
                "sequence": str(rec.get("sequence", ref.sequence)),
                "frame": int(rec.get("frame", ref.frame)),
                "id": int(rec.get("id", ref.track_id)),
            },
        }


def make_weighted_sampler(dataset: NpyJointSwitchDataset) -> WeightedRandomSampler:
    labels = dataset._labels
    counts = np.bincount(labels, minlength=2).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = 1.0 / counts[labels]
    return WeightedRandomSampler(
        weights=torch.from_numpy(weights).double(),
        num_samples=len(weights),
        replacement=True,
    )


def collate_mae_batch(batch: List[dict]) -> dict:
    """按 batch 内最长序列右对齐 padding（当前帧始终在末位）；机械臂 token 同步对齐。"""
    max_t = max(int(b["x_in"].shape[1]) for b in batch)
    max_n = max(int(b["x_in"].shape[0]) for b in batch)
    d = batch[0]["x_in"].shape[2]
    arm_d = batch[0]["arm_in"].shape[1]
    bs = len(batch)

    x_in = torch.zeros(bs, max_n, max_t, d, dtype=batch[0]["x_in"].dtype)
    hist_valid = torch.zeros(bs, max_n, max_t, dtype=torch.bool)
    arm_in = torch.zeros(bs, max_t, arm_d, dtype=batch[0]["arm_in"].dtype)
    arm_valid = torch.zeros(bs, max_t, dtype=torch.bool)
    n_tracks, labels, hist_lens = [], [], []

    for i, b in enumerate(batch):
        x = b["x_in"]
        v = b["hist_valid"]
        n, t, _ = x.shape
        x_in[i, :n, max_t - t : max_t, :] = x
        hist_valid[i, :n, max_t - t : max_t] = v
        arm_in[i, max_t - t : max_t, :] = b["arm_in"]
        arm_valid[i, max_t - t : max_t] = b["arm_valid"]
        n_tracks.append(int(b["n_tracks"]))
        labels.append(b["label"])
        hist_lens.append(int(b.get("hist_len", t - 1)))

    return {
        "x_in": x_in,
        "hist_valid": hist_valid,
        "arm_in": arm_in,
        "arm_valid": arm_valid,
        "n_tracks": torch.tensor(n_tracks, dtype=torch.long),
        "hist_len": torch.tensor(hist_lens, dtype=torch.long),
        "label": torch.stack(labels),
    }


# =============================================================================
# 模型：shared encoder + arm-aware hist/current joint classification
# =============================================================================

TIME_POS_SINUSOIDAL = "sinusoidal"
TIME_POS_LEARNABLE = "learnable"

CAT_ENCODING_ONE_HOT = "one_hot"
CAT_ENCODING_LEARNABLE = "learnable"

TOKEN_TYPE_HIST = 0
TOKEN_TYPE_CURRENT = 1
TOKEN_TYPE_ARM = 2
N_TOKEN_TYPES = 3


def build_sinusoidal_pe(max_len: int, d_model: int) -> torch.Tensor:
    """Transformer 原版正弦/余弦位置编码，shape (max_len, d_model)。"""
    positions = torch.arange(0, max_len, dtype=torch.float32)
    return _sinusoidal_pe_from_positions(positions, d_model)


def _sinusoidal_pe_from_positions(positions: torch.Tensor, d_model: int) -> torch.Tensor:
    """positions: (...) 时间步索引，返回 (..., d_model)。"""
    orig_shape = positions.shape
    pos = positions.float().unsqueeze(-1)  # (..., 1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=positions.device, dtype=torch.float32)
        * (-np.log(10000.0) / d_model)
    )
    pe = torch.zeros(*orig_shape, d_model, device=positions.device, dtype=torch.float32)
    pe[..., 0::2] = torch.sin(pos * div_term)
    if d_model > 1:
        pe[..., 1::2] = torch.cos(pos * div_term[: d_model // 2])
    return pe


class TimePositionEncoding(nn.Module):
    """时间步位置编码：默认正弦/余弦，可选可学习 Embedding。"""

    def __init__(self, d_model: int, max_len: int, mode: str = TIME_POS_SINUSOIDAL):
        super().__init__()
        if mode not in (TIME_POS_SINUSOIDAL, TIME_POS_LEARNABLE):
            raise ValueError(f"time_pos_encoding 必须是 {TIME_POS_SINUSOIDAL!r} 或 {TIME_POS_LEARNABLE!r}")
        self.mode = mode
        self.max_len = max_len
        if mode == TIME_POS_LEARNABLE:
            self.embed = nn.Embedding(max_len, d_model)
        else:
            self.register_buffer("pe", build_sinusoidal_pe(max_len, d_model), persistent=False)

    def forward(self, time_ids: torch.Tensor) -> torch.Tensor:
        if self.mode == TIME_POS_LEARNABLE:
            idx = time_ids.clamp(min=0, max=self.max_len - 1)
            return self.embed(idx)
        return _sinusoidal_pe_from_positions(time_ids, self.pe.shape[-1])


class CategoricalEncoding(nn.Module):
    """离散 id 编码：默认固定 one-hot（前 num_classes 维为 1），可选可学习 Embedding。"""

    def __init__(self, num_classes: int, d_model: int, mode: str = CAT_ENCODING_ONE_HOT):
        super().__init__()
        if mode not in (CAT_ENCODING_ONE_HOT, CAT_ENCODING_LEARNABLE):
            raise ValueError(
                f"类别编码 mode 必须是 {CAT_ENCODING_ONE_HOT!r} 或 {CAT_ENCODING_LEARNABLE!r}"
            )
        self.mode = mode
        self.num_classes = num_classes
        self.d_model = d_model
        if mode == CAT_ENCODING_LEARNABLE:
            self.embed = nn.Embedding(num_classes, d_model)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        ids = ids.clamp(min=0, max=self.num_classes - 1)
        if self.mode == CAT_ENCODING_LEARNABLE:
            return self.embed(ids)
        one_hot = F.one_hot(ids, self.num_classes).float()
        if self.d_model > self.num_classes:
            pad = torch.zeros(
                *one_hot.shape[:-1],
                self.d_model - self.num_classes,
                device=one_hot.device,
                dtype=one_hot.dtype,
            )
            return torch.cat([one_hot, pad], dim=-1)
        return one_hot[..., : self.d_model]


@dataclass
class JointEncoderConfig:
    feat_dim: int = FEAT_DIM
    arm_feat_dim: int = ARM_FEAT_DIM
    hidden: int = 256
    n_heads: int = 8
    n_layers: int = 4
    max_tracks: int = 128
    max_frames: int = 64
    dropout: float = 0.1
    use_arm: bool = True  # False=消融机械臂：arm token 完全不进 Transformer
    use_arm_cross_attn: bool = False  # True=额外做 query->arm 交叉注意力，arm-context 直送分类头
    use_temporal_decay: bool = True
    decay_rate: float = 0.08
    time_pos_encoding: str = TIME_POS_SINUSOIDAL
    track_id_encoding: str = CAT_ENCODING_ONE_HOT
    token_type_encoding: str = CAT_ENCODING_ONE_HOT
    encode_micro_batch: int = 0  # 0=按 token 数自动；>0 固定每次 forward 的 batch 子块大小
    max_encode_tokens: int = 4096


class JointTrajectoryEncoder(nn.Module):
    """
    历史帧 + 当前帧包裹坐标 token 与机械臂 token 共用 Transformer。
    机械臂作为一条额外轨迹（独立特征投影/身份/类型），供包裹 query 通过注意力感知其运动。
    返回 query 轨迹的历史 pooled 表征与当前帧 token 表征。
    """

    def __init__(self, cfg: JointEncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.hidden = cfg.hidden

        self.feat_proj = nn.Linear(cfg.feat_dim, cfg.hidden)
        self.missing_token = nn.Parameter(torch.zeros(1, 1, cfg.hidden))
        self.track_encoding = CategoricalEncoding(
            cfg.max_tracks, cfg.hidden, mode=cfg.track_id_encoding
        )
        self.time_pos = TimePositionEncoding(
            cfg.hidden, cfg.max_frames, mode=cfg.time_pos_encoding
        )
        self.token_type_encoding = CategoricalEncoding(
            N_TOKEN_TYPES, cfg.hidden, mode=cfg.token_type_encoding
        )  # 0=history, 1=current, 2=arm

        # 机械臂专用：特征投影 + 身份向量（相当于机械臂这条“轨迹”的 track 编码）
        self.arm_proj = nn.Linear(cfg.arm_feat_dim, cfg.hidden)
        self.arm_identity = nn.Parameter(torch.zeros(1, 1, cfg.hidden))

        # query->arm 交叉注意力：query 轨迹显式查询机械臂 token，输出 arm-context 直送分类头。
        # null_arm 兜底：某样本机械臂全缺失时提供一个可学习的空 key/value，避免 attention 全 mask 出 NaN。
        self.use_arm_cross_attn = bool(cfg.use_arm and cfg.use_arm_cross_attn)
        if self.use_arm_cross_attn:
            self.arm_cross_attn = nn.MultiheadAttention(
                embed_dim=cfg.hidden, num_heads=cfg.n_heads,
                dropout=cfg.dropout, batch_first=True,
            )
            self.arm_cross_norm_q = nn.LayerNorm(cfg.hidden)
            self.arm_cross_norm_kv = nn.LayerNorm(cfg.hidden)
            self.arm_ctx_norm = nn.LayerNorm(cfg.hidden)
            self.null_arm = nn.Parameter(torch.zeros(1, 1, cfg.hidden))

        if cfg.use_temporal_decay:
            self.decay_proj = nn.Linear(1, cfg.hidden)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.hidden * 4,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=cfg.n_layers, enable_nested_tensor=False
        )

    @staticmethod
    def suggest_micro_batch(batch_size: int, n_tracks: int, t_all: int, max_tokens: int = 4096) -> int:
        """按单 forward 总 token 上限估算 micro_batch，降低 attention O(L²) 显存。"""
        # 机械臂额外增加 t_all 个 token，等价于 (n_tracks + 1) 条轨迹
        tokens_per_sample = max((int(n_tracks) + 1) * int(t_all), 1)
        per_forward = max(1, int(max_tokens) // tokens_per_sample)
        return max(1, min(int(batch_size), per_forward))

    def _encode_impl(
        self,
        x_in: torch.Tensor,
        token_valid: torch.Tensor,
        arm_in: torch.Tensor,
        arm_valid: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        x_in: (B, N, T+1, D), 末位为当前帧坐标 token
        arm_in: (B, T+1, arm_D), arm_valid: (B, T+1)
        returns: hist_repr (B, hidden), cur_repr (B, hidden),
                 arm_ctx (B, hidden) 或 None（未开启 cross-attn 时）
        """
        B, N, T_all, D = x_in.shape
        T_hist = T_all - 1
        device = x_in.device

        # ---- 包裹 token ----
        x_flat = x_in.reshape(B, N * T_all, D)
        val_flat = token_valid.reshape(B, N * T_all)

        feat_emb = self.feat_proj(x_flat)
        miss_emb = self.missing_token.expand(B, N * T_all, -1)
        token_emb = torch.where(val_flat.unsqueeze(-1), feat_emb, miss_emb)

        track_ids = torch.arange(N, device=device).unsqueeze(1).expand(N, T_all).reshape(-1)
        track_ids = track_ids.unsqueeze(0).expand(B, -1)
        time_ids = torch.arange(T_all, device=device).unsqueeze(0).expand(N, -1).reshape(-1)
        time_ids = time_ids.unsqueeze(0).expand(B, -1)

        token_type = torch.zeros(N * T_all, dtype=torch.long, device=device)
        for ti in range(T_all):
            token_type[ti::T_all] = TOKEN_TYPE_CURRENT if ti == T_all - 1 else TOKEN_TYPE_HIST
        token_type = token_type.unsqueeze(0).expand(B, -1)

        h_pkg = (
            token_emb
            + self.track_encoding(track_ids)
            + self.time_pos(time_ids)
            + self.token_type_encoding(token_type)
        )
        if self.cfg.use_temporal_decay:
            rel = (time_ids.float() - (T_all - 1)) / max(T_all - 1, 1)
            h_pkg = h_pkg + self.decay_proj(rel.unsqueeze(-1))

        # ---- 机械臂 token（use_arm=False 时整条支路跳过，等价纯 bbox 版本）----
        if self.cfg.use_arm:
            arm_emb = self.arm_proj(arm_in)  # (B, T_all, hidden)
            arm_time_ids = torch.arange(T_all, device=device).unsqueeze(0).expand(B, -1)
            arm_type_ids = torch.full((B, T_all), TOKEN_TYPE_ARM, dtype=torch.long, device=device)
            h_arm = (
                arm_emb
                + self.arm_identity.expand(B, T_all, -1)
                + self.time_pos(arm_time_ids)
                + self.token_type_encoding(arm_type_ids)
            )
            if self.cfg.use_temporal_decay:
                arm_rel = (arm_time_ids.float() - (T_all - 1)) / max(T_all - 1, 1)
                h_arm = h_arm + self.decay_proj(arm_rel.unsqueeze(-1))

            # ---- 拼接过 Transformer ----
            h = torch.cat([h_pkg, h_arm], dim=1)
            pad_mask = torch.cat([~val_flat, ~arm_valid], dim=1)
        else:
            h = h_pkg
            pad_mask = ~val_flat
        h = self.encoder(h, src_key_padding_mask=pad_mask)

        h_pkg_out = h[:, : N * T_all].reshape(B, N, T_all, -1)

        query_hist = h_pkg_out[:, 0, :T_hist, :]
        hist_mask = token_valid[:, 0, :T_hist].unsqueeze(-1).float()
        if self.cfg.use_temporal_decay:
            w = self._decay_weights(T_hist, device).view(1, T_hist, 1)
            hist_mask = hist_mask * w
        hist_repr = (query_hist * hist_mask).sum(dim=1) / hist_mask.sum(dim=1).clamp(min=1e-6)

        cur_ti = T_all - 1
        cur_valid = token_valid[:, 0, cur_ti]
        cur_repr = h_pkg_out[:, 0, cur_ti, :]
        cur_repr = torch.where(cur_valid.unsqueeze(-1), cur_repr, torch.zeros_like(cur_repr))

        # ---- query->arm 交叉注意力：arm-context 直送分类头 ----
        arm_ctx = None
        if self.use_arm_cross_attn:
            arm_out = h[:, N * T_all:]  # (B, T_all, hidden) 编码后的机械臂 token
            # query = query 轨迹的 [hist_repr, cur_repr] 两个表征
            q = torch.stack([hist_repr, cur_repr], dim=1)  # (B, 2, hidden)
            q = self.arm_cross_norm_q(q)
            # key/value = arm token + null 兜底 token（永不被 mask，避免全缺失出 NaN）
            null_kv = self.null_arm.expand(B, 1, -1)
            kv = torch.cat([self.arm_cross_norm_kv(arm_out), null_kv], dim=1)  # (B, T_all+1, hidden)
            null_valid = torch.ones(B, 1, dtype=torch.bool, device=device)
            kv_pad = torch.cat([~arm_valid, ~null_valid], dim=1)  # True=屏蔽
            attn_out, _ = self.arm_cross_attn(
                q, kv, kv, key_padding_mask=kv_pad, need_weights=False
            )  # (B, 2, hidden)
            arm_ctx = self.arm_ctx_norm(attn_out.mean(dim=1))  # (B, hidden)

        return hist_repr, cur_repr, arm_ctx

    def _decay_weights(self, T: int, device: torch.device) -> torch.Tensor:
        t = torch.arange(T, device=device, dtype=torch.float32)
        return torch.exp(self.cfg.decay_rate * (t - (T - 1)))

    def encode(
        self,
        x_in: torch.Tensor,
        token_valid: torch.Tensor,
        arm_in: torch.Tensor,
        arm_valid: torch.Tensor,
        micro_batch: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        x_in: (B, N, T+1, D)。micro_batch>1 时分块过 Transformer，显著降低显存峰值。
        返回 (hist_repr, cur_repr, arm_ctx)；arm_ctx 在未开 cross-attn 时为 None。
        """
        B = x_in.shape[0]
        mb_cfg = self.cfg.encode_micro_batch if micro_batch is None else micro_batch
        if mb_cfg <= 0:
            mb = self.suggest_micro_batch(
                B, x_in.shape[1], x_in.shape[2], max_tokens=self.cfg.max_encode_tokens
            )
        else:
            mb = max(1, min(int(mb_cfg), B))

        if mb >= B:
            return self._encode_impl(x_in, token_valid, arm_in, arm_valid)

        hist_parts, cur_parts, arm_parts = [], [], []
        for start in range(0, B, mb):
            end = min(start + mb, B)
            h, c, a = self._encode_impl(
                x_in[start:end], token_valid[start:end],
                arm_in[start:end], arm_valid[start:end],
            )
            hist_parts.append(h)
            cur_parts.append(c)
            if a is not None:
                arm_parts.append(a)
        arm_ctx = torch.cat(arm_parts, dim=0) if arm_parts else None
        return torch.cat(hist_parts, dim=0), torch.cat(cur_parts, dim=0), arm_ctx


@dataclass
class JointSwitchConfig:
    max_hist_len: int = 50
    max_tracks: int = 32
    hidden: int = 256
    n_heads: int = 8
    n_layers: int = 4
    hidden_cls: int = 256
    dropout: float = 0.1
    use_arm: bool = True  # False=消融机械臂支路
    use_arm_cross_attn: bool = False  # True=query->arm 交叉注意力增强 recall
    use_temporal_decay: bool = True
    decay_rate: float = 0.08
    time_pos_encoding: str = TIME_POS_SINUSOIDAL
    track_id_encoding: str = CAT_ENCODING_ONE_HOT
    token_type_encoding: str = CAT_ENCODING_ONE_HOT
    encode_micro_batch: int = 0  # 0=按 token 数自动；>0 固定每次 forward 的 batch 子块大小
    max_encode_tokens: int = 4096


class JointSwitchClassifier(nn.Module):
    """历史 + 当前帧 + 机械臂运动 联合高维表征 -> 二分类。"""

    def __init__(self, cfg: JointSwitchConfig):
        super().__init__()
        self.cfg = cfg
        enc_cfg = JointEncoderConfig(
            feat_dim=FEAT_DIM,
            arm_feat_dim=ARM_FEAT_DIM,
            hidden=cfg.hidden,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_layers,
            max_tracks=max(cfg.max_tracks, 32),
            max_frames=max(cfg.max_hist_len + 17, 64),
            dropout=cfg.dropout,
            use_arm=cfg.use_arm,
            use_arm_cross_attn=cfg.use_arm_cross_attn,
            use_temporal_decay=cfg.use_temporal_decay,
            decay_rate=cfg.decay_rate,
            time_pos_encoding=cfg.time_pos_encoding,
            track_id_encoding=cfg.track_id_encoding,
            token_type_encoding=cfg.token_type_encoding,
            encode_micro_batch=cfg.encode_micro_batch,
            max_encode_tokens=cfg.max_encode_tokens,
        )
        self.encoder = JointTrajectoryEncoder(enc_cfg)
        self.use_arm_cross_attn = bool(cfg.use_arm and cfg.use_arm_cross_attn)
        # 开启 cross-attn 时，额外拼一个 arm-context 向量 -> joint_dim = hidden*3
        joint_dim = cfg.hidden * (3 if self.use_arm_cross_attn else 2)
        self.classifier = nn.Sequential(
            nn.LayerNorm(joint_dim),
            nn.Linear(joint_dim, cfg.hidden_cls),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_cls, cfg.hidden_cls // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_cls // 2, 2),
        )

    def forward(
        self,
        x_in: torch.Tensor,
        token_valid: torch.Tensor,
        arm_in: torch.Tensor,
        arm_valid: torch.Tensor,
        encode_micro_batch: Optional[int] = None,
    ) -> torch.Tensor:
        hist_repr, cur_repr, arm_ctx = self.encoder.encode(
            x_in, token_valid, arm_in, arm_valid, micro_batch=encode_micro_batch
        )
        if self.use_arm_cross_attn and arm_ctx is not None:
            joint = torch.cat([hist_repr, cur_repr, arm_ctx], dim=-1)
        else:
            joint = torch.cat([hist_repr, cur_repr], dim=-1)
        return self.classifier(joint)


# =============================================================================
# 可视化
# =============================================================================

def plot_error_window_trajectories(
    hist_px: np.ndarray,
    valid: np.ndarray,
    track_ids: List[int],
    query_id: int,
    window_frames: np.ndarray,
    birth_pad: np.ndarray,
    true_label: int,
    pred_label: int,
    prob_switch: float,
    error_type: str,
    save_path: str,
):
    if not HAS_MPL:
        return
    cmap = plt.get_cmap("tab20", max(len(track_ids), 1))
    fig, ax = plt.subplots(figsize=(11, 8), facecolor="white")
    ax.set_facecolor("white")

    for mi, tid in enumerate(track_ids):
        mask = valid[mi]
        if not mask.any():
            continue
        pts = hist_px[mi, mask]
        is_query = tid == query_id
        color = "crimson" if is_query else cmap(mi % 20)
        lw = 2.5 if is_query else 1.2
        ms = 5 if is_query else 3
        label = f"ID {tid}" + (" (query)" if is_query else "")
        ax.plot(pts[:, 0], pts[:, 1], "-", color=color, linewidth=lw, alpha=0.85, label=label)
        ax.scatter(pts[:, 0], pts[:, 1], c=[color], s=ms ** 2, zorder=3)
        bp = birth_pad[mi] & mask
        if bp.any():
            ax.scatter(
                hist_px[mi, bp, 0], hist_px[mi, bp, 1],
                marker="s", facecolors="none", edgecolors=color,
                s=70, linewidths=1.5, zorder=4,
            )

    ax.invert_yaxis()
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.set_title(
        f"{error_type} | true={true_label} pred={pred_label} prob={prob_switch:.3f}\n"
        f"frames {int(window_frames[0])}-{int(window_frames[-1])}  query_id={query_id}",
        fontsize=10,
    )
    ax.legend(loc="best", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.25, color="gray")
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=120, facecolor="white")
    plt.close(fig)


def plot_test_errors(
    test_ds: NpyJointSwitchDataset,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    out_dir: str,
    max_plots_per_type: int = 100,
):
    if not HAS_MPL:
        print("matplotlib 未安装，跳过 FN/FP 轨迹可视化")
        return

    fn_dir = os.path.join(out_dir, "plots_FN")
    fp_dir = os.path.join(out_dir, "plots_FP")
    os.makedirs(fn_dir, exist_ok=True)
    os.makedirs(fp_dir, exist_ok=True)

    fn_count, fp_count = 0, 0
    for idx in range(len(test_ds)):
        true_l = int(y_true[idx])
        pred_l = int(y_pred[idx])
        prob = float(y_prob[idx])
        if true_l == 1 and pred_l == 0:
            if fn_count >= max_plots_per_type:
                continue
            err_type, subdir = "FN", fn_dir
            fn_count += 1
        elif true_l == 0 and pred_l == 1:
            if fp_count >= max_plots_per_type:
                continue
            err_type, subdir = "FP", fp_dir
            fp_count += 1
        else:
            continue

        ref = test_ds.refs[idx]
        rec = np.load(ref.path, allow_pickle=True).item()
        m = int(rec["n_tracks"])
        track_ids = [int(t) for t in rec["track_ids"][:m] if int(t) >= 0]
        birth_pad = np.zeros((m, int(rec.get("window", rec["hist_valid"].shape[1]))), dtype=bool)

        fname = (
            f"{err_type}_{ref.sequence}_f{ref.frame}_id{ref.track_id}"
            f"_p{prob:.3f}.png"
        )
        plot_error_window_trajectories(
            rec["hist_bbox"][:m, :, :2],
            rec["hist_valid"][:m],
            track_ids,
            ref.track_id,
            rec["window_frames"],
            birth_pad,
            true_l, pred_l, prob, err_type,
            os.path.join(subdir, fname),
        )

    print(f"\n已保存 FN 轨迹图 {fn_count} 张 -> {fn_dir}/")
    print(f"已保存 FP 轨迹图 {fp_count} 张 -> {fp_dir}/")


# =============================================================================
# 训练与评估
# =============================================================================

def _prf_from_scores(scores: np.ndarray, y: np.ndarray, thr: float) -> Tuple[int, int, int, float, float, float]:
    """给定分数(prob 或修正后 prob)与阈值，返回 (tp, fp, fn, precision, recall, f1)。"""
    pred = (scores >= thr).astype(np.int64)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    return tp, fp, fn, prec, rec, f1


def apply_prior_correction(margin: np.ndarray, log_prior_shift: float) -> np.ndarray:
    """
    对 logit margin (= z1 - z0) 做先验修正后转回 prob_switch。
      corrected_logit = margin + log_prior_shift
      prob = sigmoid(corrected_logit)
    log_prior_shift = log(pi_target / pi_train)：训练先验高于目标先验时为负 → 压低正类分数。
    """
    z = margin.astype(np.float64) + float(log_prior_shift)
    return 1.0 / (1.0 + np.exp(-z))


def compute_prior_shift(train_pos_ratio: float, target_pos_ratio: float) -> float:
    """log(pi_target/(1-pi_target)) - log(pi_train/(1-pi_train))，即 logit 空间的先验平移量。"""
    tp = min(max(float(target_pos_ratio), 1e-12), 1 - 1e-12)
    rp = min(max(float(train_pos_ratio), 1e-12), 1 - 1e-12)
    return float(np.log(tp / (1 - tp)) - np.log(rp / (1 - rp)))


def sweep_thresholds(
    scores: np.ndarray,
    y: np.ndarray,
    target_precision: float = 0.7,
    target_recall: float = 0.85,
    n_grid: int = 512,
) -> dict:
    """
    在给定分数上扫描阈值，返回三种标准的最优阈值：
      - f1_max:            F1 最高
      - precision_target:  precision >= target_precision 中 recall 最大（默认推荐）
      - recall_target:     recall >= target_recall 中 precision 最大
    每种标准附带该阈值下的 tp/fp/fn/precision/recall/f1；无满足约束的阈值时回退到 f1_max。
    """
    uniq = np.unique(scores)
    if len(uniq) > n_grid:
        qs = np.linspace(0.0, 1.0, n_grid)
        grid = np.unique(np.quantile(uniq, qs))
    else:
        grid = uniq
    # 补上略高于最大值的阈值，覆盖"全部判负"边界
    grid = np.concatenate([grid, [min(1.0, float(grid.max()) + 1e-6)]])

    rows = []
    for thr in grid:
        tp, fp, fn, prec, rec, f1 = _prf_from_scores(scores, y, float(thr))
        rows.append({"threshold": float(thr), "tp": tp, "fp": fp, "fn": fn,
                     "precision": prec, "recall": rec, "f1": f1})

    def _best(key_ok, sort_key):
        cands = [r for r in rows if key_ok(r)]
        if not cands:
            return None
        return max(cands, key=sort_key)

    f1_max = max(rows, key=lambda r: r["f1"])
    prec_target = _best(
        lambda r: r["precision"] >= target_precision,
        lambda r: (r["recall"], r["precision"]),
    ) or dict(f1_max, fallback="no_threshold_meets_precision")
    rec_target = _best(
        lambda r: r["recall"] >= target_recall,
        lambda r: (r["precision"], r["recall"]),
    ) or dict(f1_max, fallback="no_threshold_meets_recall")

    return {
        "target_precision": target_precision,
        "target_recall": target_recall,
        "f1_max": f1_max,
        "precision_target": prec_target,
        "recall_target": rec_target,
        "recommended": "precision_target",
    }


@torch.no_grad()
def evaluate_classifier(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    encode_micro_batch: Optional[int] = None,
) -> dict:
    model.eval()
    all_y, all_p, all_prob, all_margin = [], [], [], []
    for batch in loader:
        x_in = batch["x_in"].to(device, non_blocking=True)
        token_valid = batch["hist_valid"].to(device, non_blocking=True)
        arm_in = batch["arm_in"].to(device, non_blocking=True)
        arm_valid = batch["arm_valid"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        logits = model(x_in, token_valid, arm_in, arm_valid, encode_micro_batch=encode_micro_batch)
        prob = torch.softmax(logits, dim=-1)[:, 1]
        margin = logits[:, 1] - logits[:, 0]  # = logit(prob_switch)，用于 logit 修正
        pred = logits.argmax(dim=-1)
        all_y.extend(label.cpu().numpy().tolist())
        all_p.extend(pred.cpu().numpy().tolist())
        all_prob.extend(prob.cpu().numpy().tolist())
        all_margin.extend(margin.cpu().numpy().tolist())
        if device.type == "cuda":
            del x_in, token_valid, arm_in, arm_valid, label, logits

    if not all_y:
        return {"accuracy": float("nan"), "f1": float("nan")}

    y = np.array(all_y)
    p = np.array(all_p)
    report = classification_report(y, p, labels=[0, 1], output_dict=True, zero_division=0)
    cm = confusion_matrix(y, p, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    return {
        "accuracy": report["accuracy"],
        "precision": report["1"]["precision"],
        "recall": report["1"]["recall"],
        "f1": report["1"]["f1-score"],
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "y_true": y,
        "y_pred": p,
        "y_prob": np.array(all_prob),
        "y_margin": np.array(all_margin, dtype=np.float64),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    class_weights: Optional[torch.Tensor],
    max_batches: Optional[int] = None,
    encode_micro_batch: Optional[int] = None,
) -> float:
    model.train()
    total_loss, n = 0.0, 0
    for batch in loader:
        x_in = batch["x_in"].to(device, non_blocking=True)
        hist_valid = batch["hist_valid"].to(device, non_blocking=True)
        arm_in = batch["arm_in"].to(device, non_blocking=True)
        arm_valid = batch["arm_valid"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_in, hist_valid, arm_in, arm_valid, encode_micro_batch=encode_micro_batch)
        loss = F.cross_entropy(logits, label, weight=class_weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        n += 1
        if max_batches is not None and n >= max_batches:
            break
        if device.type == "cuda":
            del x_in, hist_valid, arm_in, arm_valid, label, logits, loss
    return total_loss / max(n, 1)


def load_model_from_checkpoint(
    checkpoint_path: str, device: torch.device
) -> Tuple[JointSwitchClassifier, dict]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    c = ckpt.get("cfg", {})
    state_keys = ckpt.get("model", {})
    cfg = JointSwitchConfig(
        max_hist_len=int(c.get("max_hist_len", c.get("window", 50))),
        max_tracks=int(c.get("max_tracks", 32)),
        hidden=int(c.get("hidden", 256)),
        n_heads=int(c.get("n_heads", 8)),
        n_layers=int(c.get("n_layers", 4)),
        hidden_cls=int(c.get("hidden_cls", 256)),
        dropout=float(c.get("dropout", 0.1)),
        use_arm=bool(c.get("use_arm", c.get("arm_aware", True))),
        use_arm_cross_attn=bool(c.get("use_arm_cross_attn", False)),
        use_temporal_decay=bool(c.get("use_temporal_decay", True)),
        decay_rate=float(c.get("decay_rate", 0.08)),
        time_pos_encoding=str(
            c.get(
                "time_pos_encoding",
                TIME_POS_LEARNABLE
                if "encoder.time_embed.weight" in state_keys
                else TIME_POS_SINUSOIDAL,
            )
        ),
        track_id_encoding=str(
            c.get(
                "track_id_encoding",
                CAT_ENCODING_LEARNABLE
                if "encoder.track_embed.weight" in state_keys
                else CAT_ENCODING_ONE_HOT,
            )
        ),
        token_type_encoding=str(
            c.get(
                "token_type_encoding",
                CAT_ENCODING_LEARNABLE
                if "encoder.token_type_embed.weight" in state_keys
                else CAT_ENCODING_ONE_HOT,
            )
        ),
    )
    model = JointSwitchClassifier(cfg).to(device)
    state = dict(ckpt["model"])
    if cfg.time_pos_encoding == TIME_POS_LEARNABLE:
        if "encoder.time_embed.weight" in state and "encoder.time_pos.embed.weight" not in state:
            state["encoder.time_pos.embed.weight"] = state.pop("encoder.time_embed.weight")
    if cfg.track_id_encoding == CAT_ENCODING_LEARNABLE:
        if "encoder.track_embed.weight" in state and "encoder.track_encoding.embed.weight" not in state:
            state["encoder.track_encoding.embed.weight"] = state.pop("encoder.track_embed.weight")
    if cfg.token_type_encoding == CAT_ENCODING_LEARNABLE:
        if (
            "encoder.token_type_embed.weight" in state
            and "encoder.token_type_encoding.embed.weight" not in state
        ):
            state["encoder.token_type_encoding.embed.weight"] = state.pop(
                "encoder.token_type_embed.weight"
            )
    state = {k: v for k, v in state.items() if not k.startswith("encoder.time_embed.")}
    state = {
        k: v
        for k, v in state.items()
        if k not in ("encoder.track_embed.weight", "encoder.token_type_embed.weight")
    }
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print(
            "  checkpoint 加载提示: "
            f"missing={incompatible.missing_keys} unexpected={incompatible.unexpected_keys}"
        )
    model.eval()
    return model, ckpt


@torch.no_grad()
def run_test(
    data_dir: str,
    checkpoint_path: str,
    out_dir: str = "mae_joint_bbox_arm_npy_output",
    max_hist_len: Optional[int] = None,
    max_tracks: Optional[int] = None,
    batch_size: int = 32,
    eval_batch_size: Optional[int] = None,
    encode_micro_batch: int = 0,
    max_encode_tokens: int = 4096,
    test_max_per_class: int = 10000,
    max_error_plots: int = 100,
    device: Optional[str] = None,
    num_workers: int = 0,
    calibrate: bool = True,
    target_precision: float = 0.7,
    target_recall: float = 0.85,
    prior_correction: bool = True,
    train_pos_ratio: float = 0.5,
    target_pos_ratio: Optional[float] = None,
    use_hist_len: Optional[int] = None,
) -> dict:
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    os.makedirs(out_dir, exist_ok=True)

    model, ckpt = load_model_from_checkpoint(checkpoint_path, device_t)
    c = ckpt.get("cfg", {})
    # 测试历史截断：命令行 use_hist_len 优先，否则用 checkpoint 记录的（与训练一致）
    eff_hist_len = use_hist_len if use_hist_len is not None else c.get("use_hist_len")
    eff_hist_len = int(eff_hist_len) if eff_hist_len and int(eff_hist_len) > 0 else None
    scanned_hist, scanned_tracks = infer_model_dims(data_dir, splits=("test",))
    if eff_hist_len is not None:
        scanned_hist = min(scanned_hist, eff_hist_len)
    model_hist = int(c.get("max_hist_len", c.get("window", scanned_hist)))
    model_tracks = int(c.get("max_tracks", scanned_tracks))
    if scanned_hist > model_hist or scanned_tracks > model_tracks:
        print(
            f"  警告: 测试集扫描 max_hist={scanned_hist} max_tracks={scanned_tracks} "
            f"超过 checkpoint 容量 hist={model_hist} tracks={model_tracks}，"
            "位置编码可能被 clamp"
        )

    print(f"Checkpoint: {checkpoint_path}")
    print(f"  data_dir={data_dir}")
    print(
        f"  model_max_hist={model_hist}, test_scan_hist={scanned_hist}, "
        f"max_tracks={model_tracks}, joint_dim={c.get('hidden', 256) * 2}"
    )

    all_test_refs = collect_split_samples(data_dir, "test")
    print_sample_distribution(all_test_refs, "测试集(原始)")
    test_refs = collect_split_samples(
        data_dir, "test", max_per_class=test_max_per_class, random_state=42
    )
    print_sample_distribution(test_refs, f"测试集(每类最多{test_max_per_class})")

    print("构建测试集...")
    test_ds = NpyJointSwitchDataset(
        data_dir, "test", max_per_class=test_max_per_class,
        use_hist_len=eff_hist_len,
    )
    print(
        f"  测试集样本 hist_len 范围: "
        f"[{test_ds.max_hist_len} scanned max in split; per-batch 动态 padding]"
    )
    loader = DataLoader(
        test_ds, batch_size=eval_batch_size or max(1, batch_size // 4), shuffle=False,
        collate_fn=collate_mae_batch, num_workers=num_workers,
        pin_memory=(device_t.type == "cuda"),
    )
    if hasattr(model, "encoder") and hasattr(model.encoder, "cfg"):
        model.encoder.cfg.encode_micro_batch = int(
            c.get("encode_micro_batch", encode_micro_batch)
        )
        model.encoder.cfg.max_encode_tokens = int(
            c.get("max_encode_tokens", max_encode_tokens)
        )
    metrics = evaluate_classifier(model, loader, device_t)

    # =========================================================================
    # 阈值校准 + 先验(logit)修正：在 val 上选阈值，套用到 test
    # =========================================================================
    calib_info: Optional[dict] = None
    if calibrate:
        # 目标先验：默认对齐当前测试集正例率（约 1:100）
        test_y = metrics["y_true"]
        test_pos_ratio = float((test_y == 1).mean()) if len(test_y) else 0.5
        tgt_ratio = float(target_pos_ratio) if target_pos_ratio is not None else test_pos_ratio
        log_shift = compute_prior_shift(train_pos_ratio, tgt_ratio) if prior_correction else 0.0

        print("\n构建验证集(用于阈值校准)...")
        val_ds = NpyJointSwitchDataset(data_dir, "val", use_hist_len=eff_hist_len)
        val_loader = DataLoader(
            val_ds, batch_size=eval_batch_size or max(1, batch_size // 4), shuffle=False,
            collate_fn=collate_mae_batch, num_workers=num_workers,
            pin_memory=(device_t.type == "cuda"),
        )
        val_metrics = evaluate_classifier(model, val_loader, device_t)
        val_y = val_metrics["y_true"]
        val_scores = (
            apply_prior_correction(val_metrics["y_margin"], log_shift)
            if prior_correction else val_metrics["y_prob"]
        )
        sweep = sweep_thresholds(val_scores, val_y, target_precision, target_recall)
        chosen_key = sweep["recommended"]
        chosen_thr = float(sweep[chosen_key]["threshold"])

        print(
            f"  先验修正: prior_correction={prior_correction}  "
            f"train_pos_ratio={train_pos_ratio:.4f}  target_pos_ratio={tgt_ratio:.4f}  "
            f"log_shift={log_shift:.4f}"
        )
        print(f"  验证集阈值扫描 (n_val={len(val_y)}):")
        for key in ("f1_max", "precision_target", "recall_target"):
            r = sweep[key]
            fb = f"  [{r['fallback']}]" if "fallback" in r else ""
            star = " <-- 推荐" if key == chosen_key else ""
            print(
                f"    {key:>16}: thr={r['threshold']:.4f}  "
                f"P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f}{fb}{star}"
            )

        # 套用到 test：先验修正 test 分数，再用 val 选出的阈值判决
        test_scores = (
            apply_prior_correction(metrics["y_margin"], log_shift)
            if prior_correction else metrics["y_prob"]
        )
        cal_pred = (test_scores >= chosen_thr).astype(np.int64)
        tp, fp, fn, prec, rec, f1 = _prf_from_scores(test_scores, test_y, chosen_thr)
        tn = int(((cal_pred == 0) & (test_y == 0)).sum())
        acc = float((cal_pred == test_y).mean()) if len(test_y) else float("nan")

        # 用校准结果覆盖 metrics 的判决相关字段（概率保留修正后的分数）
        metrics["y_pred"] = cal_pred
        metrics["y_prob_calibrated"] = test_scores
        metrics["accuracy"] = acc
        metrics["precision"] = prec
        metrics["recall"] = rec
        metrics["f1"] = f1
        metrics["confusion_matrix"] = {"tn": tn, "fp": fp, "fn": fn, "tp": tp}

        calib_info = {
            "prior_correction": bool(prior_correction),
            "train_pos_ratio": float(train_pos_ratio),
            "target_pos_ratio": float(tgt_ratio),
            "log_prior_shift": float(log_shift),
            "chosen_criterion": chosen_key,
            "chosen_threshold": chosen_thr,
            "val_sweep": {
                k: sweep[k] for k in ("f1_max", "precision_target", "recall_target",
                                      "target_precision", "target_recall", "recommended")
            },
        }

    print("\n========== Test ID 切换分类 (joint bbox + arm npy) ==========")
    print(f"  有效样本: {len(test_ds)}")
    if calib_info is not None:
        print(
            f"  [已校准] 判决标准={calib_info['chosen_criterion']}  "
            f"阈值={calib_info['chosen_threshold']:.4f}  "
            f"(先验修正 log_shift={calib_info['log_prior_shift']:.4f})"
        )
    cm = metrics["confusion_matrix"]
    print("  混淆矩阵  标签0=正常 1=切换")
    print(f"                pred_0   pred_1")
    print(f"    true_0      {cm['tn']:6d}   {cm['fp']:6d}")
    print(f"    true_1      {cm['fn']:6d}   {cm['tp']:6d}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")

    rows = []
    for idx, ref in enumerate(test_ds.refs):
        row = {
            "path": ref.path,
            "sequence": ref.sequence,
            "frame": ref.frame,
            "id": ref.track_id,
            "true_label": ref.label,
            "predicted_label": int(metrics["y_pred"][idx]),
            "prob_switch": float(metrics["y_prob"][idx]),
        }
        if "y_prob_calibrated" in metrics:
            row["prob_switch_calibrated"] = float(metrics["y_prob_calibrated"][idx])
        rows.append(row)
    pred_path = os.path.join(out_dir, "joint_cls_predictions.csv")
    pd.DataFrame(rows).to_csv(pred_path, index=False)

    summary = {
        "checkpoint": checkpoint_path,
        "data_dir": data_dir,
        "max_hist_len": model_hist,
        "test_scan_max_hist": scanned_hist,
        "max_tracks": model_tracks,
        "architecture": "joint_hist_current_bbox_arm_npy",
        "test_max_per_class": test_max_per_class,
        "n_evaluated": len(test_ds),
        "calibration": calib_info,
        "metrics": {
            k: metrics[k]
            for k in ("accuracy", "precision", "recall", "f1", "confusion_matrix")
        },
    }
    metrics_path = os.path.join(out_dir, "joint_cls_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    plot_test_errors(
        test_ds=test_ds,
        y_true=metrics["y_true"],
        y_pred=metrics["y_pred"],
        y_prob=metrics["y_prob"],
        out_dir=out_dir,
        max_plots_per_type=max_error_plots,
    )

    print(f"\n预测明细 -> {pred_path}")
    print(f"指标汇总 -> {metrics_path}")
    return metrics


def train(
    data_dir: str,
    max_hist_len: Optional[int] = None,
    max_tracks: Optional[int] = None,
    hidden: int = 256,
    n_heads: int = 8,
    n_layers: int = 4,
    hidden_cls: int = 256,
    use_arm: bool = True,
    use_arm_cross_attn: bool = False,
    use_temporal_decay: bool = True,
    decay_rate: float = 0.08,
    time_pos_encoding: str = TIME_POS_SINUSOIDAL,
    track_id_encoding: str = CAT_ENCODING_ONE_HOT,
    token_type_encoding: str = CAT_ENCODING_ONE_HOT,
    epochs: int = 30,
    batch_size: int = 16,
    eval_batch_size: Optional[int] = None,
    encode_micro_batch: int = 0,
    max_encode_tokens: int = 4096,
    lr: float = 1e-4,
    balance_train: bool = True,
    use_weighted_sampler: bool = True,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    max_batches_per_epoch: Optional[int] = None,
    test_max_per_class: int = 10000,
    test_out_dir: str = "mae_joint_bbox_arm_npy_output",
    num_workers: int = 0,
    device: Optional[str] = None,
    save_path: str = "mae_switch_joint_bbox_arm_npy.pt",
    run_test_after_train: bool = True,
    max_error_plots: int = 100,
    use_hist_len: Optional[int] = None,
):
    use_hist_len = int(use_hist_len) if use_hist_len and use_hist_len > 0 else None
    inferred_hist, inferred_tracks = infer_model_dims(
        data_dir, splits=("train", "val", "test"),
        max_hist_len=max_hist_len, max_tracks=max_tracks,
    )
    if use_hist_len is not None:
        inferred_hist = min(inferred_hist, use_hist_len)
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(
        f"Device: {device_t}  data_dir={data_dir}  "
        f"max_hist_len={inferred_hist}(auto)  max_tracks={inferred_tracks}(auto)  "
        f"hidden={hidden}  joint_cls_dim={hidden * 2}  time_pos={time_pos_encoding}  "
        f"track_id={track_id_encoding}  token_type={token_type_encoding}  "
        f"use_hist_len={use_hist_len if use_hist_len is not None else 'full'}  "
        f"[{'arm-aware' if use_arm else 'NO-ARM(ablation)'}]"
    )

    train_max_per_class = None
    if max_train_samples:
        train_max_per_class = max(1, max_train_samples // 2)

    val_max_per_class = None
    if max_val_samples:
        val_max_per_class = max(1, max_val_samples // 2)

    print("构建训练集...")
    train_ds = NpyJointSwitchDataset(
        data_dir, "train",
        max_per_class=train_max_per_class, balance=balance_train,
        use_hist_len=use_hist_len,
    )
    print_sample_distribution(train_ds.refs, "训练集")
    print(f"  训练集扫描 max_hist_len={train_ds.max_hist_len}  max_tracks={train_ds.max_tracks_dim}")
    print("构建验证集...")
    val_ds = NpyJointSwitchDataset(
        data_dir, "val", max_per_class=val_max_per_class,
        use_hist_len=use_hist_len,
    )
    print_sample_distribution(val_ds.refs, "验证集")
    print(f"  验证集扫描 max_hist_len={val_ds.max_hist_len}  max_tracks={val_ds.max_tracks_dim}")
    print(f"valid samples train: {len(train_ds)}  val: {len(val_ds)}")
    if len(train_ds) == 0:
        raise RuntimeError("无有效训练样本，请检查 data_dir/train")

    eval_bs = int(eval_batch_size if eval_batch_size is not None else max(1, batch_size // 4))
    est_tokens = (inferred_tracks + 1) * (inferred_hist + 1)
    auto_mb = JointTrajectoryEncoder.suggest_micro_batch(
        batch_size, inferred_tracks, inferred_hist + 1, max_tokens=max_encode_tokens
    )
    auto_mb_eval = JointTrajectoryEncoder.suggest_micro_batch(
        eval_bs, inferred_tracks, inferred_hist + 1, max_tokens=max_encode_tokens
    )
    print(
        f"  估计 tokens/样本={est_tokens}(含机械臂)  train_batch={batch_size}  eval_batch={eval_bs}  "
        f"encode_micro_batch(auto train={auto_mb}, eval={auto_mb_eval})"
    )
    if est_tokens * batch_size > max_encode_tokens:
        print(
            f"  提示: 序列较长，已启用 micro-batch 编码（每 forward ≤{max_encode_tokens} tokens）；"
            f"若仍 OOM 请减小 --batch_size / --eval_batch_size 或 --max_encode_tokens"
        )

    sampler = make_weighted_sampler(train_ds) if use_weighted_sampler else None
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=collate_mae_batch,
        num_workers=num_workers,
        pin_memory=(device_t.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=eval_bs,
        shuffle=False,
        collate_fn=collate_mae_batch,
        num_workers=num_workers,
        pin_memory=(device_t.type == "cuda"),
    )

    labels = train_ds._labels
    counts = np.bincount(labels, minlength=2).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    class_weights = torch.tensor(
        [1.0 / counts[0], 1.0 / counts[1]], dtype=torch.float32, device=device_t
    )
    class_weights = (class_weights / class_weights.sum() * 2.0).float()

    cfg = JointSwitchConfig(
        max_hist_len=inferred_hist,
        max_tracks=inferred_tracks,
        hidden=hidden,
        n_heads=n_heads,
        n_layers=n_layers,
        hidden_cls=hidden_cls,
        use_arm=use_arm,
        use_arm_cross_attn=use_arm_cross_attn,
        use_temporal_decay=use_temporal_decay,
        decay_rate=decay_rate,
        time_pos_encoding=time_pos_encoding,
        track_id_encoding=track_id_encoding,
        token_type_encoding=token_type_encoding,
        encode_micro_batch=encode_micro_batch,
        max_encode_tokens=max_encode_tokens,
    )
    model = JointSwitchClassifier(cfg).to(device_t)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_f1 = -1.0
    for epoch in range(epochs):
        loss = train_one_epoch(
            model, train_loader, opt, device_t, class_weights, max_batches_per_epoch
        )
        if device_t.type == "cuda":
            torch.cuda.empty_cache()
        metrics = evaluate_classifier(model, val_loader, device_t)
        print(
            f"Epoch {epoch + 1}/{epochs} | loss={loss:.4f} | "
            f"val_acc={metrics['accuracy']:.4f} | val_f1={metrics['f1']:.4f} | "
            f"val_recall={metrics['recall']:.4f}"
        )
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "cfg": {
                        "max_hist_len": inferred_hist,
                        "window": inferred_hist,
                        "max_tracks": inferred_tracks,
                        "hidden": hidden,
                        "n_heads": n_heads,
                        "n_layers": n_layers,
                        "hidden_cls": hidden_cls,
                        "use_arm": use_arm,
                        "use_arm_cross_attn": use_arm_cross_attn,
                        "use_hist_len": use_hist_len,
                        "use_temporal_decay": use_temporal_decay,
                        "decay_rate": decay_rate,
                        "time_pos_encoding": time_pos_encoding,
                        "track_id_encoding": track_id_encoding,
                        "token_type_encoding": token_type_encoding,
                        "encode_micro_batch": encode_micro_batch,
                        "max_encode_tokens": max_encode_tokens,
                        "arm_aware": use_arm,
                    },
                    "epoch": epoch,
                    "val_f1": best_f1,
                },
                save_path,
            )
            print(f"  -> saved {save_path}")

    print(f"Done. Best val F1: {best_f1:.4f}")

    if run_test_after_train:
        print("\n========== 训练完成，开始测试 ==========")
        run_test(
            data_dir=data_dir,
            checkpoint_path=save_path,
            out_dir=test_out_dir,
            max_hist_len=inferred_hist,
            max_tracks=inferred_tracks,
            batch_size=batch_size,
            eval_batch_size=eval_bs,
            encode_micro_batch=encode_micro_batch,
            max_encode_tokens=max_encode_tokens,
            test_max_per_class=test_max_per_class,
            max_error_plots=max_error_plots,
            device=str(device_t),
            num_workers=num_workers,
            use_hist_len=use_hist_len,
        )


# =============================================================================
# 命令行入口
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="从预导出 .npy 加载数据的 joint bbox + 机械臂运动 ID 切换二分类"
    )
    parser.add_argument("--data_dir", type=str, default='/mnt/workspace/hzt/switch_bbox_npy_mixed', help="generate_mixed_switch_npy 输出目录")
    parser.add_argument("--device", type=str, default='cuda:1')

    sub = parser.add_subparsers(dest="command")

    p_train = sub.add_parser("train", help="训练并在结束后自动测试")
    p_train.add_argument(
        "--max_hist_len", type=int, default=None,
        help="可选：手动指定模型最大历史长度上限；默认从 train/val/test 自动扫描",
    )
    p_train.add_argument(
        "--window", type=int, default=None,
        help="同 --max_hist_len（兼容旧参数名）",
    )
    p_train.add_argument("--max_tracks", type=int, default=None, help="可选：手动指定 max_tracks 上限")
    p_train.add_argument("--hidden", type=int, default=256)
    p_train.add_argument("--n_heads", type=int, default=8)
    p_train.add_argument("--n_layers", type=int, default=4)
    p_train.add_argument("--hidden_cls", type=int, default=256)
    p_train.add_argument("--decay_rate", type=float, default=0.08)
    p_train.add_argument(
        "--time_pos_encoding",
        type=str,
        default=TIME_POS_SINUSOIDAL,
        choices=[TIME_POS_SINUSOIDAL, TIME_POS_LEARNABLE],
        help="时间步位置编码：sinusoidal=Transformer 正弦/余弦（默认），learnable=可学习 Embedding",
    )
    p_train.add_argument(
        "--track_id_encoding",
        type=str,
        default=CAT_ENCODING_ONE_HOT,
        choices=[CAT_ENCODING_ONE_HOT, CAT_ENCODING_LEARNABLE],
        help="轨迹 track id 编码：one_hot=固定 one-hot（默认），learnable=可学习 Embedding",
    )
    p_train.add_argument(
        "--token_type_encoding",
        type=str,
        default=CAT_ENCODING_ONE_HOT,
        choices=[CAT_ENCODING_ONE_HOT, CAT_ENCODING_LEARNABLE],
        help="token 类型编码（history/current/arm）：one_hot=固定 one-hot（默认），learnable=可学习 Embedding",
    )
    p_train.add_argument("--no_temporal_decay", action="store_true")
    p_train.add_argument("--no_arm", action="store_true",
                         help="消融机械臂：arm token 不进 Transformer，等价纯 bbox 版本")
    p_train.add_argument("--use_arm_cross_attn", action="store_true",
                         help="开启 query->arm 交叉注意力，arm-context 直送分类头以提高切换 recall")
    p_train.add_argument("--use_hist_len", type=int, default=None,
                         help="运行时只用最近 K 帧历史(尾部截断,当前帧保留);默认用数据集完整 window")
    p_train.add_argument("--epochs", type=int, default=30)
    p_train.add_argument("--batch_size", type=int, default=16)
    p_train.add_argument(
        "--eval_batch_size", type=int, default=None,
        help="验证/测试 batch size，默认 batch_size//4，长序列建议 1~4",
    )
    p_train.add_argument(
        "--encode_micro_batch", type=int, default=0,
        help="Transformer 分块 batch 大小，0=按 max_encode_tokens 自动（推荐）",
    )
    p_train.add_argument(
        "--max_encode_tokens", type=int, default=4096,
        help="单次 Transformer forward 允许的最大 token 数((N+1)×T 上限)",
    )
    p_train.add_argument("--lr", type=float, default=1e-4)
    p_train.add_argument("--max_train_samples", type=int, default=None)
    p_train.add_argument("--max_val_samples", type=int, default=None)
    p_train.add_argument("--max_batches_per_epoch", type=int, default=None)
    p_train.add_argument("--test_max_per_class", type=int, default=10000)
    p_train.add_argument("--test_out_dir", type=str, default="mae_joint_bbox_arm_npy_output")
    p_train.add_argument("--num_workers", type=int, default=0)
    p_train.add_argument("--no_balance_train", action="store_true")
    p_train.add_argument("--no_weighted_sampler", action="store_true")
    p_train.add_argument("--no_test_after_train", action="store_true")
    p_train.add_argument("--save_path", type=str, default="mae_switch_joint_bbox_arm_npy.pt")
    p_train.add_argument("--max_error_plots", type=int, default=100)

    p_test = sub.add_parser("test", help="单独测试")
    p_test.add_argument("--checkpoint", type=str, default="mae_switch_joint_bbox_arm_npy.pt")
    p_test.add_argument("--out_dir", type=str, default="mae_joint_bbox_arm_npy_output")
    p_test.add_argument("--max_hist_len", type=int, default=None)
    p_test.add_argument("--window", type=int, default=None, help="同 --max_hist_len")
    p_test.add_argument("--max_tracks", type=int, default=None)
    p_test.add_argument("--batch_size", type=int, default=8, help="测试 DataLoader batch（建议 1~8）")
    p_test.add_argument("--eval_batch_size", type=int, default=None)
    p_test.add_argument("--encode_micro_batch", type=int, default=0)
    p_test.add_argument("--max_encode_tokens", type=int, default=4096)
    p_test.add_argument("--test_max_per_class", type=int, default=1000)
    p_test.add_argument("--max_error_plots", type=int, default=100)
    p_test.add_argument("--num_workers", type=int, default=0)
    p_test.add_argument("--no_calibrate", action="store_true",
                        help="关闭验证集阈值校准，退回固定 0.5 阈值")
    p_test.add_argument("--target_precision", type=float, default=0.7,
                        help="校准推荐标准：precision>=该值中 recall 最大")
    p_test.add_argument("--target_recall", type=float, default=0.85,
                        help="recall_target 标准的 recall 下限")
    p_test.add_argument("--no_prior_correction", action="store_true",
                        help="关闭 logit 先验修正（仅做阈值扫描）")
    p_test.add_argument("--train_pos_ratio", type=float, default=0.5,
                        help="训练集正例率(默认 1:1 -> 0.5)，用于 logit 先验修正")
    p_test.add_argument("--target_pos_ratio", type=float, default=None,
                        help="目标先验正例率；默认对齐当前测试集(约1:100)。线上1:8300可传 0.00012")
    p_test.add_argument("--use_hist_len", type=int, default=None,
                        help="运行时只用最近 K 帧历史;默认用 checkpoint 记录的值(与训练一致)")

    args = parser.parse_args()
    if args.command is None:
        parser.error("请指定子命令: train 或 test")

    if args.command == "train":
        hist_cap = args.max_hist_len if args.max_hist_len is not None else args.window
        train(
            data_dir=args.data_dir,
            max_hist_len=hist_cap,
            max_tracks=args.max_tracks,
            hidden=args.hidden,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            hidden_cls=args.hidden_cls,
            use_arm=not args.no_arm,
            use_arm_cross_attn=args.use_arm_cross_attn,
            use_hist_len=args.use_hist_len,
            use_temporal_decay=not args.no_temporal_decay,
            decay_rate=args.decay_rate,
            time_pos_encoding=args.time_pos_encoding,
            track_id_encoding=args.track_id_encoding,
            token_type_encoding=args.token_type_encoding,
            epochs=args.epochs,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            encode_micro_batch=args.encode_micro_batch,
            max_encode_tokens=args.max_encode_tokens,
            lr=args.lr,
            balance_train=not args.no_balance_train,
            use_weighted_sampler=not args.no_weighted_sampler,
            max_train_samples=args.max_train_samples,
            max_val_samples=args.max_val_samples,
            max_batches_per_epoch=args.max_batches_per_epoch,
            test_max_per_class=args.test_max_per_class,
            test_out_dir=args.test_out_dir,
            num_workers=args.num_workers,
            device=args.device,
            save_path=args.save_path,
            run_test_after_train=not args.no_test_after_train,
            max_error_plots=args.max_error_plots,
        )
    elif args.command == "test":
        hist_cap = args.max_hist_len if args.max_hist_len is not None else args.window
        run_test(
            data_dir=args.data_dir,
            checkpoint_path=args.checkpoint,
            out_dir=args.out_dir,
            max_hist_len=hist_cap,
            max_tracks=args.max_tracks,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            encode_micro_batch=args.encode_micro_batch,
            max_encode_tokens=args.max_encode_tokens,
            test_max_per_class=args.test_max_per_class,
            max_error_plots=args.max_error_plots,
            device=args.device,
            num_workers=args.num_workers,
            calibrate=not args.no_calibrate,
            target_precision=args.target_precision,
            target_recall=args.target_recall,
            prior_correction=not args.no_prior_correction,
            train_pos_ratio=args.train_pos_ratio,
            target_pos_ratio=args.target_pos_ratio,
            use_hist_len=args.use_hist_len,
        )


if __name__ == "__main__":
    main()
