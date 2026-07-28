# TrackGuard

**Think When Lost: Adaptive Large–Small Model Collaboration for Reliable Multi-Object Tracking in Logistics**

TrackGuard is an adaptive large–small collaborative framework for long-term multi-object tracking in logistics sorting. Instead of invoking a Vision-Language Model (VLM) on every frame, TrackGuard learns **when** large-model reasoning is necessary and **how** to recover object identities using compact historical memory.

This repository contains the core implementation of TrackGuard, including the trajectory-aware router, the memory-guided VLM recovery module, and the benchmark datasets used in our paper.

---

# Repository Structure

```text
trackguard/
├── router/             Trajectory-aware routing model
├── recovery/           Memory-guided VLM recovery
├── data/               Dataset and data processing
├── scripts/            Training and evaluation scripts
├── configs/            Configuration files
└── checkpoints/        Pretrained models
```

---

# Method Overview

TrackGuard consists of two complementary components.

## Trajectory-aware Router

The router predicts **when the lightweight tracker is likely to lose the correct identity**.

Instead of relying on instantaneous confidence or heuristic rules, it jointly models the recent trajectories of the target object, neighboring objects, and the sorting arm over a temporal window. A lightweight Transformer aggregates the temporal interactions and estimates the probability of an upcoming identity switch. Only uncertain cases are forwarded to the VLM for further reasoning.

---

## Memory-guided VLM Recovery

When the router predicts a potential identity switch, TrackGuard retrieves a compact **event-centric spatial–visual memory** containing historical observations of the tracked object.

Rather than retrieving an identity from scratch, the VLM verifies the lightweight tracker's prediction using both the historical memory and the current observation. It can:

- **Accept** the current identity;
- **Correct** it to a historical candidate; or
- **Abstain** when the available evidence is insufficient.

This verification mechanism significantly improves robustness under severe occlusions, visually similar parcels, and package rotations.

---

# Quick Start

Install the package

```bash
pip install -e .
```

Train the trajectory-aware router

```bash
python scripts/train_router.py \
    --config configs/router.yaml
```

Evaluate the VLM recovery module

```bash
python scripts/eval_recovery.py \
    --config configs/recovery.yaml
```

---

# Dataset

The repository includes the benchmark datasets used in our paper.

Each sample contains

- video frames,
- object bounding boxes,
- object identities,
- router annotations, and
- VLM recovery annotations.

Please refer to `data/README.md` for the dataset format and preprocessing instructions.

---

# License

MIT License.