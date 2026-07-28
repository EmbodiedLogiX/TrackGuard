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
labeling/               Ground-truth annotation review tool
scripts/                Training and evaluation scripts
configs/                Configuration files
checkpoints/            Pretrained models
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

# Data Annotation

Ground-truth identities are reviewed and corrected with the built-in annotation
tool. It loads a sequence, renders each frame with its bounding boxes, and lets
you fix identity switches interactively: move or resize boxes, edit a track ID,
delete or rename an ID across the whole sequence, propagate a box forward to a
target frame, and undo any change. Edits are written back in place in MOT format
(`frame,id,x,y,w,h,1,-1,-1,-1`), with optional auto-save on every action.

The tool expects a sequence directory laid out as:

```text
<sequence>/
├── gt/gt.txt           MOT-format annotations (read and written in place)
└── img1/               Frame images (000001.jpg, 000002.jpg, ...)
```

Install the annotation extra (a Tk-based GUI) and launch it:

```bash
pip install -e ".[labeling]"
python scripts/label_review.py --dir <sequence>
```

If `--dir` is omitted a folder picker is shown. Add `--no_auto_save` to review
without writing until you press Save. Keyboard shortcuts: `A`/`←` and `D`/`→`
change frame, `space` toggles boxes, `Q` undoes, `E` starts quick-create, and
`W`/`Delete` removes the selected box.

The annotation logic is separated from the interface: the data model, geometry,
history, and session live in the top-level `labeling/` package as importable,
testable modules, while only the GUI layer depends on the Tk toolkit.

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

# Repository Status

This repository currently provides the core implementation of TrackGuard, together with representative training and evaluation code.

More detailed implementations, better-documented code, additional pretrained models, and non-private benchmark data will be released incrementally. 

---

# License

MIT License.
