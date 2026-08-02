# Think When Lost: Adaptive Large–Small Model Collaboration for Reliable Multi-Object Tracking in Logistics
<div align="center">
<a href="https://scholar.google.com/citations?user=mVSEBdMAAAAJ&hl=en&oi=ao">Zhaotie Hao</a><sup>1</sup>,&nbsp; <a href="https://scholar.google.com/citations?user=kXbWREkAAAAJ&hl=en">Jiawei Ma</a><sup>1</sup>,&nbsp; <a href="https://scholar.google.com/citations?hl=en&user=Yg1RBVwAAAAJ">Zhiyuan Zhou</a><sup>2</sup>,&nbsp; <a href="https://scholar.google.com/citations?user=7XOy-jkAAAAJ&hl=en">Jiangyi Fang</a><sup>3</sup>,&nbsp; <a href="http://www.zhiqinghong.one/">Zhiqing Hong</a><sup>4</sup>,&nbsp; <a herf="https://scholars.cityu.edu.hk/en/persons/xinli375/">Xin Li</a><sup>1</sup>&nbsp; <a href="https://sbuhaotian.github.io/SBUhaotian/">Haotian Wang</a><sup>5</sup>&nbsp; <a href="https://people.cs.rutgers.edu/~dz220/">Desheng Zhang</a><sup>4</sup>&nbsp; <a href="https://scholar.google.com/citations?hl=en&user=hc1m_BQAAAAJ">Tian He</a><sup>5*</sup>&nbsp;  <a href="https://scholar.google.com/citations?hl=en&user=T89V0RAAAAAJ">Zhimeng Yin</a><sup>1*</sup>&nbsp;
<br>
<sup>1</sup> City University of Hong Kong &nbsp;&nbsp;&nbsp; <sup>2</sup> Rutgers University<br> <sup>3</sup> Peking University &nbsp;&nbsp;&nbsp <sup>4</sup> Hong Kong University of Science and Technology (Guangzhou) <sup>5</sup> JD Logistics
<br>
<sup>*</sup>Corresponding Author
<br>
<!-- <a href='https://kdd2027.kdd.org/'><img src='https://img.shields.io/badge/KDD-2026-78CA2E.svg'></a> &nbsp; -->
<!-- <a href='https://openreview.net/forum?id=H6rDX4w6Al'><img src='https://img.shields.io/badge/Paper-Openreview-8D1B12.svg'></a> &nbsp; -->
<a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License"></a> &nbsp;
<a href='https://github.com/EmbodiedLogiX/TrackGuard/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;
<!-- <a href="https://arxiv.org/abs/2602.08024"><img src="https://img.shields.io/badge/arXiv-2602.08024-b31b1b.svg"></a> &nbsp; -->
<a href="https://huggingface.co/"><img src="https://img.shields.io/badge/transformers-4.57-FFD21E.svg" alt="transformers"></a> &nbsp;
<a href="https://python.org/"><img src="https://img.shields.io/badge/Python-3.10%2B-3776ab.svg" alt="Python"></a> &nbsp;
<!-- <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.5%2B-DF3411.svg" alt="PyTorch"></a> &nbsp;
<!-- <a href="#"><img src="https://img.shields.io/badge/#.svg"></a> &nbsp; -->
</div>

TrackGuard is an adaptive large–small collaborative framework for long-term multi-object tracking in logistics sorting. Instead of invoking a Vision-Language Model (VLM) on every frame, TrackGuard learns **when** large-model reasoning is necessary and **how** to recover object identities using compact historical memory.

This repository contains the core implementation of TrackGuard, including the trajectory-aware router, the memory-guided VLM recovery module, and the benchmark datasets used in our paper.

---

## 🔥News

<!-- - [2026.05.01] 🔍Fix a potential OOM bug in manual [CLS] attention extraction in Qwen2.5-VL and Qwen3-VL.
- [2026.02.10] 🚀Release our paper on arXiv.
- [2026.02.06] 🍾Our paper has been selected as an **Oral Presentation** at **ICLR 2026**.
- [2026.02.01] ✨Release FlashVID code and inference demos on *Qwen2.5-VL* and *Qwen3-VL*.
- [2026.01.31] 🚀Release this repository to the public.
- [2026.01.30] ✨Release FlashVID code and inference demos on *LLaVA-OneVision* and *LLaVA-Video*.
- [2026.01.30] 👏Initialize this GitHub repository.
- [2026.01.26] 🎉Our training-free inference acceleration method [FlashVID](https://openreview.net/forum?id=H6rDX4w6Al) has been accepted at **ICLR 2026**. -->
- [2026.07.26] 🌟Release the GitHub repository of [TrackGuard](https://github.com/EmbodiedLogiX/TrackGuard).
- [2026.07.20] 👏Initialize this GitHub repository.

## 📋Todo List

- [ ] Optimize code efficiency and improve deployment performance.
- [ ] Release the remaining TrackGuard code.
- [ ] Release demos showcasing large–small model collaboration for reliable multi-object tracking.
- [ ] Release VLM fine-tuning code based on Qwen2.5-VL and Qwen3-VL.
- [x] Release a subset of non-private datasets.
- [x] Release multi-object tracking annotation tools.
- [x] Release the Adaptive Routing module.
- [x] Initialize this GitHub repository.
<!-- - [] Release our paper on arXiv. -->

# 📦 Repository Structure

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

# 🌈 Method Overview

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

# 🚀 Quick Start

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

# 📜 Data Annotation

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

# 🔖 Dataset

The repository includes the benchmark datasets used in our paper.

Each sample contains

- video frames,
- object bounding boxes,
- object identities,
- router annotations, and
- VLM recovery annotations.

Please refer to `data/README.md` for the dataset format and preprocessing instructions.

# 📝 Repository Status

This repository currently provides the core implementation of TrackGuard, together with representative training and evaluation code.

More detailed implementations, better-documented code, additional pretrained models, and non-private benchmark data will be released incrementally. 

---

## 👏Acknowledgement

This project is built upon recent open-source works: [ByteTrack](https://github.com/FoundationVision/ByteTrack), [Qwen2.5-VL/Qwen3-VL](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct), [Gemma](https://huggingface.co/Andycurrent/Gemma-3-4B-VL-it-Gemini-Pro-Heretic-Uncensored-Thinking_GGUF), [LlamaFactory](https://github.com/hiyouga/LlamaFactory), [Transformers](https://github.com/huggingface/transformers). Thanks for their excellent work!

# License

MIT License.
