from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

VALID_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


@dataclass
class Box:
    track_id: int
    x: float
    y: float
    width: float
    height: float

    def as_dict(self) -> dict:
        return {"track_id": self.track_id, "x": self.x, "y": self.y,
                "width": self.width, "height": self.height}


def parse_gt_line(line: str) -> Optional[dict]:
    parts = line.strip().split(",")
    if len(parts) < 6:
        return None
    return {
        "frame": int(parts[0]),
        "track_id": int(parts[1]),
        "x": float(parts[2]),
        "y": float(parts[3]),
        "width": float(parts[4]),
        "height": float(parts[5]),
    }


def format_gt_line(frame: int, box: dict) -> str:
    return (f"{frame},{box['track_id']},{box['x']:.2f},{box['y']:.2f},"
            f"{box['width']:.2f},{box['height']:.2f},1,-1,-1,-1\n")


class AnnotationStore:
    def __init__(self):
        self.frames: Dict[int, List[dict]] = {}

    @classmethod
    def from_text(cls, text: str) -> "AnnotationStore":
        store = cls()
        for line in text.splitlines():
            record = parse_gt_line(line)
            if record is None:
                continue
            frame = record.pop("frame")
            store.frames.setdefault(frame, []).append(record)
        return store

    @classmethod
    def from_file(cls, path: str) -> "AnnotationStore":
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_text(handle.read())

    def to_text(self) -> str:
        lines: List[str] = []
        for frame in sorted(self.frames):
            for box in self.frames[frame]:
                lines.append(format_gt_line(frame, box))
        return "".join(lines)

    def to_file(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(self.to_text())

    def boxes_at(self, frame: int) -> List[dict]:
        return self.frames.get(frame, [])

    def all_ids(self) -> set:
        ids = set()
        for boxes in self.frames.values():
            for box in boxes:
                ids.add(box["track_id"])
        return ids

    def next_id(self, default: int = 100) -> int:
        ids = self.all_ids()
        return max(ids) + 1 if ids else default

    def add(self, frame: int, box: dict) -> None:
        self.frames.setdefault(frame, []).append(box)

    def remove(self, frame: int, index: int) -> Optional[dict]:
        boxes = self.frames.get(frame)
        if not boxes or index >= len(boxes):
            return None
        removed = boxes.pop(index)
        if not boxes:
            del self.frames[frame]
        return removed

    def delete_by_id(self, track_id: int) -> int:
        deleted = 0
        empty = []
        for frame, boxes in self.frames.items():
            kept = [b for b in boxes if b["track_id"] != track_id]
            deleted += len(boxes) - len(kept)
            if kept:
                self.frames[frame] = kept
            else:
                empty.append(frame)
        for frame in empty:
            del self.frames[frame]
        return deleted

    def rename_id(self, source_id: int, target_id: int) -> int:
        if source_id == target_id:
            return 0
        modified = 0
        for boxes in self.frames.values():
            for box in boxes:
                if box["track_id"] == source_id:
                    box["track_id"] = target_id
                    modified += 1
        return modified

    def propagate(self, track_id: int, start_frame: int, final_frame: int) -> int:
        template = next((b for b in self.boxes_at(start_frame)
                         if b["track_id"] == track_id), None)
        if template is None:
            return 0
        added = 0
        for frame in range(start_frame + 1, final_frame + 1):
            boxes = self.frames.setdefault(frame, [])
            if any(b["track_id"] == track_id for b in boxes):
                continue
            boxes.append({"track_id": track_id, "x": float(template["x"]),
                          "y": float(template["y"]), "width": float(template["width"]),
                          "height": float(template["height"])})
            added += 1
        return added

    def snapshot(self) -> Dict[int, List[dict]]:
        return copy.deepcopy(self.frames)

    def restore(self, snapshot: Dict[int, List[dict]]) -> None:
        self.frames = copy.deepcopy(snapshot)


def list_images(img_dir: str) -> List[str]:
    if not os.path.isdir(img_dir):
        return []
    return sorted(f for f in os.listdir(img_dir)
                  if f.lower().endswith(VALID_IMAGE_EXTS))
