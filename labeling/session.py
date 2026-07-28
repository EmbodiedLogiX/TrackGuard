from __future__ import annotations

import os
from typing import List, Optional

from labeling.annotations import AnnotationStore, list_images
from labeling.history import UndoHistory


def resolve_sequence(directory: str) -> dict:
    gt_path = os.path.join(directory, "gt", "gt.txt")
    img_dir = os.path.join(directory, "img1")
    if not os.path.isfile(gt_path):
        raise FileNotFoundError(gt_path)
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(img_dir)
    return {"gt_path": gt_path, "img_dir": img_dir,
            "name": os.path.basename(os.path.normpath(directory))}


class LabelingSession:
    def __init__(self, directory: str, auto_save: bool = True):
        paths = resolve_sequence(directory)
        self.name = paths["name"]
        self.gt_path = paths["gt_path"]
        self.img_dir = paths["img_dir"]
        self.auto_save = auto_save
        self.store = AnnotationStore.from_file(self.gt_path)
        self.images: List[str] = list_images(self.img_dir)
        self.frame_index = 0
        self.history = UndoHistory()

    @property
    def frame_number(self) -> int:
        return self.frame_index + 1

    @property
    def total_frames(self) -> int:
        return len(self.images)

    def image_path(self, index: Optional[int] = None) -> str:
        idx = self.frame_index if index is None else index
        return os.path.join(self.img_dir, self.images[idx])

    def current_boxes(self) -> List[dict]:
        return self.store.boxes_at(self.frame_number)

    def go_to(self, index: int) -> None:
        if 0 <= index < self.total_frames:
            self.frame_index = index

    def next_frame(self) -> None:
        self.go_to(self.frame_index + 1)

    def prev_frame(self) -> None:
        self.go_to(self.frame_index - 1)

    def _checkpoint(self) -> None:
        self.history.push(self.store.snapshot())

    def _persist(self) -> None:
        if self.auto_save:
            self.store.to_file(self.gt_path)

    def add_box(self, box: dict) -> None:
        self._checkpoint()
        self.store.add(self.frame_number, box)
        self._persist()

    def delete_box(self, index: int) -> Optional[dict]:
        self._checkpoint()
        removed = self.store.remove(self.frame_number, index)
        self._persist()
        return removed

    def update_id(self, index: int, track_id: int) -> None:
        boxes = self.current_boxes()
        if index >= len(boxes):
            return
        self._checkpoint()
        boxes[index]["track_id"] = track_id
        self._persist()

    def delete_id(self, track_id: int) -> int:
        self._checkpoint()
        count = self.store.delete_by_id(track_id)
        self._persist()
        return count

    def rename_id(self, source_id: int, target_id: int) -> int:
        self._checkpoint()
        count = self.store.rename_id(source_id, target_id)
        self._persist()
        return count

    def propagate(self, track_id: int, final_frame: int) -> int:
        final_frame = max(self.frame_number, min(final_frame, self.total_frames))
        self._checkpoint()
        count = self.store.propagate(track_id, self.frame_number, final_frame)
        self._persist()
        return count

    def undo(self) -> bool:
        snapshot = self.history.pop()
        if snapshot is None:
            return False
        self.store.restore(snapshot)
        self._persist()
        return True

    def next_id(self) -> int:
        return self.store.next_id()

    def save(self) -> None:
        self.store.to_file(self.gt_path)

    def reload(self) -> None:
        self.store = AnnotationStore.from_file(self.gt_path)
        self.history.clear()
