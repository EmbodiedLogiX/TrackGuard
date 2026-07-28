from __future__ import annotations

from typing import Optional, Tuple

RESIZE_TL = "resize_tl"
RESIZE_TR = "resize_tr"
RESIZE_BL = "resize_bl"
RESIZE_BR = "resize_br"
RESIZE_L = "resize_l"
RESIZE_R = "resize_r"
RESIZE_T = "resize_t"
RESIZE_B = "resize_b"


class ViewTransform:
    def __init__(self, canvas_width: int, canvas_height: int):
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0

    def fit(self, image_width: int, image_height: int) -> None:
        self.scale = min(self.canvas_width / image_width,
                         self.canvas_height / image_height)
        new_w = int(image_width * self.scale)
        new_h = int(image_height * self.scale)
        self.offset_x = (self.canvas_width - new_w) // 2
        self.offset_y = (self.canvas_height - new_h) // 2

    def to_canvas(self, box: dict) -> Tuple[int, int, int, int]:
        x = self.offset_x + int(box["x"] * self.scale)
        y = self.offset_y + int(box["y"] * self.scale)
        w = int(box["width"] * self.scale)
        h = int(box["height"] * self.scale)
        return x, y, w, h

    def to_image(self, cx: float, cy: float) -> Tuple[float, float]:
        return ((cx - self.offset_x) / self.scale,
                (cy - self.offset_y) / self.scale)

    def delta_to_image(self, dx: float, dy: float) -> Tuple[float, float]:
        return dx / self.scale, dy / self.scale


def hit_test(boxes, view: ViewTransform, px: float, py: float,
             tolerance: int = 5, min_side: int = 5) -> Optional[int]:
    hits = []
    for index, box in enumerate(boxes):
        x, y, w, h = view.to_canvas(box)
        if w < min_side or h < min_side:
            continue
        if (x - tolerance <= px <= x + w + tolerance and
                y - tolerance <= py <= y + h + tolerance):
            hits.append((w * h, index))
    if not hits:
        return None
    hits.sort()
    return hits[0][1]


def resize_mode_at(box_canvas: Tuple[int, int, int, int], px: float, py: float,
                   edge: int = 8) -> Optional[str]:
    x, y, w, h = box_canvas
    right = x + w
    bottom = y + h
    on_left = x <= px <= x + edge
    on_right = right - edge <= px <= right
    on_top = y <= py <= y + edge
    on_bottom = bottom - edge <= py <= bottom
    if on_right and on_bottom:
        return RESIZE_BR
    if on_right and on_top:
        return RESIZE_TR
    if on_left and on_bottom:
        return RESIZE_BL
    if on_left and on_top:
        return RESIZE_TL
    mid_v = y + edge <= py <= bottom - edge
    mid_h = x + edge <= px <= right - edge
    if on_right and mid_v:
        return RESIZE_R
    if on_left and mid_v:
        return RESIZE_L
    if on_bottom and mid_h:
        return RESIZE_B
    if on_top and mid_h:
        return RESIZE_T
    return None


def clamp_box(box: dict, image_width: int, image_height: int) -> dict:
    box["x"] = max(0.0, min(box["x"], image_width - box["width"]))
    box["y"] = max(0.0, min(box["y"], image_height - box["height"]))
    return box


def move_box(box: dict, dx_img: float, dy_img: float,
             image_width: int, image_height: int) -> dict:
    box["x"] += dx_img
    box["y"] += dy_img
    return clamp_box(box, image_width, image_height)


def resize_box(box: dict, mode: str, dx_img: float, dy_img: float,
               image_width: int, image_height: int, min_side: float = 5.0) -> dict:
    if mode in (RESIZE_BR, RESIZE_TR, RESIZE_R):
        box["width"] = max(min_side, min(box["width"] + dx_img, image_width - box["x"]))
    if mode in (RESIZE_BL, RESIZE_TL, RESIZE_L):
        new_width = max(min_side, min(box["width"] - dx_img, image_width - box["x"]))
        box["x"] += box["width"] - new_width
        box["width"] = new_width
    if mode in (RESIZE_BR, RESIZE_BL, RESIZE_B):
        box["height"] = max(min_side, min(box["height"] + dy_img, image_height - box["y"]))
    if mode in (RESIZE_TR, RESIZE_TL, RESIZE_T):
        new_height = max(min_side, min(box["height"] - dy_img, image_height - box["y"]))
        box["y"] += box["height"] - new_height
        box["height"] = new_height
    return clamp_box(box, image_width, image_height)


def new_box_from_drag(view: ViewTransform, start, end, track_id: int,
                      min_side: float = 5.0) -> Optional[dict]:
    x1, y1 = start
    x2, y2 = end
    left, right = min(x1, x2), max(x1, x2)
    top, bottom = min(y1, y2), max(y1, y2)
    ix, iy = view.to_image(left, top)
    iw = (right - left) / view.scale
    ih = (bottom - top) / view.scale
    if iw < min_side or ih < min_side:
        return None
    return {"track_id": track_id, "x": ix, "y": iy, "width": iw, "height": ih}
