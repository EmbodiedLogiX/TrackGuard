from __future__ import annotations

TRACK_PALETTE = [
    "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF",
    "#FFA500", "#800080", "#008000", "#000080", "#808000", "#800000",
]


def color_for(track_id: int) -> str:
    return TRACK_PALETTE[track_id % len(TRACK_PALETTE)]
