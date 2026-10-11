"""Greedy track association with EMA-smoothed fused image-space state."""
from math import hypot
from dataclasses import replace
from argos.geometry import PixelBox
from argos.models import TargetState


class CentroidTracker:
    def __init__(self, max_distance_px: float = 90, max_missed_frames: int = 12, ema_alpha: float = 0.3):
        if not 0 < ema_alpha <= 1:
            raise ValueError("ema_alpha must be in (0, 1]")
        self.max_distance_px = max_distance_px
        self.max_missed_frames = max_missed_frames
        self.ema_alpha = ema_alpha
        self._targets: dict[int, TargetState] = {}
        self._missed: dict[int, int] = {}
        self._next_id = 1

    def update(self, boxes: list[PixelBox], frame_size: tuple[int, int]) -> list[TargetState]:
        width, height = frame_size
        candidates = []
        for track_id, target in self._targets.items():
            for index, box in enumerate(boxes):
                c = box.center
                d = hypot(target.pixel_x - c.x, target.pixel_y - c.y)
                if d <= self.max_distance_px:
                    candidates.append((d, track_id, index))
        assigned_tracks: set[int] = set()
        assigned_boxes: set[int] = set()
        updated: dict[int, TargetState] = {}
        for _, track_id, index in sorted(candidates):
            if track_id in assigned_tracks or index in assigned_boxes:
                continue
            raw = boxes[index]
            old = self._targets[track_id].box
            a = self.ema_alpha
            smooth = PixelBox(*(round((1-a)*before + a*after) for before, after in zip(
                (old.x, old.y, old.width, old.height), (raw.x, raw.y, raw.width, raw.height))))
            state = TargetState.from_box(track_id, smooth, width, height)
            updated[track_id] = replace(state, raw_box=raw)
            assigned_tracks.add(track_id)
            assigned_boxes.add(index)
            self._missed[track_id] = 0
        for index, box in enumerate(boxes):
            if index not in assigned_boxes:
                track_id = self._next_id
                self._next_id += 1
                state = TargetState.from_box(track_id, box, width, height)
                updated[track_id] = replace(state, raw_box=box)
                self._missed[track_id] = 0
        for track_id in self._targets:
            if track_id not in assigned_tracks:
                self._missed[track_id] = self._missed.get(track_id, 0) + 1
                if self._missed[track_id] <= self.max_missed_frames:
                    updated[track_id] = self._targets[track_id]
        self._targets = updated
        self._missed = {key: self._missed.get(key, 0) for key in updated}
        return list(updated.values())
