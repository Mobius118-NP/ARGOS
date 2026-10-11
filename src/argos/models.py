"""Shared target-state data model; identity and range remain optional providers."""
from dataclasses import dataclass
from typing import Optional
from argos.geometry import PixelBox


@dataclass(frozen=True, slots=True)
class RangeEstimate:
    meters: float
    confidence: Optional[float] = None
    method: str = "unknown"


@dataclass(frozen=True, slots=True)
class TargetState:
    track_id: int
    pixel_x: float
    pixel_y: float
    normalized_x: float
    normalized_y: float
    box: PixelBox
    identity_label: str = "Unknown"
    identity_confidence: Optional[float] = None
    range_estimate: Optional[RangeEstimate] = None
    track_confidence: float = 1.0
    identity_id: str | None = None
    identity_category: str = "Unknown"
    identity_status: str = ""
    identity_suggestion: str | None = None
    identity_ambiguous: bool = False
    raw_box: PixelBox | None = None

    @classmethod
    def from_box(cls, track_id: int, box: PixelBox, image_width: int, image_height: int):
        if image_width <= 0 or image_height <= 0:
            raise ValueError("image dimensions must be positive")
        center = box.center
        return cls(track_id, center.x, center.y, center.x / image_width,
                   center.y / image_height, box)
