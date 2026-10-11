"""Coordinate helpers. Image origin is top-left; x grows right and y grows down."""
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ImagePoint:
    x: float
    y: float


@dataclass(frozen=True, slots=True)
class PixelBox:
    x: int
    y: int
    width: int
    height: int

    @property
    def center(self) -> ImagePoint:
        return ImagePoint(self.x + self.width / 2, self.y + self.height / 2)


def pixel_to_normalized(point: ImagePoint, width: int, height: int) -> ImagePoint:
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    return ImagePoint(point.x / width, point.y / height)


def normalized_to_pixel(point: ImagePoint, width: int, height: int) -> ImagePoint:
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    return ImagePoint(point.x * width, point.y * height)
