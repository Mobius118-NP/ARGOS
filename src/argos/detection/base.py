from abc import ABC, abstractmethod
from argos.geometry import PixelBox


class FaceDetector(ABC):
    @abstractmethod
    def detect(self, frame) -> list[PixelBox]:
        """Detect faces in a BGR frame."""
