import cv2
from argos.geometry import PixelBox
from .base import FaceDetector


class HaarFaceDetector(FaceDetector):
    def __init__(self, scale_factor: float = 1.1, min_neighbors: int = 5):
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(path)
        if self._cascade.empty():
            raise RuntimeError(f"Could not load Haar cascade: {path}")
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors

    def detect(self, frame) -> list[PixelBox]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found = self._cascade.detectMultiScale(gray, self.scale_factor, self.min_neighbors)
        return [PixelBox(int(x), int(y), int(w), int(h)) for x, y, w, h in found]
