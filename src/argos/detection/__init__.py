from .base import FaceDetector
from .haar import HaarFaceDetector
from .yunet import YuNetFaceDetector

__all__ = ["FaceDetector", "YuNetFaceDetector", "HaarFaceDetector"]
