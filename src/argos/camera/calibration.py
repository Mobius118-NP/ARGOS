"""Stereo calibration and rectification scaffolding using OpenCV calibration files."""
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np


@dataclass
class StereoCalibration:
    left_map: tuple
    right_map: tuple
    baseline_m: float

    @classmethod
    def load(cls, path: str | Path, image_size: tuple[int, int], baseline_m: float):
        fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            raise FileNotFoundError(f"Stereo calibration file not found: {path}")
        try:
            names = ("M1", "D1", "M2", "D2", "R", "T")
            values = [fs.getNode(name).mat() for name in names]
        finally:
            fs.release()
        if any(value is None for value in values):
            raise ValueError(f"{path} must contain matrices: {', '.join(names)}")
        m1, d1, m2, d2, rotation, translation = values
        r1, r2, p1, p2, _, _, _ = cv2.stereoRectify(m1, d1, m2, d2, image_size, rotation, translation)
        left = cv2.initUndistortRectifyMap(m1, d1, r1, p1, image_size, cv2.CV_32FC1)
        right = cv2.initUndistortRectifyMap(m2, d2, r2, p2, image_size, cv2.CV_32FC1)
        return cls(left, right, baseline_m)

    def rectify(self, left, right):
        return (cv2.remap(left, *self.left_map, cv2.INTER_LINEAR),
                cv2.remap(right, *self.right_map, cv2.INTER_LINEAR))
