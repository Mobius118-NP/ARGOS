from dataclasses import dataclass
import cv2


@dataclass(slots=True)
class StereoFrame:
    left: object
    right: object | None = None


class SingleCamera:
    """Capture from one webcam using the same frame shape as StereoCamera."""
    def __init__(self, camera_index=0, width=640, height=480, fps=30):
        self.camera_index = camera_index
        self.settings = (width, height, fps)
        self._capture = None

    def open(self):
        cap = cv2.VideoCapture(self.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.settings[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.settings[1])
        cap.set(cv2.CAP_PROP_FPS, self.settings[2])
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Could not open camera index {self.camera_index}")
        self._capture = cap
        return self

    def read(self):
        if self._capture is None:
            raise RuntimeError("SingleCamera must be opened before reading")
        ok, frame = self._capture.read()
        return StereoFrame(frame, None) if ok else None

    def close(self):
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *_):
        self.close()


class StereoCamera:
    """Two independently indexed webcams; reads are sequential, not hardware-synchronized."""
    def __init__(self, left_index=0, right_index=1, width=640, height=480, fps=30):
        self.indices = (left_index, right_index)
        self.settings = (width, height, fps)
        self._captures = []

    def open(self):
        for index in self.indices:
            cap = cv2.VideoCapture(index)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.settings[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.settings[1])
            cap.set(cv2.CAP_PROP_FPS, self.settings[2])
            if not cap.isOpened():
                cap.release()
                self.close()
                raise RuntimeError(f"Could not open camera index {index}")
            self._captures.append(cap)
        return self

    def read(self):
        if len(self._captures) != 2:
            raise RuntimeError("StereoCamera must be opened before reading")
        ok_l, left = self._captures[0].read()
        ok_r, right = self._captures[1].read()
        if not ok_l or not ok_r:
            return None
        return StereoFrame(left, right)

    def close(self):
        for cap in self._captures:
            cap.release()
        self._captures.clear()

    def __enter__(self):
        return self.open()

    def __exit__(self, *_):
        self.close()
