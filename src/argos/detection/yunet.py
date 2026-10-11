"""OpenCV YuNet detector. FaceDetectorYN applies score filtering and NMS internally."""
from pathlib import Path
from urllib.request import urlopen
import cv2
from argos.geometry import PixelBox
from .base import FaceDetector

YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/refs/heads/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"


def ensure_yunet_model(path: str | Path) -> Path:
    path = Path(path)
    if path.exists() and path.stat().st_size > 10_000:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    try:
        with urlopen(YUNET_URL, timeout=30) as response, temporary.open("wb") as out:
            while chunk := response.read(1024 * 1024):
                out.write(chunk)
        if temporary.stat().st_size < 10_000:
            raise RuntimeError("Downloaded YuNet file is unexpectedly small")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


class YuNetFaceDetector(FaceDetector):
    def __init__(self, model_path="models/face_detection_yunet_2023mar.onnx",
                 score_threshold=0.8, nms_threshold=0.3, top_k=5000, input_size=320):
        self.model_path = ensure_yunet_model(model_path)
        self.score_threshold = float(score_threshold)
        self.nms_threshold = float(nms_threshold)
        self.top_k = int(top_k)
        self.input_size = input_size
        self._detector = cv2.FaceDetectorYN.create(str(self.model_path), "", (320, 320),
                                                  self.score_threshold, self.nms_threshold, self.top_k)

    def detect_faces(self, frame):
        h, w = frame.shape[:2]
        if self.input_size in (None, "frame"):
            source, sx, sy = frame, 1.0, 1.0
        else:
            size = ((int(self.input_size), int(self.input_size)) if isinstance(self.input_size, (int, float))
                    else tuple(map(int, self.input_size)))
            source = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            sx, sy = w / size[0], h / size[1]
        self._detector.setInputSize((source.shape[1], source.shape[0]))
        _, faces = self._detector.detect(source)
        result = []
        for row in faces if faces is not None else ():
            x, y, bw, bh = row[:4]
            points = tuple((float(row[i]) * sx, float(row[i + 1]) * sy) for i in range(4, 14, 2))
            result.append((PixelBox(round(x * sx), round(y * sy), round(bw * sx), round(bh * sy)), float(row[14]), points))
        return result

    def detect(self, frame):
        return [box for box, _, _ in self.detect_faces(frame)]
