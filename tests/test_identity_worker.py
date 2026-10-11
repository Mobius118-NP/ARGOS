import time
import numpy as np
from argos.identity.worker import EmbeddingWorker


class StubRecognizer:
    def embedding(self, frame, landmarks):
        return np.array([frame[0, 0, 0], len(landmarks)], dtype=float)


def test_embedding_worker_returns_result_without_blocking_submission():
    worker = EmbeddingWorker(StubRecognizer(), max_queue=1)
    try:
        started = time.perf_counter()
        assert worker.submit(7, np.ones((4, 4, 3), dtype=np.uint8), [(1, 2)])
        assert time.perf_counter() - started < 0.1
        deadline = time.perf_counter() + 1.0
        results = []
        while not results and time.perf_counter() < deadline:
            results = worker.drain()
            time.sleep(0.005)
        assert len(results) == 1
        track_id, vector, error = results[0]
        assert track_id == 7
        assert vector.tolist() == [1.0, 1.0]
        assert error is None
    finally:
        worker.close()
