"""Periodic high-similarity checks for duplicate saved identities."""
from queue import Empty, Full, Queue
from threading import Thread
import numpy as np


def find_duplicate_pairs(snapshot, threshold):
    """Return ID pairs whose most similar stored templates clear the threshold."""
    ids = list(snapshot)
    pairs = []
    for index, left_id in enumerate(ids):
        left = [np.asarray(v, dtype=np.float32) for v in snapshot[left_id]]
        for right_id in ids[index + 1:]:
            right = [np.asarray(v, dtype=np.float32) for v in snapshot[right_id]]
            scores = []
            for a in left:
                for b in right:
                    if a.ndim == b.ndim == 1 and a.shape == b.shape and a.size:
                        scores.append(float(np.dot(a, b)))
            score = max(scores, default=-1.0)
            if score >= threshold:
                pairs.append((left_id, right_id, score))
    return sorted(pairs, key=lambda item: (-item[2], item[0], item[1]))


class SimilarityAuditWorker:
    def __init__(self):
        self.jobs = Queue(maxsize=1)
        self.results = Queue()
        self.pending = False
        self.thread = Thread(target=self._run, name="argos-catalogue-audit", daemon=True)
        self.thread.start()

    def submit(self, templates, threshold):
        if self.pending:
            return False
        snapshot = {key: [np.asarray(v, dtype=np.float32).copy() for v in item["embeddings"]]
                    for key, item in templates.items()}
        try:
            self.jobs.put_nowait((snapshot, float(threshold)))
        except Full:
            return False
        self.pending = True
        return True

    def drain(self):
        items = []
        while True:
            try:
                items.append(self.results.get_nowait())
                self.pending = False
            except Empty:
                return items

    def _run(self):
        while True:
            job = self.jobs.get()
            if job is None:
                return
            snapshot, threshold = job
            try:
                self.results.put((find_duplicate_pairs(snapshot, threshold), None))
            except Exception as exc:
                self.results.put(([], exc))

    def close(self):
        self.jobs.put(None)
        self.thread.join(timeout=5)
