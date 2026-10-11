"""One bounded background worker keeps ArcFace inference off the display loop."""
from queue import Empty, Full, Queue
from threading import Thread


class EmbeddingWorker:
    def __init__(self, recognizer, max_queue=8):
        self.recognizer = recognizer
        self.jobs = Queue(maxsize=max_queue)
        self.results = Queue()
        self.pending = set()
        self.thread = Thread(target=self._run, name="argos-arcface", daemon=True)
        self.thread.start()

    def submit(self, track_id, frame, landmarks):
        if track_id in self.pending:
            return False
        try:
            self.jobs.put_nowait((track_id, frame.copy(), tuple(landmarks)))
        except Full:
            return False
        self.pending.add(track_id)
        return True

    def drain(self):
        items = []
        while True:
            try:
                items.append(self.results.get_nowait())
            except Empty:
                return items

    def _run(self):
        while True:
            item = self.jobs.get()
            if item is None:
                return
            track_id, frame, landmarks = item
            try:
                self.results.put((track_id, self.recognizer.embedding(frame, landmarks), None))
            except Exception as exc:  # deliver model/runtime errors to the UI loop
                self.results.put((track_id, None, exc))

    def close(self):
        self.jobs.put(None)
        self.thread.join(timeout=5)
