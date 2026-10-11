"""Local ArcFace embeddings and a durable, explicitly managed identity catalogue."""
import json
from pathlib import Path
import cv2
import numpy as np

ARCFACE_TEMPLATE = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                             [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)


class ArcFaceRecognizer:
    def __init__(self, model_path, store_path="data/face_templates.json", threshold=0.45,
                 suggestion_threshold=None, ambiguity_margin=0.04):
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model_path), sess_options=options,
                                            providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.store_path = Path(store_path)
        self.threshold = float(threshold)
        self.suggestion_threshold = float(suggestion_threshold if suggestion_threshold is not None else threshold - 0.08)
        self.ambiguity_margin = float(ambiguity_margin)
        self.next_id = 1
        self.templates = self._load()

    def _load(self):
        if not self.store_path.exists():
            return {}
        try:
            data = json.loads(self.store_path.read_text(encoding="utf-8"))
            entries = data.get("identities", {}) if isinstance(data, dict) and "identities" in data else data
            if not isinstance(entries, dict):
                raise ValueError("catalogue must be a JSON object")
            cleaned = {}
            for key, entry in entries.items():
                if not isinstance(entry, dict) or not isinstance(entry.get("embeddings"), list):
                    continue
                try:
                    text_key = str(key)
                    number = int(text_key.split("-")[-1]) if text_key.startswith("FACE-") else int(text_key.split()[-1])
                    vectors = [np.asarray(v, dtype=np.float32) for v in entry["embeddings"]]
                    if not vectors or any(v.ndim != 1 or not np.isfinite(v).all() for v in vectors):
                        continue
                except (ValueError, TypeError):
                    continue
                cleaned[f"FACE-{number:04d}"] = {
                    "display_name": str(entry.get("display_name", key if not text_key.startswith("FACE-") else f"Face {number}")),
                    "category": entry.get("category", "Unknown") if entry.get("category") in ("Friendly", "Unknown") else "Unknown",
                    "embeddings": [v.tolist() for v in vectors],
                }
            saved_next = data.get("next_id", 1) if isinstance(data, dict) else 1
            self.next_id = max(int(saved_next), max((int(k.split("-")[1]) + 1 for k in cleaned), default=1))
            return cleaned
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            print(f"Could not read identity catalogue {self.store_path}: {exc}. Starting with an empty catalogue; source file preserved.")
            return {}

    def _save(self):
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.store_path.with_suffix(self.store_path.suffix + ".tmp")
        payload = {"version": 1, "next_id": self.next_id, "identities": self.templates}
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(self.store_path)

    def embedding(self, frame, landmarks):
        source = np.asarray(landmarks, dtype=np.float32)
        matrix, _ = cv2.estimateAffinePartial2D(source, ARCFACE_TEMPLATE, method=cv2.LMEDS)
        if matrix is None:
            raise ValueError("Could not align face landmarks")
        aligned = cv2.warpAffine(frame, matrix, (112, 112), borderValue=0)
        blob = ((aligned.astype(np.float32) - 127.5) / 127.5).transpose(2, 0, 1)[None, ...]
        vector = self.session.run(None, {self.input_name: blob})[0].reshape(-1).astype(np.float32)
        norm = np.linalg.norm(vector)
        if norm == 0 or not np.isfinite(norm):
            raise ValueError("ArcFace returned an invalid embedding")
        return vector / norm

    def scores(self, vector):
        query = np.asarray(vector, dtype=np.float32)
        scored = []
        for identity_id, stored in self.templates.items():
            scores = [float(np.dot(query, np.asarray(v, dtype=np.float32))) for v in stored["embeddings"]]
            if scores:
                scored.append((max(scores), identity_id))
        return sorted(scored, reverse=True)

    def identify(self, vector):
        scored = self.scores(vector)
        if not scored or scored[0][0] < self.threshold:
            return None, scored[0][0] if scored else -1.0
        return scored[0][1], scored[0][0]

    def suggestion(self, vector):
        """Return a safe explicit-approval suggestion, including close-score ambiguity."""
        scored = self.scores(vector)
        if not scored or scored[0][0] < self.suggestion_threshold:
            return None, False, scored[0][0] if scored else -1.0
        ambiguous = len(scored) > 1 and scored[0][0] - scored[1][0] < self.ambiguity_margin
        return scored[0][1], ambiguous, scored[0][0]

    def enroll(self, vector, identity_id=None):
        if identity_id is None:
            identity_id = f"FACE-{self.next_id:04d}"
            self.next_id += 1
            self.templates[identity_id] = {"display_name": f"Face {int(identity_id.split('-')[1])}",
                                           "category": "Unknown", "embeddings": []}
        if identity_id not in self.templates:
            raise KeyError(identity_id)
        self.templates[identity_id]["embeddings"].append(np.asarray(vector, dtype=np.float32).tolist())
        self._save()
        return identity_id

    def update_identity(self, identity_id, display_name=None, category=None):
        if identity_id not in self.templates:
            raise KeyError(identity_id)
        if display_name is not None:
            if not display_name.strip():
                raise ValueError("display name cannot be empty")
            self.templates[identity_id]["display_name"] = display_name.strip()
        if category is not None:
            if category not in ("Friendly", "Unknown"):
                raise ValueError("category must be Friendly or Unknown")
            self.templates[identity_id]["category"] = category
        self._save()

    def delete_identity(self, identity_id):
        """Delete one saved identity while preserving the monotonic ID counter."""
        if identity_id not in self.templates:
            raise KeyError(identity_id)
        deleted = self.templates.pop(identity_id)
        try:
            self._save()
        except Exception:
            self.templates[identity_id] = deleted
            raise
