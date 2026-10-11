import numpy as np
from argos.identity.arcface import ArcFaceRecognizer


def store(tmp_path):
    recognizer = ArcFaceRecognizer.__new__(ArcFaceRecognizer)
    recognizer.store_path = tmp_path / "faces.json"
    recognizer.threshold = 0.8
    recognizer.suggestion_threshold = 0.5
    recognizer.ambiguity_margin = 0.05
    recognizer.next_id = 1
    recognizer.templates = {}
    return recognizer


def test_sequential_ids_survive_restart_and_deleted_ids_are_not_reused(tmp_path):
    recognizer = store(tmp_path)
    first = recognizer.enroll(np.array([1.0, 0.0]))
    second = recognizer.enroll(np.array([0.0, 1.0]))
    assert (first, second) == ("FACE-0001", "FACE-0002")
    del recognizer.templates[first]
    recognizer._save()
    loaded = store(tmp_path)
    loaded.templates = loaded._load()
    assert loaded.enroll(np.array([0.7, 0.7])) == "FACE-0003"


def test_rename_category_and_explicit_association_persist(tmp_path):
    recognizer = store(tmp_path)
    identity = recognizer.enroll(np.array([1.0, 0.0]))
    recognizer.update_identity(identity, display_name="Alex", category="Friendly")
    recognizer.enroll(np.array([0.99, 0.01]), identity_id=identity)
    loaded = store(tmp_path)
    loaded.templates = loaded._load()
    entry = loaded.templates[identity]
    assert entry["display_name"] == "Alex"
    assert entry["category"] == "Friendly"
    assert len(entry["embeddings"]) == 2


def test_corrupt_catalogue_starts_empty_without_overwriting_source(tmp_path):
    path = tmp_path / "faces.json"
    path.write_text("{broken", encoding="utf-8")
    recognizer = store(tmp_path)
    recognizer.templates = recognizer._load()
    assert recognizer.templates == {}
    assert path.read_text(encoding="utf-8") == "{broken"


def test_legacy_name_catalogue_migrates_to_stable_ids(tmp_path):
    path = tmp_path / "faces.json"
    path.write_text('{"Face 2": {"embeddings": [[1.0, 0.0]]}}', encoding="utf-8")
    recognizer = store(tmp_path)
    recognizer.templates = recognizer._load()
    assert "FACE-0002" in recognizer.templates
    assert recognizer.templates["FACE-0002"]["display_name"] == "Face 2"
    assert recognizer.enroll(np.array([0.0, 1.0])) == "FACE-0003"


def test_delete_persists_and_never_reuses_id(tmp_path):
    recognizer = store(tmp_path)
    first = recognizer.enroll(np.array([1.0, 0.0]))
    second = recognizer.enroll(np.array([0.0, 1.0]))
    recognizer.delete_identity(first)
    loaded = store(tmp_path)
    loaded.templates = loaded._load()
    assert first not in loaded.templates
    assert loaded.enroll(np.array([0.7, 0.7])) == "FACE-0003"
