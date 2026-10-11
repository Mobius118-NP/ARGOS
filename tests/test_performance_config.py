from argos.app import detection_due, recheck_due, mouse_wheel_delta
from argos.identity.catalogue_ui import CatalogueUI
from argos.identity.similarity_audit import find_duplicate_pairs
import numpy as np
import cv2


def test_detection_cadence_runs_first_then_every_n_frames():
    assert [n for n in range(1, 8) if detection_due(n, 3)] == [1, 3, 6]
    assert detection_due(2, 0)


def test_unknown_track_recheck_interval():
    assert recheck_due(None, 12.0, 1.5)
    assert not recheck_due(12.0, 13.4, 1.5)
    assert recheck_due(12.0, 13.5, 1.5)


def test_mouse_wheel_delta_decodes_signed_windows_wheel_flags():
    assert mouse_wheel_delta(120 << 16) == 120
    assert mouse_wheel_delta((-120 & 0xFFFF) << 16) == -120


def test_catalogue_graphical_list_name_editor_and_association_controls():
    class Recognizer:
        templates = {"FACE-0001": {"display_name": "Face 1", "category": "Unknown", "embeddings": [[1.0]]}}

    events = []
    ui = CatalogueUI(Recognizer(), lambda *args: events.append(("rename", args)),
                     lambda *args: events.append(("category", args)),
                     lambda *args: events.append(("associate", args)),
                     lambda: events.append(("enroll", ())),
                     lambda *args: events.append(("delete", args)),
                     lambda *args: events.append(("duplicate", args)))
    ui.visible = True
    panel = np.zeros((480, 640, 3), dtype=np.uint8)
    ui.draw(panel)
    ui.mouse(cv2.EVENT_LBUTTONUP, 20, 375, 480)
    for key in map(ord, "Alex"):
        ui.key(key)
    ui.key(13)
    assert events[-1] == ("rename", ("FACE-0001", "Alex"))
    ui.mouse(cv2.EVENT_LBUTTONUP, 440, 420, 480)
    assert events[-1] == ("associate", ("FACE-0001",))
    ui.mouse(cv2.EVENT_LBUTTONUP, 560, 420, 480)
    assert ui.modal["kind"] == "delete"
    ui.mouse(cv2.EVENT_LBUTTONUP, 200, 316, 480)
    assert events[-1] == ("delete", ("FACE-0001",))


def test_duplicate_audit_only_returns_very_similar_distinct_entries():
    snapshot = {"FACE-0001": [[1.0, 0.0]],
                "FACE-0002": [[0.95, 0.3122499]],
                "FACE-0003": [[-1.0, 0.0]]}
    pairs = find_duplicate_pairs(snapshot, 0.92)
    assert len(pairs) == 1
    assert pairs[0][:2] == ("FACE-0001", "FACE-0002")
    assert pairs[0][2] >= 0.92


def test_duplicate_confirmation_targets_newer_identity_for_removal():
    class Recognizer:
        templates = {
            "FACE-0001": {"display_name": "Old", "category": "Unknown", "embeddings": [[1.0]]},
            "FACE-0002": {"display_name": "New", "category": "Unknown", "embeddings": [[1.0]]},
        }

    responses = []
    ui = CatalogueUI(Recognizer(), lambda *args: None, lambda *args: None,
                     lambda *args: None, lambda: None,
                     lambda *args: None, lambda *args: responses.append(args))
    ui.prompt_duplicate("FACE-0002", "FACE-0001", 0.99)
    ui.key(ord("y"))
    assert responses == [("FACE-0001", "FACE-0002", True)]
