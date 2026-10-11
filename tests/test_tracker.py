from argos.geometry import PixelBox
from argos.tracking import CentroidTracker


def test_tracker_keeps_id_for_nearby_detection():
    tracker = CentroidTracker(max_distance_px=40)
    first = tracker.update([PixelBox(10, 10, 20, 20)], (100, 100))
    second = tracker.update([PixelBox(13, 11, 20, 20)], (100, 100))
    assert first[0].track_id == second[0].track_id == 1


def test_tracker_keeps_raw_detection_and_smooths_track_box():
    tracker = CentroidTracker(max_distance_px=40, ema_alpha=0.3)
    tracker.update([PixelBox(0, 0, 20, 20)], (100, 100))
    target = tracker.update([PixelBox(10, 0, 20, 20)], (100, 100))[0]
    assert target.raw_box == PixelBox(10, 0, 20, 20)
    assert target.box == PixelBox(3, 0, 20, 20)
    assert target.identity_id is None
    assert target.identity_label == "Unknown"


def test_tracker_bridges_skipped_detection_frames():
    tracker = CentroidTracker(max_distance_px=40, max_missed_frames=3)
    first = tracker.update([PixelBox(10, 10, 20, 20)], (100, 100))[0]
    between = tracker.update([], (100, 100))[0]
    resumed = tracker.update([PixelBox(12, 10, 20, 20)], (100, 100))[0]
    assert between.track_id == resumed.track_id == first.track_id
