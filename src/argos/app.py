import argparse
from dataclasses import replace
from math import hypot
from pathlib import Path
from time import perf_counter
import yaml
import cv2
from argos.camera import SingleCamera, StereoCamera, StereoCalibration
from argos.detection.yunet import YuNetFaceDetector
from argos.identity.arcface import ArcFaceRecognizer
from argos.identity.catalogue_ui import CatalogueUI
from argos.identity.worker import EmbeddingWorker
from argos.identity.similarity_audit import SimilarityAuditWorker
from argos.tracking import CentroidTracker
from argos.visualization import draw_cscope


def load_config(path):
    with open(path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config.setdefault("performance", {}).setdefault("detection_cadence", 1)
    return config


def detection_due(frame_number, cadence):
    return frame_number == 1 or frame_number % max(1, int(cadence)) == 0


def recheck_due(last_checked, now, interval):
    return last_checked is None or now - last_checked >= interval


def mouse_wheel_delta(flags):
    delta = (int(flags) >> 16) & 0xFFFF
    return delta - 0x10000 if delta & 0x8000 else delta


def main():
    parser = argparse.ArgumentParser(description="Local face detection and C-scope prototype")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--mode", choices=("single", "stereo"), help="override camera mode from config")
    args = parser.parse_args()
    cfg = load_config(args.config)
    cam, detection, tracking = cfg["cameras"], cfg["detection"], cfg["tracking"]
    perf_cfg = cfg.get("performance", {})
    cadence = max(1, int(perf_cfg.get("detection_cadence", 1)))
    detector = YuNetFaceDetector(detection["model_path"], detection.get("score_threshold", 0.8),
                                 detection.get("nms_threshold", 0.3), detection.get("top_k", 5000),
                                 detection.get("input_size", "frame"))
    tracker = CentroidTracker(tracking["max_center_distance_px"], tracking["max_missed_frames"],
                              tracking["ema_alpha"])
    recognition = None
    identity = cfg.get("identity", {})
    recognition_path = Path(identity.get("model_path", "models/w600k_r50.onnx"))
    if identity.get("enabled", True) and recognition_path.exists():
        recognition = ArcFaceRecognizer(recognition_path, identity.get("templates_path", "data/face_templates.json"),
                                        identity.get("similarity_threshold", 0.45),
                                        identity.get("suggestion_threshold"), identity.get("ambiguity_margin", 0.04))
        print("ArcFace local recognition enabled. Click IDENTITIES or press C for the catalogue; Q/Esc exits.")
    elif identity.get("enabled", True):
        print(f"ArcFace weights unavailable at {recognition_path}; detection/tracking only. See README for setup and license.")
    mode = args.mode or cam.get("mode", "stereo")
    calibration_path = Path(cam["calibration_file"])
    calibration = None
    if mode == "stereo" and calibration_path.exists():
        calibration = StereoCalibration.load(calibration_path, (cam["width"], cam["height"]), cam["baseline_m"])
        print(f"Loaded stereo rectification maps; configured baseline={cam['baseline_m']:.3f} m")
    elif mode == "single":
        print("Single-camera mode: range and stereo rectification are unavailable.")
    else:
        print(f"No calibration at {calibration_path}; showing unrectified images (range unavailable).")
    capture = (SingleCamera(cam["left_index"], cam["width"], cam["height"], cam["fps"])
               if mode == "single" else
               StereoCamera(cam["left_index"], cam["right_index"], cam["width"], cam["height"], cam["fps"]))
    frame_no = 0
    recognized, embeddings, suggestions, last_embed = {}, {}, {}, {}
    worker = EmbeddingWorker(recognition) if recognition else None
    audit = SimilarityAuditWorker() if recognition else None
    duplicate_interval = max(1.0, float(identity.get("duplicate_check_interval_seconds", 60)))
    duplicate_threshold = float(identity.get("duplicate_similarity_threshold", 0.92))
    next_duplicate_check = perf_counter() + duplicate_interval
    dismissed_duplicates = set()
    latest_targets = []
    catalogue = None
    if recognition:
        def rename_identity(identity_id, name):
            try:
                recognition.update_identity(identity_id, display_name=name)
                for track_id, value in list(recognized.items()):
                    if value[0] == identity_id:
                        recognized[track_id] = (value[0], name.strip(), value[2], value[3], value[4])
                catalogue.set_status(f"Renamed {identity_id}.")
            except ValueError as exc:
                catalogue.set_status(str(exc))

        def change_category(identity_id, category):
            recognition.update_identity(identity_id, category=category)
            for track_id, value in list(recognized.items()):
                if value[0] == identity_id:
                    recognized[track_id] = (value[0], value[1], category, value[3], value[4])
            catalogue.set_status(f"Updated {identity_id} status to {category}.")

        def associate_identity(identity_id, track_id):
            vector = embeddings.get(track_id)
            if vector is None:
                catalogue.set_status(f"Waiting for Track {track_id}'s face embedding. Wait briefly, then select ASSOCIATE again.")
                return
            recognition.enroll(vector, identity_id)
            entry = recognition.templates[identity_id]
            recognized[track_id] = (identity_id, entry["display_name"], entry["category"], None, "previously_saved")
            suggestions.pop(track_id, None)
            catalogue.set_status(f"Track {track_id} associated with {identity_id}.")
            catalogue.selected_id = identity_id

        def enroll_identity(track_id):
            if track_id in recognized:
                catalogue.set_status(f"Track {track_id} already has a saved face ID. Select an unknown track to create a new ID.")
                return
            vector = embeddings.get(track_id)
            if vector is None:
                catalogue.set_status(f"Waiting for Track {track_id}'s face embedding. Wait briefly, then select ENROLL TRACK again.")
                return
            identity_id = recognition.enroll(vector)
            entry = recognition.templates[identity_id]
            recognized[track_id] = (identity_id, entry["display_name"], entry["category"], None, "")
            suggestions.pop(track_id, None)
            catalogue.set_status(f"Track {track_id} enrolled as {identity_id} {entry['display_name']} locally.")
            catalogue.selected_id = identity_id

        def delete_identity(identity_id):
            try:
                recognition.delete_identity(identity_id)
            except (KeyError, OSError, PermissionError) as exc:
                catalogue.set_status(f"Could not delete {identity_id}: {exc}")
                return False
            for track_id, value in list(recognized.items()):
                if value[0] == identity_id:
                    recognized.pop(track_id, None)
                    last_embed.pop(track_id, None)
            for track_id, value in list(suggestions.items()):
                if value[0] == identity_id:
                    suggestions.pop(track_id, None)
                    last_embed.pop(track_id, None)
            dismissed_duplicates_copy = {item for item in dismissed_duplicates
                                         if item[0] != identity_id and item[1] != identity_id}
            dismissed_duplicates.clear()
            dismissed_duplicates.update(dismissed_duplicates_copy)
            catalogue.selected_id = next(iter(recognition.templates), None)
            catalogue.set_status(f"Deleted {identity_id} and its local templates. Its ID will not be reused.")
            return True

        def duplicate_response(older, newer, same_face):
            if same_face:
                if delete_identity(newer):
                    catalogue.set_status(f"Confirmed same face. Removed newer entry {newer}; kept {older}.")
            elif older in recognition.templates and newer in recognition.templates:
                counts = (len(recognition.templates[older]["embeddings"]),
                          len(recognition.templates[newer]["embeddings"]))
                dismissed_duplicates.add((older, newer, *counts))
                catalogue.set_status(f"Kept {older} and {newer} as separate faces.")

        catalogue = CatalogueUI(recognition, rename_identity, change_category,
                               associate_identity, enroll_identity, delete_identity, duplicate_response)
    timings = {"capture": 0.0, "detect": 0.0, "track": 0.0, "recognition": 0.0, "display": 0.0}
    window_start = perf_counter()
    window_frames = 0
    fps = 0.0
    try:
        with capture as cameras:
            window_name = cfg["visualization"].get("window_name", "Argos — C-scope")
            frame_size = [cam["width"], cam["height"]]
            cv2.namedWindow(window_name)
            def on_mouse(event, x, y, flags, _userdata):
                if event != cv2.EVENT_LBUTTONUP or not catalogue:
                    if event != cv2.EVENT_MOUSEWHEEL or not catalogue or not catalogue.visible:
                        return
                if catalogue.visible:
                    wheel = mouse_wheel_delta(flags) if event == cv2.EVENT_MOUSEWHEEL else 0
                    catalogue.mouse(event, x - frame_size[0], y, frame_size[1], wheel)
                elif x >= frame_size[0] + 10 and x <= frame_size[0] + 142 and 34 <= y <= 62:
                    catalogue.toggle()
            cv2.setMouseCallback(window_name, on_mouse)
            while True:
                started = perf_counter()
                pair = cameras.read()
                timings["capture"] += perf_counter() - started
                if pair is None:
                    print("Camera frame read failed.")
                    break
                frame_no += 1
                left = pair.left
                frame_size[:] = [left.shape[1], left.shape[0]]
                if calibration is not None and pair.right is not None:
                    left, _ = calibration.rectify(left, pair.right)
                detections = []
                started = perf_counter()
                run_detection = detection_due(frame_no, cadence)
                if run_detection:
                    detections = detector.detect_faces(left)
                timings["detect"] += perf_counter() - started
                started = perf_counter()
                targets = tracker.update([box for box, _, _ in detections] if run_detection else [],
                                         (left.shape[1], left.shape[0]))
                timings["track"] += perf_counter() - started
                active_tracks = {target.track_id for target in targets}
                for cache in (recognized, embeddings, suggestions, last_embed):
                    for track_id in cache.keys() - active_tracks:
                        del cache[track_id]
                started = perf_counter()
                now = perf_counter()
                if worker and run_detection:
                    for box, _, landmarks in detections:
                        target = min(targets, key=lambda t: hypot(t.raw_box.center.x - box.center.x,
                                                                  t.raw_box.center.y - box.center.y), default=None)
                        if target is None or target.track_id in recognized:
                            continue
                        recheck = max(0.1, float(identity.get("unknown_recheck_seconds", 1.5)))
                        last_checked = last_embed.get(target.track_id)
                        if recheck_due(last_checked, now, recheck):
                            if worker.submit(target.track_id, left, landmarks):
                                last_embed[target.track_id] = now
                if worker:
                    for track_id, vector, error in worker.drain():
                        worker.pending.discard(track_id)
                        if track_id not in active_tracks:
                            continue
                        if error:
                            if catalogue:
                                catalogue.set_status(f"ArcFace failed for Track {track_id}: {error}")
                            continue
                        embeddings[track_id] = vector
                        if track_id in recognized:
                            continue
                        candidate, ambiguous, score = recognition.suggestion(vector)
                        suggestions[track_id] = (candidate, ambiguous, score)
                        label, confidence = recognition.identify(vector)
                        if label and not ambiguous:
                            entry = recognition.templates[label]
                            recognized[track_id] = (label, entry["display_name"], entry["category"], confidence, "")
                            if catalogue:
                                catalogue.set_status(f"Track {track_id} recognized as {label} {entry['display_name']}.")
                        elif candidate and catalogue:
                            prompt = "Ambiguous possible match" if ambiguous else "Possible saved match"
                            catalogue.set_status(f"Track {track_id}: {prompt} {candidate} ({score:.2f}). Review or associate in catalogue.")
                if audit:
                    for pairs, error in audit.drain():
                        if error:
                            catalogue.set_status(f"Catalogue similarity check failed: {error}")
                            continue
                        if catalogue.modal:
                            continue
                        for first_id, second_id, score in pairs:
                            older, newer = sorted((first_id, second_id), key=lambda value: int(value.split("-")[-1]))
                            if older not in recognition.templates or newer not in recognition.templates:
                                continue
                            counts = (len(recognition.templates[older]["embeddings"]),
                                      len(recognition.templates[newer]["embeddings"]))
                            if (older, newer, *counts) in dismissed_duplicates:
                                continue
                            catalogue.prompt_duplicate(older, newer, score)
                            catalogue.set_status(f"Possible duplicate: {older} and {newer}.")
                            break
                    if now >= next_duplicate_check:
                        audit.submit(recognition.templates, duplicate_threshold)
                        next_duplicate_check = now + duplicate_interval
                enriched = []
                for target in targets:
                    cached = recognized.get(target.track_id)
                    if cached:
                        identity_id, name, category, confidence, status = cached
                        target = replace(target, identity_id=identity_id, identity_label=name,
                                         identity_category=category, identity_confidence=confidence,
                                         identity_status=status)
                    suggestion = suggestions.get(target.track_id)
                    if not cached and suggestion and suggestion[0]:
                        target = replace(target, identity_suggestion=suggestion[0], identity_ambiguous=suggestion[1])
                    enriched.append(target)
                targets = enriched
                latest_targets = targets
                if catalogue:
                    catalogue.set_live_targets(targets)
                timings["recognition"] += perf_counter() - started
                window_frames += 1
                elapsed = perf_counter() - window_start
                if elapsed >= 1.0:
                    fps = window_frames / elapsed
                    window_frames, window_start = 0, perf_counter()
                    if perf_cfg.get("profile", True):
                        denom = max(1, frame_no)
                        print(f"Profile {fps:.1f} FPS | " + " | ".join(f"{k} {1000*v/denom:.1f} ms/frame" for k, v in timings.items()))
                started = perf_counter()
                cv2.imshow(window_name, draw_cscope(left, targets, fps, catalogue,
                           show_boxes=cfg["visualization"].get("show_boxes", True),
                           frame_index=frame_no))
                key = cv2.waitKeyEx(1)
                if key in (ord("q"), ord("Q")) and not (catalogue and catalogue.visible and catalogue.editing):
                    break
                elif key == 27 and not (catalogue and catalogue.visible and catalogue.editing):
                    break
                elif key in (ord("c"), ord("C")):
                    if catalogue and not catalogue.key(key):
                        catalogue.toggle()
                elif catalogue and catalogue.visible and catalogue.key(key):
                    pass
                timings["display"] += perf_counter() - started
    finally:
        if worker:
            worker.close()
        if audit:
            audit.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
