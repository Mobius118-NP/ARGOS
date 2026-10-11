import cv2
import numpy as np
from argos.models import TargetState
from argos.ui_theme import BACKGROUND, LIME, MATRIX, NEON, PANEL, TACTICAL


def _label(image, text, origin, color=TACTICAL, scale=0.43, thickness=1):
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                thickness, cv2.LINE_AA)


def _feed_bars(view, frame_index=None):
    """Subtle local-system chrome around the live camera image."""
    h, w = view.shape[:2]
    overlay = view.copy()
    cv2.rectangle(overlay, (0, 0), (w, 36), BACKGROUND, -1)
    cv2.rectangle(overlay, (0, h - 25), (w, h), BACKGROUND, -1)
    cv2.addWeighted(overlay, 0.76, view, 0.24, 0, view)
    cv2.line(view, (0, 36), (w, 36), MATRIX, 1, cv2.LINE_AA)
    cv2.line(view, (0, h - 26), (w, h - 26), MATRIX, 1, cv2.LINE_AA)
    cv2.circle(view, (15, 18), 4, NEON, -1, cv2.LINE_AA)
    _label(view, "ARGOS", (27, 22), NEON, 0.48, 1)
    _label(view, "LOCAL OPTICAL FEED  /  CONSENT-BASED", (100, 21), TACTICAL, 0.39)
    _label(view, "CAM 01", (w - 70, 21), LIME, 0.39)
    left = "TRACKING ACTIVE" if frame_index is not None else "LIVE FEED"
    _label(view, left, (10, h - 9), TACTICAL, 0.36)
    if frame_index is not None:
        _label(view, f"FRAME {frame_index:06d}", (w - 100, h - 9), TACTICAL, 0.36)


def _draw_scope_grid(panel, width, height):
    top, bottom = 72, height - 34
    left, right = 18, width - 18
    # Faint coordinate grid with brighter center axes.
    step = max(32, width // 16)
    x = left
    while x <= right:
        cv2.line(panel, (x, top), (x, bottom), MATRIX, 1, cv2.LINE_AA)
        x += step
    y = top
    while y <= bottom:
        cv2.line(panel, (left, y), (right, y), MATRIX, 1, cv2.LINE_AA)
        y += step
    cv2.line(panel, (width // 2, top), (width // 2, bottom), TACTICAL, 1, cv2.LINE_AA)
    cv2.line(panel, (left, height // 2), (right, height // 2), TACTICAL, 1, cv2.LINE_AA)
    # Corner brackets give the plot a technical instrument-panel frame.
    arm = 12
    for x0, y0, dx, dy in ((left, top, 1, 1), (right, top, -1, 1),
                            (left, bottom, 1, -1), (right, bottom, -1, -1)):
        cv2.line(panel, (x0, y0), (x0 + arm * dx, y0), NEON, 1, cv2.LINE_AA)
        cv2.line(panel, (x0, y0), (x0, y0 + arm * dy), NEON, 1, cv2.LINE_AA)


def draw_cscope(frame, targets: list[TargetState], fps=None, catalogue=None, show_boxes=True,
                frame_index=None):
    """Draw the camera view and either the C-scope or the graphical identity table."""
    h, w = frame.shape[:2]
    view = frame.copy()
    for target in targets:
        b = target.box
        if show_boxes:
            cv2.rectangle(view, (b.x, b.y), (b.x + b.width, b.y + b.height), NEON, 1, cv2.LINE_AA)
            # Small corner ticks form an uncluttered face-detection bracket.
            tick = max(5, min(b.width, b.height) // 8)
            for x0, y0, dx, dy in ((b.x, b.y, 1, 1), (b.x + b.width, b.y, -1, 1),
                                    (b.x, b.y + b.height, 1, -1),
                                    (b.x + b.width, b.y + b.height, -1, -1)):
                cv2.line(view, (x0, y0), (x0 + tick * dx, y0), LIME, 2, cv2.LINE_AA)
                cv2.line(view, (x0, y0), (x0, y0 + tick * dy), LIME, 2, cv2.LINE_AA)
        identity = f"{target.identity_id} {target.identity_label}" if target.identity_id else target.identity_label
        suffix = " · saved match" if target.identity_status == "previously_saved" else ""
        if target.identity_suggestion:
            suffix = f" · possible {target.identity_suggestion}" + (" (choose in catalogue)" if target.identity_ambiguous else "")
        cv2.rectangle(view, (b.x, max(38, b.y - 20)),
                      (min(w - 1, b.x + max(128, len(identity) * 8)), max(38, b.y - 2)), PANEL, -1)
        _label(view, f"TRK-{target.track_id:03d}  {identity}{suffix}",
               (b.x + 4, max(52, b.y - 7)), LIME, 0.4, 1)

    _feed_bars(view, frame_index)
    panel = np.full_like(view, BACKGROUND)
    if catalogue is not None and catalogue.visible:
        catalogue.draw(panel)
    else:
        cv2.rectangle(panel, (0, 0), (w, 58), PANEL, -1)
        cv2.line(panel, (0, 58), (w, 58), MATRIX, 1, cv2.LINE_AA)
        cv2.circle(panel, (16, 19), 4, NEON, -1, cv2.LINE_AA)
        _label(panel, "ARGOS  /  TRACK DATABASE", (29, 23), NEON, 0.48, 1)
        _label(panel, "LOCAL SESSION", (w - 130, 22), TACTICAL, 0.38, 1)
        _label(panel, "C-SCOPE  //  NORMALIZED IMAGE SPACE", (14, 49), TACTICAL, 0.36)

        _draw_scope_grid(panel, w, h)
        cv2.rectangle(panel, (9, h - 27), (w - 9, h - 7), PANEL, -1)
        _label(panel, f"TRACKS {len(targets):02d}", (16, h - 13), LIME, 0.38)
        _label(panel, "RANGE  N/A", (w // 2 - 38, h - 13), TACTICAL, 0.38)
        _label(panel, "IDENTITIES [C]", (w - 116, h - 13), TACTICAL, 0.36)
        if fps is not None:
            _label(panel, f"{fps:04.1f} FPS", (w - 106, 48), NEON, 0.39)

        for target in targets:
            x = int(np.clip(target.normalized_x * w, 10, w - 10))
            y = int(np.clip(target.normalized_y * h, 76, h - 42))
            cv2.circle(panel, (x, y), 8, NEON, 1, cv2.LINE_AA)
            cv2.circle(panel, (x, y), 2, LIME, -1, cv2.LINE_AA)
            cv2.line(panel, (x - 12, y), (x - 5, y), LIME, 1, cv2.LINE_AA)
            cv2.line(panel, (x + 5, y), (x + 12, y), LIME, 1, cv2.LINE_AA)
            cv2.line(panel, (x, y - 12), (x, y - 5), LIME, 1, cv2.LINE_AA)
            cv2.line(panel, (x, y + 5), (x, y + 12), LIME, 1, cv2.LINE_AA)
            cv2.putText(panel, f"TRK-{target.track_id:03d}", (x + 12, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, LIME, 1, cv2.LINE_AA)
    return np.hstack((view, panel))
