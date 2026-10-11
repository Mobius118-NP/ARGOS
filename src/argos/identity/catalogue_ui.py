"""Mouse and keyboard catalogue controls rendered in the existing OpenCV display."""
import cv2
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from argos.ui_theme import BACKGROUND, LIME, MATRIX, NEON, PANEL, TACTICAL


class CatalogueUI:
    def __init__(self, recognizer, on_rename, on_category, on_associate, on_enroll,
                 on_delete, on_duplicate_response):
        self.recognizer = recognizer
        self.on_rename = on_rename
        self.on_category = on_category
        self.on_associate = on_associate
        self.on_enroll = on_enroll
        self.on_delete = on_delete
        self.on_duplicate_response = on_duplicate_response
        self.visible = False
        self.selected_id = None
        self.selected_track_id = None
        self.live_targets = []
        self.editing = False
        self.name_buffer = ""
        self.scroll = 0
        self.track_scroll = 0
        self.visible_rows = 0
        self.visible_track_rows = 0
        self.status = "Select a live track, then enroll it or associate it with a saved face ID."
        self.modal = None
        self._texts = []
        project_root = Path(__file__).resolve().parents[3]
        self.font_regular = project_root / "static" / "FiraCode-Regular.ttf"
        self.font_bold = project_root / "static" / "FiraCode-SemiBold.ttf"
        if not self.font_regular.exists():
            self.font_regular = project_root / "static" / "FiraCode-Medium.ttf"
        if not self.font_regular.exists():
            raise FileNotFoundError(f"Fira Code font not found under {project_root / 'static'}")
        if not self.font_bold.exists():
            self.font_bold = self.font_regular
        self.font_cache = {}

    def _text(self, label, origin, size, color, bold=False):
        self._texts.append((str(label), origin, size, color, bold))

    def _flush_text(self, panel):
        if not self._texts:
            return
        image = Image.fromarray(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image)
        for label, (x, baseline), size, color, bold in self._texts:
            key = (size, bold)
            if key not in self.font_cache:
                path = self.font_bold if bold else self.font_regular
                self.font_cache[key] = ImageFont.truetype(str(path), size=size)
            draw.text((x, baseline - size), label, font=self.font_cache[key], fill=tuple(reversed(color)))
        panel[:] = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        self._texts.clear()

    def set_status(self, message):
        self.status = message

    def set_live_targets(self, targets):
        self.live_targets = list(targets)
        live_ids = {target.track_id for target in self.live_targets}
        if self.selected_track_id is not None and self.selected_track_id not in live_ids:
            expired = self.selected_track_id
            self.selected_track_id = None
            self.set_status(f"Track {expired} is no longer active. Select a current track.")
        if live_ids:
            selected_index = next((i for i, target in enumerate(self.live_targets)
                                   if target.track_id == self.selected_track_id), None)
            max_scroll = max(0, len(self.live_targets) - self.visible_track_rows)
            self.track_scroll = min(self.track_scroll, max_scroll)
            if selected_index is not None:
                if selected_index < self.track_scroll:
                    self.track_scroll = selected_index
                elif selected_index >= self.track_scroll + self.visible_track_rows:
                    self.track_scroll = selected_index - self.visible_track_rows + 1
        else:
            self.track_scroll = 0

    def toggle(self):
        self.visible = not self.visible

    def _button(self, panel, x1, y1, x2, y2, label, enabled=True):
        cv2.rectangle(panel, (x1, y1), (x2, y2), PANEL if enabled else BACKGROUND, -1)
        cv2.rectangle(panel, (x1, y1), (x2, y2), MATRIX if enabled else TACTICAL, 1)
        self._text(label, (x1 + 6, y2 - 8), 11, NEON if enabled else TACTICAL)

    @staticmethod
    def _layout(height):
        identity_top, identity_row_h = 88, 18
        live_top, live_row_h = 218, 18
        field_top, field_bottom = height - 142, height - 116
        visible_identities = max(0, min(5, (195 - identity_top - 4) // identity_row_h))
        visible_tracks = max(0, min(5, (field_top - 30 - live_top) // live_row_h))
        return {
            "identity_top": identity_top,
            "identity_row_h": identity_row_h,
            "visible_identities": visible_identities,
            "live_top": live_top,
            "live_row_h": live_row_h,
            "visible_tracks": visible_tracks,
            "field_top": field_top,
            "field_bottom": field_bottom,
            "button_top": height - 105,
            "button_bottom": height - 75,
        }

    def draw(self, panel):
        h, w = panel.shape[:2]
        layout = self._layout(h)
        self.visible_rows = layout["visible_identities"]
        self.visible_track_rows = layout["visible_tracks"]
        self._texts.clear()
        panel[:] = BACKGROUND
        cv2.rectangle(panel, (0, 0), (w, 57), PANEL, -1)
        self._text("IDENTITY CATALOGUE", (14, 27), 18, NEON, True)
        self._text("X  CLOSE", (w - 87, 27), 12, LIME, True)
        self._text("LOCAL DATABASE | choose a live track to assign a face ID",
                   (14, 49), 12, TACTICAL)
        cv2.line(panel, (12, 63), (w - 12, 63), MATRIX, 1)

        # Saved identities
        for label, x in (("FACE ID", 17), ("DISPLAY NAME", 130), ("STATUS", 390), ("TPL", 558)):
            self._text(label, (x, 81), 10, TACTICAL, True)
        entries = list(self.recognizer.templates.items())
        if self.selected_id not in self.recognizer.templates:
            self.selected_id = entries[0][0] if entries else None
        id_top, id_row_h = layout["identity_top"], layout["identity_row_h"]
        id_visible = layout["visible_identities"]
        selected_index = next((i for i, item in enumerate(entries) if item[0] == self.selected_id), 0)
        self.scroll = max(0, min(self.scroll, max(0, len(entries) - id_visible)))
        if selected_index < self.scroll:
            self.scroll = selected_index
        elif selected_index >= self.scroll + id_visible and id_visible:
            self.scroll = selected_index - id_visible + 1
        if entries:
            for row_index, (identity_id, item) in enumerate(entries[self.scroll:self.scroll + id_visible]):
                y1, y2 = id_top + row_index * id_row_h, id_top + (row_index + 1) * id_row_h
                selected = identity_id == self.selected_id
                if selected:
                    cv2.rectangle(panel, (12, y1), (w - 12, y2), PANEL, -1)
                self._text(identity_id, (17, y2 - 3), 11, LIME if selected else TACTICAL, selected)
                self._text(item["display_name"][:26], (130, y2 - 3), 11, NEON)
                self._text(item["category"], (390, y2 - 3), 11,
                           LIME if item["category"] == "Friendly" else TACTICAL)
                self._text(str(len(item["embeddings"])), (558, y2 - 3), 11, TACTICAL)
                cv2.line(panel, (12, y2), (w - 12, y2), MATRIX, 1)
            if len(entries) > id_visible:
                self._text(f"{self.scroll + 1}-{min(self.scroll + id_visible, len(entries))}/{len(entries)}",
                           (w - 75, 81), 10, MATRIX)
        else:
            self._text("NO SAVED FACE IDS", (17, id_top + 14), 11, MATRIX)

        # Current tracked faces: selection here scopes both ENROLL and ASSOCIATE.
        self._text("LIVE TRACKS", (14, 198), 12, LIME, True)
        self._text("Select the exact track to enroll or link", (145, 198), 10, TACTICAL)
        live_top, live_row_h = layout["live_top"], layout["live_row_h"]
        track_ids = [target.track_id for target in self.live_targets]
        if self.selected_track_id not in track_ids:
            self.selected_track_id = None
        self.track_scroll = max(0, min(self.track_scroll,
                                       max(0, len(self.live_targets) - self.visible_track_rows)))
        if self.live_targets:
            for row_index, target in enumerate(
                    self.live_targets[self.track_scroll:self.track_scroll + self.visible_track_rows]):
                y1 = live_top + row_index * live_row_h
                y2 = y1 + live_row_h
                selected = target.track_id == self.selected_track_id
                if selected:
                    cv2.rectangle(panel, (12, y1), (w - 12, y2), PANEL, -1)
                track_text = f"TRK-{target.track_id:03d}"
                state_text = (f"{target.identity_id}  {target.identity_label}"
                              if target.identity_id else target.identity_label)
                self._text(track_text, (17, y2 - 3), 11, LIME if selected else TACTICAL, selected)
                self._text(state_text[:58], (145, y2 - 3), 11, NEON if target.identity_id else TACTICAL)
                cv2.line(panel, (12, y2), (w - 12, y2), MATRIX, 1)
            if len(self.live_targets) > self.visible_track_rows:
                self._text(f"{self.track_scroll + 1}-{min(self.track_scroll + self.visible_track_rows, len(self.live_targets))}/{len(self.live_targets)}",
                           (w - 75, 198), 10, MATRIX)
        else:
            self._text("NO ACTIVE TRACKS — wait for a face detection", (17, live_top + 14), 11, MATRIX)

        selected_target = next((target for target in self.live_targets
                                if target.track_id == self.selected_track_id), None)
        if selected_target:
            selected_label = f"SELECTED: TRK-{selected_target.track_id:03d}  /  {selected_target.identity_label}"
        else:
            selected_label = "SELECTED: NONE  /  click a live track above"
        self._text(selected_label[:76], (14, layout["field_top"] - 21), 11,
                   LIME if selected_target else TACTICAL, bool(selected_target))

        field_y1, field_y2 = layout["field_top"], layout["field_bottom"]
        self._text("DISPLAY NAME | click field and type", (14, field_y1 - 4), 10, TACTICAL)
        cv2.rectangle(panel, (12, field_y1), (w - 12, field_y2),
                      PANEL if self.editing else BACKGROUND, -1)
        cv2.rectangle(panel, (12, field_y1), (w - 12, field_y2), NEON if self.editing else MATRIX, 1)
        shown_name = self.name_buffer if self.editing else self.recognizer.templates.get(self.selected_id, {}).get("display_name", "")
        self._text((shown_name or "Type a name...")[-68:], (18, field_y2 - 5), 12, LIME)
        button_y1, button_y2 = layout["button_top"], layout["button_bottom"]
        has_track = selected_target is not None
        actions = ((12, 128, "SAVE NAME", bool(self.selected_id)),
                   (136, 252, "STATUS", bool(self.selected_id)),
                   (260, 376, "ENROLL TRACK", has_track),
                   (384, 500, "ASSOCIATE", has_track and bool(self.selected_id)),
                   (508, w - 12, "DELETE", bool(self.selected_id)))
        for x1, x2, label, enabled in actions:
            self._button(panel, x1, button_y1, x2, button_y2, label, enabled)
        self._text(self.status[:86], (14, h - 10), 11, TACTICAL)
        if self.modal:
            self._draw_modal(panel)
        self._flush_text(panel)

    def _draw_modal(self, panel):
        h, w = panel.shape[:2]
        cv2.rectangle(panel, (35, 132), (w - 35, h - 122), BACKGROUND, -1)
        cv2.rectangle(panel, (35, 132), (w - 35, h - 122), NEON, 1)
        if self.modal["kind"] == "delete":
            identity_id = self.modal["identity_id"]
            item = self.recognizer.templates.get(identity_id, {})
            lines = ["DELETE SAVED FACE?", f"{identity_id}  {item.get('display_name', '')}",
                     "This removes its local embedding templates. This cannot be undone."]
            yes_label, no_label = "DELETE FACE", "CANCEL"
        else:
            older, newer = self.modal["older"], self.modal["newer"]
            old_name = self.recognizer.templates.get(older, {}).get("display_name", "")
            new_name = self.recognizer.templates.get(newer, {}).get("display_name", "")
            lines = ["POSSIBLE DUPLICATE FACES", f"{older} {old_name}  /  {newer} {new_name}",
                     f"Very high similarity: {self.modal['score']:.3f}",
                     f"Same person? Yes deletes the newer entry ({newer})."]
            yes_label, no_label = "SAME FACE - DELETE NEWER", "KEEP BOTH"
        start_y = 169
        for index, line in enumerate(lines):
            self._text(line, (52, start_y + index * 27), 13 if index else 15,
                       NEON if index == 0 else TACTICAL, index == 0)
        button_y = h - 174
        self._button(panel, 52, button_y, w // 2 - 8, button_y + 31, yes_label)
        self._button(panel, w // 2 + 8, button_y, w - 52, button_y + 31, no_label)

    def prompt_duplicate(self, left_id, right_id, score):
        def numeric(identity_id):
            return int(identity_id.split("-")[-1])
        older, newer = sorted((left_id, right_id), key=numeric)
        self.modal = {"kind": "duplicate", "older": older, "newer": newer, "score": score}
        self.visible = True
        self.editing = False

    def _resolve_modal(self, confirmed):
        modal = self.modal
        self.modal = None
        if modal["kind"] == "delete":
            if confirmed:
                self.on_delete(modal["identity_id"])
        else:
            self.on_duplicate_response(modal["older"], modal["newer"], confirmed)

    def mouse(self, event, x, y, height, wheel_delta=0):
        layout = self._layout(height)
        if event == cv2.EVENT_MOUSEWHEEL:
            live_end = layout["live_top"] + self.visible_track_rows * layout["live_row_h"]
            if layout["live_top"] <= y < live_end:
                max_scroll = max(0, len(self.live_targets) - self.visible_track_rows)
                direction = -1 if wheel_delta > 0 else 1
                self.track_scroll = min(max(0, self.track_scroll + direction), max_scroll)
            else:
                self.select_offset(-1 if wheel_delta > 0 else 1)
            return
        if event != cv2.EVENT_LBUTTONUP:
            return
        if self.modal:
            button_y = height - 174
            if button_y <= y <= button_y + 31:
                if 52 <= x < 320:
                    self._resolve_modal(True)
                elif 328 <= x <= 588:
                    self._resolve_modal(False)
            return
        if y < 40 and x >= 465:
            self.visible = False
            self.editing = False
            return

        entries = list(self.recognizer.templates.items())
        id_top, id_row_h = layout["identity_top"], layout["identity_row_h"]
        id_end = id_top + self.visible_rows * id_row_h
        if id_top <= y < id_end:
            index = self.scroll + (y - id_top) // id_row_h
            if 0 <= index < len(entries):
                self.selected_id = entries[index][0]
                self.editing = False
                self.name_buffer = ""
            return

        track_top, track_row_h = layout["live_top"], layout["live_row_h"]
        track_end = track_top + self.visible_track_rows * track_row_h
        if track_top <= y < track_end:
            index = self.track_scroll + (y - track_top) // track_row_h
            if 0 <= index < len(self.live_targets):
                self.selected_track_id = self.live_targets[index].track_id
                self.editing = False
                self.name_buffer = ""
                self.set_status(f"Selected Track {self.selected_track_id}; choose ENROLL TRACK or a saved ID plus ASSOCIATE.")
            return

        field_y1, field_y2 = layout["field_top"], layout["field_bottom"]
        if field_y1 <= y < field_y2:
            self.editing = True
            self.name_buffer = ""
        elif layout["button_top"] <= y < layout["button_bottom"]:
            if 12 <= x <= 128:
                self.save_name()
            elif 136 <= x <= 252 and self.selected_id:
                current = self.recognizer.templates[self.selected_id]["category"]
                self.on_category(self.selected_id, "Unknown" if current == "Friendly" else "Friendly")
            elif 260 <= x <= 376:
                if self.selected_track_id is None:
                    self.set_status("Select a live track before creating a new face ID.")
                else:
                    self.on_enroll(self.selected_track_id)
            elif 384 <= x <= 500:
                if self.selected_track_id is None:
                    self.set_status("Select a live track before associating it with a saved face ID.")
                elif not self.selected_id:
                    self.set_status("Select a saved face ID, then choose ASSOCIATE.")
                else:
                    self.on_associate(self.selected_id, self.selected_track_id)
            elif 508 <= x <= 628 and self.selected_id:
                self.modal = {"kind": "delete", "identity_id": self.selected_id}
                self.editing = False

    def save_name(self):
        if self.selected_id and self.name_buffer.strip():
            self.on_rename(self.selected_id, self.name_buffer.strip())
            self.editing = False

    def key(self, key):
        if self.modal:
            if key in (ord("y"), ord("Y"), 10, 13):
                self._resolve_modal(True)
            elif key in (ord("n"), ord("N"), 27):
                self._resolve_modal(False)
            return True
        if not self.editing:
            if key in (2490368, 65362):
                self.select_offset(-1)
                return True
            if key in (2621440, 65364):
                self.select_offset(1)
                return True
            return False
        if key in (10, 13):
            self.save_name()
        elif key in (8, 127):
            self.name_buffer = self.name_buffer[:-1]
        elif key == 27:
            self.editing = False
        elif 32 <= key <= 126 and len(self.name_buffer) < 64:
            self.name_buffer += chr(key)
        return True

    def select_offset(self, offset):
        entries = list(self.recognizer.templates)
        if not entries:
            return
        index = next((i for i, key in enumerate(entries) if key == self.selected_id), 0)
        self.selected_id = entries[min(max(index + offset, 0), len(entries) - 1)]
