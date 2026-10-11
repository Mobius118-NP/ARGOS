"""Argos green HUD palette: public RGB/HEX values and OpenCV BGR equivalents."""

COLORS = {
    "neon_cyber_green": {"rgb": (0, 255, 102), "hex": "#00FF66", "bgr": (102, 255, 0)},
    "terminal_lime": {"rgb": (57, 255, 20), "hex": "#39FF14", "bgr": (20, 255, 57)},
    "tactical_digital_green": {"rgb": (0, 170, 85), "hex": "#00AA55", "bgr": (85, 170, 0)},
    "matrix_conduit": {"rgb": (0, 143, 17), "hex": "#008F11", "bgr": (17, 143, 0)},
    "hologram_shadow": {"rgb": (0, 34, 17), "hex": "#002211", "bgr": (17, 34, 0)},
    "deep_obsidian_void": {"rgb": (5, 12, 8), "hex": "#050C08", "bgr": (8, 12, 5)},
}

NEON = COLORS["neon_cyber_green"]["bgr"]
LIME = COLORS["terminal_lime"]["bgr"]
TACTICAL = COLORS["tactical_digital_green"]["bgr"]
MATRIX = COLORS["matrix_conduit"]["bgr"]
PANEL = COLORS["hologram_shadow"]["bgr"]
BACKGROUND = COLORS["deep_obsidian_void"]["bgr"]
