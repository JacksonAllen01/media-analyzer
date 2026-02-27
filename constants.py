# =============================================================================
# constants.py
# Shared constants, file extension sets, and table column definitions.
# =============================================================================

# -- Authentication -----------------------------------------------------------
# Demo password for authorized image viewing.
# Change before deployment -- in production this should be hashed and stored securely.
DEMO_PASSWORD = "wilco2025"

# -- Ollama -------------------------------------------------------------------
OLLAMA_URL      = "http://localhost:11434/api/generate"
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL    = "llama3"
OLLAMA_LLAVA    = "llava-llama3"

# -- File extensions ----------------------------------------------------------
IMG_EXTS = frozenset({".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif"})
VID_EXTS = frozenset({".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".m4v"})

# -- Results table column definitions: (field_name, header_label, width_px) --
IMAGE_PREVIEW_COLS = [
    ("_thumb",               "Preview",      72),   # special-cased thumbnail column
    ("id",                   "ID",           40),
    ("filename",             "Filename",    160),
    ("ext",                  "Type",         45),
    ("bytes",                "Size (B)",     75),
    ("width",                "W",            50),
    ("height",               "H",            50),
    ("aspect_ratio",         "Ratio",        55),
    ("avg_brightness",       "Bright.",      65),
    ("avg_contrast",         "Contrast",     65),
    ("dominant_color_1",     "Color 1",      62),
    ("dominant_color_2",     "Color 2",      62),
    ("dominant_color_3",     "Color 3",      62),
    ("hist_bhattacharyya",   "Bhatt. Dist.", 85),
    ("detected_objects",     "Objects",     180),
    ("detection_confidence", "Conf.",        140),
    ("bucket_phash",         "pHash Grp",    70),
    ("bucket_dbscan",        "DBSCAN Grp",   75),
]

VIDEO_PREVIEW_COLS = [
    ("video_pair_1",          "Video 1",      140),
    ("video_pair_2",          "Video 2",      140),
    ("frame_id",              "Frame",         50),
    ("timestamp",             "Time (s)",      65),
    ("similarity_ssim",       "SSIM %",        65),
    ("similarity_mse",        "MSE",           65),
    ("similarity_psnr",       "PSNR (dB)",     75),
    ("motion_score",          "Motion",        65),
    ("detected_objects",      "Objects",      180),
    ("detection_confidence",  "Conf.",         140),
]

SINGLE_VIDEO_COLS = [
    ("frame_id",              "Frame",         50),
    ("timestamp",             "Time (s)",      70),
    ("motion_score",          "Motion",        70),
    ("avg_brightness",        "Brightness",    80),
    ("avg_contrast",          "Contrast",      70),
    ("dominant_color_1",      "Color 1",       65),
    ("dominant_color_2",      "Color 2",       65),
    ("dominant_color_3",      "Color 3",       65),
    ("detected_objects",      "Objects",      200),
    ("detection_confidence",  "Conf.",         150),
]
