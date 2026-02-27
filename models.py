# =============================================================================
# models.py
# Dataclasses for all analysis result types.
# =============================================================================

from dataclasses import dataclass


@dataclass
class ImageItem:
    id: int
    filename: str
    path: str
    ext: str
    bytes: int
    width: int
    height: int
    aspect_ratio: str
    color_mode: str
    phash: str
    ahash: str
    dhash: str
    avg_brightness: float
    avg_contrast: float
    dominant_color_1: str
    dominant_color_2: str
    dominant_color_3: str
    hist_bhattacharyya: float = 0.0   # mean Bhattacharyya dist to pHash bucket peers
    detected_objects: str = ""
    detection_confidence: str = ""
    bucket_phash: int = -1            # Union-Find pHash similarity group
    bucket_dbscan: int = -1           # DBSCAN cluster (-1 = noise / unique)


@dataclass
class VideoFrameResult:
    video_pair_1: str
    video_pair_2: str
    frame_id: int
    timestamp: float
    similarity_ssim: float
    similarity_mse: float
    similarity_psnr: float
    motion_score: float
    detected_objects: str = ""
    detection_confidence: str = ""


@dataclass
class SingleVideoFrame:
    """One sampled frame from a single-video analysis."""
    frame_id: int
    timestamp: float
    motion_score: float
    avg_brightness: float
    avg_contrast: float
    dominant_color_1: str
    dominant_color_2: str
    dominant_color_3: str
    detected_objects: str = ""
    detection_confidence: str = ""
