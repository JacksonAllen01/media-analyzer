# models.py
# Dataclasses for all analysis result types.

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
    bhash: str
    avg_brightness: float
    avg_contrast: float
    dominant_color_1: str
    dominant_color_2: str
    dominant_color_3: str
    hash_dist: float = 0.0            # mean Hamming distance to group peers (chosen method)
    bucket_phash: int = -1            # Union-Find similarity group (chosen hash method)
    # EXIF metadata fields (empty string if not present in file)
    exif_datetime: str = ""           # Date/time the photo was taken
    exif_camera_make: str = ""        # Camera manufacturer
    exif_camera_model: str = ""       # Camera model
    exif_gps_lat: str = ""            # GPS latitude (decimal degrees)
    exif_gps_lon: str = ""            # GPS longitude (decimal degrees)
    exif_gps_link: str = ""           # Google Maps link if GPS present


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


@dataclass
class SingleVideoFrame:
    """One sampled frame from a single-video analysis."""
    frame_id: int
    timestamp: float
    motion_score: float
    avg_brightness: float
    avg_contrast: float
