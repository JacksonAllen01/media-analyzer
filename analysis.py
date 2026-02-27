# =============================================================================
# analysis.py
# Image and video analysis: hashing, color, similarity metrics, bucketing,
# VideoAnalyzer, SingleVideoAnalyzer, and CSV export.
# =============================================================================

import os
import csv
import numpy as np
import cv2
from pathlib import Path
from typing import List, Optional, Callable
from dataclasses import asdict
from collections import defaultdict
from PIL import Image
import imagehash
from skimage.metrics import structural_similarity as ssim

from constants import IMG_EXTS, VID_EXTS
from models import ImageItem, VideoFrameResult, SingleVideoFrame
from ai import YOLODetector


# =============================================================================
# SIMILARITY METRICS
# =============================================================================

def compute_mse(f1: np.ndarray, f2: np.ndarray) -> float:
    """Mean Squared Error between two same-size grayscale frames. Lower = more similar."""
    return float(np.mean((f1.astype(np.float32) - f2.astype(np.float32)) ** 2))


def compute_psnr(mse: float, max_val: float = 255.0) -> float:
    """Peak Signal-to-Noise Ratio in dB. Returns 100.0 for identical frames."""
    if mse < 1e-10:
        return 100.0
    return round(20 * np.log10(max_val / np.sqrt(mse)), 2)


def compute_bhattacharyya(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Bhattacharyya distance between color histograms of two BGR images.
    Lower = more similar. 0.0 = identical color distributions.
    """
    dist = 0.0
    for ch in range(3):
        h1 = cv2.calcHist([img1], [ch], None, [64], [0, 256])
        h2 = cv2.calcHist([img2], [ch], None, [64], [0, 256])
        cv2.normalize(h1, h1)
        cv2.normalize(h2, h2)
        dist += cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA)
    return round(dist / 3, 4)


# =============================================================================
# COLOR / VISUAL ANALYSIS
# =============================================================================

def analyze_image_color(pil_img: Image.Image, n_colors: int = 3):
    """
    Returns (avg_brightness, avg_contrast, [hex_color_1, hex_color_2, hex_color_3])
    using a 200px thumbnail for speed.
    """
    thumb = pil_img.copy()
    thumb.thumbnail((200, 200))
    rgb  = np.array(thumb.convert("RGB"), dtype=np.float32)
    gray = np.array(thumb.convert("L"),   dtype=np.float32)

    avg_brightness = float(np.mean(gray))
    avg_contrast   = float(np.std(gray))

    pixels   = rgb.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    k        = min(n_colors, len(pixels))
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 3, cv2.KMEANS_RANDOM_CENTERS)

    counts = np.bincount(labels.flatten())
    order  = np.argsort(-counts)
    hex_colors = []
    for idx in order[:n_colors]:
        r, g, b = [int(c) for c in centers[idx]]
        hex_colors.append(f"#{r:02X}{g:02X}{b:02X}")
    while len(hex_colors) < n_colors:
        hex_colors.append("")

    return round(avg_brightness, 2), round(avg_contrast, 2), hex_colors


# =============================================================================
# IMAGE HANDLING
# =============================================================================

def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def aspect_ratio_str(w: int, h: int) -> str:
    if w == 0 or h == 0:
        return "0:0"
    g = _gcd(w, h)
    return f"{w // g}:{h // g}"


def iter_image_paths(root: Path):
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p


def analyze_single_image(img_path: Path,
                         detector: Optional[YOLODetector] = None) -> Optional[ImageItem]:
    """Analyze one image file and return an ImageItem, or None on failure."""
    try:
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            w, h = im.size
            ph   = str(imagehash.phash(im))
            ah   = str(imagehash.average_hash(im))
            dh   = str(imagehash.dhash(im))
            brightness, contrast, dom_colors = analyze_image_color(im)
            det_objects = det_conf = ""
            if detector:
                det_objects, det_conf = detector.detect_pil(im)
    except Exception:
        return None

    return ImageItem(
        id=1,
        filename=img_path.name,
        path=str(img_path),
        ext=img_path.suffix.lower(),
        bytes=img_path.stat().st_size,
        width=w,
        height=h,
        aspect_ratio=aspect_ratio_str(w, h),
        color_mode="RGB",
        phash=ph,
        ahash=ah,
        dhash=dh,
        avg_brightness=brightness,
        avg_contrast=contrast,
        dominant_color_1=dom_colors[0],
        dominant_color_2=dom_colors[1],
        dominant_color_3=dom_colors[2],
        detected_objects=det_objects,
        detection_confidence=det_conf,
    )


def build_image_items(root_dir: str,
                      progress_cb: Optional[Callable] = None,
                      detector: Optional[YOLODetector] = None) -> List[ImageItem]:
    """Scan a folder recursively and return a list of ImageItems."""
    root      = Path(root_dir).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Invalid folder: {root}")
    all_paths = list(iter_image_paths(root))
    total     = len(all_paths)
    items: List[ImageItem] = []

    for i, img_path in enumerate(all_paths):
        item = analyze_single_image(img_path, detector=detector)
        if item is None:
            continue
        item.id = len(items) + 1
        items.append(item)
        if progress_cb:
            progress_cb(i + 1, total, img_path.name)

    return items


def assign_image_buckets(items: List[ImageItem], threshold: int = 6):
    """
    Two-pass similarity grouping:
      Pass 1 -- Union-Find on pHash distance    -> bucket_phash
      Pass 2 -- DBSCAN on aHash+dHash vectors   -> bucket_dbscan
    Also computes mean Bhattacharyya distance for images within pHash groups.
    """
    if not items:
        return

    # -- Pass 1: pHash Union-Find ---------------------------------------------
    hashes = [imagehash.hex_to_hash(it.phash) for it in items]
    parent = list(range(len(items)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if hashes[i] - hashes[j] <= threshold:
                union(i, j)

    root_to_bucket: dict = {}
    next_bucket = 1
    for i in range(len(items)):
        r = find(i)
        if r not in root_to_bucket:
            root_to_bucket[r] = next_bucket
            next_bucket += 1
        items[i].bucket_phash = root_to_bucket[r]

    # -- Pass 2: DBSCAN on combined hash features -----------------------------
    try:
        from sklearn.cluster import DBSCAN

        def hash_to_bits(h_str: str) -> np.ndarray:
            h = imagehash.hex_to_hash(h_str)
            return h.hash.flatten().astype(np.float32)

        features = np.array([
            np.concatenate([hash_to_bits(it.ahash), hash_to_bits(it.dhash)])
            for it in items
        ])
        db = DBSCAN(eps=10, min_samples=2, metric="hamming").fit(features)
        for i, label in enumerate(db.labels_):
            items[i].bucket_dbscan = int(label)
    except ImportError:
        pass   # sklearn not installed -- skip silently

    # -- Bhattacharyya distance within pHash groups ---------------------------
    bucket_members: dict = defaultdict(list)
    for it in items:
        bucket_members[it.bucket_phash].append(it)

    for members in bucket_members.values():
        if len(members) < 2:
            continue
        bgr_imgs = []
        for m in members:
            try:
                img = cv2.imread(m.path)
                if img is not None:
                    bgr_imgs.append((m, img))
            except Exception:
                pass
        for m, img in bgr_imgs:
            dists = [compute_bhattacharyya(img, other_img)
                     for om, other_img in bgr_imgs if om is not m]
            if dists:
                m.hist_bhattacharyya = round(sum(dists) / len(dists), 4)


def write_image_csv(items: List[ImageItem], out_csv: str):
    out_path = Path(out_csv).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(x) for x in items]
    if not rows:
        raise ValueError("No images to write.")
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# =============================================================================
# VIDEO HANDLING
# =============================================================================

class VideoAnalyzer:
    """Compares pairs of videos frame-by-frame using SSIM, MSE, and PSNR."""

    def __init__(self):
        self.results: List[VideoFrameResult] = []

    @staticmethod
    def frame_to_gray(frame, size=(300, 300)):
        return cv2.cvtColor(cv2.resize(frame, size), cv2.COLOR_BGR2GRAY)

    @staticmethod
    def motion_score(frame: np.ndarray) -> float:
        return float(np.mean(cv2.absdiff(frame, cv2.blur(frame, (5, 5)))))

    def compare_videos(self, path1: str, path2: str,
                       interval: float = 0.5,
                       progress_cb: Optional[Callable] = None,
                       detector: Optional[YOLODetector] = None):
        cap1 = cv2.VideoCapture(path1)
        cap2 = cv2.VideoCapture(path2)
        fps1 = cap1.get(cv2.CAP_PROP_FPS)
        fps2 = cap2.get(cv2.CAP_PROP_FPS)
        if fps1 == 0 or fps2 == 0:
            cap1.release(); cap2.release()
            raise ValueError("Could not read FPS from one or both videos.")

        dur1        = cap1.get(cv2.CAP_PROP_FRAME_COUNT) / fps1
        dur2        = cap2.get(cv2.CAP_PROP_FRAME_COUNT) / fps2
        total_steps = int(min(dur1, dur2) / interval) + 1
        frame_count = 0

        while True:
            time_ms = frame_count * interval * 1000
            cap1.set(cv2.CAP_PROP_POS_MSEC, time_ms)
            cap2.set(cv2.CAP_PROP_POS_MSEC, time_ms)
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            if not ret1 or not ret2:
                break

            gray1      = self.frame_to_gray(frame1)
            gray2      = self.frame_to_gray(frame2)
            ssim_score = ssim(gray1, gray2)
            mse_val    = compute_mse(gray1, gray2)
            psnr_val   = compute_psnr(mse_val)
            motion     = self.motion_score(gray1)

            det_objects = det_conf = ""
            if detector:
                det_objects, det_conf = detector.detect(frame1)

            self.results.append(VideoFrameResult(
                video_pair_1=os.path.basename(path1),
                video_pair_2=os.path.basename(path2),
                frame_id=frame_count,
                timestamp=round(frame_count * interval, 2),
                similarity_ssim=round(ssim_score * 100, 2),
                similarity_mse=round(mse_val, 2),
                similarity_psnr=psnr_val,
                motion_score=round(motion, 2),
                detected_objects=det_objects,
                detection_confidence=det_conf,
            ))
            frame_count += 1
            if progress_cb:
                progress_cb(frame_count, total_steps,
                            f"{os.path.basename(path1)} vs {os.path.basename(path2)}")

        cap1.release()
        cap2.release()

    def write_csv(self, out_csv: str) -> Path:
        out_path = Path(out_csv).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.results:
            raise ValueError("No video results to write.")
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(self.results[0]).keys()))
            writer.writeheader()
            for r in self.results:
                writer.writerow(asdict(r))
        return out_path


class SingleVideoAnalyzer:
    """Analyzes a single video file: metadata plus per-frame motion, color, and YOLO."""

    def __init__(self):
        self.filename: str = ""
        self.duration: float = 0.0
        self.fps: float = 0.0
        self.width: int = 0
        self.height: int = 0
        self.total_frames: int = 0
        self.codec: str = ""
        self.frames: List[SingleVideoFrame] = []
        self._source_path: str = ""   # full path to original file

    def analyze(self, path: str, interval: float = 0.5,
                progress_cb: Optional[Callable] = None,
                detector: Optional[YOLODetector] = None):
        cap = cv2.VideoCapture(path)
        self._source_path = path
        self.filename     = os.path.basename(path)
        self.fps          = cap.get(cv2.CAP_PROP_FPS) or 0.0
        self.width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration     = round(self.total_frames / self.fps, 2) if self.fps > 0 else 0.0

        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        self.codec = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)]).strip()

        if self.fps == 0:
            cap.release()
            raise ValueError("Could not read FPS from video.")

        total_steps = int(self.duration / interval) + 1
        frame_count = 0

        while True:
            time_ms = frame_count * interval * 1000
            cap.set(cv2.CAP_PROP_POS_MSEC, time_ms)
            ret, frame = cap.read()
            if not ret:
                break

            motion    = float(np.mean(cv2.absdiff(frame, cv2.blur(frame, (5, 5)))))
            pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            brightness, contrast, dom_colors = analyze_image_color(pil_frame)

            det_objects = det_conf = ""
            if detector:
                det_objects, det_conf = detector.detect(frame)

            self.frames.append(SingleVideoFrame(
                frame_id=frame_count,
                timestamp=round(frame_count * interval, 2),
                motion_score=round(motion, 2),
                avg_brightness=brightness,
                avg_contrast=contrast,
                dominant_color_1=dom_colors[0],
                dominant_color_2=dom_colors[1],
                dominant_color_3=dom_colors[2],
                detected_objects=det_objects,
                detection_confidence=det_conf,
            ))
            frame_count += 1
            if progress_cb:
                progress_cb(frame_count, total_steps, f"Frame {frame_count}")

        cap.release()

    def write_csv(self, out_csv: str) -> Path:
        out_path = Path(out_csv).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.frames:
            raise ValueError("No frames to write.")
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(self.frames[0]).keys()))
            writer.writeheader()
            for fr in self.frames:
                writer.writerow(asdict(fr))
        return out_path
