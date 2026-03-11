# ai.py
# YOLO object detection wrapper, Ollama/Llama3 AI summary generation,
# and LLaVA visual description with threat flagging.

import numpy as np
from dataclasses import asdict
from typing import List, Optional
from collections import defaultdict
from PIL import Image
import cv2

from constants import OLLAMA_URL, OLLAMA_CHAT_URL, OLLAMA_MODEL, OLLAMA_LLAVA
from models import ImageItem, VideoFrameResult


# YOLO DETECTOR

class YOLODetector:
    MODEL_OPTIONS = {
        "Nano: fastest, less accurate":  "yolov8n.pt",
        "Small: good balance":            "yolov8s.pt",
        "Medium: most accurate, slower":  "yolov8m.pt",
    }

    def __init__(self, model_size: str = "yolov8n.pt"):
        self.model_size = model_size
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from ultralytics import YOLO
                self._model = YOLO(self.model_size)
            except ImportError:
                raise RuntimeError(
                    "ultralytics is not installed. Run: pip install ultralytics"
                )

    def detect(self, image: np.ndarray, confidence: float = 0.35):
        """
        Run detection on a BGR numpy array (OpenCV format).
        Returns (labels_str, confidences_str) as comma-separated strings.
        """
        self._load()
        results = self._model(image, verbose=False)[0]
        labels, confs = [], []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            if conf >= confidence:
                labels.append(results.names[cls_id])
                confs.append(str(round(conf, 2)))
        return (
            ", ".join(labels) if labels else "",
            ", ".join(confs)  if confs  else "",
        )

    def detect_pil(self, pil_img: Image.Image, confidence: float = 0.35):
        """Convenience wrapper for PIL images."""
        bgr = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
        return self.detect(bgr, confidence)



# OLLAMA AI SUMMARY


def build_ai_summary(mode: str,
                     image_items: Optional[List[ImageItem]] = None,
                     video_results: Optional[List[VideoFrameResult]] = None) -> str:
    """
    Builds a structured findings prompt from analysis results,
    sends it to the local Ollama/Llama3 instance, and returns
    a plain-English forensic report.
    """
    import urllib.request, urllib.error, json

    # -- Build findings string -------------------------------------------------
    if mode in ("Image", "Single Image") and image_items:
        rows  = [asdict(it) for it in image_items]
        total = len(rows)

        # Correctly compute groups... -1 means unique, anything else is a group bucket
        bc: dict = defaultdict(int)
        for r in rows:
            bid = r.get("bucket_phash", -1)
            if bid != -1:
                bc[bid] += 1
        n_groups    = len(bc)
        n_in_groups = sum(bc.values())
        n_unique    = total - n_in_groups
        largest     = max(bc.values(), default=0)

        bvals = [r["avg_brightness"] for r in rows if "avg_brightness" in r]
        avg_b = round(sum(bvals) / len(bvals), 1) if bvals else 0

        findings = (
            f"Mode: Image analysis\n"
            f"Total images analyzed: {total}\n"
            f"Images with no similarity match (unique): {n_unique}\n"
            f"pHash similarity groups found: {n_groups}\n"
            f"Images that belong to a similarity group: {n_in_groups}\n"
            f"Largest group size: {largest if largest > 0 else 'N/A'}\n"
            f"Average image brightness (0-255): {avg_b}\n"
        )

    elif mode in ("Video", "Single Video") and video_results:
        rows    = [asdict(r) for r in video_results]
        total   = len(rows)
        pairs   = list({(r["video_pair_1"], r["video_pair_2"]) for r in rows})
        sims    = [r["similarity_ssim"] for r in rows]
        motions = [r["motion_score"]    for r in rows]

        findings = (
            f"Mode: Video analysis\n"
            f"Video pairs compared: {len(pairs)}\n"
            f"Total frame comparisons: {total}\n"
            f"Average SSIM similarity: {round(sum(sims)/len(sims), 1)}%\n"
            f"Max similarity: {round(max(sims), 1)}%   Min: {round(min(sims), 1)}%\n"
            f"Average motion score: {round(sum(motions)/len(motions), 2)}\n"
        )
    else:
        return "No data available to summarize."

    # -- Build prompt ----------------------------------------------------------
    prompt = (
        "You are a forensic media analyst assistant helping law enforcement investigators "
        "understand the results of an automated media analysis tool.\n\n"
        "The FINDINGS section below contains the actual recorded values from this specific analysis. "
        "Use those exact numbers in your report. Do not use placeholders like [insert value], as "
        "the real numbers are already provided and you must reference them directly.\n\n"
        "Write a thorough plain-English report interpreting every data point. "
        "For each metric, state the actual recorded value, explain what it measures, "
        "and describe what significance it may have for an investigation. "
        "Write in clear paragraphs. Do not use bullet points. "
        "Be thorough but accessible, as the reader may not have a technical background.\n\n"
        f"FINDINGS:\n{findings}\n\n"
        "Using only the values in the FINDINGS above, write the forensic report now:\n\n"
        "FORENSIC MEDIA ANALYSIS REPORT:"
    )

    # -- Send to Ollama --------------------------------------------------------
    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_ctx": 2048,
        }
    }).encode()

    req = urllib.request.Request(
        OLLAMA_URL, data=payload,
        headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            return json.loads(resp.read().decode()).get("response", "").strip()
    except urllib.error.HTTPError as e:
        try:
            detail = e.read().decode("utf-8", errors="replace")
        except Exception:
            detail = str(e)
        raise RuntimeError(
            f"Ollama returned HTTP {e.code}.\n"
            f"Detail: {detail}\n\n"
            f"Try running: ollama pull llama3"
        )
    except Exception as e:
        raise RuntimeError(
            f"Could not reach Ollama at {OLLAMA_URL}.\n"
            f"Make sure Ollama is running (ollama serve) and llama3 is pulled.\n\nDetail: {e}"
        )



# LLAVA VISUAL DESCRIPTION AND THREAT FLAGGING


def _encode_pil_for_llava(pil_img: Image.Image) -> str:
    """Resize to max 640px wide and base64-encode as JPEG for Ollama."""
    import base64, io
    img = pil_img.convert("RGB")
    if img.width > 640:
        ratio = 640 / img.width
        img = img.resize((640, int(img.height * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _send_to_llava(b64: str, prompt: str, timeout: int = 300) -> str:
    """
    Send a base64 image and prompt to a vision model via the generate endpoint.
    Tries OLLAMA_LLAVA first. If that times out or fails, automatically falls
    back to moondream, which is smaller and faster on less powerful machines.
    """
    import urllib.request, urllib.error, json

    def _try_model(model_name: str) -> str:
        payload = json.dumps({
            "model": model_name,
            "prompt": prompt,
            "images": [b64],
            "stream": False,
            "options": {"temperature": 0.2, "num_ctx": 2048},
        }).encode()
        req = urllib.request.Request(
            OLLAMA_URL, data=payload,
            headers={"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
            return data.get("response", "").strip()

    # First attempt with preferred model
    try:
        return _try_model(OLLAMA_LLAVA)
    except Exception as primary_err:
        # If primary model timed out or failed, try moondream as fallback
        if OLLAMA_LLAVA != "moondream":
            try:
                return _try_model("moondream")
            except Exception:
                pass
        # Both failed -- raise the original error
        raise RuntimeError(
            f"Could not reach Ollama at {OLLAMA_URL}.\n"
            f"Make sure Ollama is running and a vision model is pulled "
            f"(llava-llama3 or moondream).\n\nDetail: {primary_err}"
        )


def analyze_image_llava(pil_img: Image.Image, filename: str) -> dict:
    """
    Send a single still image to LLaVA for forensic description and threat flagging.

    Returns a dict with:
        description  (str)  -- detailed plain-English description of the image
        flagged      (bool) -- True if weapons, drugs, contraband, or suspicious
                               activity were detected
        flag_details (str)  -- what specifically was flagged, or empty string
    """
    b64 = _encode_pil_for_llava(pil_img)

    prompt = (
        "You are a law enforcement forensic investigator examining evidence. "
        "Look at this image carefully and do two things: "
        "First, describe in detail what you see: any people, their clothing and activity, "
        "any objects, the setting, lighting, and any visible text. "
        "Second, state whether the image contains anything of serious investigative concern "
        "such as weapons, drugs, contraband, or violence. "
        "Be factual and specific. Write in plain sentences."
    )

    raw = _send_to_llava(b64, prompt)

    # -- Clean tokenizer artifacts out of the response ------------------------
    cleaned = []
    for line in raw.splitlines():
        stripped = line.strip()
        if (stripped
                and not all(c in "<>/ \t" for c in stripped)
                and stripped not in ("<unk>", "<s>", "</s>", "<|im_end|>")):
            cleaned.append(stripped)
    description = " ".join(cleaned).strip() or "(no description returned)"

    # -- Flag detection via keyword scan --------------------------------------
    flagged      = False
    flag_details = ""
    flag_keywords = {
        "weapon":      "weapon detected",
        "firearm":     "firearm detected",
        "gun":         "firearm detected",
        "knife":       "knife detected",
        "drug":        "drug-related content detected",
        "narcotic":    "drug-related content detected",
        "contraband":  "contraband detected",
        "violence":    "violent content detected",
        "blood":       "possible violent content",
        "altercation": "altercation detected",
        "fight":       "altercation detected",
        "suspicious":  "suspicious activity noted",
        "explosive":   "explosive device detected",
    }
    lower = description.lower()
    hits  = []
    for kw, label in flag_keywords.items():
        if kw in lower and label not in hits:
            hits.append(label)
    if hits:
        flagged      = True
        flag_details = "; ".join(hits)

    return {
        "description":  description,
        "flagged":      flagged,
        "flag_details": flag_details,
    }


def describe_frame_llava(pil_img: Image.Image,
                         timestamp: float,
                         frame_id: int) -> str:
    """
    Send a single video frame to LLaVA and get a forensic description.
    Returns the description string.
    """
    b64 = _encode_pil_for_llava(pil_img)
    prompt = (
        "You are assisting a law enforcement forensic investigator. "
        "Describe this video frame in precise, objective detail as if writing a formal report. "
        "Include any people present (clothing, appearance, position, activity), "
        "any objects of potential investigative interest, the environment and setting, "
        "lighting conditions, and any visible text or signage. "
        "Be specific and factual. Do not speculate beyond what is visible. "
        "Write in complete sentences, 3-5 sentences maximum."
    )
    result = _send_to_llava(b64, prompt)
    return result if result else "(no description returned)"


def build_llava_narrative(frames_data: list,
                          filename: str,
                          progress_cb=None) -> str:
    """
    Describes each sampled frame with LLaVA, then sends all descriptions to
    Llama 3 to synthesize into a coherent forensic narrative.

    frames_data: list of (timestamp, frame_id, pil_img) tuples
    progress_cb: optional callable(current, total, label)
    Returns the final synthesized narrative string.
    """
    if not frames_data:
        return "No frames available to describe."

    # -- Step 1: LLaVA describes each frame -----------------------------------
    frame_descriptions = []
    total = len(frames_data)

    for i, (timestamp, frame_id, pil_img) in enumerate(frames_data):
        if progress_cb:
            progress_cb(i + 1, total, f"Describing frame {frame_id} ({timestamp}s)...")
        try:
            desc = describe_frame_llava(pil_img, timestamp, frame_id)
        except RuntimeError as e:
            raise RuntimeError(str(e))
        frame_descriptions.append(
            f"[{timestamp}s - Frame {frame_id}]: {desc}"
        )

    # -- Step 2: Llama 3 synthesizes all descriptions into a narrative --------
    if progress_cb:
        progress_cb(total, total, "Synthesizing narrative with Llama 3...")

    combined = "\n\n".join(frame_descriptions)
    prompt = (
        "You are a forensic media analyst writing an official report for law enforcement. "
        "Below are per-frame descriptions of a video file. Each entry shows the timestamp "
        "and a visual description of what was observed in that frame.\n\n"
        "Your task is to synthesize these into a single, coherent forensic narrative. "
        "Describe what happens across the video chronologically; who is present, what they "
        "are doing, how the scene changes over time, and any details of investigative "
        "significance. Write in clear, formal prose paragraphs. Do not use bullet points. "
        "Be specific about timing where relevant (e.g. 'At approximately 4 seconds...'). "
        "If multiple frames show the same thing, summarize rather than repeat. "
        "End with a brief overall summary of the video's content.\n\n"
        f"VIDEO FILE: {filename}\n\n"
        f"FRAME DESCRIPTIONS:\n{combined}\n\n"
        "FORENSIC NARRATIVE:"
    )

    import urllib.request, urllib.error, json
    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_ctx": 4096,
        }
    }).encode()

    req = urllib.request.Request(
        OLLAMA_URL, data=payload,
        headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            return json.loads(resp.read().decode()).get("response", "").strip()
    except urllib.error.HTTPError as e:
        try:
            detail = e.read().decode("utf-8", errors="replace")
        except Exception:
            detail = str(e)
        raise RuntimeError(
            f"Llama 3 synthesis returned HTTP {e.code}.\nDetail: {detail}"
        )
    except Exception as e:
        raise RuntimeError(
            f"Could not reach Ollama during synthesis.\nDetail: {e}"
        )