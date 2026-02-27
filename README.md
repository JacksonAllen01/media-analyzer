# Media Analyzer

A forensic media analysis tool built for law enforcement investigations. 100% local. No data ever leaves the machine.

Developed as a capstone project in partnership with the **Williamson County District Attorney's Office** (Georgetown, TX).

---

## What It Does

Media Analyzer helps investigators make sense of large collections of images and videos by automating the tedious work of comparison, grouping, flagging, and documentation.

**Image Analysis**
- Computes three independent perceptual hashes (pHash, aHash, dHash) and groups visually similar images automatically
- DBSCAN clustering as a second-pass similarity check independent of hashing
- Bhattacharyya color histogram distance between grouped images
- K-means dominant color extraction per image, displayed as color swatches in the results table
- Brightness and contrast profiling

**Video Analysis**
- Frame-by-frame comparison of video pairs using SSIM, MSE, and PSNR
- Per-frame motion scoring, brightness, contrast, and dominant color tracking
- Single-video frame profiling mode

**AI Tools** *(optional, local only)*
- YOLO v8 object detection (nano / small / medium models)
- Llama 3 plain-English forensic narrative via Ollama. No API keys, no internet required.
- LLaVA visual descriptions for images and video frames in natural language
- **Automatic threat flagging** -- LLaVA scans images and video frames for weapons, drugs, contraband, and suspicious activity. Flagged content triggers a popup alert, turns the summary panel red, and highlights the row in the results table.

**Security and Documentation**
- All image thumbnails blurred by default. Password required to unlock viewing.
- Double-click any row to preview the full image (authorized users only)
- One-click CSV export for every analysis mode
- One-click PDF report with cover page, statistics table, AI narrative, and full findings

---

## Screenshots

*(Add screenshots here once you have them. Drag and drop into the GitHub editor.)*

---

## Requirements

- **Python 3.9+**
- **Ollama** *(optional -- only needed for AI summaries, visual descriptions, and threat flagging)*

---

## Installation

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/media-analyzer.git
cd media-analyzer
```

### 2. Run the setup script

```bash
python install.py
```

This will:
- Verify your Python version
- Install all Python dependencies automatically
- Check whether Ollama is installed and running
- Pull llama3 and llava-llama3 if Ollama is available

That's it. If anything can't be done automatically, the script will tell you exactly what to do.

---

### Manual install (if you prefer)

```bash
pip install -r requirements.txt
```

---

### Ollama setup *(for AI features)*

Ollama is a separate application that runs language models locally. It is **not required** to use Media Analyzer. All analysis, export, and blur features work without it. Ollama is only needed for AI summaries, LLaVA visual descriptions, and threat flagging.

**Install Ollama:**

| Platform | Instructions |
|---|---|
| macOS | `brew install ollama` or [download from ollama.com](https://ollama.com/download) |
| Windows | [Download OllamaSetup.exe](https://ollama.com/download) and run it |
| Linux | `curl -fsSL https://ollama.com/install.sh \| sh` |

**Then pull the models and start the server:**

```bash
ollama pull llama3
ollama pull llava-llama3
ollama serve
```

**Note:** `llava-llama3` is the recommended vision model. The older `llava` build may produce garbled output depending on your Ollama version. If you already have Ollama running in the background (which is common on Windows), you do not need to run `ollama serve` manually.

---

## Usage

```bash
python main.py
```

### Modes

| Mode | What it does |
|---|---|
| **Images (folder)** | Analyzes all images in a folder. Hashing, grouping, color analysis, optional LLaVA flagging. |
| **Single Image** | Full profile of one image file, including LLaVA visual description and threat flagging. |
| **Videos (folder)** | Pairwise comparison of all videos in a folder. |
| **Single Video** | Per-frame analysis of one video file with optional LLaVA visual description. |

### Options

- **Interval (s):** how often to sample frames in video modes (default 0.5s)
- **pHash threshold:** how similar two images must be to be grouped (default 6; lower = stricter)
- **YOLO detection:** enable AI object detection with your choice of model size

### LLaVA Visual Analysis and Threat Flagging

After running any image or video analysis, the results window includes AI buttons in the summary panel:

- **"Analyze with LLaVA"** (image modes): sends the image to LLaVA for a detailed forensic description. LLaVA also scans for weapons, drugs, contraband, violence, and other suspicious content. If anything is flagged, a popup alert fires immediately, the summary panel turns red, and the image's row in the results table is highlighted red.
- **"Generate Visual Description"** (video modes): LLaVA describes each sampled frame, then Llama 3 synthesizes all descriptions into a chronological forensic narrative.

All LLaVA processing runs locally through Ollama. No image data is transmitted externally.

### Authorization

Image content is blurred by default. Click **Authorize Access** in the results window and enter the investigator password to unlock viewing.

> Default demo password: `wilco2025`
> Change this in `constants.py` before deployment.

---

## Project Structure

```
media_analyzer/
|
|-- main.py           Entry point. Run this.
|-- install.py        One-shot setup script
|-- requirements.txt  Python dependencies
|
|-- constants.py      Password, Ollama config, column definitions
|-- models.py         ImageItem, VideoFrameResult, SingleVideoFrame dataclasses
|-- analysis.py       Image/video analysis, metrics, hashing, bucketing, CSV export
|-- ai.py             YOLO, Llama 3 summary, LLaVA visual analysis and threat flagging
|-- pdf_report.py     PDF report generator (ReportLab)
|-- gui.py            ResultsWindow and MediaAnalyzerGUI (tkinter)
```

---

## Similarity Metrics Explained

### Image Metrics

| Metric | What it measures | Investigative use |
|---|---|---|
| **pHash** | Perceptual hash, tolerant of resize and compression | Groups visually similar images even across different saves |
| **aHash** | Average hash, fast, good for near-duplicates | Cross-checks pHash groupings |
| **dHash** | Difference hash, edge-sensitive | Catches edits that preserve average tone but change content |
| **DBSCAN** | Density-based clustering on aHash and dHash | Finds groups pHash misses. -1 means unique image with no close matches. |
| **Bhattacharyya** | Color histogram distance | Measures how similarly colored images within a group are |

### Video Metrics

| Metric | What it measures | Scale |
|---|---|---|
| **SSIM %** | Structural similarity | 100% = identical, 0% = completely different |
| **MSE** | Mean squared pixel error | Lower = more similar |
| **PSNR (dB)** | Signal-to-noise ratio | 100 dB = identical, above 50 dB = near-exact copy |
| **Motion Score** | Frame activity | Higher = more movement |

---

## Dependencies

| Package | Purpose |
|---|---|
| `opencv-python` | Image/video I/O, color analysis, frame processing |
| `Pillow` | Image loading, thumbnail generation, blur |
| `imagehash` | pHash, aHash, dHash computation |
| `scikit-image` | SSIM calculation |
| `scikit-learn` | DBSCAN clustering |
| `numpy` | Numerical operations across pixel data |
| `ultralytics` | YOLOv8 object detection |
| `reportlab` | PDF report generation |
| `ollama` *(external)* | Local LLM runtime for Llama 3 and LLaVA |

---

## Legal and Privacy

- All processing is **100% local**. No images, videos, or analysis results are ever transmitted to any external server.
- Original files are **never modified**.
- Image content is **blurred by default** and requires password authorization to view.
- AI-generated descriptions and threat flags are analytical aids only. They require human verification before use in legal proceedings.
- Designed for use by authorized law enforcement personnel only.

---

## Acknowledgements

Built as part of the Texas State University Computer Science capstone program in collaboration with the Williamson County District Attorney's Office.

Previous capstone cohort tools that informed this work: the Virgil audio transcription tool and the original image comparison tool (Spring 2025).
