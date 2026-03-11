# Media Analyzer

A forensic media analysis tool for law enforcement investigations. All processing is 100% local. No data ever leaves the machine.

Media Analyzer helps investigators make sense of large collections of seized images and videos by automating comparison, grouping, flagging, and documentation.

**Image Analysis**

- Computes four independent perceptual hashes per image: pHash, aHash, dHash, and bHash (wavelet)
- Groups visually similar images using Union-Find with a tunable Hamming distance threshold
- Mean hash distance computed for every image within a group, so you can see how tight or loose a group is
- Brightness and contrast profiling per image
- Full EXIF metadata extraction: date taken, camera make and model, GPS coordinates, and a clickable Google Maps link when GPS is present
- HEIC/HEIF support for iPhone photos

**Video Analysis**

- Frame-by-frame comparison of video pairs using SSIM, MSE, and PSNR
- Per-frame motion scoring and brightness/contrast tracking
- Single-video frame profiling mode

**AI Tools** (optional, local only)

- LLaVA visual descriptions for individual images and video frames in natural language
- Automatic threat flagging: LLaVA scans images and video frames for weapons, drugs, contraband, and suspicious activity. Flagged content triggers a popup alert, turns the summary panel red, and highlights the row in the results table.
- Llama 3 plain-English forensic narrative via Ollama. No API keys, no internet required.

**Security and Documentation**

- All image thumbnails blurred by default. Password required to unlock full viewing.
- Double-click any row to preview the full image (authorized users only)
- Live threshold slider in the results window so investigators can tune grouping without re-running the analysis
- View Groups popup: scrollable side-by-side thumbnail view of every similarity group
- Export grouped images only: CSV and PDF exports can be filtered to grouped images exclusively
- One-click CSV export for every analysis mode
- One-click PDF report with cover page, statistics table, AI narrative section, and full findings table

---

## Requirements

- Python 3.9+
- Ollama (optional, only needed for AI summaries, LLaVA descriptions, and threat flagging)

---

## Installation

### 1. Clone the repo

```bash
git clone https://github.com/AI-Dev-WilCo-Capstone/media-analyzer.git
cd media-analyzer
```

### 2. Run the setup script

```bash
python install.py
```

This will verify your Python version, install all dependencies, check whether Ollama is running, and pull the required models if available. If anything cannot be done automatically, the script will tell you exactly what to do.

### Manual install

```bash
pip install -r requirements.txt
```

### Ollama setup (for AI features)

Ollama is a separate application that runs language models locally. It is not required to use Media Analyzer. All analysis, hashing, grouping, export, and blur features work without it. Ollama is only needed for AI summaries, LLaVA visual descriptions, and threat flagging.

**Install Ollama:**

| Platform | Instructions |
|---|---|
| macOS | `brew install ollama` or download from ollama.com |
| Windows | Download OllamaSetup.exe from ollama.com and run it |
| Linux | `curl -fsSL https://ollama.com/install.sh \| sh` |

**Pull the models and start the server:**

```bash
ollama pull llama3
ollama pull llava-llama3
ollama serve
```

Note: `llava-llama3` is the recommended vision model. If you already have Ollama running in the background (common on Windows), you do not need to run `ollama serve` manually.

---

## Usage

```bash
python main.py
```

### Analysis Modes

| Mode | What it does |
|---|---|
| Images (folder) | Analyzes all images in a folder. Hashing, grouping, EXIF, optional LLaVA flagging. |
| Single Image | Full profile of one image file including LLaVA visual description and threat flagging. |
| Videos (folder) | Pairwise comparison of all videos in a folder using SSIM, MSE, and PSNR. |
| Single Video | Per-frame analysis of one video file with optional LLaVA visual description. |

### Options

- **Interval (s):** how often to sample frames in video modes (default 0.5s)
- **Hash method:** which hash algorithm to use for grouping. Options are pHash, dHash, aHash, bHash, and Combined (min). See Hash Methods below.
- **Threshold:** maximum Hamming distance allowed between two images before they are considered too different to group (default 6, lower is stricter). This is a real unit: the number of bits allowed to differ between two 64-bit hash fingerprints.

### Hash Methods

| Method | Best for |
|---|---|
| pHash | General use. Tolerant of compression, minor resizing, slight color shifts. |
| dHash | Catching crops and translations. Sensitive to structural edge shifts. |
| aHash | Fast near-duplicate detection. Less precise, good for obvious matches. |
| bHash | Resistant to localized edits like watermarks or corner logos. |
| Combined (min) | Conservative grouping. Images group only if at least one hash considers them similar. |

The Hamming distance threshold has the same unit regardless of which method is active: bits differing out of 64.

### Live Threshold Slider

After running an image analysis, the results window includes a live threshold slider. Dragging it re-runs the Union-Find grouping instantly without re-analyzing the images. This lets investigators tune sensitivity in real time.

### View Groups

The View Groups button opens a scrollable popup showing every similarity group side by side as thumbnails, sorted by group size. Each group header shows the group ID, image count, and mean hash distance. Thumbnails are blurred until access is authorized.

### LLaVA Visual Analysis and Threat Flagging

After running any image or video analysis, the results window includes AI buttons in the summary panel:

- **Analyze with LLaVA** (image modes): sends the selected image to LLaVA for a detailed forensic description. LLaVA also scans for weapons, drugs, contraband, violence, and other suspicious content. If anything is flagged, a popup alert fires, the summary panel turns red, and the row in the results table is highlighted red.
- **Generate Visual Description** (video modes): LLaVA describes each sampled frame, then Llama 3 synthesizes all descriptions into a chronological forensic narrative.

All LLaVA processing runs locally through Ollama. No image data is transmitted externally.

### Authorization

Image content is blurred by default. Click Authorize Access in the results window and enter the investigator password to unlock viewing.

Default demo password: `wilco2025`

Change this in `constants.py` before deployment.

### Exporting

**CSV:** exports all columns for every analyzed image or video frame as a flat CSV file.

**PDF Report:** generates a formatted report with a cover banner, summary statistics, AI narrative (if available), and a full findings table.

**Export grouped images only:** checking this box before exporting filters both the CSV and PDF to only images that belong to a similarity group. If no groups exist at the current threshold, the export is cancelled with a message rather than producing an empty file. The PDF cover will note "Scope: Grouped Images Only" so the report is self-documenting.

---

## Project Structure

```
media_analyzer/
|
|-- main.py           Entry point
|-- install.py        One-shot setup script
|-- requirements.txt  Python dependencies
|
|-- constants.py      Password, Ollama config, column definitions, hash methods
|-- models.py         ImageItem, VideoFrameResult, SingleVideoFrame dataclasses
|-- analysis.py       All image/video analysis, hashing, bucketing, EXIF, CSV export
|-- ai.py             Llama 3 summary, LLaVA visual analysis and threat flagging
|-- pdf_report.py     PDF report generator (ReportLab)
|-- gui.py            ResultsWindow and main MediaAnalyzerGUI (tkinter)
```

---

## Similarity Metrics Explained

### Image Hashing

All four hashes produce a 64-bit fingerprint. The Hamming distance between two fingerprints counts how many bits differ. The threshold controls how many bit differences are tolerated before two images are considered different.

| Hash | Algorithm | Strengths |
|---|---|---|
| pHash | Discrete cosine transform on an 8x8 frequency grid | Best general-purpose, handles compression and minor edits |
| dHash | Compares adjacent pixel gradients left-to-right | Sensitive to crops and translations |
| aHash | Pixels above/below mean brightness | Fast, catches obvious duplicates |
| bHash | Wavelet transform | Robust against localized edits |

### Video Metrics

| Metric | What it measures | Scale |
|---|---|---|
| SSIM % | Structural similarity between frames | 100% identical, 0% completely different |
| MSE | Mean squared pixel error | Lower means more similar |
| PSNR (dB) | Signal-to-noise ratio | Above 50 dB is a near-exact copy |
| Motion Score | Frame activity vs blurred version | Higher means more movement or scene change |

---

## Dependencies

| Package | Purpose |
|---|---|
| opencv-python | Image and video I/O, color analysis, frame processing |
| Pillow | Image loading, thumbnail generation, blur |
| imagehash | pHash, aHash, dHash, bHash computation |
| scikit-image | SSIM calculation for video comparison |
| numpy | Numerical operations across pixel data |
| reportlab | PDF report generation |
| pillow-heif | HEIC/HEIF support for iPhone photos |
| exifread | EXIF metadata parsing |
| ollama (external) | Local LLM runtime for Llama 3 and LLaVA |

---
