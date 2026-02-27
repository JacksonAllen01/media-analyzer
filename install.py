# =============================================================================
# install.py
# One-shot setup script for Media Analyzer.
# Run this once before launching the app for the first time.
#
#   python install.py
#
# What it does:
#   1. Checks Python version (3.9+ required)
#   2. Installs all Python packages from requirements.txt
#   3. Checks if Ollama is installed and running
#   4. Pulls llama3 and llava-llama3 if Ollama is available
#   5. Prints a summary of what is ready and what needs manual action
# =============================================================================

import sys
import os
import subprocess
import platform
import shutil
import urllib.request
import json

# -- Colour helpers (work on all platforms) -----------------------------------
RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"

def ok(msg):     print(f"  {GREEN}ok{RESET}  {msg}")
def warn(msg):   print(f"  {YELLOW}warn{RESET}  {msg}")
def err(msg):    print(f"  {RED}err{RESET}  {msg}")
def info(msg):   print(f"  {CYAN}->{RESET}  {msg}")
def header(msg): print(f"\n{BOLD}{msg}{RESET}")


# =============================================================================
# STEP 1 -- Python version
# =============================================================================

def check_python():
    header("Checking Python version...")
    major, minor = sys.version_info[:2]
    version_str  = f"{major}.{minor}.{sys.version_info[2]}"
    if major < 3 or (major == 3 and minor < 9):
        err(f"Python {version_str} detected -- Media Analyzer requires Python 3.9 or newer.")
        err("Download the latest Python from https://python.org/downloads")
        sys.exit(1)
    ok(f"Python {version_str} -- good to go.")


# =============================================================================
# STEP 2 -- Python packages
# =============================================================================

def install_packages():
    header("Installing Python packages...")
    req_file = os.path.join(os.path.dirname(__file__), "requirements.txt")
    if not os.path.exists(req_file):
        err("requirements.txt not found. Make sure you are running this script "
            "from inside the media_analyzer folder.")
        sys.exit(1)

    info("Running: pip install -r requirements.txt")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", req_file, "--upgrade"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        err("pip install failed. Output:")
        print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
        print(result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
        sys.exit(1)

    packages = {
        "cv2":         "opencv-python",
        "PIL":         "Pillow",
        "numpy":       "numpy",
        "imagehash":   "imagehash",
        "skimage":     "scikit-image",
        "sklearn":     "scikit-learn",
        "ultralytics": "ultralytics",
        "reportlab":   "reportlab",
    }
    all_good = True
    for import_name, pip_name in packages.items():
        try:
            __import__(import_name)
            ok(pip_name)
        except ImportError:
            err(f"{pip_name} failed to install -- try running: pip install {pip_name}")
            all_good = False

    if not all_good:
        warn("Some packages failed. Try re-running this script or installing them manually.")
    else:
        ok("All Python packages installed successfully.")


# =============================================================================
# STEP 3 -- Ollama
# =============================================================================

OLLAMA_API_URL = "http://localhost:11434/api/tags"

def _ollama_binary_present() -> bool:
    return shutil.which("ollama") is not None

def _ollama_running() -> bool:
    try:
        with urllib.request.urlopen(OLLAMA_API_URL, timeout=3) as r:
            return r.status == 200
    except Exception:
        return False

def _model_pulled(model: str) -> bool:
    try:
        with urllib.request.urlopen(OLLAMA_API_URL, timeout=3) as r:
            data = json.loads(r.read().decode())
            return any(m.get("name", "").startswith(model)
                       for m in data.get("models", []))
    except Exception:
        return False

def _pull_model(model: str) -> bool:
    info(f"Pulling {model} -- this may take several minutes depending on model size...")
    result = subprocess.run(["ollama", "pull", model], capture_output=False)
    return result.returncode == 0

def check_ollama():
    header("Checking Ollama (required for AI summaries, LLaVA, and threat flagging)...")

    if not _ollama_binary_present():
        err("Ollama is not installed on this system.")
        _print_ollama_install_instructions()
        return

    ok("Ollama binary found on PATH.")

    if not _ollama_running():
        warn("Ollama is installed but the server is not running.")
        info("Start it with:  ollama serve")
        info("Then re-run this script to pull the models, or pull them manually:")
        info("  ollama pull llama3")
        info("  ollama pull llava-llama3")
        return

    ok("Ollama server is running.")

    # Check and pull llama3
    if _model_pulled("llama3"):
        ok("llama3 is already downloaded.")
    else:
        info("llama3 not found locally -- pulling now...")
        if _pull_model("llama3"):
            ok("llama3 downloaded successfully.")
        else:
            err("Failed to pull llama3.")
            info("Try manually:  ollama pull llama3")

    # Check and pull llava-llama3 (recommended over base llava)
    if _model_pulled("llava-llama3"):
        ok("llava-llama3 is already downloaded.")
    else:
        info("llava-llama3 not found locally -- pulling now (approx 5.5 GB)...")
        if _pull_model("llava-llama3"):
            ok("llava-llama3 downloaded successfully.")
        else:
            err("Failed to pull llava-llama3.")
            info("Try manually:  ollama pull llava-llama3")
            warn("Visual description and threat flagging will not work without this model.")


def _print_ollama_install_instructions():
    system = platform.system()
    print()
    print(f"  {BOLD}To install Ollama:{RESET}")
    if system == "Darwin":
        info("macOS:   brew install ollama")
        info("         OR download from https://ollama.com/download")
    elif system == "Windows":
        info("Windows: Download the installer from https://ollama.com/download")
        info("         Run OllamaSetup.exe, then restart this script.")
    else:
        info("Linux:   curl -fsSL https://ollama.com/install.sh | sh")
        info("         OR visit https://ollama.com/download")
    print()
    info("After installing, start the server with:  ollama serve")
    info("Then pull the models with:")
    info("  ollama pull llama3")
    info("  ollama pull llava-llama3")
    warn("AI features will not work until Ollama is set up, but all other "
         "features (analysis, PDF export, blur/auth) work fine without it.")


# =============================================================================
# STEP 4 -- Final summary
# =============================================================================

def print_summary():
    header("Setup complete!")
    print()
    print(f"  {BOLD}To launch Media Analyzer:{RESET}")
    print()
    print(f"    {CYAN}python main.py{RESET}")
    print()
    print(f"  {BOLD}Quick notes:{RESET}")
    info("The first time you enable YOLO object detection it will download")
    info("  model weights automatically (approx 6 MB for nano, 22 MB for small).")
    info("AI summaries and LLaVA require Ollama to be running (ollama serve).")
    info("LLaVA visual analysis uses the llava-llama3 model.")
    info(f"The demo password for image authorization is: {BOLD}wilco2025{RESET}")
    print()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print(f"\n{BOLD}{'=' * 60}{RESET}")
    print(f"{BOLD}  Media Analyzer -- Setup{RESET}")
    print(f"{BOLD}{'=' * 60}{RESET}")

    check_python()
    install_packages()
    check_ollama()
    print_summary()
