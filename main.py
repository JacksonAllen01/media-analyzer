# =============================================================================
# main.py
# Entry point -- run this file to launch Media Analyzer.
#
# Usage:
#   python main.py
#
# Dependencies:
#   pip install opencv-python scikit-image Pillow imagehash numpy
#               ultralytics scikit-learn reportlab
# =============================================================================

from gui import MediaAnalyzerGUI

if __name__ == "__main__":
    app = MediaAnalyzerGUI()
    app.run()
