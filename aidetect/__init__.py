"""aidetect — deterministic detection of Adobe AI provenance in images."""
from .detector import detect_bytes, detect_file
from .models import DetectionResult, Evidence, Verdict

__all__ = ["detect_file", "detect_bytes", "DetectionResult", "Evidence", "Verdict"]
__version__ = "1.0.0"
