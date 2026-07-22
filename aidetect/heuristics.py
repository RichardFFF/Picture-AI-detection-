"""Three classic pixel-forensics tools: ELA, spectral analysis, noise-floor map.

These are ADVISORY heuristics — statistical signals, not proof. They run only
when explicitly requested (CLI --heuristics / UI checkbox) and never change
the deterministic provenance verdict. Each returns a score in [0, 1] where
higher means "more consistent with AI generation or local manipulation".
"""
from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter

ANALYSIS_SIZE = 512  # cap the long edge for speed; scores are size-normalized
BLOCK = 16


@dataclass
class HeuristicResult:
    tool: str
    score: float  # 0..1, higher = more suspicious
    summary: str

    def to_dict(self) -> dict:
        return {"tool": self.tool, "score": round(self.score, 3), "summary": self.summary}


def _load_gray(path: str) -> tuple[Image.Image, np.ndarray]:
    img = Image.open(path).convert("RGB")
    if max(img.size) > ANALYSIS_SIZE:
        img.thumbnail((ANALYSIS_SIZE, ANALYSIS_SIZE))
    gray = np.asarray(img.convert("L"), dtype=np.float64)
    return img, gray


def _block_stats(arr: np.ndarray) -> np.ndarray:
    """Mean of each BLOCK x BLOCK tile."""
    h, w = arr.shape
    h, w = h - h % BLOCK, w - w % BLOCK
    if h == 0 or w == 0:
        return np.array([[arr.mean()]])
    tiles = arr[:h, :w].reshape(h // BLOCK, BLOCK, w // BLOCK, BLOCK)
    return tiles.mean(axis=(1, 3))


def error_level_analysis(path: str) -> HeuristicResult:
    """ELA: recompress at a known quality and study the residual.

    Regions that were pasted/generated at a different compression history
    stand out from their surroundings; a high spread of per-block error
    levels indicates a spliced or locally regenerated area.
    """
    img, _ = _load_gray(path)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    buf.seek(0)
    resaved = Image.open(buf).convert("RGB")
    diff = np.abs(
        np.asarray(img, dtype=np.float64) - np.asarray(resaved, dtype=np.float64)
    ).mean(axis=2)
    blocks = _block_stats(diff)
    spread = float(blocks.std())
    mean = float(blocks.mean()) + 1e-6
    # Relative spread: uniform compression history -> low; splices -> high.
    score = min(1.0, (spread / mean) / 2.5)
    return HeuristicResult(
        "ELA",
        score,
        f"error-level block spread/mean = {spread:.2f}/{mean:.2f} "
        f"({'uneven — possible local edit' if score > 0.5 else 'uniform compression history'})",
    )


def spectral_analysis(path: str) -> HeuristicResult:
    """FFT spectrum shape: generative upsamplers leave periodic peaks and an
    unnaturally clean high-frequency band compared with camera images."""
    _, gray = _load_gray(path)
    f = np.abs(np.fft.fftshift(np.fft.fft2(gray - gray.mean())))
    h, w = f.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(yy - cy, xx - cx)
    rmax = r.max() + 1e-6
    low = f[r < 0.25 * rmax].mean()
    high = f[r > 0.60 * rmax].mean() + 1e-9
    ratio = low / high
    # Camera JPEGs land roughly in ratio 20-400; synthetic/AI renders are often
    # far outside: extremely clean high band (huge ratio) or periodic peaks.
    peak = float(f[r > 0.60 * rmax].max() / high)  # peak-to-mean in the high band
    peakiness = min(1.0, peak / 500.0)
    cleanness = min(1.0, max(0.0, (ratio - 400.0) / 4000.0))
    score = max(peakiness * 0.4, cleanness)
    return HeuristicResult(
        "Spectral",
        score,
        f"low/high frequency energy ratio = {ratio:.0f}, high-band peakiness = {peak:.0f} "
        f"({'atypical spectrum for a camera image' if score > 0.5 else 'spectrum in natural range'})",
    )


def noise_floor_analysis(path: str) -> HeuristicResult:
    """Sensor-noise uniformity: real captures carry noise everywhere; AI
    output and inpainted regions are often unnaturally smooth in patches."""
    img, _ = _load_gray(path)
    blurred = img.filter(ImageFilter.MedianFilter(3))
    residual = np.abs(
        np.asarray(img.convert("L"), dtype=np.float64)
        - np.asarray(blurred.convert("L"), dtype=np.float64)
    )
    blocks = _block_stats(residual)
    dead = float((blocks < 0.30).mean())        # fraction of noise-free tiles
    spread = float(blocks.std() / (blocks.mean() + 1e-6))
    score = min(1.0, max(dead, min(1.0, spread / 3.0) * 0.8))
    return HeuristicResult(
        "NoiseMap",
        score,
        f"noise-free tile fraction = {dead:.2f}, noise unevenness = {spread:.2f} "
        f"({'smooth/uneven noise floor — possible synthesis' if score > 0.5 else 'natural noise floor'})",
    )


TOOLS = (error_level_analysis, spectral_analysis, noise_floor_analysis)


def run_heuristics(path: str) -> dict:
    """Run all three tools; return scores plus an explicitly advisory summary."""
    results = [tool(path) for tool in TOOLS]
    flagged = sum(1 for r in results if r.score > 0.5)
    if flagged >= 2:
        assessment = "heuristics lean toward AI generation / manipulation"
    elif flagged == 1:
        assessment = "one heuristic raised a flag — inconclusive"
    else:
        assessment = "no heuristic flags"
    return {
        "tools": [r.to_dict() for r in results],
        "flagged": flagged,
        "assessment": assessment,
        "disclaimer": (
            "Heuristic pixel analysis is statistical and advisory only — it can "
            "be wrong in both directions and never overrides the provenance verdict."
        ),
    }
