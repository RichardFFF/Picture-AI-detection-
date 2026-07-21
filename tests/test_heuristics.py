import numpy as np
import pytest
from PIL import Image

from aidetect.heuristics import (
    error_level_analysis,
    noise_floor_analysis,
    run_heuristics,
    spectral_analysis,
)


@pytest.fixture(scope="module")
def noisy_photo(tmp_path_factory):
    """Camera-like image: full-frame gaussian noise over a gradient."""
    rng = np.random.default_rng(42)
    base = np.linspace(40, 210, 256, dtype=np.float64)
    arr = np.tile(base, (256, 1)) + rng.normal(0, 6, (256, 256))
    img = Image.fromarray(np.clip(arr, 0, 255).astype("uint8")).convert("RGB")
    p = tmp_path_factory.mktemp("h") / "noisy.jpg"
    img.save(p, "JPEG", quality=90)
    return str(p)


@pytest.fixture(scope="module")
def flat_synthetic(tmp_path_factory):
    """AI-render-like image: perfectly smooth flat regions, no sensor noise."""
    arr = np.zeros((256, 256), dtype="uint8")
    arr[:128] = 90
    arr[128:] = 200
    img = Image.fromarray(arr).convert("RGB")
    p = tmp_path_factory.mktemp("h") / "flat.png"
    img.save(p, "PNG")
    return str(p)


def test_scores_in_range(noisy_photo, flat_synthetic):
    for path in (noisy_photo, flat_synthetic):
        for tool in (error_level_analysis, spectral_analysis, noise_floor_analysis):
            r = tool(path)
            assert 0.0 <= r.score <= 1.0
            assert r.summary


def test_noise_map_separates_flat_from_noisy(noisy_photo, flat_synthetic):
    assert noise_floor_analysis(flat_synthetic).score > noise_floor_analysis(noisy_photo).score


def test_run_heuristics_shape(noisy_photo):
    out = run_heuristics(noisy_photo)
    assert {t["tool"] for t in out["tools"]} == {"ELA", "Spectral", "NoiseMap"}
    assert "advisory" in out["disclaimer"]
    assert isinstance(out["flagged"], int)


def test_deterministic(noisy_photo):
    assert run_heuristics(noisy_photo) == run_heuristics(noisy_photo)
