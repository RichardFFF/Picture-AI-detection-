"""ML layer tests: model file integrity, deterministic inference, and a
wiring gate over the committed held-out feature matrix.

Note: heldout_features.json rows were a CV test fold during training, but the
shipped model was then refit on all data — so the accuracy floor here guards
against broken exports/feature drift, not generalization (generalization is
measured by scripts/train_ml.py CV and scripts/benchmark_pipeline.py).
"""
import json
from pathlib import Path

import numpy as np
import pytest

from aidetect.features import FEATURE_NAMES
from aidetect.ml_detector import _model, ml_assess, tile_probs

REPO = Path(__file__).resolve().parent.parent
HELDOUT = REPO / "tests" / "heldout_features.json"
FIXTURES = REPO / "fixtures"

model = _model()
if model is None:
    pytest.skip("ml_model.json missing — run scripts/train_ml.py",
                allow_module_level=True)


def test_model_structure():
    assert model["schema"] == 3
    assert model["feature_names"] == FEATURE_NAMES
    assert model["models"]
    for mj in model["models"]:
        assert mj["type"] in ("gb", "linear")
    assert 0.0 <= model["threshold_lo"] <= model["threshold"] \
        <= model["threshold_hi"] <= 1.0
    assert model["cv_image_accuracy"] > 0.80
    assert model["cv_selective_accuracy"] >= 0.90


def test_calibration_monotonic():
    for mj in model["models"]:
        assert mj["calibration"]["a"] > 0, \
            "calibration must preserve score ordering"


def test_heldout_accuracy_floor():
    data = json.loads(HELDOUT.read_text())
    X = np.array(data["X"])
    y = np.array(data["y"])
    assert X.shape[1] == len(FEATURE_NAMES) + model["n_embedding_features"]
    pred = (tile_probs(model, X) >= 0.5).astype(int)
    acc = float((pred == y).mean())
    assert acc >= 0.75, f"heldout wiring accuracy dropped to {acc:.3f}"


def test_inference_deterministic():
    path = str(FIXTURES / "clean.jpg")
    a = ml_assess(path)
    b = ml_assess(path)
    assert a == b
    if not a["available"]:
        pytest.skip(a["note"])  # backbone not fetched in this environment
    assert 0.0 <= a["probability_ai"] <= 1.0
    assert a["confident_decision"] in ("ai", "not_ai", "uncertain")
    assert a["verdict_hint"] in ("likely_ai", "likely_not_ai", "uncertain")
    assert isinstance(a["decision"], bool)
