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
from aidetect.ml_detector import _gb_decision, _gb_model, ml_assess

REPO = Path(__file__).resolve().parent.parent
HELDOUT = REPO / "tests" / "heldout_features.json"
FIXTURES = REPO / "fixtures"

model = _gb_model()
if model is None:
    pytest.skip("ml_model.json missing — run scripts/train_ml.py",
                allow_module_level=True)


def test_model_structure():
    assert model["feature_names"] == FEATURE_NAMES
    assert model["trees"] and model["learning_rate"] > 0
    assert 0.3 <= model["threshold"] <= 0.7
    assert model["cv_tile_accuracy"] > 0.75


def test_heldout_accuracy_floor():
    data = json.loads(HELDOUT.read_text())
    X = np.array(data["X"])
    y = np.array(data["y"])
    assert X.shape[1] == len(FEATURE_NAMES)
    raw = _gb_decision(model, X)
    pred = (1.0 / (1.0 + np.exp(-raw)) >= 0.5).astype(int)
    acc = float((pred == y).mean())
    assert acc >= 0.80, f"heldout wiring accuracy dropped to {acc:.3f}"


def test_inference_deterministic():
    path = str(FIXTURES / "clean.jpg")
    a = ml_assess(path)
    b = ml_assess(path)
    assert a == b
    assert a["available"]
    assert 0.0 <= a["probability_ai"] <= 1.0
    assert a["verdict_hint"] in ("likely_ai", "likely_not_ai", "uncertain")
    assert isinstance(a["decision"], bool)
