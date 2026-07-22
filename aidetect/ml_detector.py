"""ML classifier layer: statistical AI-image assessment for provenance-less
images.

Two models, both trained on the corpus assembled by scripts/build_corpus.py
(genuine Stable Diffusion/ControlNet outputs vs real photographs), evaluated
on tiles from parent images never seen in training:

- gradient-boosted trees on forensic features (aidetect/ml_model.json,
  no runtime dependency beyond numpy), and
- a residual CNN (aidetect/cnn_model.pt, used automatically when torch is
  installed).

The reported probability is the mean over image tiles; when both models are
available their probabilities are averaged. This layer is STATISTICAL — its
accuracy is a measured estimate, not a guarantee — and it never overrides
provenance evidence.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from .features import image_features

_DIR = Path(__file__).resolve().parent
GB_MODEL_PATH = _DIR / "ml_model.json"
CNN_MODEL_PATH = _DIR / "cnn_model.pt"


@lru_cache(maxsize=1)
def _gb_model() -> dict | None:
    if not GB_MODEL_PATH.exists():
        return None
    return json.loads(GB_MODEL_PATH.read_text())


def _gb_decision(model: dict, X: np.ndarray) -> np.ndarray:
    X32 = np.asarray(X, dtype=np.float32)
    out = np.full(len(X), model["baseline"], dtype=np.float64)
    lr = model["learning_rate"]
    for tree in model["trees"]:
        cl = tree["children_left"]
        cr = tree["children_right"]
        feat = tree["feature"]
        thr = tree["threshold"]
        val = tree["value"]
        for i, x in enumerate(X32):
            node = 0
            while cl[node] != -1:
                node = cl[node] if x[feat[node]] <= thr[node] else cr[node]
            out[i] += lr * val[node]
    return out


def gb_probability(path: str) -> float | None:
    """Mean per-tile AI probability from the gradient-boosted model."""
    model = _gb_model()
    if model is None:
        return None
    X = image_features(path)
    raw = _gb_decision(model, X)
    return float(np.mean(1.0 / (1.0 + np.exp(-raw))))


@lru_cache(maxsize=1)
def _cnn():
    if not CNN_MODEL_PATH.exists():
        return None
    try:
        import torch

        from .cnn import ResidualNet

        payload = torch.load(CNN_MODEL_PATH, map_location="cpu", weights_only=True)
        model = ResidualNet()
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model
    except Exception:
        return None


def cnn_probability(path: str) -> float | None:
    model = _cnn()
    if model is None:
        return None
    import torch

    from .cnn import residual_tensor
    from .features import image_tiles
    from PIL import Image

    probs = []
    with torch.no_grad():
        for tile in image_tiles(path):
            img = Image.fromarray(tile)
            w, h = img.size
            img = img.crop(((w - 128) // 2, (h - 128) // 2,
                            (w - 128) // 2 + 128, (h - 128) // 2 + 128))
            logits = model(residual_tensor(img).unsqueeze(0))
            probs.append(float(torch.softmax(logits, dim=1)[0, 1]))
    return float(np.mean(probs)) if probs else None


def ml_assess(path: str) -> dict:
    """Combined ML assessment for an image file."""
    components: dict[str, float] = {}
    gb = gb_probability(path)
    if gb is not None:
        components["gradient_boosting"] = round(gb, 3)
    cnn = cnn_probability(path)
    if cnn is not None:
        components["residual_cnn"] = round(cnn, 3)

    if not components:
        return {
            "available": False,
            "note": "No trained model found — run scripts/build_corpus.py "
                    "and scripts/train_ml.py.",
        }

    prob = float(np.mean(list(components.values())))
    model = _gb_model() or {}
    threshold = float(model.get("threshold", 0.5))
    if prob >= threshold + 0.15:
        hint = "likely_ai"
    elif prob <= threshold - 0.15:
        hint = "likely_not_ai"
    else:
        hint = "uncertain"

    return {
        "available": True,
        "probability_ai": round(prob, 3),
        "decision": prob >= threshold,
        "threshold": threshold,
        "verdict_hint": hint,
        "components": components,
        "accuracy_note": (
            "Statistical estimate from models trained on diffusion outputs vs "
            "real photographs; measured on held-out images "
            f"(gradient boosting CV tile accuracy "
            f"{model.get('cv_tile_accuracy', 'n/a')}). Advisory — never "
            "overrides provenance evidence."
        ),
    }
