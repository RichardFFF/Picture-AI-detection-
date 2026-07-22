"""ML classifier layer: statistical AI-image assessment for provenance-less
images.

The shipped model (aidetect/ml_model.json, schema 2) is a head trained over
ResNet18 ImageNet embeddings (ONNX Model Zoo backbone, Apache-2.0 — see
aidetect/embeddings.py) concatenated with 76 forensic pixel features,
Platt-calibrated, with a selective decision band chosen on cross-validation.

Assessment of an image:
- per-tile combined features -> per-tile raw scores (bag-averaged)
- image score = trimmed mean of tile scores -> calibrated probability
- forced decision at `threshold`; confident decision commits only outside
  the [threshold_lo, threshold_hi] band, otherwise "uncertain".

This layer is STATISTICAL — its accuracy is a measured estimate, not a
guarantee — and it never overrides provenance evidence.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import embeddings
from .features import FEATURE_NAMES, image_tiles, tile_features

_DIR = Path(__file__).resolve().parent
MODEL_PATH = _DIR / "ml_model.json"


@lru_cache(maxsize=1)
def _model() -> dict | None:
    if not MODEL_PATH.exists():
        return None
    return json.loads(MODEL_PATH.read_text())


def _gb_scores(mj: dict, X: np.ndarray) -> np.ndarray:
    X32 = np.asarray(X, dtype=np.float32)
    out = np.full(len(X), mj["baseline"], dtype=np.float64)
    lr = mj["learning_rate"]
    for tree in mj["trees"]:
        cl, cr = tree["children_left"], tree["children_right"]
        feat, thr, val = tree["feature"], tree["threshold"], tree["value"]
        for i, x in enumerate(X32):
            node = 0
            while cl[node] != -1:
                node = cl[node] if x[feat[node]] <= thr[node] else cr[node]
            out[i] += lr * val[node]
    return out


def _linear_scores(mj: dict, X: np.ndarray) -> np.ndarray:
    mean = np.asarray(mj["scaler_mean"])
    std = np.asarray(mj["scaler_std"])
    return ((X - mean) / std) @ np.asarray(mj["coef"]) + mj["intercept"]


def _raw_scores(model: dict, X: np.ndarray) -> np.ndarray:
    per_model = []
    for mj in model["models"]:
        if mj["type"] == "gb":
            per_model.append(_gb_scores(mj, X))
        else:
            per_model.append(_linear_scores(mj, X))
    return np.mean(per_model, axis=0)


def _tile_matrix(path: str, model: dict) -> np.ndarray | None:
    needs_emb = model.get("n_embedding_features", 0) > 0
    if needs_emb and not embeddings.available():
        return None
    rows = []
    for tile in image_tiles(path, max_tiles=16):
        vec = tile_features(tile)
        if needs_emb:
            vec = np.concatenate([vec, embeddings.tile_embedding(tile)])
        rows.append(vec)
    return np.stack(rows)


def _trimmed_mean(values: np.ndarray) -> float:
    if len(values) >= 4:
        values = np.sort(values)[1:-1]
    return float(values.mean())


def ml_assess(path: str) -> dict:
    """Combined ML assessment for an image file."""
    model = _model()
    if model is None:
        return {
            "available": False,
            "note": "No trained model found — run scripts/build_corpus.py "
                    "and scripts/train_ml.py.",
        }
    if model.get("schema") != 2:
        return {"available": False,
                "note": "ml_model.json has an unsupported schema — retrain "
                        "with scripts/train_ml.py."}

    X = _tile_matrix(path, model)
    if X is None:
        return {
            "available": False,
            "note": "Model requires the ImageNet backbone; run "
                    "scripts/fetch_backbone.py (and pip install onnxruntime).",
        }
    assert X.shape[1] == len(FEATURE_NAMES) + model["n_embedding_features"]

    score = _trimmed_mean(_raw_scores(model, X))
    cal = model["calibration"]
    prob = float(1.0 / (1.0 + np.exp(-(cal["a"] * score + cal["b"]))))

    threshold = model["threshold"]
    lo, hi = model["threshold_lo"], model["threshold_hi"]
    if prob >= hi:
        confident = "ai"
    elif prob <= lo:
        confident = "not_ai"
    else:
        confident = "uncertain"

    return {
        "available": True,
        "probability_ai": round(prob, 3),
        "decision": prob >= threshold,
        "confident_decision": confident,
        "threshold": threshold,
        "band": [lo, hi],
        "verdict_hint": {"ai": "likely_ai", "not_ai": "likely_not_ai",
                         "uncertain": "uncertain"}[confident],
        "components": {model["head"]: round(prob, 3)},
        "accuracy_note": (
            "Statistical estimate (ImageNet-embedding transfer learning + "
            "forensic features). Cross-validated on unseen source images: "
            f"forced-decision image accuracy "
            f"{model.get('cv_image_accuracy', 'n/a')}, selective accuracy "
            f"{model.get('cv_selective_accuracy', 'n/a')} at coverage "
            f"{model.get('cv_coverage', 'n/a')}. Advisory — never overrides "
            "provenance evidence."
        ),
    }
