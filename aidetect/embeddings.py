"""ImageNet-backbone embeddings for transfer learning (open-source model).

Uses ResNet18 from the ONNX Model Zoo (Apache-2.0) via onnxruntime. The
graph's penultimate (global-average-pool / flatten) output is exposed by
graph surgery so tiles map to 512-d feature vectors; if surgery fails the
1000-d logits are used instead. Everything is optional: when onnxruntime or
the model file is missing, callers fall back to forensic features only.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "resnet18-v1-7.onnx"

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@lru_cache(maxsize=1)
def _session():
    """Return (ort_session, output_name, dim) or None when unavailable."""
    if not MODEL_PATH.exists():
        return None
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        return None

    try:
        model = onnx.load(str(MODEL_PATH))
        # find the penultimate tensor: input to the final Gemm/MatMul
        final_dense = None
        for node in model.graph.node:
            if node.op_type in ("Gemm", "MatMul"):
                final_dense = node
        if final_dense is not None:
            feat_name = final_dense.input[0]
            model.graph.output.append(
                onnx.helper.make_tensor_value_info(
                    feat_name, onnx.TensorProto.FLOAT, None
                )
            )
            sess = ort.InferenceSession(
                model.SerializeToString(), providers=["CPUExecutionProvider"]
            )
            return sess, feat_name
    except Exception:
        pass

    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(MODEL_PATH),
                                    providers=["CPUExecutionProvider"])
        return sess, sess.get_outputs()[0].name
    except Exception:
        return None


def available() -> bool:
    return _session() is not None


def _preprocess(rgb: np.ndarray) -> np.ndarray:
    img = Image.fromarray(rgb).resize((224, 224))
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - _MEAN) / _STD
    return x.transpose(2, 0, 1)[None]


def tile_embedding(rgb: np.ndarray) -> np.ndarray | None:
    """512-d (or logits) embedding for one RGB uint8 tile array."""
    handle = _session()
    if handle is None:
        return None
    sess, out_name = handle
    input_name = sess.get_inputs()[0].name
    out = sess.run([out_name], {input_name: _preprocess(rgb)})[0]
    return np.asarray(out, dtype=np.float64).reshape(-1)


def embedding_dim() -> int | None:
    handle = _session()
    if handle is None:
        return None
    probe = tile_embedding(np.zeros((32, 32, 3), dtype=np.uint8))
    return None if probe is None else int(probe.shape[0])
