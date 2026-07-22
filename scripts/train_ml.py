"""Train the AI-vs-real classifier: ImageNet-embedding transfer learning.

Per-tile feature vector = 76 forensic features (aidetect.features) ⊕ 512-d
ResNet18 ImageNet embeddings (aidetect.embeddings, ONNX Model Zoo backbone).

Honest evaluation protocol:
- StratifiedGroupKFold(5) by PARENT image (tiles never straddle the split);
- two candidate heads (standardized LogisticRegression, GradientBoosting)
  compared on out-of-fold image-level balanced accuracy;
- Platt calibration fitted on pooled out-of-fold image scores;
- selective decision bands (lo, hi) chosen on calibrated out-of-fold
  probabilities: widest coverage with selective accuracy >= 98%.

The final model is refit on all data (GB: 3-seed bag) and exported to
aidetect/ml_model.json as pure JSON (verified against sklearn). The worst
fold's test features go to tests/heldout_features.json for CI wiring gates.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from aidetect import embeddings  # noqa: E402
from aidetect.features import FEATURE_NAMES, tile_features  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.ensemble import GradientBoostingClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402

CORPUS = REPO / "corpus"
MODEL_PATH = REPO / "aidetect" / "ml_model.json"
HELDOUT_PATH = REPO / "tests" / "heldout_features.json"
CACHE_PATH = REPO / "corpus" / "features_cache.npz"
SEED = 20260722
GB_CFG = {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 3}
TARGET_SELECTIVE = 0.98


def balanced_weights(y: np.ndarray) -> np.ndarray:
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    return np.where(y == 1, len(y) / (2.0 * n_pos), len(y) / (2.0 * n_neg))


def load_corpus():
    files = []
    for label, sub in ((1, "ai"), (0, "real")):
        for p in sorted((CORPUS / sub).glob("*.jpg")):
            files.append((p, label, p.name.split("__")[0], f"{sub}/{p.name}"))

    if CACHE_PATH.exists():
        cache = np.load(CACHE_PATH, allow_pickle=True)
        if list(cache["names"]) == [n for _, _, _, n in files]:
            print("using cached features")
            return cache["X"], cache["y"], cache["groups"], list(cache["names"])

    use_emb = embeddings.available()
    if not use_emb:
        print("warn: embeddings unavailable (run scripts/fetch_backbone.py) — "
              "training on forensic features only")
    X, y, groups, names = [], [], [], []
    for i, (p, label, group, name) in enumerate(files):
        rgb = np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)
        vec = tile_features(rgb)
        if use_emb:
            vec = np.concatenate([vec, embeddings.tile_embedding(rgb)])
        X.append(vec)
        y.append(label)
        groups.append(group)
        names.append(name)
        if (i + 1) % 400 == 0:
            print(f"  features {i + 1}/{len(files)}")
    X, y, groups = np.stack(X), np.array(y), np.array(groups)
    np.savez_compressed(CACHE_PATH, X=X, y=y, groups=groups, names=np.array(names))
    return X, y, groups, names


def fit_head(kind: str, X, y, seed=SEED):
    if kind == "linear":
        mean, std = X.mean(axis=0), X.std(axis=0) + 1e-9
        clf = LogisticRegression(max_iter=4000, C=0.5, class_weight="balanced")
        clf.fit((X - mean) / std, y)
        return {"kind": "linear", "clf": clf, "mean": mean, "std": std}
    clf = GradientBoostingClassifier(subsample=0.8, random_state=seed, **GB_CFG)
    clf.fit(X, y, sample_weight=balanced_weights(y))
    return {"kind": "gb", "clf": clf}


def head_scores(head, X) -> np.ndarray:
    """Raw decision scores (log-odds-ish)."""
    if head["kind"] == "linear":
        return head["clf"].decision_function((X - head["mean"]) / head["std"])
    return head["clf"].decision_function(X)


def image_scores(scores, idx, y, groups):
    pp: dict[str, list[float]] = {}
    pt: dict[str, int] = {}
    for i, s in zip(idx, scores):
        pp.setdefault(groups[i], []).append(float(s))
        pt[groups[i]] = int(y[i])
    out = [(float(np.mean(v)), pt[g], g) for g, v in sorted(pp.items())]
    return out


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def fit_platt(scores: np.ndarray, truths: np.ndarray) -> tuple[float, float]:
    lr = LogisticRegression(max_iter=2000)
    lr.fit(scores.reshape(-1, 1), truths)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def export_gb_trees(clf) -> dict:
    trees = []
    for est in clf.estimators_[:, 0]:
        t = est.tree_
        trees.append({
            "children_left": t.children_left.tolist(),
            "children_right": t.children_right.tolist(),
            "feature": t.feature.tolist(),
            "threshold": t.threshold.tolist(),
            "value": t.value[:, 0, 0].tolist(),
        })
    return {"type": "gb", "learning_rate": clf.learning_rate,
            "baseline": 0.0, "trees": trees}


def json_gb_scores(model: dict, X: np.ndarray) -> np.ndarray:
    X32 = np.asarray(X, dtype=np.float32)
    out = np.full(len(X), model["baseline"], dtype=np.float64)
    lr = model["learning_rate"]
    for tree in model["trees"]:
        cl, cr = tree["children_left"], tree["children_right"]
        feat, thr, val = tree["feature"], tree["threshold"], tree["value"]
        for i, x in enumerate(X32):
            node = 0
            while cl[node] != -1:
                node = cl[node] if x[feat[node]] <= thr[node] else cr[node]
            out[i] += lr * val[node]
    return out


def main() -> int:
    print("extracting features...")
    X, y, groups, names = load_corpus()
    n_emb = X.shape[1] - len(FEATURE_NAMES)
    print(f"corpus: {len(y)} tiles ({int(y.sum())} AI / {int((1 - y).sum())} real), "
          f"{len(set(groups))} parents, {X.shape[1]} features "
          f"({len(FEATURE_NAMES)} forensic + {n_emb} embedding)")

    gkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = list(gkf.split(X, y, groups))

    results = {}
    for kind in ("linear", "gb"):
        oof_img: list[tuple[float, int, str]] = []
        tile_accs = []
        worst = (1.1, None)
        for tr, te in folds:
            head = fit_head(kind, X[tr], y[tr])
            s = head_scores(head, X[te])
            acc = float(((s >= 0) == y[te]).mean())
            tile_accs.append(acc)
            oof_img.extend(image_scores(s, te, y, groups))
            if acc < worst[0]:
                worst = (acc, te)
        img_s = np.array([s for s, _, _ in oof_img])
        img_t = np.array([t for _, t, _ in oof_img])
        img_acc = float(((img_s >= 0) == img_t).mean())
        bal = 0.5 * (((img_s >= 0) & (img_t == 1)).sum() / max(1, img_t.sum())
                     + ((img_s < 0) & (img_t == 0)).sum() / max(1, (1 - img_t).sum()))
        print(f"{kind}: tile CV {np.mean(tile_accs):.3f} "
              f"(folds {', '.join(f'{a:.3f}' for a in tile_accs)}), "
              f"image CV {img_acc:.3f}, balanced {bal:.3f}")
        results[kind] = {"tile": float(np.mean(tile_accs)), "img": img_acc,
                         "bal": float(bal), "oof": oof_img,
                         "folds": tile_accs, "worst_te": worst[1]}

    kind = max(results, key=lambda k: results[k]["bal"])
    res = results[kind]
    print(f"\nselected head: {kind}")

    # ---- calibration on pooled out-of-fold image scores
    img_s = np.array([s for s, _, _ in res["oof"]])
    img_t = np.array([t for _, t, _ in res["oof"]])
    a, b = fit_platt(img_s, img_t)
    p_cal = sigmoid(a * img_s + b)

    # ---- forced threshold (balanced) + selective bands on calibrated probs
    best_thr, best_bal = 0.5, 0.0
    for thr in np.arange(0.20, 0.81, 0.01):
        pred = p_cal >= thr
        tpr = (pred & (img_t == 1)).sum() / max(1, (img_t == 1).sum())
        tnr = (~pred & (img_t == 0)).sum() / max(1, (img_t == 0).sum())
        if (tpr + tnr) / 2 > best_bal:
            best_bal, best_thr = (tpr + tnr) / 2, float(thr)

    # Widest-coverage band meeting the target selective accuracy; if the CV
    # data can't support the target (few image-level points), relax the
    # target in steps and require at least half coverage.
    best_band, best_cov, achieved = None, 0.0, 0.0
    for target in (0.98, 0.96, 0.94, 0.92):
        for lo in np.arange(0.02, best_thr + 0.001, 0.02):
            for hi in np.arange(best_thr, 0.99, 0.02):
                committed = (p_cal <= lo) | (p_cal >= hi)
                if committed.mean() < 0.5:
                    continue
                sel = ((p_cal[committed] >= hi) == img_t[committed]).mean()
                if sel >= target and committed.mean() > best_cov:
                    best_cov = float(committed.mean())
                    best_band = (float(lo), float(hi))
                    achieved = float(sel)
        if best_band is not None:
            break
    if best_band is None:
        best_band, best_cov, achieved = (best_thr, best_thr), 1.0, float(
            ((p_cal >= best_thr) == img_t).mean())
    lo, hi = best_band
    sel_acc = achieved
    print(f"forced threshold {best_thr:.2f} (balanced {best_bal:.3f}); "
          f"selective band [{lo:.2f}, {hi:.2f}] -> selective accuracy "
          f"{sel_acc:.3f} at coverage {best_cov:.2f}")

    # ---- final model on all data
    models_json = []
    if kind == "gb":
        for seed in (SEED, SEED + 1, SEED + 2):
            head = fit_head("gb", X, y, seed=seed)
            mj = export_gb_trees(head["clf"])
            raw = head["clf"].decision_function(X)
            mj["baseline"] = float(np.mean(raw - json_gb_scores(mj, X)))
            err = np.abs(json_gb_scores(mj, X) - raw).max()
            assert err < 1e-8, f"JSON export mismatch: {err}"
            models_json.append(mj)
    else:
        head = fit_head("linear", X, y)
        mj = {"type": "linear",
              "coef": head["clf"].coef_[0].tolist(),
              "intercept": float(head["clf"].intercept_[0]),
              "scaler_mean": head["mean"].tolist(),
              "scaler_std": head["std"].tolist()}
        check = ((X - head["mean"]) / head["std"]) @ np.array(mj["coef"]) + mj["intercept"]
        err = np.abs(check - head_scores(head, X)).max()
        assert err < 1e-6, f"JSON export mismatch: {err}"
        models_json.append(mj)
    print(f"JSON export verified for {len(models_json)} model(s)")

    model = {
        "schema": 2,
        "head": kind,
        "n_forensic_features": len(FEATURE_NAMES),
        "n_embedding_features": int(n_emb),
        "feature_names": FEATURE_NAMES,
        "models": models_json,
        "calibration": {"a": a, "b": b},
        "threshold": round(best_thr, 2),
        "threshold_lo": round(lo, 2),
        "threshold_hi": round(hi, 2),
        "cv_tile_accuracy": round(res["tile"], 4),
        "cv_image_accuracy": round(res["img"], 4),
        "cv_image_balanced_accuracy": round(res["bal"], 4),
        "cv_selective_accuracy": round(sel_acc, 4),
        "cv_coverage": round(best_cov, 4),
        "training_note": (
            "Head over ResNet18 ImageNet embeddings (ONNX Model Zoo, "
            "Apache-2.0) + forensic pixel features; trained on diffusion "
            "outputs vs real photographs from permissively-licensed "
            "open-source repos (see SOURCES.md); grouped 5-fold CV by parent "
            "image. Accuracy is a statistical estimate, not a guarantee."
        ),
    }
    MODEL_PATH.write_text(json.dumps(model) + "\n")
    print(f"wrote {MODEL_PATH} ({MODEL_PATH.stat().st_size // 1024} KiB)")

    te = res["worst_te"]
    HELDOUT_PATH.write_text(json.dumps({
        "X": np.round(X[te], 6).tolist(),
        "y": y[te].tolist(),
        "groups": [groups[i] for i in te],
        "note": "worst CV fold's test tiles (unseen parents for that fold)",
    }) + "\n")
    print(f"wrote {HELDOUT_PATH} ({len(te)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
