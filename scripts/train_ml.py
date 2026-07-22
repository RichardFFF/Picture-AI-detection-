"""Train the AI-vs-real classifier on the corpus and export it as JSON.

- Extracts forensic features (aidetect.features) for every corpus tile.
- Reports honest generalization via 5-fold GROUP cross-validation: folds are
  split by PARENT image, so tiles of one source never appear in both train
  and test.
- Trains the final GradientBoosting model on all data and exports every tree
  (children/feature/threshold/value arrays) to aidetect/ml_model.json — no
  pickle. The JSON predictor is verified against sklearn's output before
  writing.
- Saves the features of the worst CV fold's test split to
  tests/heldout_features.json so CI re-verifies accuracy without network.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from aidetect.features import FEATURE_NAMES, tile_features  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.ensemble import GradientBoostingClassifier  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402

CORPUS = REPO / "corpus"
MODEL_PATH = REPO / "aidetect" / "ml_model.json"
HELDOUT_PATH = REPO / "tests" / "heldout_features.json"
SEED = 20260722


CONFIGS = [
    {"n_estimators": 400, "learning_rate": 0.05, "max_depth": 3},
    {"n_estimators": 500, "learning_rate": 0.04, "max_depth": 4},
    {"n_estimators": 300, "learning_rate": 0.08, "max_depth": 2},
]


def make_clf(cfg: dict) -> GradientBoostingClassifier:
    return GradientBoostingClassifier(subsample=0.8, random_state=SEED, **cfg)


def balanced_weights(y: np.ndarray) -> np.ndarray:
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    w_pos = len(y) / (2.0 * n_pos)
    w_neg = len(y) / (2.0 * n_neg)
    return np.where(y == 1, w_pos, w_neg)


def load_corpus():
    X, y, groups, names = [], [], [], []
    for label, sub in ((1, "ai"), (0, "real")):
        for p in sorted((CORPUS / sub).glob("*.jpg")):
            rgb = np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)
            X.append(tile_features(rgb))
            y.append(label)
            groups.append(p.name.split("__")[0])
            names.append(f"{sub}/{p.name}")
    return np.stack(X), np.array(y), np.array(groups), names


def export_trees(clf: GradientBoostingClassifier) -> list[dict]:
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
    return trees


def json_decision_function(model: dict, X: np.ndarray) -> np.ndarray:
    # sklearn casts inputs to float32 before traversal — match it exactly
    X32 = np.asarray(X, dtype=np.float32)
    out = np.full(len(X), model["baseline"], dtype=np.float64)
    lr = model["learning_rate"]
    for tree in model["trees"]:
        cl = tree["children_left"]
        cr = tree["children_right"]
        feat = tree["feature"]
        thr = tree["threshold"]  # float64, compared against float32-cast x
        val = tree["value"]
        for i, x in enumerate(X32):
            node = 0
            while cl[node] != -1:
                node = cl[node] if x[feat[node]] <= thr[node] else cr[node]
            out[i] += lr * val[node]
    return out


def main() -> int:
    print("extracting features...")
    X, y, groups, names = load_corpus()
    print(f"corpus: {len(y)} tiles, {len(set(groups))} parent images, "
          f"{int(y.sum())} AI / {int((1 - y).sum())} real")

    # ---- honest generalization estimate: 5-fold grouped CV per config
    gkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    best = None
    for cfg in CONFIGS:
        tile_accs, img_accs = [], []
        worst = (1.1, None)
        cv_img_probs: list[tuple[float, int]] = []
        for fold, (tr, te) in enumerate(gkf.split(X, y, groups)):
            clf = make_clf(cfg).fit(X[tr], y[tr],
                                    sample_weight=balanced_weights(y[tr]))
            proba = clf.predict_proba(X[te])[:, 1]
            pred = (proba >= 0.5).astype(int)
            acc = float((pred == y[te]).mean())
            tile_accs.append(acc)

            pp: dict[str, list[float]] = {}
            pt: dict[str, int] = {}
            for i, p in zip(te, proba):
                pp.setdefault(groups[i], []).append(float(p))
                pt[groups[i]] = int(y[i])
            cv_img_probs.extend((float(np.mean(v)), pt[g]) for g, v in pp.items())
            img_ok = sum((np.mean(v) >= 0.5) == pt[g] for g, v in pp.items())
            img_accs.append(img_ok / len(pp))
            if acc < worst[0]:
                worst = (acc, te)
        mean_tile = float(np.mean(tile_accs))
        mean_img = float(np.mean(img_accs))
        print(f"config {cfg}: tile {mean_tile:.3f} "
              f"(folds {', '.join(f'{a:.3f}' for a in tile_accs)}), "
              f"image {mean_img:.3f}")
        if best is None or mean_tile > best["tile"]:
            best = {"cfg": cfg, "tile": mean_tile, "img": mean_img,
                    "folds": tile_accs, "worst_te": worst[1],
                    "img_probs": cv_img_probs}

    # ---- calibrate the decision threshold on CV image-level probabilities
    # (maximize balanced accuracy: robust to the corpus' class imbalance)
    probs = np.array([p for p, _ in best["img_probs"]])
    truths = np.array([t for _, t in best["img_probs"]])
    best_thr, best_bal = 0.5, 0.0
    for thr in np.arange(0.30, 0.71, 0.01):
        pred = probs >= thr
        tpr = (pred & (truths == 1)).sum() / max(1, (truths == 1).sum())
        tnr = (~pred & (truths == 0)).sum() / max(1, (truths == 0).sum())
        bal = (tpr + tnr) / 2
        if bal > best_bal:
            best_bal, best_thr = bal, float(thr)
    print(f"calibrated threshold: {best_thr:.2f} "
          f"(CV image-level balanced accuracy {best_bal:.3f})")

    cfg = best["cfg"]
    tile_accs = best["folds"]
    worst = (None, best["worst_te"])
    print(f"\nbest config: {cfg}")
    print(f"CV tile-level accuracy:  {best['tile']:.3f}")
    print(f"CV image-level accuracy: {best['img']:.3f}")

    # ---- final model on all data, exported as JSON trees
    clf = make_clf(cfg).fit(X, y, sample_weight=balanced_weights(y))
    trees = export_trees(clf)
    raw = clf.decision_function(X)
    model = {
        "feature_names": FEATURE_NAMES,
        "learning_rate": clf.learning_rate,
        "baseline": 0.0,
        "trees": trees,
        "threshold": round(best_thr, 2),
        "cv_image_balanced_accuracy": round(best_bal, 4),
        "cv_tile_accuracy": round(best["tile"], 4),
        "cv_image_accuracy": round(best["img"], 4),
        "cv_folds": [round(a, 4) for a in tile_accs],
        "config": cfg,
        "training_note": (
            "Gradient-boosted trees on forensic pixel features; trained on "
            "Stable Diffusion/ControlNet outputs vs real photographs "
            "(open-source sample assets), grouped 5-fold CV by parent image. "
            "Accuracy is a statistical estimate for this distribution, not a "
            "guarantee."
        ),
    }
    # solve for baseline so the JSON predictor exactly matches sklearn
    model["baseline"] = float(np.mean(raw - json_decision_function(model, X)))
    err = np.abs(json_decision_function(model, X) - raw).max()
    assert err < 1e-8, f"JSON export mismatch: {err}"
    print(f"JSON predictor verified against sklearn (max diff {err:.2e})")

    MODEL_PATH.write_text(json.dumps(model) + "\n")
    print(f"wrote {MODEL_PATH} ({MODEL_PATH.stat().st_size // 1024} KiB)")

    te = worst[1]
    HELDOUT_PATH.write_text(json.dumps({
        "X": X[te].tolist(),
        "y": y[te].tolist(),
        "groups": [groups[i] for i in te],
        "names": [names[i] for i in te],
        "note": "worst CV fold's test tiles (unseen parents for that fold)",
    }) + "\n")
    print(f"wrote {HELDOUT_PATH} ({len(te)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
