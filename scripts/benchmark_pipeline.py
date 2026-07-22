"""Evaluate the FULL detection pipeline on unseen data.

Two benchmark pools:
1. benchmark/ (ML-holdout images, scripts/make_benchmark.py) — sources the
   ML models never trained on.
2. The provenance-labeled sets (dataset/, testset/) — Adobe-style AI images
   and clean originals, decided deterministically.

Pipeline decision per image:
  1. provenance evidence (C2PA / XMP / generator metadata) -> its verdict
  2. otherwise ML probability >= 0.5 -> AI, else real

Reports accuracy per pool, per decision layer, and overall.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from aidetect import Verdict, detect_file  # noqa: E402
from aidetect.ml_detector import ml_assess  # noqa: E402

AI_VERDICTS = {Verdict.AI_GENERATED, Verdict.AI_MODIFIED}


def pipeline_decide(path: Path, use_ml: bool):
    """Return (is_ai_prediction, decision_layer, confident_decision).

    use_ml=False is the app's default deterministic pipeline: no provenance
    evidence means "no evidence of AI" (counted as not-AI). use_ml=True adds
    the statistical fallback for photograph-domain pools. confident_decision
    is "ai"/"not_ai"/"uncertain" for ML-decided items, else mirrors the
    deterministic decision.
    """
    result = detect_file(path)
    if result.evidence:
        is_ai = result.verdict in AI_VERDICTS
        return is_ai, "provenance", ("ai" if is_ai else "not_ai")
    if use_ml:
        ml = ml_assess(str(path))
        if ml.get("available"):
            return bool(ml["decision"]), "ml", ml["confident_decision"]
    return False, "no-evidence", "not_ai"


def main() -> int:
    items: list[tuple[Path, bool, str]] = []  # path, truth_is_ai, pool

    bench_labels = json.loads((REPO / "benchmark" / "labels.json").read_text())
    for rel, label in sorted(bench_labels.items()):
        items.append((REPO / "benchmark" / rel, label == "ai", "ml-holdout"))

    for setdir, labels_file in (("dataset", "labels.json"), ("testset", "labels.json")):
        labels = json.loads((REPO / setdir / labels_file).read_text())
        for rel, meta in sorted(labels.items()):
            if meta["set"] == "special":
                continue
            truth_ai = meta["expected"] in ("AI_GENERATED", "AI_MODIFIED")
            items.append((REPO / setdir / rel, truth_ai, "provenance-sets"))

    stats: dict[str, list[int]] = {}
    layer_stats: dict[str, list[int]] = {}
    rows = []
    for path, truth_ai, pool in items:
        # The ML fallback is a photograph-domain model; the provenance sets'
        # synthetic rendered originals are decided by the deterministic
        # default pipeline (no evidence -> not AI), exactly as the app does.
        pred_ai, layer, confident = pipeline_decide(
            path, use_ml=(pool == "ml-holdout"))
        ok = pred_ai == truth_ai
        stats.setdefault(pool, []).append(ok)
        layer_stats.setdefault(layer, []).append(ok)
        rows.append((str(path.relative_to(REPO)), truth_ai, pred_ai, layer,
                     confident, ok))

    w = max(len(r[0]) for r in rows)
    print(f"{'image':<{w}}  {'truth':<6} {'pred':<6} {'layer':<11} "
          f"{'confident':<10} ok")
    print("-" * (w + 44))
    for rel, truth, pred, layer, confident, ok in rows:
        print(f"{rel:<{w}}  {'AI' if truth else 'real':<6} "
              f"{'AI' if pred else 'real':<6} {layer:<11} {confident:<10} "
              f"{'OK' if ok else 'WRONG'}")

    print()
    total_ok = total_n = 0
    for pool, oks in sorted(stats.items()):
        total_ok += sum(oks)
        total_n += len(oks)
        print(f"{pool:<16} accuracy: {sum(oks)}/{len(oks)} "
              f"({100.0 * sum(oks) / len(oks):.1f}%)")
    for layer, oks in sorted(layer_stats.items()):
        print(f"decided by {layer:<11}: {sum(oks)}/{len(oks)} correct")

    # ---- selective mode: commit only outside the uncertainty band
    committed = [(t, c, ok) for (_, t, _, _, c, ok) in rows if c != "uncertain"]
    sel_ok = sum(1 for t, c, _ in committed if (c == "ai") == t)
    coverage = 100.0 * len(committed) / len(rows)
    sel_pct = 100.0 * sel_ok / len(committed) if committed else 0.0
    print(f"\nSELECTIVE mode: {sel_ok}/{len(committed)} committed decisions "
          f"correct ({sel_pct:.1f}%) at {coverage:.0f}% coverage "
          f"({len(rows) - len(committed)} answered UNCERTAIN)")

    pct = 100.0 * total_ok / total_n
    print(f"OVERALL forced-decision accuracy: {total_ok}/{total_n} ({pct:.1f}%)")
    # Gates sit slightly below the measured values so benign re-download/
    # re-train fluctuations pass while real regressions fail.
    return 0 if (pct >= 88.0 and sel_pct >= 92.0) else 1


if __name__ == "__main__":
    sys.exit(main())
