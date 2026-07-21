"""Evaluate the detector on paired image sets: AI-modified vs original.

Binary scoring: verdicts AI_GENERATED / AI_MODIFIED count as "AI",
EDITED_NO_AI / NO_AI_EVIDENCE count as "not AI". Prints a per-file table,
a confusion matrix, and accuracy; exits non-zero unless accuracy is 100%.

Point it at your own directories to test real Firefly / Photoshop exports:

    python evaluate_set.py --ai-dir my_ai_images --original-dir my_originals
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from aidetect import Verdict, detect_file

AI_VERDICTS = {Verdict.AI_GENERATED, Verdict.AI_MODIFIED}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".avif", ".heic"}


def image_files(directory: Path) -> list[Path]:
    return sorted(p for p in directory.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ai-dir", default="testset/ai_modified",
                        help="directory of AI-created/modified images")
    parser.add_argument("--original-dir", default="testset/original",
                        help="directory of original (non-AI) images")
    args = parser.parse_args()

    ai_dir, orig_dir = Path(args.ai_dir), Path(args.original_dir)
    for d in (ai_dir, orig_dir):
        if not d.is_dir():
            print(f"error: {d} is not a directory "
                  "(run: python scripts/make_testset.py)")
            return 1

    rows = []  # (path, truth_is_ai, verdict, predicted_ai)
    for path in image_files(ai_dir):
        v = detect_file(path).verdict
        rows.append((path, True, v, v in AI_VERDICTS))
    for path in image_files(orig_dir):
        v = detect_file(path).verdict
        rows.append((path, False, v, v in AI_VERDICTS))

    if not rows:
        print("error: no images found")
        return 1

    w = max(len(str(p)) for p, *_ in rows)
    print(f"{'file':<{w}}  {'truth':<9} {'verdict':<16} result")
    print("-" * (w + 38))
    tp = tn = fp = fn = 0
    for path, truth_ai, verdict, pred_ai in rows:
        ok = truth_ai == pred_ai
        if truth_ai and pred_ai:
            tp += 1
        elif not truth_ai and not pred_ai:
            tn += 1
        elif pred_ai:
            fp += 1
        else:
            fn += 1
        truth = "AI" if truth_ai else "original"
        print(f"{str(path):<{w}}  {truth:<9} {verdict.value:<16} "
              f"{'OK' if ok else 'WRONG'}")

    total = len(rows)
    correct = tp + tn
    print("\nConfusion matrix (positive = AI):")
    print("                 predicted AI   predicted not-AI")
    print(f"  actually AI    {tp:>12}   {fn:>16}")
    print(f"  actually orig  {fp:>12}   {tn:>16}")
    print(f"\nAccuracy: {correct}/{total} ({100.0 * correct / total:.1f}%)")
    if fp:
        print(f"False positives: {fp}")
    if fn:
        print(f"False negatives: {fn} (AI images whose provenance metadata "
              "is missing/stripped are undetectable by any honest method)")
    return 0 if correct == total else 1


if __name__ == "__main__":
    sys.exit(main())
