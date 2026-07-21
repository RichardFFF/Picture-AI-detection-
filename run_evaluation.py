"""Run the detector over the labeled fixture set and report accuracy.

Exits non-zero unless accuracy is 100%.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from aidetect import detect_file

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def main() -> int:
    labels_path = FIXTURES / "labels.json"
    if not labels_path.exists():
        print("fixtures/labels.json not found — run: python scripts/make_fixtures.py")
        return 1
    labels: dict[str, str] = json.loads(labels_path.read_text())

    rows = []
    correct = 0
    for fname, expected in sorted(labels.items()):
        result = detect_file(FIXTURES / fname)
        got = result.verdict.value
        ok = got == expected
        correct += ok
        rows.append((fname, expected, got, len(result.evidence), "OK" if ok else "MISMATCH"))

    w = max(len(r[0]) for r in rows)
    print(f"{'file':<{w}}  {'expected':<16} {'got':<16} {'evidence':<8} status")
    print("-" * (w + 52))
    for fname, expected, got, n_ev, status in rows:
        print(f"{fname:<{w}}  {expected:<16} {got:<16} {n_ev:<8} {status}")

    total = len(rows)
    pct = 100.0 * correct / total
    print(f"\nAccuracy: {correct}/{total} ({pct:.1f}%)")
    print(
        "\nNote: verdicts are based on declared C2PA/XMP provenance metadata. "
        "Images whose metadata has been stripped (see stripped_ai.jpg) are "
        "reported as NO_AI_EVIDENCE — the detector never guesses."
    )
    return 0 if correct == total else 1


if __name__ == "__main__":
    sys.exit(main())
