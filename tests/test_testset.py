"""Gate over the paired testset (original vs AI-modified): exact verdicts."""
import json
from pathlib import Path

import pytest

from aidetect import detect_file

TESTSET = Path(__file__).resolve().parent.parent / "testset"
LABELS_PATH = TESTSET / "labels.json"

if not LABELS_PATH.exists():
    pytest.skip("testset missing — run: python scripts/make_testset.py",
                allow_module_level=True)

LABELS = json.loads(LABELS_PATH.read_text())


@pytest.mark.parametrize("relpath,meta", sorted(LABELS.items()))
def test_testset_classifies_exactly(relpath, meta):
    result = detect_file(TESTSET / relpath)
    assert result.verdict.value == meta["expected"], (
        f"{relpath} ({meta['set']}): expected {meta['expected']}, "
        f"got {result.verdict.value}"
    )
