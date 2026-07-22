"""Gate over the new dataset: exact verdicts for original/ and ai/ entries."""
import json
from pathlib import Path

import pytest

from aidetect import detect_file

DATASET = Path(__file__).resolve().parent.parent / "dataset"
LABELS_PATH = DATASET / "labels.json"

if not LABELS_PATH.exists():
    pytest.skip("dataset missing — run: python scripts/make_dataset.py",
                allow_module_level=True)

LABELS = json.loads(LABELS_PATH.read_text())
GATED = sorted((k, v) for k, v in LABELS.items() if v["set"] in ("original", "ai"))


@pytest.mark.parametrize("relpath,meta", GATED)
def test_dataset_classifies_exactly(relpath, meta):
    result = detect_file(DATASET / relpath)
    assert result.verdict.value == meta["expected"], (
        f"{relpath} ({meta['set']}): expected {meta['expected']}, "
        f"got {result.verdict.value}"
    )
