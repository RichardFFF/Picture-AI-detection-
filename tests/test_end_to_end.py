"""The 100% gate: every labeled fixture must classify exactly as labeled."""
import json
from pathlib import Path

import pytest

from aidetect import detect_file
from aidetect.detector import detect_bytes

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
LABELS_PATH = FIXTURES / "labels.json"

if not LABELS_PATH.exists():
    pytest.fail("fixtures/labels.json missing — run: python scripts/make_fixtures.py", pytrace=False)

LABELS = json.loads(LABELS_PATH.read_text())


@pytest.mark.parametrize("fname,expected", sorted(LABELS.items()))
def test_fixture_classifies_exactly(fname, expected):
    result = detect_file(FIXTURES / fname)
    assert result.verdict.value == expected, (
        f"{fname}: expected {expected}, got {result.verdict.value} "
        f"(evidence: {[e.to_dict() for e in result.evidence]})"
    )


@pytest.mark.parametrize("fname,expected", sorted(LABELS.items()))
def test_fixture_classifies_without_c2pa_library(fname, expected, monkeypatch):
    """The byte-level fallback must reach the same verdicts without c2pa-python."""
    from aidetect import c2pa_reader

    monkeypatch.setattr(c2pa_reader, "c2pa", None)
    data = (FIXTURES / fname).read_bytes()
    result = detect_bytes(data, path=fname, c2pa_checked=False)
    assert result.verdict.value == expected
