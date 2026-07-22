from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from aidetect.webapp import app  # noqa: E402

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
client = TestClient(app)


def test_index_serves_page():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "AI Image Detector" in resp.text


@pytest.mark.parametrize(
    "fname,expected",
    [("firefly_gen.jpg", "AI_GENERATED"),
     ("genfill.png", "AI_MODIFIED"),
     ("clean.jpg", "NO_AI_EVIDENCE")],
)
def test_upload_detection(fname, expected):
    with open(FIXTURES / fname, "rb") as f:
        resp = client.post("/api/detect", files={"file": (fname, f, "image/jpeg")})
    assert resp.status_code == 200
    data = resp.json()
    assert data["verdict"] == expected
    assert data["path"] == fname
