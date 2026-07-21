"""Durable-credential path: TrustMark decode + Content Credentials Cloud lookup.

The real TrustMark models and Adobe's cloud are unavailable in CI, so the
watermark decoder is faked, but the cloud lookup runs against a REAL local
HTTP server serving the dataset's cloud_store fixture — the full client code
path (URL building, fetch, JSON parse, manifest classification) executes.
"""
import functools
import http.server
import json
import threading
from pathlib import Path

import pytest

from aidetect import softbinding
from aidetect.detector import detect_file
from aidetect.models import Verdict

DATASET = Path(__file__).resolve().parent.parent / "dataset"
CLOUD_STORE = DATASET / "cloud_store"
STRIPPED = DATASET / "special" / "stripped_recoverable.jpg"
SB_ID = "A7X9QK2M"

if not STRIPPED.exists():
    pytest.skip("dataset missing — run: python scripts/make_dataset.py",
                allow_module_level=True)


@pytest.fixture()
def cloud_server():
    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler, directory=str(CLOUD_STORE)
    )
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/"
    server.shutdown()


def test_query_manifest_cloud_roundtrip(cloud_server):
    store, note = softbinding.query_manifest_cloud(f"{SB_ID}.json", endpoint=cloud_server)
    assert note is None
    assert store and store.get("manifests")


def test_query_manifest_cloud_missing_id(cloud_server):
    store, note = softbinding.query_manifest_cloud("UNKNOWN.json", endpoint=cloud_server)
    assert store is None
    assert "No recovered manifest" in note


def test_query_manifest_cloud_unreachable():
    store, note = softbinding.query_manifest_cloud(
        SB_ID, endpoint="http://127.0.0.1:9/"  # discard port: always refused
    )
    assert store is None
    assert "unreachable" in note


def test_full_durable_recovery_classifies_stripped_image(cloud_server, monkeypatch):
    # Without durable recovery, the stripped image honestly yields no evidence.
    plain = detect_file(STRIPPED)
    assert plain.verdict == Verdict.NO_AI_EVIDENCE

    # Fake only the watermark decode (models are proxy-blocked in CI);
    # everything downstream — cloud query, manifest classification — is real.
    monkeypatch.setattr(
        softbinding, "decode_trustmark",
        lambda path: (f"{SB_ID}.json", None),
    )
    monkeypatch.setenv(softbinding.ENDPOINT_ENV, cloud_server)
    recovered = detect_file(STRIPPED, durable=True)
    assert recovered.verdict == Verdict.AI_GENERATED
    assert any(e.source == "c2pa-cloud" for e in recovered.evidence)
    assert any("recovered from Content Credentials Cloud" in n for n in recovered.notes)


def test_durable_flag_degrades_gracefully_without_trustmark(monkeypatch):
    # Simulate trustmark being uninstalled/unloadable: verdict unchanged, note added.
    monkeypatch.setattr(softbinding, "_tm_instance", None)
    monkeypatch.setattr(softbinding, "_tm_error", None)
    import builtins

    real_import = builtins.__import__

    def no_trustmark(name, *args, **kwargs):
        if name == "trustmark":
            raise ImportError("blocked")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_trustmark)
    result = detect_file(STRIPPED, durable=True)
    assert result.verdict == Verdict.NO_AI_EVIDENCE
    assert any("watermark decoding skipped" in n for n in result.notes)


def test_labels_json_documents_cloud_expectation():
    labels = json.loads((DATASET / "labels.json").read_text())
    meta = labels["special/stripped_recoverable.jpg"]
    assert meta["expected"] == "NO_AI_EVIDENCE"
    assert meta["expected_with_cloud"] == "AI_GENERATED"
