import struct

from aidetect.models import Verdict
from aidetect.raw_scan import scan_raw

TRAINED = b"http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"
COMPOSITE = b"http://cv.iptc.org/newscodes/digitalsourcetype/compositeWithTrainedAlgorithmicMedia"


def jpeg_with_app11(payload: bytes) -> bytes:
    seg = b"\xff\xeb" + struct.pack(">H", len(payload) + 2) + payload
    return b"\xff\xd8" + seg + b"\xff\xd9"


def test_app11_trained_uri_generated():
    data = jpeg_with_app11(b"jumbc2pa c2pa.actions c2pa.created " + TRAINED)
    ev = scan_raw(data)
    assert any(e.implies == Verdict.AI_GENERATED for e in ev)


def test_app11_composite_modified():
    data = jpeg_with_app11(b"jumbc2pa c2pa.actions c2pa.placed " + COMPOSITE)
    ev = scan_raw(data)
    assert any(e.implies == Verdict.AI_MODIFIED for e in ev)
    assert not any(e.implies == Verdict.AI_GENERATED for e in ev)


def test_app11_firefly_agent():
    data = jpeg_with_app11(b"jumbc2pa c2pa.actions c2pa.placed Adobe Firefly")
    ev = scan_raw(data)
    assert any(e.signal == "softwareAgent" and e.implies == Verdict.AI_MODIFIED for e in ev)


def test_app11_manifest_without_ai_is_edited():
    data = jpeg_with_app11(b"jumbc2pa c2pa.actions c2pa.color_adjustments")
    ev = scan_raw(data)
    assert [e.implies for e in ev] == [Verdict.EDITED_NO_AI]


def test_whole_file_fallback_only_matches_uris():
    assert scan_raw(b"\xff\xd8 Adobe Firefly \xff\xd9") == []
    ev = scan_raw(b"random blob " + TRAINED)
    assert any(e.implies == Verdict.AI_GENERATED for e in ev)


def test_clean_jpeg_no_evidence():
    assert scan_raw(b"\xff\xd8\xff\xdb\x00\x04\x00\x00\xff\xd9") == []
