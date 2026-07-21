from aidetect.models import Verdict
from aidetect.xmp_scan import scan_xmp

TRAINED = "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"
COMPOSITE = "http://cv.iptc.org/newscodes/digitalsourcetype/compositeWithTrainedAlgorithmicMedia"


def packet(body: str) -> bytes:
    return (
        '<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>'
        '<x:xmpmeta xmlns:x="adobe:ns:meta/">'
        f"{body}"
        '</x:xmpmeta><?xpacket end="w"?>'
    ).encode()


def test_trained_uri_implies_generated():
    ev = scan_xmp(packet(f"<Iptc4xmpExt:DigitalSourceType>{TRAINED}</Iptc4xmpExt:DigitalSourceType>"))
    assert any(e.implies == Verdict.AI_GENERATED for e in ev)


def test_composite_checked_before_plain_uri():
    ev = scan_xmp(packet(f"<Iptc4xmpExt:DigitalSourceType>{COMPOSITE}</Iptc4xmpExt:DigitalSourceType>"))
    dst = [e for e in ev if e.signal == "DigitalSourceType"]
    assert dst and dst[0].implies == Verdict.AI_MODIFIED


def test_firefly_creator_tool_implies_generated():
    ev = scan_xmp(packet("<xmp:CreatorTool>Adobe Firefly 1.0</xmp:CreatorTool>"))
    assert any(e.implies == Verdict.AI_GENERATED and e.signal == "CreatorTool" for e in ev)


def test_firefly_elsewhere_implies_modified():
    ev = scan_xmp(packet("<stEvt:softwareAgent>Adobe Firefly</stEvt:softwareAgent>"))
    assert any(e.implies == Verdict.AI_MODIFIED for e in ev)


def test_photoshop_creator_tool_implies_edited():
    ev = scan_xmp(packet("<xmp:CreatorTool>Adobe Photoshop 25.0</xmp:CreatorTool>"))
    assert [e.implies for e in ev] == [Verdict.EDITED_NO_AI]


def test_clean_bytes_no_evidence():
    assert scan_xmp(b"\xff\xd8 just image bytes, mentions Adobe Firefly outside xmp") == []
