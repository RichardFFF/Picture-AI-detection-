"""Pure-stdlib byte-level fallback for C2PA (JUMBF) provenance.

Used when the c2pa library is unavailable or finds no manifest. C2PA
assertion values are embedded as text strings inside CBOR, so a substring
search over the JUMBF region is reliable for the specific signals we need.
"""
from __future__ import annotations

from .models import Evidence, Verdict

COMPOSITE_URI = b"compositeWithTrainedAlgorithmicMedia"
TRAINED_URI_TAIL = b"/trainedAlgorithmicMedia"
FIREFLY = b"Adobe Firefly"

_PNG_SIG = b"\x89PNG\r\n\x1a\n"


def _jpeg_jumbf_regions(data: bytes) -> list[bytes]:
    """Collect payloads of APP11 (0xFFEB) segments — where C2PA JUMBF lives."""
    regions: list[bytes] = []
    i = 2  # skip SOI
    n = len(data)
    while i + 4 <= n:
        if data[i] != 0xFF:
            break
        marker = data[i + 1]
        if marker == 0xD9:  # EOI
            break
        if 0xD0 <= marker <= 0xD7 or marker == 0x01:  # standalone markers
            i += 2
            continue
        if marker == 0xDA:  # SOS — entropy-coded data follows; stop scanning
            break
        seglen = int.from_bytes(data[i + 2 : i + 4], "big")
        payload = data[i + 4 : i + 2 + seglen]
        if marker == 0xEB:
            regions.append(payload)
        i += 2 + seglen
    return regions


def _png_jumbf_regions(data: bytes) -> list[bytes]:
    """Collect caBX chunk payloads from a PNG stream."""
    regions: list[bytes] = []
    i = len(_PNG_SIG)
    n = len(data)
    while i + 8 <= n:
        length = int.from_bytes(data[i : i + 4], "big")
        ctype = data[i + 4 : i + 8]
        payload = data[i + 8 : i + 8 + length]
        if ctype == b"caBX":
            regions.append(payload)
        if ctype == b"IEND":
            break
        i += 12 + length  # length + type + payload + crc
    return regions


def find_jumbf_regions(data: bytes) -> list[bytes]:
    if data[:2] == b"\xff\xd8":
        return _jpeg_jumbf_regions(data)
    if data[: len(_PNG_SIG)] == _PNG_SIG:
        return _png_jumbf_regions(data)
    return []


def scan_raw(data: bytes) -> list[Evidence]:
    """Search the C2PA JUMBF region (or whole file as last resort) for AI signals."""
    evidence: list[Evidence] = []
    regions = find_jumbf_regions(data)
    if regions:
        blob = b"".join(regions)
        if COMPOSITE_URI in blob:
            evidence.append(
                Evidence("raw-scan", "digitalSourceType",
                         COMPOSITE_URI.decode(), Verdict.AI_MODIFIED)
            )
        elif TRAINED_URI_TAIL in blob:
            evidence.append(
                Evidence("raw-scan", "digitalSourceType",
                         "trainedAlgorithmicMedia", Verdict.AI_GENERATED)
            )
        if FIREFLY in blob:
            implies = (
                Verdict.AI_GENERATED
                if b"c2pa.created" in blob and TRAINED_URI_TAIL in blob
                else Verdict.AI_MODIFIED
            )
            evidence.append(
                Evidence("raw-scan", "softwareAgent", FIREFLY.decode(), implies)
            )
        if not evidence and (b"c2pa.actions" in blob or b"c2pa" in blob):
            evidence.append(
                Evidence("raw-scan", "c2pa_manifest",
                         "manifest present, no AI actions", Verdict.EDITED_NO_AI)
            )
    else:
        # No JUMBF container found: only trust the two IPTC URIs — they are
        # too specific to appear by accident in image data.
        if COMPOSITE_URI in data:
            evidence.append(
                Evidence("raw-scan", "digitalSourceType",
                         COMPOSITE_URI.decode(), Verdict.AI_MODIFIED)
            )
        elif TRAINED_URI_TAIL in data:
            evidence.append(
                Evidence("raw-scan", "digitalSourceType",
                         "trainedAlgorithmicMedia", Verdict.AI_GENERATED)
            )
    return evidence
