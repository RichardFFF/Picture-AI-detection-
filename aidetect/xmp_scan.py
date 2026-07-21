"""Pure-stdlib scan for XMP metadata declaring Adobe AI provenance.

Extracts XMP packets from the raw file bytes (works for JPEG APP1 segments,
PNG iTXt chunks, and any other container that embeds the packet verbatim)
and matches the fields Adobe Firefly / Photoshop write.
"""
from __future__ import annotations

import re

from .models import Evidence, Verdict

COMPOSITE_URI = "compositeWithTrainedAlgorithmicMedia"
COMPOSITE_SYNTHETIC_URI = "compositeSynthetic"
TRAINED_URI = "digitalsourcetype/trainedAlgorithmicMedia"

_XMPMETA_RE = re.compile(rb"<x:xmpmeta.*?</x:xmpmeta>", re.DOTALL)
_XPACKET_RE = re.compile(rb"<\?xpacket begin=.*?<\?xpacket end.*?\?>", re.DOTALL)
_CREATOR_TOOL_RE = re.compile(
    r"(?:<xmp:CreatorTool>(.*?)</xmp:CreatorTool>|xmp:CreatorTool=\"(.*?)\")",
    re.DOTALL,
)


def extract_xmp_packets(data: bytes) -> list[str]:
    """Return every XMP packet found in the raw bytes, decoded to text."""
    spans: list[tuple[int, int]] = []
    for regex in (_XPACKET_RE, _XMPMETA_RE):
        for m in regex.finditer(data):
            # Skip packets fully contained in an already-collected span
            # (xmpmeta lives inside xpacket).
            if any(m.start() >= s and m.end() <= e for s, e in spans):
                continue
            spans.append((m.start(), m.end()))
    return [data[s:e].decode("utf-8", errors="replace") for s, e in spans]


def _creator_tool(packet: str) -> str | None:
    m = _CREATOR_TOOL_RE.search(packet)
    if m:
        return (m.group(1) or m.group(2) or "").strip()
    return None


def scan_xmp(data: bytes) -> list[Evidence]:
    """Match Adobe AI provenance signals in any XMP packet in the file."""
    evidence: list[Evidence] = []
    for packet in extract_xmp_packets(data):
        # Composite forms must be checked before the plain URI: the plain
        # trainedAlgorithmicMedia URI is a substring of the composite one.
        if COMPOSITE_URI in packet or COMPOSITE_SYNTHETIC_URI in packet:
            uri = COMPOSITE_URI if COMPOSITE_URI in packet else COMPOSITE_SYNTHETIC_URI
            evidence.append(
                Evidence("xmp", "DigitalSourceType", uri, Verdict.AI_MODIFIED)
            )
        elif TRAINED_URI.lower() in packet.lower():
            evidence.append(
                Evidence("xmp", "DigitalSourceType", "trainedAlgorithmicMedia", Verdict.AI_GENERATED)
            )

        tool = _creator_tool(packet)
        if "Adobe Firefly" in packet:
            if tool and "Adobe Firefly" in tool:
                evidence.append(
                    Evidence("xmp", "CreatorTool", tool, Verdict.AI_GENERATED)
                )
            else:
                evidence.append(
                    Evidence("xmp", "agent", "Adobe Firefly", Verdict.AI_MODIFIED)
                )
        elif tool and "Adobe Photoshop" in tool:
            evidence.append(
                Evidence("xmp", "CreatorTool", tool, Verdict.EDITED_NO_AI)
            )
    return evidence
