"""Deterministic detection of non-Adobe AI-generator metadata.

Many popular generators embed identifying metadata in their outputs:

- Stable Diffusion WebUI (AUTOMATIC1111): PNG tEXt/iTXt chunk keyed
  "parameters" containing "Steps:", "Sampler:", "CFG scale:" (also copied
  into EXIF UserComment for JPEG exports)
- ComfyUI: PNG chunks keyed "prompt" / "workflow" holding the node-graph JSON
- NovelAI: chunks "Software" = NovelAI plus a JSON "Comment"
- Midjourney: XMP description carrying a "Job ID" and author "Midjourney"
- InvokeAI: "invokeai_metadata" / "sd-metadata" chunks

Like the C2PA/XMP pass this is exact: the signals are compound and specific,
so a match is proof of generator metadata (which can, as always, be stripped).
"""
from __future__ import annotations

import re
import zlib

from .models import Evidence, Verdict

_PNG_SIG = b"\x89PNG\r\n\x1a\n"


def _png_text_chunks(data: bytes) -> list[tuple[str, bytes]]:
    """Yield (keyword, text_payload) from tEXt/iTXt/zTXt chunks."""
    out: list[tuple[str, bytes]] = []
    if data[: len(_PNG_SIG)] != _PNG_SIG:
        return out
    i = len(_PNG_SIG)
    n = len(data)
    while i + 8 <= n:
        length = int.from_bytes(data[i : i + 4], "big")
        ctype = data[i + 4 : i + 8]
        payload = data[i + 8 : i + 8 + length]
        if ctype in (b"tEXt", b"iTXt", b"zTXt"):
            try:
                key, _, rest = payload.partition(b"\x00")
                if ctype == b"tEXt":
                    text = rest
                elif ctype == b"zTXt":
                    text = zlib.decompress(rest[1:])
                else:  # iTXt: compression flag+method, lang, translated key
                    comp_flag = rest[:1]
                    body = rest[2:]
                    body = body.split(b"\x00", 2)[-1]
                    text = zlib.decompress(body) if comp_flag == b"\x01" else body
                out.append((key.decode("latin-1"), text))
            except Exception:
                pass
        if ctype == b"IEND":
            break
        i += 12 + length
    return out


_A1111_MARKERS = (b"Steps:", b"Sampler:", b"CFG scale:")
_COMFY_NODE_RE = re.compile(rb'"class_type"\s*:')
_MIDJOURNEY_RE = re.compile(rb"Midjourney", re.IGNORECASE)
_JOB_ID_RE = re.compile(rb"Job ID:\s*[0-9a-f-]{20,}", re.IGNORECASE)


def scan_genai_metadata(data: bytes) -> list[Evidence]:
    evidence: list[Evidence] = []

    for key, text in _png_text_chunks(data):
        lowered = key.lower()
        if lowered == "parameters" and sum(m in text for m in _A1111_MARKERS) >= 2:
            evidence.append(Evidence(
                "genai-metadata", "png:parameters",
                "Stable Diffusion WebUI generation parameters",
                Verdict.AI_GENERATED,
            ))
        elif lowered in ("prompt", "workflow") and _COMFY_NODE_RE.search(text):
            evidence.append(Evidence(
                "genai-metadata", f"png:{lowered}",
                "ComfyUI workflow graph", Verdict.AI_GENERATED,
            ))
        elif lowered == "software" and b"NovelAI" in text:
            evidence.append(Evidence(
                "genai-metadata", "png:Software", "NovelAI", Verdict.AI_GENERATED,
            ))
        elif lowered in ("invokeai_metadata", "sd-metadata"):
            evidence.append(Evidence(
                "genai-metadata", f"png:{lowered}",
                "InvokeAI generation metadata", Verdict.AI_GENERATED,
            ))

    # JPEG path: A1111 copies the parameters block into EXIF UserComment;
    # Midjourney stamps its job id + name in XMP/EXIF description fields.
    if data[:2] == b"\xff\xd8":
        head = data[:262144]
        if sum(m in head for m in _A1111_MARKERS) >= 3:
            evidence.append(Evidence(
                "genai-metadata", "exif:UserComment",
                "Stable Diffusion generation parameters",
                Verdict.AI_GENERATED,
            ))
        if _MIDJOURNEY_RE.search(head) and _JOB_ID_RE.search(head):
            evidence.append(Evidence(
                "genai-metadata", "xmp:description",
                "Midjourney job metadata", Verdict.AI_GENERATED,
            ))

    return evidence
