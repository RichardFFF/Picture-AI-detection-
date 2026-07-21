"""Data model for detection results."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Verdict(str, Enum):
    AI_GENERATED = "AI_GENERATED"
    AI_MODIFIED = "AI_MODIFIED"
    EDITED_NO_AI = "EDITED_NO_AI"
    NO_AI_EVIDENCE = "NO_AI_EVIDENCE"


# Higher number wins when merging evidence from multiple sources.
SEVERITY = {
    Verdict.AI_GENERATED: 3,
    Verdict.AI_MODIFIED: 2,
    Verdict.EDITED_NO_AI: 1,
    Verdict.NO_AI_EVIDENCE: 0,
}

VERDICT_DESCRIPTIONS = {
    Verdict.AI_GENERATED: "Image was created by an AI tool (e.g. Adobe Firefly text-to-image).",
    Verdict.AI_MODIFIED: "Image was modified with AI (e.g. Photoshop Generative Fill / Firefly composite).",
    Verdict.EDITED_NO_AI: "Image carries edit provenance (e.g. Photoshop) with no AI actions declared.",
    Verdict.NO_AI_EVIDENCE: "No provenance metadata found; AI involvement can neither be confirmed nor ruled out.",
}


@dataclass
class Evidence:
    source: str  # "c2pa" | "xmp" | "raw-scan"
    signal: str  # e.g. "digitalSourceType", "softwareAgent", "claim_generator", "CreatorTool"
    value: str  # the matched string
    implies: Verdict

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "signal": self.signal,
            "value": self.value,
            "implies": self.implies.value,
        }


@dataclass
class DetectionResult:
    path: str
    verdict: Verdict
    evidence: list[Evidence] = field(default_factory=list)
    c2pa_manifest_present: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def description(self) -> str:
        return VERDICT_DESCRIPTIONS[self.verdict]

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "verdict": self.verdict.value,
            "description": self.description,
            "c2pa_manifest_present": self.c2pa_manifest_present,
            "evidence": [e.to_dict() for e in self.evidence],
            "notes": self.notes,
        }
