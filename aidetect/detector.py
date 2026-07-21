"""Detection orchestration: C2PA structured pass, raw-scan fallback, XMP pass.

All rules are deterministic — no ML, no thresholds. The final verdict is the
highest-severity verdict implied by any piece of collected evidence.
"""
from __future__ import annotations

from pathlib import Path

from . import c2pa_reader, raw_scan, xmp_scan
from .models import SEVERITY, DetectionResult, Evidence, Verdict

TRAINED_URI_TAIL = "/trainedAlgorithmicMedia"
COMPOSITE_URI = "compositeWithTrainedAlgorithmicMedia"


def _software_agent_name(agent) -> str:
    if isinstance(agent, dict):
        return str(agent.get("name", ""))
    return str(agent) if agent else ""


def _iter_manifests(store: dict):
    manifests = store.get("manifests", {})
    if isinstance(manifests, dict):
        yield from manifests.values()
    elif isinstance(manifests, list):
        yield from manifests


def _c2pa_evidence(store: dict) -> list[Evidence]:
    evidence: list[Evidence] = []
    for manifest in _iter_manifests(store):
        if not isinstance(manifest, dict):
            continue

        # Claim generator (string form and structured form)
        generators: list[str] = []
        cg = manifest.get("claim_generator")
        if cg:
            generators.append(str(cg))
        for info in manifest.get("claim_generator_info") or []:
            if isinstance(info, dict) and info.get("name"):
                generators.append(str(info["name"]))
        for gen in generators:
            if "Adobe Firefly" in gen:
                evidence.append(
                    Evidence("c2pa", "claim_generator", gen, Verdict.AI_GENERATED)
                )
            elif "Photoshop" in gen:
                evidence.append(
                    Evidence("c2pa", "claim_generator", gen, Verdict.EDITED_NO_AI)
                )

        # Actions assertions on the manifest
        for assertion in manifest.get("assertions") or []:
            if not isinstance(assertion, dict):
                continue
            label = str(assertion.get("label", ""))
            if not label.startswith("c2pa.actions"):
                continue
            actions = (assertion.get("data") or {}).get("actions") or []
            for action in actions:
                if not isinstance(action, dict):
                    continue
                act = str(action.get("action", ""))
                dst = str(action.get("digitalSourceType", ""))
                agent = _software_agent_name(action.get("softwareAgent"))
                if COMPOSITE_URI in dst:
                    evidence.append(
                        Evidence("c2pa", "digitalSourceType", dst, Verdict.AI_MODIFIED)
                    )
                elif dst.endswith(TRAINED_URI_TAIL):
                    implies = (
                        Verdict.AI_GENERATED
                        if act == "c2pa.created"
                        else Verdict.AI_MODIFIED
                    )
                    evidence.append(
                        Evidence("c2pa", "digitalSourceType", dst, implies)
                    )
                if "Adobe Firefly" in agent:
                    implies = (
                        Verdict.AI_GENERATED
                        if act == "c2pa.created"
                        else Verdict.AI_MODIFIED
                    )
                    evidence.append(
                        Evidence("c2pa", "softwareAgent", agent, implies)
                    )

        # Ingredients (e.g. a Firefly-generated asset placed into a composite)
        for ingredient in manifest.get("ingredients") or []:
            if not isinstance(ingredient, dict):
                continue
            dst = str(ingredient.get("digitalSourceType", ""))
            if dst.endswith(TRAINED_URI_TAIL) or COMPOSITE_URI in dst:
                evidence.append(
                    Evidence("c2pa", "ingredient.digitalSourceType", dst, Verdict.AI_MODIFIED)
                )

        # Manifest exists at all
        if not evidence:
            evidence.append(
                Evidence("c2pa", "c2pa_manifest",
                         "manifest present, no AI actions", Verdict.EDITED_NO_AI)
            )
    return evidence


def detect_bytes(data: bytes, path: str = "<bytes>",
                 store: dict | None = None,
                 validation_state: str | None = None,
                 c2pa_checked: bool = False) -> DetectionResult:
    """Classify raw image bytes (C2PA store may be pre-parsed by caller)."""
    evidence: list[Evidence] = []
    notes: list[str] = []
    manifest_present = False

    if store:
        manifest_present = True
        evidence.extend(_c2pa_evidence(store))
        if validation_state and validation_state.lower() not in ("trusted",):
            notes.append(
                f"C2PA signature validation state: {validation_state} — "
                "provenance is declared by the manifest, not verified against "
                "a public trust list."
            )
    elif not c2pa_checked or store is None:
        # Raw-scan fallback: c2pa lib unavailable or found no manifest.
        raw_evidence = raw_scan.scan_raw(data)
        if raw_evidence:
            manifest_present = True
            if not c2pa_reader.library_available():
                notes.append(
                    "c2pa library unavailable; classification based on "
                    "byte-level JUMBF scan."
                )
        evidence.extend(raw_evidence)

    evidence.extend(xmp_scan.scan_xmp(data))

    if evidence:
        verdict = max((e.implies for e in evidence), key=lambda v: SEVERITY[v])
    else:
        verdict = Verdict.NO_AI_EVIDENCE
        notes.append(
            "No C2PA or XMP provenance metadata found. Metadata may have been "
            "stripped (e.g. by re-saving or social media processing); absence "
            "of evidence is not proof the image is authentic."
        )

    return DetectionResult(
        path=path,
        verdict=verdict,
        evidence=evidence,
        c2pa_manifest_present=manifest_present,
        notes=notes,
    )


def detect_file(path: str | Path, durable: bool = False) -> DetectionResult:
    """Classify an image file on disk.

    With durable=True, an image with no embedded provenance is additionally
    checked for a durable credential: TrustMark watermark decode followed by
    a Content Credentials Cloud manifest lookup (see aidetect.softbinding).
    """
    path = str(path)
    data = Path(path).read_bytes()
    store, state = c2pa_reader.read_manifest_store(path)
    result = detect_bytes(
        data,
        path=path,
        store=store,
        validation_state=state,
        c2pa_checked=c2pa_reader.library_available(),
    )

    if durable and not result.evidence:
        from . import softbinding

        cloud_store, notes = softbinding.recover_credentials(path)
        result.notes.extend(notes)
        if cloud_store:
            cloud_evidence = _c2pa_evidence(cloud_store)
            for ev in cloud_evidence:
                ev.source = "c2pa-cloud"
            result.evidence.extend(cloud_evidence)
            if cloud_evidence:
                result.verdict = max(
                    (e.implies for e in cloud_evidence),
                    key=lambda v: SEVERITY[v],
                )
                result.c2pa_manifest_present = True
    return result
