import pytest

from aidetect.detector import _c2pa_evidence, detect_bytes
from aidetect.models import Verdict

TRAINED = "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"
COMPOSITE = "http://cv.iptc.org/newscodes/digitalsourcetype/compositeWithTrainedAlgorithmicMedia"


def store_with(manifest: dict) -> dict:
    return {"active_manifest": "m1", "manifests": {"m1": manifest}}


def actions_manifest(actions, generator="Adobe Photoshop"):
    return {
        "claim_generator_info": [{"name": generator}],
        "assertions": [{"label": "c2pa.actions.v2", "data": {"actions": actions}}],
    }


def test_firefly_created_is_generated():
    store = store_with(actions_manifest(
        [{"action": "c2pa.created", "softwareAgent": {"name": "Adobe Firefly"},
          "digitalSourceType": TRAINED}],
        generator="Adobe Firefly",
    ))
    result = detect_bytes(b"", store=store, c2pa_checked=True)
    assert result.verdict == Verdict.AI_GENERATED
    assert result.c2pa_manifest_present


def test_generative_fill_is_modified():
    store = store_with(actions_manifest(
        [{"action": "c2pa.opened"},
         {"action": "c2pa.placed", "softwareAgent": {"name": "Adobe Firefly"},
          "digitalSourceType": COMPOSITE}],
    ))
    result = detect_bytes(b"", store=store, c2pa_checked=True)
    assert result.verdict == Verdict.AI_MODIFIED


def test_plain_photoshop_manifest_is_edited():
    store = store_with(actions_manifest(
        [{"action": "c2pa.opened"}, {"action": "c2pa.color_adjustments"}],
    ))
    result = detect_bytes(b"", store=store, c2pa_checked=True)
    assert result.verdict == Verdict.EDITED_NO_AI


def test_string_software_agent_form():
    store = store_with(actions_manifest(
        [{"action": "c2pa.placed", "softwareAgent": "Adobe Firefly 1.0"}],
    ))
    result = detect_bytes(b"", store=store, c2pa_checked=True)
    assert result.verdict == Verdict.AI_MODIFIED


def test_ai_ingredient_is_modified():
    store = store_with({
        "claim_generator": "Adobe Photoshop/25.0",
        "ingredients": [{"title": "gen.png", "digitalSourceType": TRAINED}],
    })
    result = detect_bytes(b"", store=store, c2pa_checked=True)
    assert result.verdict == Verdict.AI_MODIFIED


def test_c2pa_beats_weaker_xmp_signal():
    # C2PA says Firefly-created; XMP only says Photoshop CreatorTool.
    store = store_with(actions_manifest(
        [{"action": "c2pa.created", "digitalSourceType": TRAINED}],
        generator="Adobe Firefly",
    ))
    xmp = (b'<x:xmpmeta xmlns:x="adobe:ns:meta/">'
           b"<xmp:CreatorTool>Adobe Photoshop 25.0</xmp:CreatorTool>"
           b"</x:xmpmeta>")
    result = detect_bytes(xmp, store=store, c2pa_checked=True)
    assert result.verdict == Verdict.AI_GENERATED


def test_untrusted_signature_still_classifies_with_note():
    store = store_with(actions_manifest(
        [{"action": "c2pa.created", "digitalSourceType": TRAINED}],
        generator="Adobe Firefly",
    ))
    result = detect_bytes(b"", store=store, validation_state="Invalid", c2pa_checked=True)
    assert result.verdict == Verdict.AI_GENERATED
    assert any("validation state" in n for n in result.notes)


def test_no_evidence_verdict_and_note():
    result = detect_bytes(b"\xff\xd8\xff\xd9", c2pa_checked=True)
    assert result.verdict == Verdict.NO_AI_EVIDENCE
    assert result.evidence == []
    assert any("stripped" in n for n in result.notes)


@pytest.mark.parametrize(
    "generator,expected",
    [("Adobe Firefly", Verdict.AI_GENERATED),
     ("Adobe Photoshop", Verdict.EDITED_NO_AI)],
)
def test_claim_generator_alone(generator, expected):
    store = store_with({"claim_generator_info": [{"name": generator}]})
    result = detect_bytes(b"", store=store, c2pa_checked=True)
    assert result.verdict == expected


def test_c2pa_evidence_handles_malformed_store():
    assert _c2pa_evidence({"manifests": {"m": "not-a-dict"}}) == []
    assert _c2pa_evidence({}) == []
