import io

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from aidetect.detector import detect_bytes
from aidetect.genai_metadata import scan_genai_metadata
from aidetect.models import Verdict

A1111_PARAMS = (
    "a cat in space\nNegative prompt: blurry\n"
    "Steps: 30, Sampler: DPM++ 2M Karras, CFG scale: 7, Seed: 12345, "
    "Size: 512x512, Model: sd_v1-5"
)
COMFY_PROMPT = '{"3": {"class_type": "KSampler", "inputs": {"seed": 5}}}'


def png_with_text(key: str, value: str) -> bytes:
    img = Image.new("RGB", (32, 32), (10, 120, 200))
    info = PngInfo()
    info.add_text(key, value)
    buf = io.BytesIO()
    img.save(buf, "PNG", pnginfo=info)
    return buf.getvalue()


def test_a1111_parameters_chunk():
    ev = scan_genai_metadata(png_with_text("parameters", A1111_PARAMS))
    assert [e.implies for e in ev] == [Verdict.AI_GENERATED]
    assert ev[0].source == "genai-metadata"


def test_comfyui_workflow_chunk():
    ev = scan_genai_metadata(png_with_text("prompt", COMFY_PROMPT))
    assert any("ComfyUI" in e.value for e in ev)


def test_novelai_software_chunk():
    ev = scan_genai_metadata(png_with_text("Software", "NovelAI"))
    assert any(e.value == "NovelAI" for e in ev)


def test_plain_png_text_not_flagged():
    ev = scan_genai_metadata(png_with_text("Comment", "holiday photo, Steps: none"))
    assert ev == []


def test_jpeg_a1111_usercomment():
    img = Image.new("RGB", (32, 32), (200, 50, 50))
    exif = Image.Exif()
    exif[0x9286] = A1111_PARAMS  # UserComment
    buf = io.BytesIO()
    img.save(buf, "JPEG", exif=exif)
    ev = scan_genai_metadata(buf.getvalue())
    assert any(e.implies == Verdict.AI_GENERATED for e in ev)


def test_midjourney_jpeg_metadata():
    img = Image.new("RGB", (32, 32), (5, 5, 5))
    exif = Image.Exif()
    exif[0x010E] = "spaceship over city --v 6 Job ID: 1f2e3d4c-5b6a-7980-abcd-ef0123456789"
    exif[0x013B] = "Midjourney"
    buf = io.BytesIO()
    img.save(buf, "JPEG", exif=exif)
    ev = scan_genai_metadata(buf.getvalue())
    assert any("Midjourney" in e.value for e in ev)


def test_detector_integration():
    result = detect_bytes(png_with_text("parameters", A1111_PARAMS), c2pa_checked=True)
    assert result.verdict == Verdict.AI_GENERATED
    assert any(e.source == "genai-metadata" for e in result.evidence)
