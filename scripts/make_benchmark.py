"""Build the pipeline benchmark set (benchmark/) — ML-holdout images only.

Every image here comes from a source that scripts/build_corpus.py EXCLUDES
from ML training, so the pipeline (provenance -> generator metadata -> ML)
is evaluated on genuinely unseen data:

- benchmark/ai/: individual samples sliced from SD's merged-0007 grid (each a
  standalone genuine AI image), mountains-3 (img2img), upscaling-out, plus
  three metadata-carrying AI images (A1111 / ComfyUI / EXIF variants built
  from held-out AI pixels).
- benchmark/real/: 10 held-out BSDS photographs, Real-ESRGAN input 0030,
  scikit-image coffee — saved byte-for-byte as downloaded.

Writes benchmark/labels.json: relpath -> "ai" | "real".
"""
from __future__ import annotations

import io
import json
import sys
import urllib.request
from pathlib import Path

from PIL import Image
from PIL.PngImagePlugin import PngInfo

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_corpus import BSDS, BSDS_HOLDOUT  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
BENCH = REPO / "benchmark"

SD = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets"

A1111_PARAMS = (
    "portrait, dramatic light\nNegative prompt: lowres\n"
    "Steps: 28, Sampler: DPM++ 2M Karras, CFG scale: 7, Seed: 991, "
    "Size: 512x512, Model: sd_xl_base"
)
COMFY_PROMPT = '{"7": {"class_type": "KSampler", "inputs": {"steps": 20}}}'


def fetch_bytes(url: str) -> bytes | None:
    try:
        return urllib.request.urlopen(url, timeout=60).read()
    except Exception as exc:
        print(f"warn: {url}: {exc}")
        return None


def main() -> int:
    (BENCH / "ai").mkdir(parents=True, exist_ok=True)
    (BENCH / "real").mkdir(parents=True, exist_ok=True)
    labels: dict[str, str] = {}

    # ---- AI: slice merged-0007 into its individual 512x512 samples
    raw = fetch_bytes(f"{SD}/stable-samples/txt2img/merged-0007.png")
    if raw:
        grid = Image.open(io.BytesIO(raw)).convert("RGB")
        w, h = grid.size
        n = 0
        for gy in range(h // 512):
            for gx in range(w // 512):
                if n >= 6:
                    break
                sample = grid.crop((gx * 512, gy * 512, (gx + 1) * 512, (gy + 1) * 512))
                fname = f"sd_sample_{n:02d}.png"
                sample.save(BENCH / "ai" / fname, "PNG")
                labels[f"ai/{fname}"] = "ai"
                n += 1
        print(f"sliced {n} SD samples from merged-0007")

    for name, url in [
        ("sd_mountains3.png", f"{SD}/stable-samples/img2img/mountains-3.png"),
        ("sd_upscaled.png", f"{SD}/stable-samples/img2img/upscaling-out.png"),
    ]:
        raw = fetch_bytes(url)
        if raw:
            (BENCH / "ai" / name).write_bytes(raw)
            labels[f"ai/{name}"] = "ai"

    # ---- AI with generator metadata (from held-out AI pixels): these test
    # the deterministic genai-metadata layer inside the pipeline
    src = BENCH / "ai" / "sd_sample_00.png"
    if src.exists():
        base = Image.open(src).convert("RGB")
        info = PngInfo()
        info.add_text("parameters", A1111_PARAMS)
        base.save(BENCH / "ai" / "sd_meta_a1111.png", "PNG", pnginfo=info)
        labels["ai/sd_meta_a1111.png"] = "ai"

        info = PngInfo()
        info.add_text("prompt", COMFY_PROMPT)
        base.save(BENCH / "ai" / "sd_meta_comfy.png", "PNG", pnginfo=info)
        labels["ai/sd_meta_comfy.png"] = "ai"

        exif = Image.Exif()
        exif[0x9286] = A1111_PARAMS
        base.save(BENCH / "ai" / "sd_meta_exif.jpg", "JPEG", quality=92, exif=exif)
        labels["ai/sd_meta_exif.jpg"] = "ai"

    # ---- real: held-out photographs, byte-for-byte
    for bid in BSDS_HOLDOUT:
        for split in ("test", "val", "train"):
            raw = fetch_bytes(f"{BSDS}/{split}/{bid}.jpg")
            if raw:
                fname = f"bsds_{bid}.jpg"
                (BENCH / "real" / fname).write_bytes(raw)
                labels[f"real/{fname}"] = "real"
                break

    for name, url in [
        ("esr_0030.jpg", "https://raw.githubusercontent.com/xinntao/Real-ESRGAN/master/inputs/0030.jpg"),
        ("ski_coffee.png", "https://raw.githubusercontent.com/scikit-image/scikit-image/v0.19.3/skimage/data/coffee.png"),
    ]:
        raw = fetch_bytes(url)
        if raw:
            (BENCH / "real" / name).write_bytes(raw)
            labels[f"real/{name}"] = "real"

    (BENCH / "labels.json").write_text(json.dumps(labels, indent=2) + "\n")
    n_ai = sum(1 for v in labels.values() if v == "ai")
    print(f"benchmark: {n_ai} AI + {len(labels) - n_ai} real images in {BENCH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
