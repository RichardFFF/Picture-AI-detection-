"""Download "wild" sample images for manual testing (not committed to git).

AI images are genuine generative-model outputs published in open-source
repos (Stable Diffusion / ControlNet sample assets); controls are a
human-drawn sketch from the same repo and real photographs from the
Content Authenticity Initiative fixtures.

None of the AI images are Adobe outputs, so the provenance layer is
EXPECTED to report NO_AI_EVIDENCE for them — they exercise the documented
limitation (non-Adobe generators embed no Content Credentials) and the
heuristic tools. Run:

    python scripts/fetch_wild_samples.py
    python -m aidetect --heuristics wild/*
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

DEST = Path(__file__).resolve().parent.parent / "wild"

SD = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples"
CAI = "https://raw.githubusercontent.com/contentauth/c2pa-rs/main/sdk/tests/fixtures"

SAMPLES = {
    # --- genuine AI-generated (Stable Diffusion official sample outputs)
    "ai_sd_txt2img_05.png": f"{SD}/txt2img/merged-0005.png",
    "ai_sd_txt2img_06.png": f"{SD}/txt2img/merged-0006.png",
    "ai_sd_txt2img_07.png": f"{SD}/txt2img/merged-0007.png",
    "ai_sd_img2img_mountains.png": f"{SD}/img2img/mountains-3.png",
    "ai_sd_upscaled.png": f"{SD}/img2img/upscaling-out.png",
    "ai_controlnet_sample.png":
        "https://raw.githubusercontent.com/lllyasviel/ControlNet/main/github_page/p1.png",
    # --- non-AI controls
    "control_human_sketch.jpg": f"{SD}/img2img/sketch-mountains-input.jpg",
    "control_photo_C.jpg": f"{CAI}/C.jpg",
    "control_photo_CA.jpg": f"{CAI}/CA.jpg",
}


def main() -> int:
    DEST.mkdir(exist_ok=True)
    ok = 0
    for name, url in SAMPLES.items():
        try:
            data = urllib.request.urlopen(url, timeout=60).read()
            (DEST / name).write_bytes(data)
            print(f"downloaded {name} ({len(data)} bytes)")
            ok += 1
        except Exception as exc:
            print(f"warn: {name}: {exc}")
    print(f"\n{ok}/{len(SAMPLES)} wild samples in {DEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
