"""Build a fresh evaluation dataset in dataset/ (distinct from testset/).

Layout:
  dataset/original/     8 clean images (new crops of real CAI photographs +
                        newly rendered scenes) — no provenance metadata
  dataset/ai/           8 AI images: signed C2PA Firefly creations, signed
                        Generative Fill composites (JPEG + PNG), XMP-only
  dataset/special/      stripped_recoverable.jpg — an AI image whose metadata
                        was stripped; its manifest lives in dataset/cloud_store/
                        keyed by a soft-binding ID, exercising the durable-
                        credential (TrustMark + Content Credentials Cloud) path
  dataset/cloud_store/  <soft-binding-id>.json manifest store for the mock cloud
  dataset/labels.json   filename -> {"set", "expected"}

Run the binary evaluation with:
  python evaluate_set.py --ai-dir dataset/ai --original-dir dataset/original
"""
from __future__ import annotations

import json
import sys
import tempfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from make_fixtures import (  # noqa: E402
    COMPOSITE,
    get_signing_credentials,
    inject_xmp_jpeg,
    make_base_image,
    manifest_firefly_generated,
    manifest_generative_fill,
    sign_file,
    xmp_packet,
)
from make_testset import apply_generative_patch  # noqa: E402

from PIL import Image  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
DATASET = REPO / "dataset"

SOFT_BINDING_ID = "A7X9QK2M"  # id "decoded" from the TrustMark watermark

CAI_BASE = "https://raw.githubusercontent.com/contentauth/c2pa-rs/main/sdk/tests/fixtures"
CAI_PHOTOS = ["C.jpg", "CA.jpg", "CIE-sig-CA.jpg", "C_with_CAWG_data.jpg"]


def real_photo_crops(tmpdir: Path) -> list[tuple[str, Image.Image]]:
    """New images: center/corner crops of the real CAI photographs."""
    out = []
    for i, name in enumerate(CAI_PHOTOS):
        cached = REPO / "fixtures" / "external" / name
        try:
            raw = cached.read_bytes() if cached.exists() else urllib.request.urlopen(
                f"{CAI_BASE}/{name}", timeout=30).read()
            p = tmpdir / name
            p.write_bytes(raw)
            img = Image.open(p).convert("RGB")
            w, h = img.size
            # a different crop per image so the dataset differs from testset
            crops = [
                (0, 0, w * 2 // 3, h * 2 // 3),
                (w // 3, 0, w, h * 2 // 3),
                (0, h // 3, w * 2 // 3, h),
                (w // 6, h // 6, w * 5 // 6, h * 5 // 6),
            ]
            crop = img.crop(crops[i % 4]).copy()
            stem = name.rsplit(".", 1)[0].lower().replace("-", "_")
            out.append((f"crop_{stem}", crop))
            print(f"real photo crop: {name} -> {crop.size[0]}x{crop.size[1]}")
        except Exception as exc:
            print(f"warn: skipping {name}: {exc}")
    return out


def main() -> int:
    for sub in ("original", "ai", "special", "cloud_store"):
        (DATASET / sub).mkdir(parents=True, exist_ok=True)
    labels: dict[str, dict] = {}

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        certs, key = get_signing_credentials(tmpdir)

        bases = real_photo_crops(tmpdir)
        bases += [(f"render_{i:02d}", make_base_image(300 + i * 7)) for i in range(1, 5)]

        # ---- originals: clean saves
        for name, img in bases:
            fname = f"{name}.jpg"
            img.save(DATASET / "original" / fname, "JPEG", quality=91)
            labels[f"original/{fname}"] = {"set": "original", "expected": "NO_AI_EVIDENCE"}

        # ---- AI images
        specs = [
            ("firefly", "AI_GENERATED"),  # signed Firefly creation, JPEG
            ("firefly_png", "AI_GENERATED"),  # signed Firefly creation, PNG
            ("genfill", "AI_MODIFIED"),  # signed Generative Fill
            ("genfill", "AI_MODIFIED"),
            ("genfill_png", "AI_MODIFIED"),  # signed Generative Fill, PNG
            ("genfill", "AI_MODIFIED"),
            ("xmp", "AI_MODIFIED"),  # XMP-only composite
            ("xmp_gen", "AI_GENERATED"),  # XMP-only Firefly creation
        ]
        for i, (kind, expected) in enumerate(specs):
            base_name, base_img = bases[i % len(bases)]
            ext = "png" if kind.endswith("_png") else "jpg"
            fname = f"ai_{i:02d}_{kind}.{ext}"
            dst = DATASET / "ai" / fname

            if kind.startswith("firefly"):
                rendered = make_base_image(400 + i * 11)
                src = tmpdir / f"gen_{i}.{ext}"
                rendered.save(src, "PNG" if ext == "png" else "JPEG", quality=92)
                sign_file(manifest_firefly_generated(fname), src, dst, certs, key)
            elif kind.startswith("genfill"):
                edited = apply_generative_patch(base_img, 50 + i)
                src = tmpdir / f"edit_{i}.{ext}"
                edited.save(src, "PNG" if ext == "png" else "JPEG", quality=92)
                sign_file(manifest_generative_fill(fname), src, dst, certs, key)
            elif kind == "xmp":
                edited = apply_generative_patch(base_img, 50 + i)
                src = tmpdir / f"edit_{i}.jpg"
                edited.save(src, "JPEG", quality=92)
                packet = xmp_packet(creator_tool="Adobe Photoshop 25.0",
                                    digital_source_type=COMPOSITE)
                inject_xmp_jpeg(src, dst, packet)
            else:  # xmp_gen
                rendered = make_base_image(500 + i * 13)
                src = tmpdir / f"gen_{i}.jpg"
                rendered.save(src, "JPEG", quality=92)
                packet = xmp_packet(
                    creator_tool="Adobe Firefly 1.0",
                    digital_source_type="http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia",
                )
                inject_xmp_jpeg(src, dst, packet)
            labels[f"ai/{fname}"] = {"set": "ai", "expected": expected}
            print(f"ai: {fname} [{expected}]")

        # ---- durable-credential special case: stripped AI image + cloud store
        signed = DATASET / "ai" / "ai_00_firefly.jpg"
        stripped_src = Image.open(signed)
        stripped_src.load()
        clean = Image.new("RGB", stripped_src.size)
        clean.paste(stripped_src)
        clean.save(DATASET / "special" / "stripped_recoverable.jpg", "JPEG", quality=92)
        labels["special/stripped_recoverable.jpg"] = {
            "set": "special",
            "expected": "NO_AI_EVIDENCE",  # without cloud recovery
            "expected_with_cloud": "AI_GENERATED",
            "soft_binding_id": SOFT_BINDING_ID,
        }
        # The manifest store the cloud serves for its soft-binding ID: the same
        # store the signed original carries (read back via the c2pa library).
        from aidetect.c2pa_reader import read_manifest_store  # noqa: E402
        store, _state = read_manifest_store(str(signed))
        if store:
            (DATASET / "cloud_store" / f"{SOFT_BINDING_ID}.json").write_text(
                json.dumps(store, indent=1) + "\n"
            )
            print(f"cloud_store: {SOFT_BINDING_ID}.json (recovered-manifest fixture)")
        else:
            print("warn: could not read manifest back for cloud_store fixture")

    (DATASET / "labels.json").write_text(json.dumps(labels, indent=2) + "\n")
    n = {s: sum(1 for v in labels.values() if v["set"] == s)
         for s in ("original", "ai", "special")}
    print(f"\nWrote dataset: {n['original']} originals, {n['ai']} AI, "
          f"{n['special']} special to {DATASET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
