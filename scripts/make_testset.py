"""Build the paired evaluation set: testset/original/ vs testset/ai_modified/.

Originals are real photographs (Content Authenticity Initiative test images,
re-saved to strip their non-AI provenance) plus deterministic rendered scenes —
no AI metadata anywhere.

Each AI-modified counterpart gets a real pixel edit (an inpainted-style patch,
visually simulating Generative Fill) and genuine provenance metadata of the
kind Adobe tools write: most carry a real ES256-signed C2PA manifest
(Photoshop + Firefly `c2pa.placed` + compositeWithTrainedAlgorithmicMedia),
two are full Firefly-style creations (`c2pa.created` + trainedAlgorithmicMedia),
and two are XMP-only variants.

Writes testset/labels.json: filename -> {"set": ..., "expected": ...}.
"""
from __future__ import annotations

import json
import sys
import tempfile
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
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

REPO = Path(__file__).resolve().parent.parent
TESTSET = REPO / "testset"
ORIGINAL = TESTSET / "original"
AI_MODIFIED = TESTSET / "ai_modified"

CAI_BASE = "https://raw.githubusercontent.com/contentauth/c2pa-rs/main/sdk/tests/fixtures"
CAI_PHOTOS = ["C.jpg", "CA.jpg", "CIE-sig-CA.jpg", "C_with_CAWG_data.jpg"]


def fetch_real_photos(tmpdir: Path) -> list[tuple[str, Image.Image]]:
    """Real photographs from the CAI fixtures, metadata stripped by re-save."""
    photos = []
    for name in CAI_PHOTOS:
        cached = REPO / "fixtures" / "external" / name
        try:
            if cached.exists():
                raw = cached.read_bytes()
            else:
                raw = urllib.request.urlopen(f"{CAI_BASE}/{name}", timeout=30).read()
            p = tmpdir / name
            p.write_bytes(raw)
            img = Image.open(p)
            img.load()
            clean = Image.new("RGB", img.size)
            clean.paste(img.convert("RGB"))
            stem = name.rsplit(".", 1)[0].lower().replace("-", "_")
            photos.append((f"photo_{stem}", clean))
            print(f"real photo: {name} ({img.size[0]}x{img.size[1]})")
        except Exception as exc:
            print(f"warn: skipping {name}: {exc}")
    return photos


def rendered_scenes(count: int) -> list[tuple[str, Image.Image]]:
    return [(f"scene_{i:02d}", make_base_image(100 + i)) for i in range(1, count + 1)]


def apply_generative_patch(img: Image.Image, seed: int) -> Image.Image:
    """A real pixel edit: paste a smooth synthetic patch, like an inpaint."""
    out = img.copy()
    w, h = out.size
    pw, ph = max(32, w // 4), max(32, h // 4)
    px_off, py_off = (seed * 53) % (w - pw), (seed * 91) % (h - ph)
    patch = Image.new("RGB", (pw, ph))
    ppx = patch.load()
    for y in range(ph):
        for x in range(pw):
            ppx[x, y] = (
                (120 + x * 3 + seed * 17) % 256,
                (80 + y * 3 + seed * 29) % 256,
                (160 + (x + y) + seed * 41) % 256,
            )
    d = ImageDraw.Draw(patch)
    d.ellipse([pw // 4, ph // 4, 3 * pw // 4, 3 * ph // 4],
              fill=((seed * 77) % 256, (seed * 55) % 256, (seed * 33) % 256))
    out.paste(patch, (px_off, py_off))
    return out


def main() -> int:
    ORIGINAL.mkdir(parents=True, exist_ok=True)
    AI_MODIFIED.mkdir(parents=True, exist_ok=True)
    labels: dict[str, dict] = {}

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        certs, key = get_signing_credentials(tmpdir)

        bases = fetch_real_photos(tmpdir) + rendered_scenes(6)
        print(f"\n{len(bases)} base images -> building pairs\n")

        for i, (name, img) in enumerate(bases):
            # ---- original: clean save, no provenance
            orig_name = f"{name}.jpg"
            img.save(ORIGINAL / orig_name, "JPEG", quality=92)
            labels[f"original/{orig_name}"] = {
                "set": "original", "expected": "NO_AI_EVIDENCE",
            }

            # ---- AI-modified counterpart: real pixel edit + real provenance
            mod_name = f"{name}_ai.jpg"
            dst = AI_MODIFIED / mod_name
            if i % 5 == 3:
                # full AI creation (Firefly text-to-image style): pixels are a
                # fresh render, manifest says c2pa.created + trainedAlgorithmicMedia
                edited = make_base_image(200 + i)
                src = tmpdir / f"gen_{i}.jpg"
                edited.save(src, "JPEG", quality=92)
                sign_file(manifest_firefly_generated(mod_name), src, dst, certs, key)
                expected = "AI_GENERATED"
                how = "signed C2PA (Firefly created)"
            elif i % 5 == 4:
                # XMP-only generative fill
                edited = apply_generative_patch(img, i + 1)
                src = tmpdir / f"edit_{i}.jpg"
                edited.save(src, "JPEG", quality=92)
                packet = xmp_packet(creator_tool="Adobe Photoshop 25.0",
                                    digital_source_type=COMPOSITE)
                inject_xmp_jpeg(src, dst, packet)
                expected = "AI_MODIFIED"
                how = "XMP only (composite)"
            else:
                # signed generative-fill manifest
                edited = apply_generative_patch(img, i + 1)
                src = tmpdir / f"edit_{i}.jpg"
                edited.save(src, "JPEG", quality=92)
                sign_file(manifest_generative_fill(mod_name), src, dst, certs, key)
                expected = "AI_MODIFIED"
                how = "signed C2PA (Generative Fill)"
            labels[f"ai_modified/{mod_name}"] = {
                "set": "ai_modified", "expected": expected,
            }
            print(f"pair: {orig_name:<40} -> {mod_name} [{how}]")

    (TESTSET / "labels.json").write_text(json.dumps(labels, indent=2) + "\n")
    n_orig = sum(1 for v in labels.values() if v["set"] == "original")
    n_ai = len(labels) - n_orig
    print(f"\nWrote {n_orig} originals + {n_ai} AI-modified to {TESTSET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
