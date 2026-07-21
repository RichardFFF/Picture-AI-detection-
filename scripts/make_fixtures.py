"""Generate the labeled test picture set in fixtures/.

Builds deterministic base images with Pillow, then:
- signs real C2PA manifests (Firefly-style AI actions) with ES256 test certs,
- injects Adobe-style XMP packets for the metadata-only variants,
- produces clean negatives and a metadata-stripped AI image.

Writes fixtures/labels.json mapping filename -> expected verdict.

Signing certs: tries to download the Content Authenticity Initiative public
test certs from GitHub; falls back to generating a local mini-CA + leaf that
satisfies the C2PA certificate profile (digitalSignature + emailProtection,
CA:FALSE).
"""
from __future__ import annotations

import json
import struct
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw
from PIL.PngImagePlugin import PngInfo

REPO = Path(__file__).resolve().parent.parent
FIXTURES = REPO / "fixtures"

TRAINED = "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"
COMPOSITE = "http://cv.iptc.org/newscodes/digitalsourcetype/compositeWithTrainedAlgorithmicMedia"

CERT_URLS = [
    (
        "https://raw.githubusercontent.com/contentauth/c2pa-python/main/tests/fixtures/es256_certs.pem",
        "https://raw.githubusercontent.com/contentauth/c2pa-python/main/tests/fixtures/es256_private.key",
    ),
    (
        "https://raw.githubusercontent.com/contentauth/c2pa-rs/main/sdk/tests/fixtures/certs/es256.pub",
        "https://raw.githubusercontent.com/contentauth/c2pa-rs/main/sdk/tests/fixtures/certs/es256.pem",
    ),
]


# ---------------------------------------------------------------- base images
def make_base_image(seed: int) -> Image.Image:
    """Deterministic 256x256 gradient + shapes, distinct per seed."""
    img = Image.new("RGB", (256, 256))
    px = img.load()
    for y in range(256):
        for x in range(256):
            px[x, y] = (
                (x + seed * 37) % 256,
                (y + seed * 71) % 256,
                (x + y + seed * 13) % 256,
            )
    draw = ImageDraw.Draw(img)
    draw.ellipse(
        [40 + seed * 5, 40, 160 + seed * 5, 160],
        fill=((seed * 60) % 256, (seed * 90) % 256, (seed * 120) % 256),
    )
    draw.rectangle([150, 150 + (seed * 7) % 40, 230, 230], outline=(255, 255, 255), width=3)
    return img


# ---------------------------------------------------------------- certificates
def get_signing_credentials(workdir: Path) -> tuple[bytes, bytes]:
    """Return (cert_chain_pem, private_key_pem)."""
    for cert_url, key_url in CERT_URLS:
        try:
            certs = urllib.request.urlopen(cert_url, timeout=30).read()
            key = urllib.request.urlopen(key_url, timeout=30).read()
            if b"BEGIN CERTIFICATE" in certs and b"PRIVATE KEY" in key:
                print(f"Using downloaded test certs from {cert_url}")
                return certs, key
        except Exception as exc:
            print(f"warn: cert download failed ({cert_url}): {exc}")

    print("Generating local ES256 mini-CA + leaf certificate")
    ext = workdir / "ext.cnf"
    ext.write_text(
        "basicConstraints=CA:FALSE\n"
        "keyUsage=digitalSignature\n"
        "extendedKeyUsage=emailProtection\n"
    )

    def run(*cmd: str) -> None:
        subprocess.run(cmd, check=True, cwd=workdir, capture_output=True)

    run("openssl", "ecparam", "-name", "prime256v1", "-genkey", "-noout", "-out", "ca.key")
    run("openssl", "req", "-x509", "-new", "-key", "ca.key", "-days", "3650",
        "-subj", "/O=AIDetect Test/CN=AIDetect Test CA", "-out", "ca.pem")
    run("openssl", "ecparam", "-name", "prime256v1", "-genkey", "-noout", "-out", "leaf.key")
    run("openssl", "req", "-new", "-key", "leaf.key",
        "-subj", "/O=AIDetect Test/CN=AIDetect Test Signer", "-out", "leaf.csr")
    run("openssl", "x509", "-req", "-in", "leaf.csr", "-CA", "ca.pem", "-CAkey", "ca.key",
        "-CAcreateserial", "-days", "3650", "-extfile", "ext.cnf", "-out", "leaf.pem")
    # PKCS#8 key format is what c2pa expects
    run("openssl", "pkcs8", "-topk8", "-nocrypt", "-in", "leaf.key", "-out", "leaf.pk8")
    chain = (workdir / "leaf.pem").read_bytes() + (workdir / "ca.pem").read_bytes()
    return chain, (workdir / "leaf.pk8").read_bytes()


# ---------------------------------------------------------------- manifests
def manifest_firefly_generated(title: str) -> dict:
    return {
        "claim_generator_info": [{"name": "Adobe Firefly", "version": "1.0.0"}],
        "title": title,
        "assertions": [
            {
                "label": "c2pa.actions",
                "data": {
                    "actions": [
                        {
                            "action": "c2pa.created",
                            "softwareAgent": {"name": "Adobe Firefly"},
                            "digitalSourceType": TRAINED,
                        }
                    ]
                },
            }
        ],
    }


def manifest_generative_fill(title: str) -> dict:
    return {
        "claim_generator_info": [{"name": "Adobe Photoshop", "version": "25.0"}],
        "title": title,
        "assertions": [
            {
                "label": "c2pa.actions",
                "data": {
                    "actions": [
                        {"action": "c2pa.opened"},
                        {
                            "action": "c2pa.placed",
                            "softwareAgent": {"name": "Adobe Firefly"},
                            "digitalSourceType": COMPOSITE,
                        },
                    ]
                },
            }
        ],
    }


def manifest_ps_edit(title: str) -> dict:
    return {
        "claim_generator_info": [{"name": "Adobe Photoshop", "version": "25.0"}],
        "title": title,
        "assertions": [
            {
                "label": "c2pa.actions",
                "data": {
                    "actions": [
                        {"action": "c2pa.opened"},
                        {"action": "c2pa.color_adjustments"},
                    ]
                },
            }
        ],
    }


# ---------------------------------------------------------------- XMP packets
def xmp_packet(creator_tool: str | None = None, digital_source_type: str | None = None) -> str:
    attrs = ['xmlns:xmp="http://ns.adobe.com/xap/1.0/"',
             'xmlns:Iptc4xmpExt="http://iptc.org/std/Iptc4xmpExt/2008-02-29/"']
    fields = []
    if creator_tool:
        fields.append(f"<xmp:CreatorTool>{creator_tool}</xmp:CreatorTool>")
    if digital_source_type:
        fields.append(
            f"<Iptc4xmpExt:DigitalSourceType>{digital_source_type}</Iptc4xmpExt:DigitalSourceType>"
        )
    return (
        '<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
        '<x:xmpmeta xmlns:x="adobe:ns:meta/">\n'
        ' <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">\n'
        f'  <rdf:Description rdf:about="" {" ".join(attrs)}>\n'
        f'   {"".join(fields)}\n'
        "  </rdf:Description>\n"
        " </rdf:RDF>\n"
        "</x:xmpmeta>\n"
        '<?xpacket end="w"?>'
    )


def inject_xmp_jpeg(src: Path, dst: Path, packet: str) -> None:
    """Splice an APP1 XMP segment right after SOI (and any APP0/JFIF)."""
    data = src.read_bytes()
    assert data[:2] == b"\xff\xd8", "not a JPEG"
    insert_at = 2
    # keep APP0 (JFIF) first if present, as Adobe tools do
    if data[2:4] == b"\xff\xe0":
        seglen = int.from_bytes(data[4:6], "big")
        insert_at = 4 + seglen
    payload = b"http://ns.adobe.com/xap/1.0/\x00" + packet.encode("utf-8")
    segment = b"\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload
    dst.write_bytes(data[:insert_at] + segment + data[insert_at:])


def save_png_with_xmp(img: Image.Image, dst: Path, packet: str) -> None:
    info = PngInfo()
    info.add_itxt("XML:com.adobe.xmp", packet, zip=False)
    img.save(dst, "PNG", pnginfo=info)


# ---------------------------------------------------------------- signing
def sign_file(manifest: dict, src: Path, dst: Path, certs: bytes, key: bytes) -> None:
    import ctypes

    import c2pa

    # The wrapper's __init__ rejects ta_url=None, but the native struct wants a
    # NULL pointer to skip timestamping (an empty string fails with
    # "Signature: empty string"). Build the struct directly.
    signer_info = c2pa.C2paSignerInfo.__new__(c2pa.C2paSignerInfo)
    ctypes.Structure.__init__(signer_info, b"es256", certs, key, None)
    signer = c2pa.Signer.from_info(signer_info)
    builder = c2pa.Builder.from_json(json.dumps(manifest))
    builder.sign_file(str(src), str(dst), signer)


# ---------------------------------------------------------------- main
def main() -> int:
    FIXTURES.mkdir(exist_ok=True)
    labels: dict[str, str] = {}

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        certs, key = get_signing_credentials(tmpdir)

        # --- base images (unsigned sources for the signed variants)
        bases = {}
        for i, name in enumerate(
            ["firefly", "genfill", "psedit", "clean_a", "clean_b", "exif", "tricky"]
        ):
            bases[name] = make_base_image(i + 1)

        src_jpg = {}
        for name, img in bases.items():
            p = tmpdir / f"{name}.jpg"
            img.save(p, "JPEG", quality=90)
            src_jpg[name] = p

        # --- signed C2PA fixtures
        signed_specs = [
            ("firefly_gen.jpg", "firefly", manifest_firefly_generated, "AI_GENERATED"),
            ("genfill.jpg", "genfill", manifest_generative_fill, "AI_MODIFIED"),
            ("ps_edit.jpg", "psedit", manifest_ps_edit, "EDITED_NO_AI"),
        ]
        for fname, base, mk_manifest, label in signed_specs:
            sign_file(mk_manifest(fname), src_jpg[base], FIXTURES / fname, certs, key)
            labels[fname] = label
            print(f"signed  {fname}")

        # PNG signed variants
        for fname, base, mk_manifest, label in [
            ("firefly_gen.png", "firefly", manifest_firefly_generated, "AI_GENERATED"),
            ("genfill.png", "genfill", manifest_generative_fill, "AI_MODIFIED"),
        ]:
            src_png = tmpdir / f"{base}.png"
            bases[base].save(src_png, "PNG")
            sign_file(mk_manifest(fname), src_png, FIXTURES / fname, certs, key)
            labels[fname] = label
            print(f"signed  {fname}")

        # --- XMP-only fixtures (no C2PA manifest)
        packet = xmp_packet(creator_tool="Adobe Firefly 1.0", digital_source_type=TRAINED)
        inject_xmp_jpeg(src_jpg["firefly"], FIXTURES / "firefly_gen_xmp_only.jpg", packet)
        labels["firefly_gen_xmp_only.jpg"] = "AI_GENERATED"

        packet = xmp_packet(creator_tool="Adobe Photoshop 25.0", digital_source_type=COMPOSITE)
        inject_xmp_jpeg(src_jpg["genfill"], FIXTURES / "genfill_xmp_only.jpg", packet)
        labels["genfill_xmp_only.jpg"] = "AI_MODIFIED"

        packet = xmp_packet(creator_tool="Adobe Photoshop 25.0")
        inject_xmp_jpeg(src_jpg["psedit"], FIXTURES / "ps_edit_xmp_only.jpg", packet)
        labels["ps_edit_xmp_only.jpg"] = "EDITED_NO_AI"

        # --- negatives
        bases["clean_a"].save(FIXTURES / "clean.jpg", "JPEG", quality=90)
        labels["clean.jpg"] = "NO_AI_EVIDENCE"

        bases["clean_b"].save(FIXTURES / "clean.png", "PNG")
        labels["clean.png"] = "NO_AI_EVIDENCE"

        # camera-style EXIF only (no XMP, no C2PA)
        exif = Image.Exif()
        exif[0x010F] = "Canon"          # Make
        exif[0x0110] = "Canon EOS R5"   # Model
        exif[0x0132] = "2024:06:01 12:00:00"
        bases["exif"].save(FIXTURES / "clean_exif.jpg", "JPEG", quality=90, exif=exif)
        labels["clean_exif.jpg"] = "NO_AI_EVIDENCE"

        # AI image with metadata stripped by re-saving — documents the limitation
        stripped = Image.open(FIXTURES / "firefly_gen.jpg")
        stripped.load()
        clean_copy = Image.new("RGB", stripped.size)
        clean_copy.paste(stripped)
        clean_copy.save(FIXTURES / "stripped_ai.jpg", "JPEG", quality=90)
        labels["stripped_ai.jpg"] = "NO_AI_EVIDENCE"

        # adversarial filename: proves we classify on metadata, not names
        bases["tricky"].save(FIXTURES / "clean_tricky_name_firefly.png", "PNG")
        labels["clean_tricky_name_firefly.png"] = "NO_AI_EVIDENCE"

    (FIXTURES / "labels.json").write_text(json.dumps(labels, indent=2) + "\n")
    print(f"\nWrote {len(labels)} fixtures + labels.json to {FIXTURES}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
