"""Best-effort download of real-world C2PA sample images for manual demos.

These come from the Content Authenticity Initiative's open-source test
fixtures. They are NOT part of the labeled evaluation set (their provenance
is not authored by this project) — they exist so you can point the detector
at images signed by real tooling:

    python -m aidetect fixtures/external/*
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

DEST = Path(__file__).resolve().parent.parent / "fixtures" / "external"

BASE = "https://raw.githubusercontent.com/contentauth/c2pa-rs/main/sdk/tests/fixtures"
SAMPLES = ["C.jpg", "CA.jpg", "CIE-sig-CA.jpg", "C_with_CAWG_data.jpg"]


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    ok = 0
    for name in SAMPLES:
        url = f"{BASE}/{name}"
        try:
            data = urllib.request.urlopen(url, timeout=30).read()
            (DEST / name).write_bytes(data)
            print(f"downloaded {name} ({len(data)} bytes)")
            ok += 1
        except Exception as exc:
            print(f"warn: could not download {name}: {exc}")
    print(f"\n{ok}/{len(SAMPLES)} samples in {DEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
