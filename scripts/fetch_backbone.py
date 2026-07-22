"""Download the open-source ImageNet backbone used for transfer learning.

Model: ResNet18 v1.7 from the ONNX Model Zoo (github.com/onnx/models,
Apache-2.0). The file is stored in the repo via git-LFS, so the actual bytes
are served from media.githubusercontent.com.

Saved to models/resnet18-v1-7.onnx (gitignored; ~45 MB).
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEST = REPO / "models" / "resnet18-v1-7.onnx"

URL = ("https://media.githubusercontent.com/media/onnx/models/main/"
       "validated/vision/classification/resnet/model/resnet18-v1-7.onnx")


def main() -> int:
    DEST.parent.mkdir(exist_ok=True)
    if DEST.exists() and DEST.stat().st_size > 10_000_000:
        print(f"already present: {DEST} ({DEST.stat().st_size // 1_000_000} MB)")
        return 0
    print(f"downloading {URL} ...")
    data = urllib.request.urlopen(URL, timeout=300).read()
    if len(data) < 10_000_000 or data[:7] == b"version":
        print("error: got an LFS pointer or truncated file, not the model")
        return 1
    DEST.write_bytes(data)
    print(f"saved {DEST} ({len(data) // 1_000_000} MB, "
          f"sha256 {hashlib.sha256(data).hexdigest()[:16]}...)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
