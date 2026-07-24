"""Add your own local image folders to the training corpus.

Use this to train on datasets that can't be fetched inside a network-
restricted environment (HuggingFace, Kaggle, Google Drive, ...): download and
unzip them on your own machine, then point this script at two folders — one
of AI-generated images, one of real images.

    python scripts/ingest_local.py --ai-dir path/to/ai --real-dir path/to/real
    python scripts/train_ml.py

Images are tiled to 256x256 and augmented exactly like scripts/build_corpus.py
(q95 + a q75 recompressed variant), and each source image becomes its own
parent group so the train/test split never leaks tiles. Small images are
upscaled so their short side reaches one full tile. Existing corpus contents
are kept — run scripts/build_corpus.py first if you also want the GitHub
base corpus, or use --fresh to start from only your local images.
"""
from __future__ import annotations

import argparse
import io
import shutil
import sys
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.build_corpus import (  # noqa: E402
    CORPUS,
    MAX_TILES_PER_SOURCE,
    TILE,
    tiles_from,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def load_upscaled(path: Path) -> Image.Image | None:
    try:
        img = Image.open(path).convert("RGB")
        img.load()
    except Exception as exc:
        print(f"warn: skip {path.name}: {exc}")
        return None
    w, h = img.size
    if min(w, h) < TILE:
        s = TILE / min(w, h)
        img = img.resize((round(w * s), round(h * s)))
    return img


def ingest(src_dir: Path, out_sub: str, prefix: str, limit: int | None) -> int:
    outdir = CORPUS / out_sub
    outdir.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in src_dir.rglob("*")
                   if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if limit:
        files = files[:limit]
    n_tiles = 0
    for i, path in enumerate(files):
        img = load_upscaled(path)
        if img is None:
            continue
        # unique, filesystem-safe parent id per source image
        pid = f"{prefix}_{i:05d}"
        for r, c, tile in tiles_from(img, max_tiles=MAX_TILES_PER_SOURCE):
            tile.save(outdir / f"{pid}__r{r}c{c}.jpg", "JPEG", quality=95)
            n_tiles += 1
            buf = io.BytesIO()
            tile.save(buf, "JPEG", quality=75)
            buf.seek(0)
            Image.open(buf).convert("RGB").save(
                outdir / f"{pid}__r{r}c{c}_q75.jpg", "JPEG", quality=95
            )
            n_tiles += 1
    print(f"{out_sub}: {len(files)} images from {src_dir} -> {n_tiles} tiles")
    return n_tiles


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ai-dir", type=Path, help="folder of AI-generated images")
    ap.add_argument("--real-dir", type=Path, help="folder of real images")
    ap.add_argument("--prefix", default="local",
                    help="parent-id prefix for these images (default 'local')")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap images taken per folder")
    ap.add_argument("--fresh", action="store_true",
                    help="delete any existing corpus first (only your images)")
    args = ap.parse_args()

    if not args.ai_dir and not args.real_dir:
        ap.error("give at least one of --ai-dir / --real-dir")
    for d in (args.ai_dir, args.real_dir):
        if d and not d.is_dir():
            ap.error(f"{d} is not a directory")

    if args.fresh and CORPUS.exists():
        shutil.rmtree(CORPUS)
        print(f"removed existing {CORPUS}")

    n = 0
    if args.ai_dir:
        n += ingest(args.ai_dir, "ai", f"{args.prefix}_ai", args.limit)
    if args.real_dir:
        n += ingest(args.real_dir, "real", f"{args.prefix}_real", args.limit)

    n_ai = len(list((CORPUS / "ai").glob("*.jpg"))) if (CORPUS / "ai").exists() else 0
    n_real = len(list((CORPUS / "real").glob("*.jpg"))) if (CORPUS / "real").exists() else 0
    print(f"\nCorpus now: {n_ai} AI tiles, {n_real} real tiles in {CORPUS}")
    print("Next: python scripts/train_ml.py")
    return 0 if n else 1


if __name__ == "__main__":
    sys.exit(main())
