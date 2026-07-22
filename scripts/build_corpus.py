"""Assemble the ML training corpus: genuine AI images vs real photographs.

Sources (all public open-source repo assets on raw.githubusercontent.com):
- AI class: Stable Diffusion official sample outputs (CompVis/stable-diffusion),
  latent-diffusion samples, ControlNet result pages (lllyasviel/ControlNet).
  Held out for the pipeline benchmark (never trained on): merged-0007,
  mountains-3, upscaling-out.
- Real class: OpenCV sample photographs, CAI test photographs, Pillow test
  photos.

Each source image is sliced into 256x256 tiles (sampled on a grid) so the
classifier sees many patches per image. Tiles inherit their parent image's
label AND parent id — the train/test split is done at PARENT level to
prevent leakage.

Outputs: corpus/ai/*.png, corpus/real/*.png  (tile files named
<parent>__r<row>c<col>.png)
"""
from __future__ import annotations

import io
import sys
import urllib.request
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parent.parent
CORPUS = REPO / "corpus"

SD = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets"
CN = "https://raw.githubusercontent.com/lllyasviel/ControlNet/main"
CV = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data"
CAI = "https://raw.githubusercontent.com/contentauth/c2pa-rs/main/sdk/tests/fixtures"
PIL_T = "https://raw.githubusercontent.com/python-pillow/Pillow/main/Tests/images"
SKI = "https://raw.githubusercontent.com/scikit-image/scikit-image/v0.19.3/skimage/data"

AI_SOURCES = {
    "sd_txt2img_0005": f"{SD}/stable-samples/txt2img/merged-0005.png",
    "sd_txt2img_0006": f"{SD}/stable-samples/txt2img/merged-0006.png",

    "sd_img2img_mountains1": f"{SD}/stable-samples/img2img/mountains-1.png",
    "sd_img2img_mountains2": f"{SD}/stable-samples/img2img/mountains-2.png",


    "sd_preview": f"{SD}/txt2img-preview.png",
    "sd_convsample": f"{SD}/txt2img-convsample.png",
    "ld_birdhouse": "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/assets/birdhouse.png",
    # ControlNet result pages: collages dominated by generated outputs.
    # Conditioning inputs are filtered out (leftmost column dropped +
    # near-grayscale map tiles skipped); residual label noise is accepted.
    **{f"cn_p{i}": f"{CN}/github_page/p{i}.png" for i in range(1, 22)},
}

REAL_SOURCES = {
    "cv_baboon": f"{CV}/baboon.jpg",
    "cv_fruits": f"{CV}/fruits.jpg",
    "cv_building": f"{CV}/building.jpg",
    "cv_lena": f"{CV}/lena.jpg",
    "cv_left01": f"{CV}/left01.jpg",
    "cv_aloel": f"{CV}/aloeL.jpg",
    "cv_home": f"{CV}/home.jpg",
    "cv_messi": f"{CV}/messi5.jpg",
    "cv_starry": f"{CV}/starry_night.jpg",
    "cv_graf1": f"{CV}/graf1.png",
    "cv_box_in_scene": f"{CV}/box_in_scene.png",
    "cai_c": f"{CAI}/C.jpg",
    "cai_ca": f"{CAI}/CA.jpg",
    "cai_cie": f"{CAI}/CIE-sig-CA.jpg",
    "cai_cawg": f"{CAI}/C_with_CAWG_data.jpg",
    "pil_flower": f"{PIL_T}/flower.jpg",
    "cv_stuff": f"{CV}/stuff.jpg",
    "cv_butterfly": f"{CV}/butterfly.jpg",
    "cv_chicky": f"{CV}/chicky_512.png",
    "cv_orange": f"{CV}/orange.jpg",
    "cv_right01": f"{CV}/right01.jpg",
    "cv_smarties": f"{CV}/smarties.png",
    "cv_sudoku": f"{CV}/sudoku.png",
    "cv_text_defocus": f"{CV}/text_defocus.jpg",
    "cv_lena_tmpl": f"{CV}/lena_tmpl.jpg",
    "cv_apple": f"{CV}/apple.jpg",
    "cv_blox": f"{CV}/blox.jpg",
    "cv_board": f"{CV}/board.jpg",
    "cv_happyfish": f"{CV}/HappyFish.jpg",
    "cv_rubberwhale": f"{CV}/rubberwhale1.png",
    "tv_dog1": "https://raw.githubusercontent.com/pytorch/vision/main/gallery/assets/dog1.jpg",
    "tv_dog2": "https://raw.githubusercontent.com/pytorch/vision/main/gallery/assets/dog2.jpg",
    "tv_hopper": "https://raw.githubusercontent.com/pytorch/vision/main/test/assets/encode_jpeg/grace_hopper_517x606.jpg",
    "esr_0014": "https://raw.githubusercontent.com/xinntao/Real-ESRGAN/master/inputs/0014.jpg",

    "esr_ade": "https://raw.githubusercontent.com/xinntao/Real-ESRGAN/master/inputs/ADE_val_00000114.jpg",
    # pristine (never-JPEG) camera photos from scikit-image's data folder
    "ski_astronaut": f"{SKI}/astronaut.png",

    "ski_chelsea": f"{SKI}/chelsea.png",
    "ski_rocket": f"{SKI}/rocket.jpg",
    "ski_hubble": f"{SKI}/hubble_deep_field.jpg",
    "ski_ihc": f"{SKI}/ihc.png",
    "ski_retina": f"{SKI}/retina.jpg",
}

# BSDS500: 481x321 real photographs (Berkeley segmentation dataset, GitHub
# mirror). IDs are the dataset's standard image numbers; each is tried in
# test/, val/ and train/ since the split assignment varies by id.
BSDS = "https://raw.githubusercontent.com/BIDS/BSDS500/master/BSDS500/data/images"
BSDS_IDS = [
    3096, 8023, 12084, 14037, 16077, 21077, 24077, 33039, 37073, 38082,
    41033, 42049, 43074, 45096, 54082, 55073, 58060, 62096, 65033, 66053,
    69015, 69040, 76053, 78004, 85048, 86000, 86068, 87046, 89072, 97033,
    100007, 100075, 100099, 101085, 101087, 102061, 103070, 105025, 106024,
    108005, 118035, 119082, 123074, 126007, 130026, 134035, 143090, 145086,
    148089, 156065, 157055, 159008, 160068, 163085, 167062, 170057, 175043,
    182053, 189080, 196073, 197017, 208001, 210088, 216081, 219090, 220075,
    223061, 227092, 229036, 236037, 241004, 253027, 260058, 271035, 285079,
    291000, 295087, 296007, 299086, 300091, 302008, 304034, 306005,
]
BSDS_MAX = 80

# The LAST 10 BSDS ids are never used for training — they are the real-photo
# half of the pipeline benchmark (scripts/make_benchmark.py)
BSDS_HOLDOUT = BSDS_IDS[-10:]

TILE = 256
MAX_TILES_PER_SOURCE = 16
MAX_TILES_PURE_AI = 40  # the SD grids hold many independent samples each
MIN_TILE_STD = 12.0  # skip near-uniform tiles (white margins, text panels)


def tiles_from(img: Image.Image, skip_first_col: bool = False,
               max_tiles: int = MAX_TILES_PER_SOURCE):
    import numpy as np

    w, h = img.size
    cols = max(1, w // TILE)
    rows = max(1, h // TILE)
    # sample a uniform grid, capped per source so huge collages don't dominate
    # anchor positions: regular grid plus right/bottom edges, so images just
    # over one tile wide (e.g. BSDS 481x321) still contribute two crops
    xs = sorted({c * TILE for c in range(cols) if c * TILE + TILE <= w} | ({w - TILE} if w >= TILE else set()))
    ys = sorted({r * TILE for r in range(rows) if r * TILE + TILE <= h} | ({h - TILE} if h >= TILE else set()))
    if skip_first_col and len(xs) > 1:
        xs = xs[1:]
    coords = [(y, x) for y in ys for x in xs]
    step = max(1, len(coords) // max_tiles)
    emitted = 0
    for y, x in coords[::step]:
        if emitted >= max_tiles:
            break
        tile = img.crop((x, y, x + TILE, y + TILE))
        if float(np.asarray(tile.convert("L"), dtype=np.float64).std()) < MIN_TILE_STD:
            continue
        if skip_first_col:
            # ControlNet collages: also drop conditioning maps (canny/depth/
            # pose renders are near-grayscale) wherever they appear
            arr = np.asarray(tile, dtype=np.float64)
            sat = ((arr.max(axis=2) - arr.min(axis=2)) / (arr.max(axis=2) + 1e-9)).mean()
            if sat < 0.08:
                continue
        emitted += 1
        yield y // TILE, x // TILE, tile


def fetch(url: str) -> Image.Image | None:
    try:
        data = urllib.request.urlopen(url, timeout=60).read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
        img.load()
        return img
    except Exception as exc:
        print(f"warn: {url}: {exc}")
        return None


def build(sources: dict[str, str], outdir: Path, augment_compression: bool = False) -> int:
    import io as _io

    outdir.mkdir(parents=True, exist_ok=True)
    n = 0
    for name, url in sources.items():
        img = fetch(url)
        if img is None:
            continue
        count = 0
        is_pure_ai_grid = name.startswith(("sd_", "ld_"))
        for r, c, tile in tiles_from(
            img,
            skip_first_col=name.startswith("cn_"),
            max_tiles=MAX_TILES_PURE_AI if is_pure_ai_grid else MAX_TILES_PER_SOURCE,
        ):
            # uniform re-encode: both classes become JPEG q95 so the model
            # cannot cheat by learning container/compression differences
            tile.save(outdir / f"{name}__r{r}c{c}.jpg", "JPEG", quality=95)
            count += 1
            if augment_compression:
                # AI renders are pristine while many real photos carry heavy
                # prior JPEG history; adding a q75-then-q95 variant of every
                # AI tile stops the model from learning "compression level"
                # as a shortcut for "real"
                buf = _io.BytesIO()
                tile.save(buf, "JPEG", quality=75)
                buf.seek(0)
                Image.open(buf).convert("RGB").save(
                    outdir / f"{name}__r{r}c{c}_q75.jpg", "JPEG", quality=95
                )
                count += 1
        n += count
        print(f"{name}: {count} tiles ({img.size[0]}x{img.size[1]})")
    return n


def build_bsds(outdir: Path) -> int:
    outdir.mkdir(parents=True, exist_ok=True)
    n = ok = 0
    for bid in [b for b in BSDS_IDS if b not in BSDS_HOLDOUT]:
        if ok >= BSDS_MAX:
            break
        img = None
        for split in ("test", "val", "train"):
            img = fetch(f"{BSDS}/{split}/{bid}.jpg")
            if img is not None:
                break
        if img is None:
            continue
        ok += 1
        import io as _io
        for r, c, tile in tiles_from(img):
            tile.save(outdir / f"bsds_{bid}__r{r}c{c}.jpg", "JPEG", quality=95)
            n += 1
            buf = _io.BytesIO()
            tile.save(buf, "JPEG", quality=75)
            buf.seek(0)
            Image.open(buf).convert("RGB").save(
                outdir / f"bsds_{bid}__r{r}c{c}_q75.jpg", "JPEG", quality=95)
            n += 1
    print(f"bsds: {ok} photos -> {n} tiles")
    return n


def main() -> int:
    n_ai = build(AI_SOURCES, CORPUS / "ai", augment_compression=True)
    n_real = build(REAL_SOURCES, CORPUS / "real", augment_compression=True)
    n_real += build_bsds(CORPUS / "real")
    print(f"\nCorpus: {n_ai} AI tiles, {n_real} real tiles in {CORPUS}")
    return 0 if n_ai and n_real else 1


if __name__ == "__main__":
    sys.exit(main())
