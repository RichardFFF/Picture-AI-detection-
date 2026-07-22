"""Assemble the ML training corpus: genuine AI images vs real photographs.

All sources are files distributed inside permissively-licensed open-source
repositories (see SOURCES.md for the per-source license table). Sources with
research-only or unclear image copyright (BSDS500/Corel, lena, celebrity
photos, ADE20K) are deliberately NOT used.

Held out from training entirely (used by scripts/make_benchmark.py):
- AI: merged-0007, mountains-3, upscaling-out (CompVis samples)
- Real: the HOLDOUT_REAL_SOURCES listed below

Each source image is sliced into 256x256 tiles; tiles inherit their parent
image's label AND parent id, and the train/test split is done at PARENT level
to prevent leakage. Both classes are re-encoded JPEG q95, and every tile also
gets a q75-recompressed variant so the model cannot use compression level as
a class shortcut.

Outputs: corpus/ai/*.jpg, corpus/real/*.jpg
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
TV = "https://raw.githubusercontent.com/pytorch/vision/main"
ESR = "https://raw.githubusercontent.com/xinntao/Real-ESRGAN/master/inputs"

AI_SOURCES = {
    # CompVis/stable-diffusion (MIT repo): official SD sample outputs.
    # merged-0007, mountains-3 and upscaling-out are benchmark holdouts.
    "sd_txt2img_0005": f"{SD}/stable-samples/txt2img/merged-0005.png",
    "sd_txt2img_0006": f"{SD}/stable-samples/txt2img/merged-0006.png",
    "sd_img2img_mountains1": f"{SD}/stable-samples/img2img/mountains-1.png",
    "sd_img2img_mountains2": f"{SD}/stable-samples/img2img/mountains-2.png",
    "sd_preview": f"{SD}/txt2img-preview.png",
    "sd_convsample": f"{SD}/txt2img-convsample.png",
    "ld_birdhouse": "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/assets/birdhouse.png",
    # lllyasviel/ControlNet (Apache-2.0): result collages dominated by
    # generated outputs; conditioning inputs filtered in tiles_from
    **{f"cn_p{i}": f"{CN}/github_page/p{i}.png" for i in range(1, 22)},
}

REAL_SOURCES = {
    # opencv/opencv samples/data (Apache-2.0)
    "cv_baboon": f"{CV}/baboon.jpg",
    "cv_fruits": f"{CV}/fruits.jpg",
    "cv_building": f"{CV}/building.jpg",
    "cv_left01": f"{CV}/left01.jpg",
    "cv_aloel": f"{CV}/aloeL.jpg",
    "cv_home": f"{CV}/home.jpg",
    "cv_starry": f"{CV}/starry_night.jpg",
    "cv_graf1": f"{CV}/graf1.png",
    "cv_box_in_scene": f"{CV}/box_in_scene.png",
    "cv_right01": f"{CV}/right01.jpg",
    "cv_smarties": f"{CV}/smarties.png",
    "cv_sudoku": f"{CV}/sudoku.png",
    "cv_text_defocus": f"{CV}/text_defocus.jpg",
    "cv_blox": f"{CV}/blox.jpg",
    "cv_board": f"{CV}/board.jpg",
    "cv_happyfish": f"{CV}/HappyFish.jpg",
    "cv_rubberwhale": f"{CV}/rubberwhale1.png",
    "cv_chicky": f"{CV}/chicky_512.png",
    # contentauth/c2pa-rs fixtures (MIT/Apache-2.0)
    "cai_c": f"{CAI}/C.jpg",
    "cai_ca": f"{CAI}/CA.jpg",
    "cai_cie": f"{CAI}/CIE-sig-CA.jpg",
    "cai_cawg": f"{CAI}/C_with_CAWG_data.jpg",
    # pytorch/vision (BSD-3)
    "tv_dog1": f"{TV}/gallery/assets/dog1.jpg",
    "tv_hopper": f"{TV}/test/assets/encode_jpeg/grace_hopper_517x606.jpg",
    # xinntao/Real-ESRGAN inputs (BSD-3); ADE20K-derived input excluded
    "esr_0014": f"{ESR}/0014.jpg",
    # scikit-image data (BSD-3 repo; several NASA public-domain images)
    "ski_astronaut": f"{SKI}/astronaut.png",
    "ski_chelsea": f"{SKI}/chelsea.png",
    "ski_hubble": f"{SKI}/hubble_deep_field.jpg",
    "ski_ihc": f"{SKI}/ihc.png",
    "ski_retina": f"{SKI}/retina.jpg",
    "ski_camera": f"{SKI}/camera.png",
    "ski_moon": f"{SKI}/moon.png",
    "ski_coins": f"{SKI}/coins.png",
    "ski_page": f"{SKI}/page.png",
    "ski_brick": f"{SKI}/brick.png",
    "ski_grass": f"{SKI}/grass.png",
    "ski_gravel": f"{SKI}/gravel.png",
    "ski_clock": f"{SKI}/clock_motion.png",
}

# Never used for training — the real-photo half of the pipeline benchmark
# (scripts/make_benchmark.py). Same licensing constraints as above.
HOLDOUT_REAL_SOURCES = {
    "cv_stuff": f"{CV}/stuff.jpg",
    "cv_butterfly": f"{CV}/butterfly.jpg",
    "cv_orange": f"{CV}/orange.jpg",
    "cv_apple": f"{CV}/apple.jpg",
    "tv_dog2": f"{TV}/gallery/assets/dog2.jpg",
    "esr_0030": f"{ESR}/0030.jpg",
    "pil_flower": f"{PIL_T}/flower.jpg",
    "ski_coffee": f"{SKI}/coffee.png",
    "ski_rocket": f"{SKI}/rocket.jpg",
    "ski_motorcycle": f"{SKI}/motorcycle_left.png",
}

TILE = 256
MAX_TILES_PER_SOURCE = 16
MAX_TILES_PURE_AI = 48  # the SD grids hold many independent samples each
MIN_TILE_STD = 12.0  # skip near-uniform tiles (white margins, text panels)


def tiles_from(img: Image.Image, skip_first_col: bool = False,
               max_tiles: int = MAX_TILES_PER_SOURCE, stride: int = TILE):
    import numpy as np

    w, h = img.size
    xs = sorted({x for x in range(0, max(1, w - TILE + 1), stride)}
                | ({w - TILE} if w >= TILE else set()))
    ys = sorted({y for y in range(0, max(1, h - TILE + 1), stride)}
                | ({h - TILE} if h >= TILE else set()))
    if skip_first_col and len(xs) > 1:
        xs = [x for x in xs if x >= TILE]
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
        yield y // 64, x // 64, tile


def fetch(url: str) -> Image.Image | None:
    try:
        data = urllib.request.urlopen(url, timeout=60).read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
        img.load()
        return img
    except Exception as exc:
        print(f"warn: {url}: {exc}")
        return None


def build(sources: dict[str, str], outdir: Path) -> int:
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
            stride=128 if is_pure_ai_grid else TILE,
        ):
            # uniform re-encode: both classes become JPEG q95 so the model
            # cannot cheat on container differences
            tile.save(outdir / f"{name}__r{r}c{c}.jpg", "JPEG", quality=95)
            count += 1
            # ...and a q75-then-q95 variant so compression level is not a
            # usable class shortcut either
            buf = io.BytesIO()
            tile.save(buf, "JPEG", quality=75)
            buf.seek(0)
            Image.open(buf).convert("RGB").save(
                outdir / f"{name}__r{r}c{c}_q75.jpg", "JPEG", quality=95
            )
            count += 1
        n += count
        print(f"{name}: {count} tiles ({img.size[0]}x{img.size[1]})")
    return n


def main() -> int:
    n_ai = build(AI_SOURCES, CORPUS / "ai")
    n_real = build(REAL_SOURCES, CORPUS / "real")
    print(f"\nCorpus: {n_ai} AI tiles, {n_real} real tiles in {CORPUS}")
    return 0 if n_ai and n_real else 1


if __name__ == "__main__":
    sys.exit(main())
