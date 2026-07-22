# Image & model sources and licenses

Every training/benchmark image and model is a file distributed inside a
permissively-licensed open-source repository; the repository's license is the
grant we rely on. Sources with research-only or unclear image copyright were
deliberately removed (BSDS500/Corel photographs, `lena`, celebrity press
photos, ADE20K-derived inputs).

| Source repository | License | Used for |
|---|---|---|
| [CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion) | CreativeML Open RAIL-M; its §6 claims **no rights over generated outputs**, and the sample assets we use are such outputs | AI class: official txt2img/img2img sample outputs; benchmark AI holdouts (`merged-0007`, `mountains-3`, `upscaling-out`) |
| [CompVis/latent-diffusion](https://github.com/CompVis/latent-diffusion) | MIT | AI class: `birdhouse.png` sample |
| [lllyasviel/ControlNet](https://github.com/lllyasviel/ControlNet) | Apache-2.0 | AI class: result-page collages (conditioning inputs filtered out) |
| [opencv/opencv](https://github.com/opencv/opencv) `samples/data` | Apache-2.0 | Real class + benchmark holdouts (`stuff`, `butterfly`, `orange`, `apple`) |
| [scikit-image/scikit-image](https://github.com/scikit-image/scikit-image) `v0.19.3 skimage/data` | BSD-3; several images NASA public domain (`astronaut`, `rocket`, `moon`, `hubble_deep_field`) | Real class + benchmark holdouts (`coffee`, `rocket`, `motorcycle_left`) |
| [contentauth/c2pa-rs](https://github.com/contentauth/c2pa-rs) test fixtures | MIT / Apache-2.0 | Real class (CAI photographs); C2PA demo samples |
| [contentauth/c2pa-python](https://github.com/contentauth/c2pa-python) test fixtures | MIT / Apache-2.0 | ES256 test signing certificates |
| [pytorch/vision](https://github.com/pytorch/vision) | BSD-3 | Real class (`dog1`, `grace_hopper`); benchmark holdout `dog2` |
| [python-pillow/Pillow](https://github.com/python-pillow/Pillow) test images | MIT-CMU (HPND) | Benchmark holdout `flower` |
| [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) inputs | BSD-3 (ADE20K-derived input excluded) | Real class (`0014`); benchmark holdout (`0030`) |
| [onnx/models](https://github.com/onnx/models) | Apache-2.0 | `resnet18-v1-7.onnx` ImageNet backbone for transfer learning |
| [adobe/trustmark](https://pypi.org/project/trustmark/) | MIT | Optional watermark decoding (durable credentials) |

## Commercial-use notes

- Every license above (MIT, BSD-3, Apache-2.0, HPND) permits commercial use.
  CreativeML Open RAIL-M also permits commercial use of the model subject to
  use restrictions, and disclaims rights over outputs (§6) — we only use
  outputs.
- The ResNet18 weights are distributed by `onnx/models` under Apache-2.0.
  As with all ImageNet-pretrained weights, the underlying training dataset
  (ImageNet) has research-oriented access terms; relying on the
  distributor's Apache-2.0 grant for the weights is standard industry
  practice, noted here for completeness.
- Python dependencies: c2pa-python (MIT/Apache-2.0), Pillow (HPND), numpy
  (BSD-3; wheel bundles OpenBLAS/LAPACK BSD and GCC runtime libs under
  GPL-with-linking-exception / LGPL — standard and commercially safe),
  onnxruntime (MIT), onnx (Apache-2.0), fastapi (MIT), uvicorn (BSD-3),
  python-multipart (Apache-2.0), pytest (MIT), httpx (BSD-3),
  scikit-learn (BSD-3, training only), trustmark (MIT, optional).

Notes:
- `fixtures/`, `testset/`, and `dataset/` images are generated locally
  (Pillow renders and crops of the CAI photographs above) with manifests we
  sign ourselves.
- `corpus/` (training tiles) and `wild/` are gitignored — reproducible via
  `scripts/build_corpus.py` and `scripts/fetch_wild_samples.py`.
- Committed `benchmark/` images come only from the repositories listed above.
