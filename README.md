# AI Image Detector

Identifies pictures **created or modified by AI** by reading the provenance
metadata generators embed, with optional statistical analysis for images
that carry none. Detected generator families:

- **Adobe Firefly / Photoshop Generative Fill & Expand** — C2PA Content
  Credentials + XMP (the flagship deterministic feature)
- **OpenAI DALL·E / ChatGPT images** — C2PA claim generators
- **Stable Diffusion WebUI, ComfyUI, NovelAI, InvokeAI, Midjourney** —
  generator metadata (PNG chunks, EXIF, XMP)

Ships with a CLI (single images or batch scans of whole directories), a
drag-and-drop web UI with multi-file batches, and labeled test sets where
the deterministic layers score **100%**.

## How it works — and why metadata detection can be exact

Statistical/pixel-based "AI detectors" guess, and guessing can never be 100%
accurate. The metadata layers of this app don't guess. Adobe Firefly and
Photoshop's generative features embed **C2PA Content Credentials** —
cryptographically signed provenance manifests — plus XMP metadata that
explicitly declare AI involvement:

- `digitalSourceType: …/trainedAlgorithmicMedia` (fully AI-generated)
- `digitalSourceType: …/compositeWithTrainedAlgorithmicMedia` (AI composite, e.g. Generative Fill)
- `softwareAgent: Adobe Firefly`, claim generator `Adobe Firefly` / `Adobe Photoshop`

Reading these declarations is **deterministic**: when the metadata is present,
classification is exact — no thresholds, no ML, no false positives. When no
provenance exists, the app honestly reports `NO_AI_EVIDENCE` rather than
fabricating a verdict.

### Verdicts

| Verdict | Meaning |
|---|---|
| `AI_GENERATED` | Created by AI (e.g. Firefly text-to-image) |
| `AI_MODIFIED` | Modified with AI (e.g. Photoshop Generative Fill) |
| `EDITED_NO_AI` | Edit provenance present (e.g. Photoshop) with no AI actions |
| `NO_AI_EVIDENCE` | No provenance metadata found — AI use can neither be confirmed nor ruled out |

### Detection pipeline (all deterministic)

1. **C2PA structured pass** (`aidetect/c2pa_reader.py` + `detector.py`) — parses
   the manifest store with the official [`c2pa-python`](https://pypi.org/project/c2pa-python/)
   library; walks actions assertions, software agents, claim generators, and
   ingredients.
2. **Byte-level fallback** (`aidetect/raw_scan.py`) — if the c2pa library is
   unavailable or finds nothing, locates the JUMBF container (JPEG APP11
   segments / PNG `caBX` chunk) and searches it for the IPTC digital-source-type
   URIs and Firefly agent strings.
3. **XMP pass** (`aidetect/xmp_scan.py`) — always runs; extracts XMP packets and
   matches `DigitalSourceType`, `CreatorTool`, and Firefly agent fields.
4. **Generator-metadata pass** (`aidetect/genai_metadata.py`) — always runs;
   deterministically matches the metadata other AI generators embed:
   Stable Diffusion WebUI `parameters` chunks (PNG and EXIF), ComfyUI
   workflow graphs, NovelAI, InvokeAI, and Midjourney job metadata; C2PA
   claim generators from OpenAI/DALL·E are matched in pass 1.
5. **Merge** — the final verdict is the highest-severity verdict implied by any
   evidence; every matched signal is reported back to the user as evidence.

### Durable credentials (TrustMark + Content Credentials Cloud)

Adobe's *durable* Content Credentials survive metadata stripping: an invisible
**TrustMark watermark** carries a soft-binding ID that the **Content
Credentials Cloud** can resolve back to the original manifest. With
`--durable` (CLI) or the UI checkbox, images that carry no embedded
provenance go through this recovery path (`aidetect/softbinding.py`):

1. decode the TrustMark watermark (optional `pip install trustmark`; model
   weights download from Adobe on first use),
2. query the manifest-recovery endpoint (defaults to Adobe's
   `cai-manifests.adobe.com`; override with `AIDETECT_CC_CLOUD_ENDPOINT`),
3. classify the recovered manifest exactly like an embedded one (evidence
   source `c2pa-cloud`).

Every unavailable step degrades to an explanatory note — the verdict is never
guessed. The full recovery path (cloud query → manifest classification) is
exercised in `tests/test_softbinding.py` against a real local HTTP server
serving `dataset/cloud_store/`.

### Heuristic pixel analysis (3 advisory tools)

`--heuristics` (CLI) or the UI checkbox runs three classic pixel-forensics
tools (`aidetect/heuristics.py`) — **advisory only**, never part of the
deterministic verdict:

| Tool | What it looks for |
|---|---|
| **ELA** (Error Level Analysis) | recompression residuals that differ across regions — splices/local regeneration |
| **Spectral** (FFT) | periodic peaks and unnaturally clean high-frequency bands typical of generative upsamplers |
| **NoiseMap** | missing/uneven sensor noise floor — synthesized or inpainted patches |

Each returns a 0–1 score with an explanation; ≥2 flags ⇒ "heuristics lean
toward AI". These are statistical signals that can be wrong in both
directions, which is exactly why the provenance verdict stays separate.

### ML classifier for provenance-less images (transfer learning, ~89% forced / ~98% selective)

Images from generators that embed no metadata (or had it stripped) cannot be
classified deterministically — for those the optional **ML layer**
(`--ml` / UI checkbox, `aidetect/ml_detector.py`) reports a statistical AI
probability. Architecture: **ImageNet transfer learning** — 2048-d embeddings
from the open-source ResNet50 backbone (ONNX Model Zoo, Apache-2.0;
`scripts/fetch_backbone.py`, `aidetect/embeddings.py`) concatenated with 76
forensic pixel features (`aidetect/features.py`), classified by an ensemble
of a calibrated logistic head and a calibrated 3-seed bag of
gradient-boosted trees (probabilities averaged). Training data comes only
from permissively-licensed open-source repos (see `SOURCES.md`), built by
`scripts/build_corpus.py`, trained by `scripts/train_ml.py` with group-aware
5-fold cross-validation (tiles of one source image never span train/test).
The model ships as portable JSON — no pickle, no torch.

The classifier has **two decision modes**:
- **forced** — always answers at the calibrated threshold;
- **selective** — commits only when the calibrated probability is outside
  the cross-validation-chosen uncertainty band, otherwise answers
  `UNCERTAIN`. Committed decisions are substantially more reliable.

**Measured results** (`scripts/benchmark_pipeline.py`; every benchmark image
comes from sources excluded from training):

| Evaluation | Result |
|---|---|
| CV image-level accuracy (grouped 5-fold, ensemble) | 89.1% |
| CV selective accuracy / coverage | 96.4% / 57% |
| Full pipeline, forced decisions (56 images) | **87.5%** |
| Full pipeline, selective mode | **97.8%** correct at 80% coverage |
| — provenance/metadata-decided images | 100% |
| — ML-holdout pool (20 unseen photos/SD images), forced | 70.0% |
| External shoe held-out (MIT dataset, unseen), forced | **97.1%** (0 false positives) |

Provenance and generator metadata decide deterministically wherever evidence
exists; only evidence-free photographs fall through to the statistical
model. A residual CNN (`scripts/train_cnn.py`) underperformed and is not
shipped. **No statistical model can be 100% accurate** — this layer is
separate, advisory, reports a probability, and says `UNCERTAIN` rather than
guessing when the evidence is thin.

## Quick start

```bash
pip install -r requirements.txt

# CLI — single images or batch scans (files and/or directories)
python -m aidetect fixtures/firefly_gen.jpg fixtures/clean.jpg
python -m aidetect --json fixtures/*.jpg
python -m aidetect testset dataset --jobs 4 --csv scan_results.csv
python -m aidetect --heuristics --durable dataset/special/stripped_recoverable.jpg
python -m aidetect --ml wild/ai_sd_txt2img_05.png   # statistical layer
python scripts/benchmark_pipeline.py                # full-pipeline accuracy report

# Web UI (drag & drop)
uvicorn aidetect.webapp:app --port 8000   # then open http://localhost:8000
```

Example CLI output:

```
AI_GENERATED     fixtures/firefly_gen.jpg
    Image was created by an AI tool (e.g. Adobe Firefly text-to-image).
    - [c2pa] claim_generator = 'Adobe Firefly' -> AI_GENERATED
    - [c2pa] digitalSourceType = 'http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia' -> AI_GENERATED
    - [c2pa] softwareAgent = 'Adobe Firefly' -> AI_GENERATED
```

## The test picture set

`fixtures/` contains 13 labeled images (`fixtures/labels.json`), regenerable
with `python scripts/make_fixtures.py`:

- **Signed C2PA fixtures** (`firefly_gen.jpg/.png`, `genfill.jpg/.png`,
  `ps_edit.jpg`) — real, cryptographically **signed C2PA manifests** (ES256,
  Content Authenticity Initiative test certificates) carrying exactly the
  actions Adobe tools write: `c2pa.created` + `trainedAlgorithmicMedia` for
  Firefly, `c2pa.placed` + `compositeWithTrainedAlgorithmicMedia` +
  Firefly agent for Generative Fill, and plain edit actions for the non-AI case.
- **XMP-only fixtures** (`*_xmp_only.jpg`) — Adobe-style XMP packets without a
  C2PA manifest.
- **Negatives** — clean JPEG/PNG, a camera-EXIF-only image, and
  `clean_tricky_name_firefly.png` (proves classification uses metadata, not
  filenames).
- **`stripped_ai.jpg`** — the Firefly image re-saved so its metadata is gone.
  Labeled `NO_AI_EVIDENCE` on purpose: it demonstrates the fundamental limit of
  *any* detector (see below).

### Run the evaluation

```bash
python run_evaluation.py   # prints per-file table; exits non-zero unless 100%
python -m pytest           # 56 tests incl. the same 100% gate + a no-c2pa-library fallback gate
```

Current result: **Accuracy: 13/13 (100.0%)** — also holds with the c2pa
library disabled (byte-level fallback path).

CI (`.github/workflows/ci.yml`) runs the full suite, both accuracy gates, and
a lint pass on every push and pull request.

### Paired sets: AI-modified vs original

`python scripts/make_testset.py` builds `testset/original/` (10 clean images —
real CAI photographs with provenance stripped, plus rendered scenes) and
`testset/ai_modified/` (10 counterparts with real pixel edits and genuine
Adobe-style provenance: signed C2PA Generative Fill manifests, Firefly-created
manifests, and XMP-only variants). Score it with:

```bash
python evaluate_set.py            # per-file table + confusion matrix
```

Current result: **20/20 (100.0%)** — zero false positives, zero false
negatives. `evaluate_set.py --ai-dir DIR --original-dir DIR` works on any two
directories, so you can point it at your own Firefly/Photoshop exports and
untouched photos.

### Training on your own image folders

Datasets hosted on HuggingFace / Kaggle / Google Drive can't be fetched from a
network-restricted environment. Download and unzip one on your own machine,
then point `scripts/ingest_local.py` at two folders (AI-generated and real):

```bash
python scripts/build_corpus.py                                   # optional: GitHub base corpus
python scripts/ingest_local.py --ai-dir my_ai --real-dir my_real # add your images
python scripts/train_ml.py                                       # retrain
python scripts/benchmark_pipeline.py                             # measure
```

Images are tiled and augmented exactly like the base corpus, each source image
is its own train/test group (no leakage), and small images are upscaled to one
tile. Good commercial-safe sources: **ArtiFact_240K** (MIT — faces/animals/
vehicles), **DiffusionDB** (CC0 — Stable Diffusion), **Unsplash Lite** (real
photos). See `SOURCES.md`.

### The `dataset/` evaluation set

`python scripts/make_dataset.py` builds a second, independent set: 8 originals
(fresh crops of the real CAI photographs + new renders), 8 AI images (signed
Firefly creations and Generative Fill composites in JPEG *and* PNG, plus
XMP-only variants), and `dataset/special/stripped_recoverable.jpg` — an AI
image with stripped metadata whose manifest lives in `dataset/cloud_store/`,
demonstrating durable-credential recovery. Score it with:

```bash
python evaluate_set.py --ai-dir dataset/ai --original-dir dataset/original
```

Current result: **16/16 (100.0%)**.

### Real-world samples

`python scripts/download_samples.py` fetches C2PA-signed sample images from the
Content Authenticity Initiative's open-source test fixtures into
`fixtures/external/` (excluded from the eval — their provenance isn't authored
here) so you can try the detector on files signed by real tooling.

## Honest limitations

- **Stripped metadata**: screenshots, re-encoding, and most social-media
  pipelines remove C2PA/XMP metadata. Such images yield `NO_AI_EVIDENCE` —
  the app tells you it found nothing, it does not certify authenticity.
  No detector of any kind can be 100% accurate on metadata-stripped images;
  this app is 100% accurate *about what the image's provenance declares*.
- **Non-Adobe generators**: images from tools that embed no provenance
  (or had it removed) are out of scope of the AI verdicts.
- **Trust**: verdicts reflect the *declared* manifest. Signature validation
  state is surfaced in `notes`; chain-of-trust verification against the C2PA
  public trust list is not performed.

## Batch performance (measured on this machine, CPU-only)

`python -m aidetect DIR --jobs N --csv out.csv` scans whole directories;
the web UI accepts multi-file drops. Measured with
`scripts/measure_throughput.py` over the committed test sets:

| Mode | ms / image | 1,000 images |
|---|---|---|
| Metadata layers only (deterministic) | ~16 | **~16 s** |
| Metadata, 4 worker processes | ~4 | ~4 s |
| Metadata + heuristic tools | ~68 | ~1 min 10 s |
| Metadata + ML classifier | ~486 | ~8 min |
| Metadata + ML, 4 worker processes | ~195 | **~3 min 15 s** |

## Key open-source libraries

The libraries with the biggest impact on what this tool can do:

| Library | Role | URL |
|---|---|---|
| **c2pa-python** (Content Authenticity Initiative) | Reading & signing C2PA Content Credentials — the core of deterministic detection | https://github.com/contentauth/c2pa-python |
| **ONNX Runtime** | CPU inference of the ImageNet backbone for the ML layer | https://onnxruntime.ai |
| **ONNX Model Zoo** | Source of the Apache-2.0 ResNet50 backbone weights | https://github.com/onnx/models |
| **scikit-learn** | Training the classifier ensemble (logistic + gradient boosting) | https://scikit-learn.org |
| **NumPy** | Forensic feature extraction (FFT, noise statistics) | https://numpy.org |
| **Pillow** | Image decoding, tiling, fixture generation | https://github.com/python-pillow/Pillow |
| **FastAPI** + **Uvicorn** | Web UI and API | https://fastapi.tiangolo.com · https://www.uvicorn.org |
| **TrustMark** (Adobe) | Durable-credential watermark decoding (optional) | https://github.com/adobe/trustmark |
| **pytest** | The 117-test quality gate | https://pytest.org |

## License

MIT (see `LICENSE`). All dependencies, models, and bundled test images come
from open-source projects whose licenses permit commercial use — the full
per-source audit is in `SOURCES.md`.

## Project layout

```
aidetect/            detection library + CLI (python -m aidetect) + FastAPI web UI
scripts/             make_fixtures.py (build/sign test set), download_samples.py
fixtures/            labeled test picture set + labels.json
tests/               unit + end-to-end suite (pytest)
run_evaluation.py    accuracy report over the labeled set
```
