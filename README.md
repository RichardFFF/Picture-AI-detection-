# Adobe AI Image Detector

Detects pictures **created or modified with Adobe AI tools** — Adobe Firefly
(text-to-image) and Photoshop's AI features (Generative Fill / Generative
Expand) — by reading the provenance metadata those tools embed. Ships with a
CLI, a drag-and-drop web UI, and a labeled test picture set on which it scores
**100% accuracy** (`13/13`, verified by `run_evaluation.py` and the test suite).

## How it works — and why it can be exact

Statistical/pixel-based "AI detectors" guess, and guessing can never be 100%
accurate. This app doesn't guess. Adobe Firefly and Photoshop's generative
features embed **C2PA Content Credentials** — cryptographically signed
provenance manifests — plus XMP metadata that explicitly declare AI
involvement:

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

### ML classifier for provenance-less images (~90% pipeline accuracy)

Images from generators that embed no metadata (or had it stripped) cannot be
classified deterministically — for those the optional **ML layer**
(`--ml` / UI checkbox, `aidetect/ml_detector.py`) reports a statistical
AI probability. It is gradient-boosted trees over 76 forensic pixel features
(noise statistics and cross-channel correlation, FFT spectrum shape and
latent-grid energy, gradients, saturation — `aidetect/features.py`), trained
on a corpus of genuine Stable Diffusion / ControlNet outputs vs real
photographs (BSDS500, OpenCV, CAI and other open datasets), built by
`scripts/build_corpus.py` and trained by `scripts/train_ml.py` with
group-aware 5-fold cross-validation (tiles of one source image never span
train and test). The model ships as portable JSON (`aidetect/ml_model.json`)
— no pickle, no torch needed at inference.

**Measured results** (see `scripts/benchmark_pipeline.py`; the benchmark
images come from sources excluded from training):

| Evaluation | Result |
|---|---|
| CV tile-level accuracy (grouped 5-fold) | 83.2% |
| CV image-level balanced accuracy | 89.0% |
| Full pipeline on all test pools (58 images) | **89.7%** |
| — provenance-decided images | 100% (57/57 across all layers' deterministic decisions) |
| — ML-holdout pool (22 unseen photos/SD images) | 72.7% |

The pipeline number is honest and reproducible: provenance and generator
metadata decide deterministically wherever evidence exists; only
evidence-free photographs fall through to the statistical model. A residual
CNN was also trained (`scripts/train_cnn.py`) but underperformed the tree
model (74.8% held-out) and is not shipped. **No statistical model can be
100% accurate** — that is why this layer is separate, advisory, and reports
a probability with its measured accuracy attached.

## Quick start

```bash
pip install -r requirements.txt

# CLI
python -m aidetect fixtures/firefly_gen.jpg fixtures/clean.jpg
python -m aidetect --json fixtures/*.jpg
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

## Project layout

```
aidetect/            detection library + CLI (python -m aidetect) + FastAPI web UI
scripts/             make_fixtures.py (build/sign test set), download_samples.py
fixtures/            labeled test picture set + labels.json
tests/               unit + end-to-end suite (pytest)
run_evaluation.py    accuracy report over the labeled set
```
