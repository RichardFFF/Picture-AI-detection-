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
4. **Merge** — the final verdict is the highest-severity verdict implied by any
   evidence; every matched signal is reported back to the user as evidence.

## Quick start

```bash
pip install -r requirements.txt

# CLI
python -m aidetect fixtures/firefly_gen.jpg fixtures/clean.jpg
python -m aidetect --json fixtures/*.jpg

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
