"""Minimal web UI: drag-drop an image, get the provenance verdict.

Run with: uvicorn aidetect.webapp:app --port 8000
"""
from __future__ import annotations

import os
import tempfile

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse

from .detector import detect_file

app = FastAPI(title="Adobe AI Image Detector")

PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Adobe AI Image Detector</title>
<style>
  :root { color-scheme: light dark; }
  body { font-family: system-ui, sans-serif; max-width: 720px; margin: 2rem auto; padding: 0 1rem; }
  h1 { font-size: 1.4rem; }
  #drop { border: 2px dashed #888; border-radius: 12px; padding: 3rem 1rem; text-align: center;
          cursor: pointer; transition: background .15s; }
  #drop.hover { background: rgba(100,140,255,.12); border-color: #58f; }
  .badge { display: inline-block; padding: .35rem .8rem; border-radius: 999px; color: #fff;
           font-weight: 600; margin: .8rem 0 .4rem; }
  .AI_GENERATED { background: #c0392b; }
  .AI_MODIFIED { background: #e67e22; }
  .EDITED_NO_AI { background: #2980b9; }
  .NO_AI_EVIDENCE { background: #27ae60; }
  table { border-collapse: collapse; width: 100%; margin-top: .6rem; font-size: .9rem; }
  th, td { border: 1px solid #8884; padding: .35rem .5rem; text-align: left; }
  pre { background: #8881; padding: .8rem; border-radius: 8px; overflow-x: auto; font-size: .8rem; }
  .note { font-size: .85rem; opacity: .8; margin-top: .5rem; }
  footer { margin-top: 2rem; font-size: .8rem; opacity: .7; }
</style>
</head>
<body>
<h1>Adobe AI Image Detector</h1>
<p>Detects images created or modified with Adobe AI tools (Firefly, Photoshop
Generative Fill) by reading their C2PA Content Credentials and XMP provenance
metadata. Deterministic — no guessing.</p>
<div id="drop">Drop an image here or click to choose a file
  <input id="file" type="file" accept="image/*" hidden>
</div>
<div id="result"></div>
<footer>Verdicts: AI_GENERATED (fully AI-created) &middot; AI_MODIFIED (AI edits
such as Generative Fill) &middot; EDITED_NO_AI (edit provenance, no AI actions)
&middot; NO_AI_EVIDENCE (no provenance metadata &mdash; not proof of authenticity,
metadata can be stripped).</footer>
<script>
const drop = document.getElementById("drop");
const input = document.getElementById("file");
const result = document.getElementById("result");
drop.addEventListener("click", () => input.click());
["dragover", "dragenter"].forEach(ev =>
  drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add("hover"); }));
["dragleave", "drop"].forEach(ev =>
  drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove("hover"); }));
drop.addEventListener("drop", e => { if (e.dataTransfer.files[0]) analyze(e.dataTransfer.files[0]); });
input.addEventListener("change", () => { if (input.files[0]) analyze(input.files[0]); });

async function analyze(file) {
  result.innerHTML = "<p>Analyzing " + file.name + "…</p>";
  const body = new FormData();
  body.append("file", file);
  try {
    const resp = await fetch("/api/detect", { method: "POST", body });
    const data = await resp.json();
    render(file.name, data);
  } catch (err) {
    result.innerHTML = "<p>Error: " + err + "</p>";
  }
}

function esc(s) { const d = document.createElement("div"); d.textContent = String(s); return d.innerHTML; }

function render(name, r) {
  let html = "<h2>" + esc(name) + "</h2>";
  html += '<span class="badge ' + esc(r.verdict) + '">' + esc(r.verdict) + "</span>";
  html += "<p>" + esc(r.description) + "</p>";
  if (r.evidence.length) {
    html += "<table><tr><th>Source</th><th>Signal</th><th>Value</th><th>Implies</th></tr>";
    for (const e of r.evidence) {
      html += "<tr><td>" + esc(e.source) + "</td><td>" + esc(e.signal) + "</td><td>"
            + esc(e.value) + "</td><td>" + esc(e.implies) + "</td></tr>";
    }
    html += "</table>";
  }
  for (const n of r.notes) html += '<p class="note">Note: ' + esc(n) + "</p>";
  html += "<details><summary>Raw JSON</summary><pre>" + esc(JSON.stringify(r, null, 2)) + "</pre></details>";
  result.innerHTML = html;
}
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return PAGE


@app.post("/api/detect")
async def api_detect(file: UploadFile = File(...)) -> dict:
    suffix = os.path.splitext(file.filename or "upload")[1] or ".bin"
    data = await file.read()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        result = detect_file(tmp_path)
        payload = result.to_dict()
        payload["path"] = file.filename or "upload"
        return payload
    finally:
        os.unlink(tmp_path)
