"""Batch processing: scan many pictures (files and directories) at once.

`scan()` expands the given paths, runs the detection pipeline on every image
(optionally with the durable/heuristics/ML layers), in parallel across
processes when jobs > 1, and returns per-image payloads plus a summary
(verdict counts, throughput).
"""
from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".avif",
              ".heic", ".bmp", ".gif"}


def expand_paths(paths: list[str | Path], recursive: bool = True) -> list[Path]:
    """Resolve files and directories into a sorted list of image files."""
    out: list[Path] = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            it = p.rglob("*") if recursive else p.glob("*")
            out.extend(f for f in it
                       if f.is_file() and f.suffix.lower() in IMAGE_EXTS)
        elif p.is_file():
            out.append(p)
    return sorted(set(out))


def analyze_one(path: str, durable: bool = False, heuristics: bool = False,
                ml: bool = False) -> dict:
    """Full per-image payload (importable at module level for ProcessPool)."""
    from .detector import detect_file

    try:
        result = detect_file(path, durable=durable)
    except OSError as exc:
        return {"path": str(path), "error": str(exc)}
    payload = result.to_dict()
    if heuristics:
        from .heuristics import run_heuristics

        payload["heuristics"] = run_heuristics(path)
    if ml:
        from .ml_detector import ml_assess

        payload["ml"] = ml_assess(path)
    return payload


def _worker(args) -> dict:
    path, durable, heuristics, ml = args
    return analyze_one(path, durable=durable, heuristics=heuristics, ml=ml)


def scan(paths: list[str | Path], durable: bool = False,
         heuristics: bool = False, ml: bool = False, jobs: int = 1,
         recursive: bool = True, progress=None) -> tuple[list[dict], dict]:
    """Analyze every image under `paths`. Returns (payloads, summary)."""
    files = expand_paths(paths, recursive=recursive)
    started = time.monotonic()
    payloads: list[dict] = []

    tasks = [(str(f), durable, heuristics, ml) for f in files]
    if jobs > 1 and len(files) > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            for payload in pool.map(_worker, tasks):
                payloads.append(payload)
                if progress:
                    progress(len(payloads), len(files), payload)
    else:
        for task in tasks:
            payloads.append(_worker(task))
            if progress:
                progress(len(payloads), len(files), payloads[-1])

    elapsed = time.monotonic() - started
    counts: dict[str, int] = {}
    uncertain = 0
    errors = 0
    for p in payloads:
        if "error" in p:
            errors += 1
            continue
        counts[p["verdict"]] = counts.get(p["verdict"], 0) + 1
        if p.get("ml", {}).get("confident_decision") == "uncertain":
            uncertain += 1
    summary = {
        "images": len(files),
        "errors": errors,
        "verdicts": counts,
        "ml_uncertain": uncertain if ml else None,
        "elapsed_seconds": round(elapsed, 2),
        "images_per_second": round(len(files) / elapsed, 2) if elapsed > 0 else None,
    }
    return payloads, summary
