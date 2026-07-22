"""Measure per-image latency of each analysis mode and project batch times.

Runs the batch scanner over the committed image sets in three modes
(metadata-only, +heuristics, +ML) and prints measured ms/image plus the
extrapolated wall time for 1,000 images, single-process and with --jobs 4.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from aidetect.batch import expand_paths, scan  # noqa: E402

SETS = [REPO / "fixtures", REPO / "testset", REPO / "dataset", REPO / "benchmark"]


def run(mode: str, jobs: int = 1, **kwargs) -> tuple[float, int]:
    files = expand_paths(SETS)
    start = time.monotonic()
    _, summary = scan(SETS, jobs=jobs, **kwargs)
    elapsed = time.monotonic() - start
    assert summary["errors"] == 0
    return elapsed, len(files)


def main() -> int:
    # warm up model/backbone caches so ML timing reflects steady state
    from aidetect.ml_detector import ml_assess

    ml_assess(str(REPO / "fixtures" / "clean.jpg"))

    print(f"{'mode':<28} {'images':>6} {'total s':>8} {'ms/image':>9} "
          f"{'1000 imgs':>10}")
    print("-" * 66)
    rows = []
    for label, kwargs, jobs in [
        ("metadata-only", {}, 1),
        ("metadata-only, jobs=4", {}, 4),
        ("metadata + heuristics", {"heuristics": True}, 1),
        ("metadata + ML", {"ml": True}, 1),
        ("metadata + ML, jobs=4", {"ml": True}, 4),
    ]:
        elapsed, n = run(label, jobs=jobs, **kwargs)
        ms = 1000.0 * elapsed / n
        proj = ms  # seconds for 1000 images
        rows.append((label, n, elapsed, ms, proj))
        mins, secs = divmod(int(proj), 60)
        proj_str = f"{mins}m {secs:02d}s" if mins else f"{secs}s"
        print(f"{label:<28} {n:>6} {elapsed:>8.1f} {ms:>9.1f} {proj_str:>10}")

    print("\n'1000 imgs' extrapolates the measured per-image latency; "
          "first-run costs (model load, backbone init) are excluded by the "
          "warm-up call.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
