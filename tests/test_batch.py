import csv
import json
import subprocess
import sys
from pathlib import Path

from aidetect.batch import expand_paths, scan

REPO = Path(__file__).resolve().parent.parent
FIXTURES = REPO / "fixtures"
LABELS = json.loads((FIXTURES / "labels.json").read_text())


def test_expand_paths_directory():
    files = expand_paths([FIXTURES])
    names = {f.name for f in files}
    assert set(LABELS) <= names


def test_scan_matches_labels():
    payloads, summary = scan([FIXTURES])
    assert summary["errors"] == 0
    by_name = {Path(p["path"]).name: p["verdict"] for p in payloads}
    for fname, expected in LABELS.items():
        assert by_name[fname] == expected
    assert summary["images"] == len(payloads)
    assert sum(summary["verdicts"].values()) == summary["images"]


def test_parallel_equals_serial():
    serial, _ = scan([FIXTURES])
    parallel, _ = scan([FIXTURES], jobs=2)
    assert [(p["path"], p["verdict"]) for p in serial] == \
           [(p["path"], p["verdict"]) for p in parallel]


def test_cli_batch_csv(tmp_path):
    out = tmp_path / "scan.csv"
    proc = subprocess.run(
        [sys.executable, "-m", "aidetect", str(FIXTURES), "--csv", str(out)],
        capture_output=True, text=True, cwd=REPO,
    )
    assert proc.returncode == 0
    assert "Scanned" in proc.stdout
    rows = list(csv.DictReader(out.open()))
    # recursive scan may also pick up fixtures/external/ demo downloads
    assert len(rows) >= len(LABELS)
    verdicts = {Path(r["path"]).name: r["verdict"] for r in rows}
    for fname, expected in LABELS.items():
        assert verdicts[fname] == expected
