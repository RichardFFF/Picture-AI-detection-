import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FIXTURES = REPO / "fixtures"


def run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "aidetect", *args],
        capture_output=True, text=True, cwd=REPO,
    )


def test_json_output_schema():
    proc = run_cli("--json", str(FIXTURES / "firefly_gen.jpg"), str(FIXTURES / "clean.jpg"))
    assert proc.returncode == 0
    results = json.loads(proc.stdout)
    assert len(results) == 2
    by_name = {Path(r["path"]).name: r for r in results}
    assert by_name["firefly_gen.jpg"]["verdict"] == "AI_GENERATED"
    assert by_name["firefly_gen.jpg"]["c2pa_manifest_present"] is True
    assert by_name["firefly_gen.jpg"]["evidence"]
    assert by_name["clean.jpg"]["verdict"] == "NO_AI_EVIDENCE"
    for r in results:
        assert set(r) == {"path", "verdict", "description",
                          "c2pa_manifest_present", "evidence", "notes"}


def test_human_output():
    proc = run_cli(str(FIXTURES / "genfill.jpg"))
    assert proc.returncode == 0
    assert "AI_MODIFIED" in proc.stdout
    assert "genfill.jpg" in proc.stdout


def test_missing_file_exit_code():
    proc = run_cli("does_not_exist.jpg")
    assert proc.returncode == 2
    assert "ERROR" in proc.stderr
