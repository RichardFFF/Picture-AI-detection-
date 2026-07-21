"""Command-line interface: python -m aidetect FILE... [--json]"""
from __future__ import annotations

import argparse
import json
import sys

from .detector import detect_file


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="aidetect",
        description="Detect Adobe AI (Firefly / Photoshop generative) provenance in images.",
    )
    parser.add_argument("files", nargs="+", help="image files to analyze")
    parser.add_argument("--json", action="store_true", help="emit JSON output")
    args = parser.parse_args(argv)

    results = []
    had_error = False
    for path in args.files:
        try:
            results.append(detect_file(path))
        except OSError as exc:
            had_error = True
            print(f"ERROR  {path}: {exc}", file=sys.stderr)

    if args.json:
        print(json.dumps([r.to_dict() for r in results], indent=2))
    else:
        for r in results:
            print(f"{r.verdict.value:<16} {r.path}")
            print(f"    {r.description}")
            for e in r.evidence:
                print(f"    - [{e.source}] {e.signal} = {e.value!r} -> {e.implies.value}")
            for note in r.notes:
                print(f"    note: {note}")

    return 2 if had_error else 0


if __name__ == "__main__":
    sys.exit(main())
