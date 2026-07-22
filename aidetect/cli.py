"""Command-line interface: python -m aidetect FILE... [--json] [--durable] [--heuristics]"""
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
    parser.add_argument(
        "--durable", action="store_true",
        help="for images without embedded provenance, attempt durable-credential "
             "recovery (TrustMark watermark + Content Credentials Cloud lookup)",
    )
    parser.add_argument(
        "--heuristics", action="store_true",
        help="also run the 3 advisory pixel-forensics tools (ELA, Spectral, NoiseMap)",
    )
    parser.add_argument(
        "--ml", action="store_true",
        help="also run the trained ML classifier (statistical AI probability "
             "for images without provenance metadata)",
    )
    args = parser.parse_args(argv)

    payloads = []
    had_error = False
    for path in args.files:
        try:
            result = detect_file(path, durable=args.durable)
        except OSError as exc:
            had_error = True
            print(f"ERROR  {path}: {exc}", file=sys.stderr)
            continue
        payload = result.to_dict()
        if args.heuristics:
            from .heuristics import run_heuristics

            payload["heuristics"] = run_heuristics(path)
        if args.ml:
            from .ml_detector import ml_assess

            payload["ml"] = ml_assess(path)
        payloads.append((result, payload))

    if args.json:
        print(json.dumps([p for _, p in payloads], indent=2))
    else:
        for result, payload in payloads:
            print(f"{result.verdict.value:<16} {result.path}")
            print(f"    {result.description}")
            for e in result.evidence:
                print(f"    - [{e.source}] {e.signal} = {e.value!r} -> {e.implies.value}")
            for note in result.notes:
                print(f"    note: {note}")
            heur = payload.get("heuristics")
            if heur:
                print(f"    heuristics ({heur['assessment']}):")
                for tool in heur["tools"]:
                    print(f"    - {tool['tool']}: score {tool['score']:.2f} — {tool['summary']}")
            ml = payload.get("ml")
            if ml:
                if ml.get("available"):
                    print(f"    ml: P(AI) = {ml['probability_ai']:.2f} "
                          f"[{ml['verdict_hint']}] "
                          f"(components: {ml['components']})")
                else:
                    print(f"    ml: {ml['note']}")

    return 2 if had_error else 0


if __name__ == "__main__":
    sys.exit(main())
