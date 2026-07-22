"""Command-line interface.

    python -m aidetect FILE_OR_DIR... [--json] [--csv OUT] [--jobs N]
                       [--durable] [--heuristics] [--ml] [--no-recursive]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys


def _print_human(payload: dict, compact: bool) -> None:
    if "error" in payload:
        print(f"ERROR            {payload['path']}: {payload['error']}")
        return
    ml = payload.get("ml") or {}
    if compact:
        extra = ""
        if ml.get("available"):
            extra = (f"  P(AI)={ml['probability_ai']:.2f} "
                     f"[{ml['confident_decision']}]")
        print(f"{payload['verdict']:<16} {payload['path']}{extra}")
        return
    print(f"{payload['verdict']:<16} {payload['path']}")
    print(f"    {payload['description']}")
    for e in payload["evidence"]:
        print(f"    - [{e['source']}] {e['signal']} = {e['value']!r} "
              f"-> {e['implies']}")
    for note in payload["notes"]:
        print(f"    note: {note}")
    heur = payload.get("heuristics")
    if heur:
        print(f"    heuristics ({heur['assessment']}):")
        for tool in heur["tools"]:
            print(f"    - {tool['tool']}: score {tool['score']:.2f} — {tool['summary']}")
    if ml:
        if ml.get("available"):
            print(f"    ml: P(AI) = {ml['probability_ai']:.2f} "
                  f"[{ml['confident_decision'].upper()}] "
                  f"(band {ml['band']}, forced: "
                  f"{'AI' if ml['decision'] else 'not AI'})")
        else:
            print(f"    ml: {ml['note']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="aidetect",
        description="AI Image Detector — identifies AI-generated/modified "
                    "pictures via provenance metadata (Adobe Firefly/Photoshop "
                    "C2PA, DALL·E, Stable Diffusion, ComfyUI, NovelAI, "
                    "Midjourney, ...), durable credentials, and optional "
                    "statistical analysis.",
    )
    parser.add_argument("paths", nargs="+", metavar="FILE_OR_DIR",
                        help="image files and/or directories to analyze")
    parser.add_argument("--json", action="store_true", help="emit JSON output")
    parser.add_argument("--csv", metavar="PATH",
                        help="write results as CSV to PATH")
    parser.add_argument("--jobs", type=int, default=1, metavar="N",
                        help="parallel worker processes for batch scans (default 1)")
    parser.add_argument("--no-recursive", action="store_true",
                        help="do not descend into subdirectories")
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

    from .batch import expand_paths, scan

    files = expand_paths(args.paths, recursive=not args.no_recursive)
    if not files:
        print("No image files found.", file=sys.stderr)
        return 2

    payloads, summary = scan(
        args.paths, durable=args.durable, heuristics=args.heuristics,
        ml=args.ml, jobs=args.jobs, recursive=not args.no_recursive,
    )

    if args.json:
        print(json.dumps(payloads, indent=2))
    else:
        compact = len(payloads) > 20
        for payload in payloads:
            _print_human(payload, compact)
        if len(payloads) > 1:
            print(f"\nScanned {summary['images']} images in "
                  f"{summary['elapsed_seconds']}s "
                  f"({summary['images_per_second']} images/s)"
                  + (f", {summary['errors']} errors" if summary["errors"] else ""))
            for verdict, n in sorted(summary["verdicts"].items()):
                print(f"  {verdict:<16} {n}")
            if args.ml and summary["ml_uncertain"] is not None:
                print(f"  {'ML UNCERTAIN':<16} {summary['ml_uncertain']}")

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["path", "verdict", "probability_ai",
                             "confident_decision", "evidence_count", "error"])
            for p in payloads:
                ml = p.get("ml") or {}
                writer.writerow([
                    p.get("path", ""),
                    p.get("verdict", ""),
                    ml.get("probability_ai", ""),
                    ml.get("confident_decision", ""),
                    len(p.get("evidence", [])),
                    p.get("error", ""),
                ])
        print(f"CSV written to {args.csv}", file=sys.stderr)

    return 2 if summary["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
