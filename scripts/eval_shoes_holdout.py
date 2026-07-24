"""External generalization test on the MIT shoe dataset (held-out slice).

Downloads the shoe images that scripts/build_corpus.py does NOT train on
(real 100-149 and the first 20 Midjourney AI filenames), runs the ML layer,
and reports forced + selective accuracy. Because these images are excluded
from the training corpus, this is an honest out-of-domain generalization
measurement.

Source: github.com/sunkakar/dataset-shoes-ai-generated (MIT).
"""
from __future__ import annotations

import sys
import tempfile
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from aidetect.ml_detector import ml_assess  # noqa: E402

SHOE = "https://raw.githubusercontent.com/sunkakar/dataset-shoes-ai-generated/main"
REAL_IDS = list(range(100, 150))
AI_FILES = [
    "aadwlchsdhy", "aaidtlwuezt", "abgdjvgwjym", "acaepajyzuu", "acnwnitrdrm",
    "adkwdhfulmy", "adswvmoymdh", "aeheiwyinva", "aeptapltanf", "aetfmeycjid",
    "affqhqolhvi", "afnjqggzcbj", "agshkoujown", "ahaevflguwr", "ahhgblpoqwp",
    "ahkgfypdcbo", "ahvcqgvfwuj", "aiaxsrjbrys", "ajaccfuzzbk", "albbpeafqlp",
]


def download(url: str, dest: Path) -> bool:
    try:
        dest.write_bytes(urllib.request.urlopen(url, timeout=30).read())
        return True
    except Exception:
        return False


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cases = []  # (path, truth_is_ai)
        for n in REAL_IDS:
            p = tmp / f"real_{n}.jpg"
            if download(f"{SHOE}/real/{n}.jpg", p):
                cases.append((p, 0))
        for f in AI_FILES:
            p = tmp / f"ai_{f}.jpg"
            if download(f"{SHOE}/ai-midjourney/{f}.jpg", p):
                cases.append((p, 1))

        if not cases:
            print("Could not download shoe images (network to GitHub blocked?).")
            return 1

        forced_ok = committed = sel_ok = 0
        tp = tn = fp = fn = 0
        for path, truth in cases:
            r = ml_assess(str(path))
            if not r.get("available"):
                print("ML model unavailable:", r.get("note"))
                return 1
            pred = 1 if r["decision"] else 0
            forced_ok += pred == truth
            tp += truth == 1 and pred == 1
            tn += truth == 0 and pred == 0
            fp += truth == 0 and pred == 1
            fn += truth == 1 and pred == 0
            if r["confident_decision"] != "uncertain":
                committed += 1
                sel_ok += (r["confident_decision"] == "ai") == (truth == 1)

    n = len(cases)
    n_ai = sum(t for _, t in cases)
    print(f"Shoe held-out set: {n} images ({n_ai} AI / {n - n_ai} real)")
    print(f"Forced:    {forced_ok}/{n} = {100 * forced_ok / n:.1f}%  "
          f"(TP {tp} TN {tn} FP {fp} FN {fn})")
    if committed:
        print(f"Selective: {sel_ok}/{committed} = {100 * sel_ok / committed:.1f}% "
              f"at {100 * committed / n:.0f}% coverage "
              f"({n - committed} UNCERTAIN)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
