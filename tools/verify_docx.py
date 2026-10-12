#!/usr/bin/env python3
"""
verify_docx.py
==============
Cross-platform verification script (Linux, macOS, Windows).
Verifies that fabricated / stale ablation values are completely gone from
the compiled .docx binaries, and that the corrected empirical values are present.

Checks both:
- paper/CSE432_Bangla_Sentiment_BERT_Paper.docx
- docs/assets/CSE432_Bangla_Sentiment_BERT_Paper.docx
"""

import sys
import zipfile
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

BAD_STRINGS = ["0.5768", "+2.17", "-2.17", "2.17%"]
GOOD_STRINGS = ["0.6228", "2.43", "negative delta"]

TARGETS = [
    ROOT / "paper" / "CSE432_Bangla_Sentiment_BERT_Paper.docx",
    ROOT / "docs" / "assets" / "CSE432_Bangla_Sentiment_BERT_Paper.docx",
]

def verify_docx(path: Path) -> bool:
    if not path.exists():
        print(f"[FAIL] Missing file: {path}")
        return False

    print(f"\n=== Verifying: {path.relative_to(ROOT)} ({path.stat().st_size:,} bytes) ===")
    try:
        with zipfile.ZipFile(path) as z:
            if "word/document.xml" not in z.namelist():
                print("  [FAIL] word/document.xml not found in archive")
                return False
            xml_content = z.read("word/document.xml").decode("utf-8", errors="replace")
    except Exception as e:
        print(f"  [FAIL] Could not read docx: {e}")
        return False

    # Strip XML tags to get raw text
    text = re.sub(r"<[^>]+>", "", xml_content)
    print(f"  Body text length: {len(text):,} characters")

    all_passed = True

    for bad in BAD_STRINGS:
        if bad in text:
            print(f"  [FAIL] Fabricated/stale string still present: '{bad}'")
            all_passed = False
        else:
            print(f"  [OK] Absent: '{bad}'")

    for good in GOOD_STRINGS:
        if good in text:
            print(f"  [OK] Present: '{good}'")
        else:
            print(f"  [FAIL] Corrected string missing: '{good}'")
            all_passed = False

    return all_passed

def main():
    overall_ok = True
    for target in TARGETS:
        ok = verify_docx(target)
        if not ok:
            overall_ok = False

    print("\n" + "=" * 50)
    if overall_ok:
        print("RESULT: ALL DOCX CHECKS PASSED")
        sys.exit(0)
    else:
        print("RESULT: VERIFICATION FAILED")
        sys.exit(1)

if __name__ == "__main__":
    main()
