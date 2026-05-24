#!/usr/bin/env python3
"""Splice redesigned CSS and HTML into generate_final_report.py.

Replaces:
  - Lines 631–933 (HTML_STYLE CSS)   with _new_css.txt
  - Lines 1047–1215 (interview + HTML) with _new_html.txt

Keeps everything else (data functions, markdown builder, main) unchanged.
"""
from pathlib import Path

base = Path(__file__).resolve().parent
target = base / "generate_final_report.py"

original = target.read_text(encoding="utf-8").splitlines(keepends=True)
css_lines = (base / "_new_css.txt").read_text(encoding="utf-8").splitlines(keepends=True)
html_lines = (base / "_new_html.txt").read_text(encoding="utf-8").splitlines(keepends=True)

print(f"  Original: {len(original)} lines")
print(f"  New CSS:  {len(css_lines)} lines  (replaces {933-631+1} original)")
print(f"  New HTML: {len(html_lines)} lines  (replaces {1215-1047+1} original)")

# Splice using 0-indexed positions
# original[:630]      = lines 1–630     (imports, data functions, markdown builder)
# css_lines            = new CSS         (replaces lines 631–933)
# original[933:1046]  = lines 934–1046  (utility funcs, build_html data prep)
# html_lines           = new HTML        (replaces lines 1047–1215)
# original[1215:]     = lines 1216+     (main function)
result = (
    original[:630]
    + css_lines
    + original[933:1046]
    + html_lines
    + original[1215:]
)

target.write_text("".join(result), encoding="utf-8")
print(f"  ✅ Written {len(result)} lines to {target.name}")
