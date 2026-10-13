#!/usr/bin/env python3
"""Render the cut-flow markdown report to PDF."""

from __future__ import annotations

import re
from pathlib import Path

import markdown
from weasyprint import HTML

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "results" / "report"
SOURCE = REPORT_DIR / "quirk_cutflow_report.md"
OUTPUT = REPORT_DIR / "quirk_cutflow_report.pdf"

CSS = """
@page {
  size: A4;
  margin: 16mm 14mm 18mm 14mm;
  @bottom-center {
    content: counter(page);
    font-size: 9pt;
    color: #444;
  }
}
body {
  font-family: "Liberation Serif", "DejaVu Serif", serif;
  font-size: 11pt;
  line-height: 1.35;
  color: #1a1a1a;
}
h1 { font-size: 18pt; margin-bottom: 0.4em; }
h2 { font-size: 14pt; margin-top: 1.2em; page-break-after: avoid; }
h3 { font-size: 12pt; margin-top: 1em; page-break-after: avoid; }
p { margin: 0.45em 0; }
table {
  border-collapse: collapse;
  width: 100%;
  margin: 0.7em 0 1em 0;
  font-size: 8.5pt;
  page-break-inside: auto;
}
th, td {
  border: 1px solid #b5b5b5;
  padding: 3px 5px;
  vertical-align: top;
  text-align: left;
}
th { background: #f2f2f2; }
tr { page-break-inside: avoid; }
img { width: 100%; margin: 0.4em 0 0.8em 0; }
code { font-family: "DejaVu Sans Mono", monospace; font-size: 9pt; }
.equation {
  font-family: "DejaVu Sans", sans-serif;
  background: #f7f7f7;
  border: 1px solid #e0e0e0;
  padding: 8px 10px;
  margin: 0.6em 0;
  font-size: 10pt;
}
"""


def latex_inline(match: re.Match) -> str:
    text = match.group(1)
    replacements = (
        (r"\varepsilon", "ε"),
        (r"\Delta", "Δ"),
        (r"\phi", "φ"),
        (r"\hat p", "p̂"),
        (r"\hat{p}", "p̂"),
        (r"\mathrm", ""),
        (r"\,", " "),
        (r"\;", " "),
        (r"\pm", "±"),
        (r"\sqrt", "√"),
        (r"\frac", ""),
        (r"\left", ""),
        (r"\right", ""),
        (r"\hbar", "ℏ"),
    )
    for old, new in replacements:
        text = text.replace(old, new)
    text = text.replace("{", "").replace("}", "")
    text = text.replace("\\", "")
    return text


def prepare(markdown_text: str) -> str:
    display = re.search(r"\\\[(.*?)\\\]", markdown_text, flags=re.S)
    equation = ""
    if display:
        equation = latex_inline(display)
        markdown_text = markdown_text[: display.start()] + "\n\nEQUATION_BLOCK\n\n" + markdown_text[display.end() :]
    markdown_text = re.sub(r"\\\((.*?)\\\)", latex_inline, markdown_text)
    return markdown_text, equation


def main() -> None:
    raw = SOURCE.read_text()
    body, equation = prepare(raw)
    html_body = markdown.markdown(body, extensions=["tables", "sane_lists"])
    if equation:
        html_body = html_body.replace("<p>EQUATION_BLOCK</p>", f'<div class="equation">{equation}</div>')
    html = f"<!DOCTYPE html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{html_body}</body></html>"
    HTML(string=html, base_url=str(REPORT_DIR)).write_pdf(str(OUTPUT))
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
