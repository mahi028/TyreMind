"""Render `docs/data_doc.md` to a printable HTML page and a PDF.

The data document is the one we hand to mentors and read from in the room, so it
has to survive being printed: no table cut in half across a page break, no code
block orphaned from its heading, and headings that start a new page rather than
sitting two lines from the bottom.

`scripts/render_pdfs.py` renders the pitch documents, which are hand-written HTML.
This one starts from Markdown instead, so it needs its own converter and its own
print stylesheet.

    python scripts/render_data_doc.py

Needs Chromium via Playwright, the same dependency the other renderer uses:

    python -m playwright install chromium
"""

from __future__ import annotations

import sys
from pathlib import Path

SOURCE = Path("docs/data_doc.md")
HTML = Path("docs/data_doc.html")
PDF = Path("docs/data_doc.pdf")

# Print rules matter more than screen rules here. `break-inside: avoid` on tables
# and code blocks is what stops a five-row table being split across two pages,
# which is the failure mode that makes a printed document hard to read aloud from.
STYLE = """
:root {
  --ink: #14171c;
  --muted: #5b6472;
  --rule: #d8dde5;
  --accent: #c8102e;
  --panel: #f6f8fa;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  color: var(--ink);
  background: #fff;
  font: 10.5pt/1.55 "Segoe UI", system-ui, -apple-system, sans-serif;
}
.page { max-width: 190mm; margin: 0 auto; padding: 0 2mm; }

h1, h2, h3, h4 { line-height: 1.25; margin: 1.4em 0 0.5em; break-after: avoid; }
h1 {
  font-size: 19pt;
  border-bottom: 2.5px solid var(--accent);
  padding-bottom: 0.28em;
  break-before: page;
  margin-top: 0;
}
h1:first-of-type { break-before: avoid; }
h2 { font-size: 14pt; color: #0d1117; border-bottom: 1px solid var(--rule); padding-bottom: 0.2em; }
h3 { font-size: 11.8pt; color: #1d2530; }
h4 { font-size: 10.8pt; color: var(--muted); }

p, li { orphans: 3; widows: 3; }
ul, ol { padding-left: 1.3em; }
li { margin: 0.22em 0; }

strong { color: #000; }
em { color: #2a3038; }

a { color: #0b5fd0; text-decoration: none; }

hr { border: 0; border-top: 1px solid var(--rule); margin: 1.6em 0; }

table {
  border-collapse: collapse;
  width: 100%;
  margin: 0.9em 0;
  font-size: 9.2pt;
  break-inside: avoid;
}
th, td {
  border: 1px solid var(--rule);
  padding: 4.5px 7px;
  text-align: left;
  vertical-align: top;
}
th { background: var(--panel); font-weight: 600; }
tbody tr:nth-child(even) td { background: #fbfcfd; }

code {
  font: 9pt/1.45 "Cascadia Mono", Consolas, monospace;
  background: var(--panel);
  padding: 1px 4px;
  border-radius: 3px;
}
pre {
  background: var(--panel);
  border: 1px solid var(--rule);
  border-left: 3px solid var(--accent);
  padding: 9px 11px;
  overflow-x: auto;
  break-inside: avoid;
  font-size: 8.8pt;
  line-height: 1.45;
}
pre code { background: none; padding: 0; font-size: inherit; }

blockquote {
  margin: 1em 0;
  padding: 0.6em 1em;
  border-left: 3px solid var(--accent);
  background: #fff8f8;
  break-inside: avoid;
}
blockquote p { margin: 0.4em 0; }

@page { size: A4; margin: 15mm 12mm 14mm; }
@media print { body { font-size: 10pt; } }
"""


def to_html(markdown_text: str) -> str:
    """Convert the document to standalone HTML.

    Tables and fenced code are both load-bearing in this document -- the
    experiment results are tables and the reproduction steps are shell blocks --
    so the converter has to support GitHub-flavoured Markdown, not the original
    1.0 syntax.
    """
    from markdown_it import MarkdownIt

    md = MarkdownIt("commonmark", {"html": False, "linkify": False})
    md.enable(["table", "strikethrough"])
    body = md.render(markdown_text)
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<title>TyreMind — Data &amp; Progress Document</title>"
        f"<style>{STYLE}</style></head><body><div class='page'>{body}</div></body></html>"
    )


def render_pdf(source_html: Path, destination: Path, *, timeout_ms: int = 120_000) -> int:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            page = browser.new_page()
            page.goto(source_html.resolve().as_uri(), wait_until="networkidle",
                      timeout=timeout_ms)
            page.pdf(
                path=str(destination),
                print_background=True,
                prefer_css_page_size=True,
                display_header_footer=True,
                header_template="<div></div>",
                # A read-aloud document needs page numbers: "turn to page 14" is
                # the only way four people share a place in a 60-page file.
                footer_template=(
                    "<div style='width:100%;font-size:8px;color:#8a94a3;"
                    "padding:0 12mm;display:flex;justify-content:space-between'>"
                    "<span>TyreMind — internal</span>"
                    "<span class='pageNumber'></span></div>"
                ),
            )
        finally:
            browser.close()
    return destination.stat().st_size


def main() -> int:
    if not SOURCE.exists():
        print(f"missing {SOURCE}", file=sys.stderr)
        return 1

    HTML.write_text(to_html(SOURCE.read_text(encoding="utf-8")), encoding="utf-8")
    print(f"{HTML}  {HTML.stat().st_size / 1024:.0f} KB")

    size = render_pdf(HTML, PDF)
    print(f"{PDF}  {size / 1024:.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
