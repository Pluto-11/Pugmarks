"""Gallery → print-ready PDF via WeasyPrint.

Wraps render_html() with print-optimized CSS:
  - A4 size, 1.8cm margins
  - Cover page with book title + book type (genre/period/setting/themes)
  - Per-type section with H2 + grid of cards
  - page-break-inside: avoid on cards (no card split across pages)
  - Print-friendly typography (serif body, sans heads)

AI-generated images embedded via file:// URLs (WeasyPrint reads them natively).
"""
from __future__ import annotations

import logging
from pathlib import Path

from pugmark.render import render_html
from pugmark.schemas import Gallery

logger = logging.getLogger(__name__)


PRINT_CSS = """
@page {
  size: A4;
  margin: 1.8cm 1.6cm;
  @bottom-center {
    content: "Pugmark — " counter(page) " / " counter(pages);
    font-size: 8pt;
    color: #888;
  }
}

@page :first {
  @bottom-center { content: ""; }
}

html, body {
  font-family: "Georgia", "Liberation Serif", "DejaVu Serif", serif;
  font-size: 10.5pt;
  line-height: 1.45;
  color: #222;
}

h1, h2, h3 {
  font-family: "Helvetica", "DejaVu Sans", sans-serif;
  color: #1a1a1a;
}

h1 { font-size: 22pt; }
h2.type-header {
  font-size: 16pt;
  border-bottom: 1px solid #aaa;
  padding-bottom: 4pt;
  margin-top: 18pt;
  margin-bottom: 12pt;
  page-break-after: avoid;
}

.meta {
  font-size: 9pt;
  color: #666;
  margin-bottom: 14pt;
}

.cover {
  page-break-after: always;
  padding-top: 4cm;
}
.cover h1 {
  font-size: 32pt;
  margin-bottom: 0.2cm;
}
.cover .subtitle {
  font-size: 13pt;
  color: #555;
  margin-bottom: 1.4cm;
  font-style: italic;
}
.cover dl {
  margin: 1.2cm 0;
  font-size: 11pt;
}
.cover dt {
  font-weight: bold;
  margin-top: 8pt;
  color: #444;
}
.cover dd {
  margin: 0 0 0 1.2cm;
  color: #222;
}

.grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 14pt;
}

article.card {
  page-break-inside: avoid;
  border: 0.5pt solid #d0d0d0;
  background: #fafafa;
  border-radius: 4pt;
  overflow: hidden;
}
article.card img {
  width: 100%;
  height: 6.5cm;
  object-fit: cover;
  display: block;
}
article.card .no-image-placeholder {
  height: 6.5cm;
  display: flex; align-items: center; justify-content: center;
  background: linear-gradient(135deg, #e8eef5, #cfd9e7);
  font-size: 24pt; color: #678;
  font-family: serif;
}
.card-body {
  padding: 10pt 12pt 12pt;
}
.card-body h3 {
  margin: 0 0 6pt;
  font-size: 12pt;
}
.card-body p {
  margin: 0 0 6pt;
  font-size: 9.5pt;
}
.card-body .badge {
  display: inline-block;
  padding: 1pt 6pt;
  border-radius: 3pt;
  background: #eaeaea;
  color: #444;
  font-size: 7.5pt;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
.card-body .ai-badge {
  background: #fce4e4;
  color: #800;
}
.attribution {
  font-size: 7pt;
  color: #888;
  margin-top: 6pt;
  font-style: italic;
}
.unresolved {
  margin-top: 1.8cm;
  padding: 12pt;
  background: #fff8e1;
  border-left: 3pt solid #fb0;
  font-size: 9pt;
  page-break-inside: avoid;
}
"""


def _cover_html(gallery: Gallery) -> str:
    """Single-page cover with title + book type metadata."""
    bt = getattr(getattr(gallery, "book_schema", None), "book_type", None)
    chapter = gallery.chapter
    parts = [
        '<section class="cover">',
        '  <div class="subtitle">A Pugmark gallery</div>',
        f"  <h1>{_escape(chapter.book)}</h1>",
        f'  <div class="subtitle">Chapter {chapter.number}: '
        f"{_escape(chapter.title)} · pp. {chapter.page_start}–{chapter.page_end}</div>",
    ]
    if bt is not None:
        parts.append("  <dl>")
        parts.append(f"    <dt>Genre</dt><dd>{_escape(bt.genre)}</dd>")
        parts.append(f"    <dt>Period</dt><dd>{_escape(bt.period)}</dd>")
        parts.append(f"    <dt>Setting</dt><dd>{_escape(bt.setting)}</dd>")
        if bt.themes:
            parts.append(
                "    <dt>Themes</dt><dd>"
                + _escape(", ".join(bt.themes))
                + "</dd>"
            )
        if bt.summary:
            parts.append(f"    <dt>Summary</dt><dd>{_escape(bt.summary)}</dd>")
        parts.append("  </dl>")
    n_cards = sum(len(cs) for cs in gallery.cards_by_type.values())
    n_unresolved = len(gallery.unresolved)
    parts.append(
        f'  <div class="meta">Generated {gallery.generated_at:%Y-%m-%d %H:%M} · '
        f"{n_cards} cards · {n_unresolved} unresolved · "
        f"Pugmark {gallery.pugmark_version}</div>"
    )
    parts.append("</section>")
    return "\n".join(parts)


def _escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def render_pdf(gallery: Gallery, out_path: Path) -> Path:
    """Render a Gallery to a PDF at `out_path`.

    Composes:
      1. Cover page (book + book_type)
      2. Existing render_html() body
    Applies print CSS overlay on top of whatever render_html ships.
    Returns the path written.
    """
    import weasyprint  # local import so HTML-only users don't pay the cost

    body_html = render_html(gallery)
    # Replace render_html's <head>/<body> wrapper: inject cover + override CSS.
    # render_html already produces a full document; we wrap it conservatively.
    full_html = (
        '<!doctype html><html><head><meta charset="utf-8">'
        f"<style>{PRINT_CSS}</style>"
        "</head><body>"
        + _cover_html(gallery)
        + _strip_outer_document(body_html)
        + "</body></html>"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    weasyprint.HTML(string=full_html, base_url=str(Path.cwd())).write_pdf(str(out_path))
    logger.info(
        f"pdf-export: wrote {out_path} ({out_path.stat().st_size / 1024:.0f} KB)"
    )
    return out_path


def _strip_outer_document(html: str) -> str:
    """Pull the body content out of render_html's full <html><body>...</body></html>.

    render_html ships a full document; we need just the body content to splice
    after our cover page. Robust fallback: if the markers aren't found, return
    the original (weasyprint handles nested documents acceptably).
    """
    start = html.find("<body>")
    end = html.rfind("</body>")
    if start == -1 or end == -1 or end < start:
        return html
    return html[start + len("<body>") : end]
