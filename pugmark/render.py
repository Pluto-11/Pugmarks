"""TaxonCard → HTML / Gradio.

Two render functions, same Gallery input. The Gradio path uses gr.Gallery
with a click-through detail view; the HTML path produces a single self-contained file.
"""
from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from pugmark.schemas import Gallery

_TEMPLATES_DIR = Path(__file__).parent / "templates"


def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(_TEMPLATES_DIR),
        autoescape=select_autoescape(["html"]),
    )


def render_html(gallery: Gallery) -> str:
    template = _env().get_template("gallery.html.j2")
    return template.render(gallery=gallery)


def render_gradio(gallery: Gallery):  # type: ignore[no-untyped-def]
    """Build a Gradio Blocks UI for the given Gallery.

    Imported lazily so non-UI usage (CLI, tests) doesn't pay Gradio startup cost.
    """
    import gradio as gr

    image_items = [
        (str(c.primary_image.url), c.taxon.vernacular) for c in gallery.cards
    ]

    def card_detail(idx: int | None) -> str:
        if idx is None or idx >= len(gallery.cards):
            return "Click a card to see details."
        c = gallery.cards[idx]
        return (
            f"## {c.taxon.vernacular}\n\n"
            f"*{c.taxon.canonical_name}* · {c.taxon.rank}\n\n"
            f"{c.wikipedia_summary}\n\n"
            f"**Pages:** {', '.join(str(s.page) for s in c.sightings)}\n\n"
            f"**License:** {c.primary_image.license} · {c.primary_image.attribution}\n\n"
            f"[Wikipedia →]({c.wikipedia_url})"
        )

    with gr.Blocks(title=f"Pugmark — {gallery.chapter.title}") as blocks:
        gr.Markdown(f"# {gallery.chapter.title}\n*{gallery.chapter.book}*")
        gallery_widget = gr.Gallery(
            value=image_items,
            columns=3,
            height="auto",
            object_fit="cover",
        )
        detail = gr.Markdown("Click a card to see details.")
        gallery_widget.select(
            fn=lambda evt: card_detail(evt.index if hasattr(evt, "index") else None),
            inputs=None,
            outputs=detail,
        )
    return blocks
