"""Gradio HF Space entry point for Pugmark v2."""
from __future__ import annotations

import asyncio
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

from pugmark.api import extract_gallery
from pugmark.ingest import list_chapters
from pugmark.observability import init_observability

PUGMARK_VERSION = "0.2.0"

load_dotenv()
init_observability()


def list_chapters_for_dropdown(pdf_file: gr.File | None) -> gr.Dropdown:
    if pdf_file is None:
        return gr.Dropdown(choices=[], value=None)
    chapters = list_chapters(Path(pdf_file.name))
    choices = [(f"{c['number']}. {c['title']}", c["number"]) for c in chapters]
    return gr.Dropdown(choices=choices, value=choices[0][1] if choices else None)


def run_pipeline(pdf_file, chapter_number: int):
    if pdf_file is None or chapter_number is None:
        return [], "Upload a PDF and select a chapter."
    gallery = asyncio.run(extract_gallery(pdf_file.name, int(chapter_number)))

    # Flatten cards_by_type to a single image grid with type-prefixed captions
    image_items: list[tuple[str, str]] = []
    for type_name, cards in gallery.cards_by_type.items():
        for card in cards:
            if card.primary_image is not None:
                image_items.append(
                    (str(card.primary_image.url), f"[{type_name}] {card.entity.vernacular}")
                )

    types_summary = ", ".join(
        f"{t}: {len(cards)}" for t, cards in gallery.cards_by_type.items()
    )
    summary = (
        f"**{gallery.chapter.title}** -- {types_summary} - "
        f"{len(gallery.unresolved)} unresolved."
    )
    return image_items, summary


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Pugmark -- Universal Bestiary") as app:
        gr.Markdown(
            "# Pugmark\n"
            "*Turn any book PDF into an illustrated, evaluated bestiary -- "
            "entity types auto-detected per book.*"
        )
        with gr.Row():
            pdf_in = gr.File(label="Upload PDF", file_types=[".pdf"])
            chapter_dd = gr.Dropdown(label="Chapter", choices=[], value=None)
        run_btn = gr.Button("Build gallery", variant="primary")
        summary_md = gr.Markdown()
        gallery_widget = gr.Gallery(
            label="Gallery", columns=3, height="auto", object_fit="cover"
        )

        pdf_in.change(list_chapters_for_dropdown, inputs=pdf_in, outputs=chapter_dd)
        run_btn.click(
            run_pipeline, inputs=[pdf_in, chapter_dd], outputs=[gallery_widget, summary_md]
        )
    return app


if __name__ == "__main__":
    build_app().launch()
