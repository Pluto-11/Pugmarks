"""Gradio HF Space entry point for Pugmark.

Launches a UI: upload a PDF, pick a chapter, run pipeline, see gallery.
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

from pugmark.cache import Cache
from pugmark.enrich import enrich_taxa
from pugmark.extract import extract_candidates
from pugmark.ingest import list_chapters, load_chapter
from pugmark.llm import LLMConfig
from pugmark.observability import init_observability
from pugmark.schemas import Gallery
from pugmark.validate import validate_candidates

PUGMARK_VERSION = "0.1.0"

load_dotenv()
init_observability()


async def _run(pdf_path: str, chapter_number: int) -> Gallery:
    pdf = Path(pdf_path)
    cache = Cache.from_env()
    llm_config = LLMConfig.from_env()

    chapter = load_chapter(pdf, chapter_number)
    candidates = await extract_candidates(
        chapter, llm_config=llm_config, prompt_dir=Path("prompts"), cache=cache
    )
    confirmed, unresolved = await validate_candidates(candidates, cache=cache)
    cards = await enrich_taxa(confirmed, chapter=chapter, cache=cache)
    return Gallery(
        chapter=chapter,
        cards=cards,
        unresolved=unresolved,
        generated_at=datetime.now(),
        pugmark_version=PUGMARK_VERSION,
        eval_metrics=None,
    )


def list_chapters_for_dropdown(pdf_file: gr.File | None) -> gr.Dropdown:
    if pdf_file is None:
        return gr.Dropdown(choices=[], value=None)
    chapters = list_chapters(Path(pdf_file.name))
    choices = [(f"{c['number']}. {c['title']}", c["number"]) for c in chapters]
    return gr.Dropdown(choices=choices, value=choices[0][1] if choices else None)


def run_pipeline(pdf_file, chapter_number: int):
    if pdf_file is None or chapter_number is None:
        return [], "Upload a PDF and select a chapter."
    gallery = asyncio.run(_run(pdf_file.name, int(chapter_number)))
    image_items = [
        (str(c.primary_image.url), c.taxon.vernacular) for c in gallery.cards
    ]
    summary = (
        f"**{gallery.chapter.title}** — {len(gallery.cards)} taxa, "
        f"{len(gallery.unresolved)} unresolved."
    )
    return image_items, summary


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Pugmark — Illustrated Bestiaries") as app:
        gr.Markdown(
            "# 🐅 Pugmark\n"
            "*Turn hunting and natural-history novels into illustrated bestiaries.*"
        )
        with gr.Row():
            pdf_in = gr.File(label="Upload PDF", file_types=[".pdf"])
            chapter_dd = gr.Dropdown(label="Chapter", choices=[], value=None)
        run_btn = gr.Button("Build gallery", variant="primary")
        summary_md = gr.Markdown()
        gallery_widget = gr.Gallery(label="Gallery", columns=3, height="auto", object_fit="cover")

        pdf_in.change(list_chapters_for_dropdown, inputs=pdf_in, outputs=chapter_dd)
        run_btn.click(
            run_pipeline, inputs=[pdf_in, chapter_dd], outputs=[gallery_widget, summary_md]
        )
    return app


if __name__ == "__main__":
    build_app().launch()
