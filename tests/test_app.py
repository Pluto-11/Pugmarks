"""Smoke test for the Gradio app — does it build a Blocks instance?"""
from __future__ import annotations


def test_app_module_importable() -> None:
    import app

    assert hasattr(app, "build_app")


def test_build_app_returns_blocks() -> None:
    import gradio as gr

    import app

    blocks = app.build_app()
    assert isinstance(blocks, gr.Blocks)


def test_run_pipeline_flat_image_list_with_type_prefix(monkeypatch) -> None:
    """run_pipeline returns image items prefixed by type name for the gradio gallery."""
    from datetime import datetime
    from pathlib import Path

    import app
    from pugmark.schemas import (
        Candidate,
        Chapter,
        ConfirmedEntity,
        EntityCard,
        Gallery,
        ImageRef,
    )

    chapter = Chapter(
        book="x", number=1, title="T", source_pdf=Path("/tmp/x.pdf"),
        page_start=1, page_end=1, raw_text="x", normalized_text="x",
        page_offsets=[0], ingest_version="v1",
    )
    cand = Candidate(
        surface_form="tiger", proposed_name="tiger", entity_type="taxa",
        context_sentence="x", context_window="x", char_offset=0, page=1,
        llm_confidence=0.9, extractor_version="v2",
    )
    entity = ConfirmedEntity(
        canonical_name="Panthera tigris", vernacular="Tiger", entity_type="taxa",
        wikidata_qid="Q15324", rank="species", attributes={},
        validation_method="sparql_exact", source_candidates=[cand],
    )
    card = EntityCard(
        entity=entity, wikipedia_url="https://x", wikipedia_summary="x",
        summary_source="wikipedia",
        primary_image=ImageRef(
            url="https://example.org/tiger.jpg", license="CC0",
            attribution="x", source="wikimedia",
        ),
        alt_images=[], sightings=[], enrich_version="v2",
    )
    fake_gallery = Gallery(
        chapter=chapter,
        cards_by_type={"taxa": [card]},
        unresolved=[],
        generated_at=datetime.now(),
        pugmark_version="0.2.0",
        book_schema=None,
        eval_metrics=None,
    )

    async def fake_extract_gallery(*args, **kwargs):
        return fake_gallery

    monkeypatch.setattr("app.extract_gallery", fake_extract_gallery)

    # Stub gr.File-like object
    class _F:
        name = "tests/fixtures/sample_chapter.pdf"

    images, summary = app.run_pipeline(_F(), 1)
    assert len(images) == 1
    # New: caption prefixes the type
    assert images[0][1].startswith("taxa:") or images[0][1].startswith("[taxa]")
