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
