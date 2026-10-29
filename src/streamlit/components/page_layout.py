"""Shared UI helpers for dashboard pages."""

from pathlib import Path

import streamlit as st


def render_page_header(title: str, description: str) -> None:
    """Render a compact analytics header with a decorative project illustration."""
    illustration = (
        Path(__file__).resolve().parents[1] / "assets" / "dashboard_signal.svg"
    )
    title_col, art_col = st.columns([1.7, 1], gap="medium")
    with title_col:
        st.markdown(
            '<div class="dashboard-heading"><div class="eyebrow">'
            f"{title}</div><h1>{description}</h1></div>",
            unsafe_allow_html=True,
        )
    if illustration.is_file():
        with art_col:
            # Pass SVG markup directly to Streamlit's image element so the
            # browser renders the vector asset without an HTML data URI.
            st.image(illustration.read_text(encoding="utf-8"), width=350)
    st.markdown('<div class="dashboard-header-divider"></div>', unsafe_allow_html=True)


def render_section_intro(title: str, description: str = "") -> None:
    """Render a consistent section heading and optional supporting text."""
    st.subheader(title)
    if description:
        st.markdown(
            f'<p class="section-note">{description}</p>', unsafe_allow_html=True
        )
