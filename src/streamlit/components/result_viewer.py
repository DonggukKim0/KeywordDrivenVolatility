"""Reusable renderers for generated EDA and model result artifacts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import streamlit as st
from components.csv_loader import load_csv


def render_figure(path: Path, caption: str | None = None) -> None:
    if not path.is_file():
        st.info(f"그림 파일을 찾지 못했습니다: `{path}`")
        return
    st.image(str(path), width="stretch")
    if caption:
        st.caption(caption)


def render_table(
    path: Path,
    *,
    key: str | None = None,
    preview_rows: int = 2000,
    height: int | None = None,
    hidden_columns: tuple[str, ...] = (),
    column_config: dict | None = None,
) -> pd.DataFrame | None:
    frame = load_csv(path)
    if frame is None:
        st.info(f"결과 파일을 찾지 못했습니다: `{path}`")
        return None
    if frame.empty:
        st.info(f"결과 파일에 표시할 행이 없습니다: `{path.name}`")
        return frame

    displayed = frame.head(preview_rows) if len(frame) > preview_rows else frame
    displayed = displayed.drop(columns=list(hidden_columns), errors="ignore")
    if len(displayed) < len(frame):
        st.caption(f"전체 {len(frame):,}행 중 처음 {len(displayed):,}행을 표시합니다.")
    display_options = {"width": "stretch", "hide_index": True}
    if column_config is not None:
        display_options["column_config"] = column_config
    if height is not None:
        display_options["height"] = height
    st.dataframe(displayed, **display_options)
    return frame
