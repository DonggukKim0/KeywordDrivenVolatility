"""Cached readers for existing dashboard result files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import streamlit as st


@st.cache_data(show_spinner=False)
def _read_csv(path: str, modified_ns: int) -> pd.DataFrame:
    """Read one CSV; its modification time invalidates the cached copy."""
    del modified_ns
    for encoding in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, low_memory=False)


def load_csv(path: Path) -> pd.DataFrame | None:
    """Load an existing CSV once per file version, or return None if absent."""
    if not path.is_file():
        return None
    return _read_csv(str(path.resolve()), path.stat().st_mtime_ns)


@st.cache_data(show_spinner=False)
def _summarize_csv(path: str, date_column: str, modified_ns: int) -> tuple[int, str]:
    """Count rows and find a date range without retaining the full source table."""
    del modified_ns
    row_count = 0
    minimum = None
    maximum = None

    for encoding in ("utf-8-sig", "cp949", None):
        row_count = 0
        minimum = None
        maximum = None
        try:
            chunks = pd.read_csv(
                path,
                usecols=[date_column],
                chunksize=100_000,
                encoding=encoding,
                low_memory=False,
            )
            for chunk in chunks:
                row_count += len(chunk)
                values = chunk[date_column]
                if date_column == "bsns_year":
                    valid = pd.to_numeric(values, errors="coerce").dropna()
                else:
                    valid = pd.to_datetime(values, errors="coerce").dropna()
                if valid.empty:
                    continue
                chunk_min, chunk_max = valid.min(), valid.max()
                minimum = chunk_min if minimum is None else min(minimum, chunk_min)
                maximum = chunk_max if maximum is None else max(maximum, chunk_max)
            break
        except UnicodeDecodeError:
            continue

    if minimum is None or maximum is None:
        period = "기간 정보 없음"
    elif date_column == "bsns_year":
        period = f"{minimum:.0f}–{maximum:.0f}년"
    else:
        period = f"{minimum:%Y-%m-%d}–{maximum:%Y-%m-%d}"
    return row_count, period


def summarize_csv(path: Path, date_column: str) -> tuple[int, str] | None:
    """Return a cached row count/date range for an existing CSV."""
    if not path.is_file():
        return None
    return _summarize_csv(str(path.resolve()), date_column, path.stat().st_mtime_ns)


@st.cache_data(show_spinner=False)
def _count_unique_csv(path: str, column: str, modified_ns: int) -> int:
    """Cache a distinct count without retaining the full source table."""
    del modified_ns
    for encoding in ("utf-8-sig", "cp949"):
        identities = set()
        try:
            with pd.read_csv(
                path,
                usecols=[column],
                dtype={column: "string"},
                encoding=encoding,
                chunksize=100_000,
            ) as chunks:
                for chunk in chunks:
                    identities.update(chunk[column].dropna().unique())
            return len(identities)
        except UnicodeDecodeError:
            if encoding == "cp949":
                raise


def count_unique_csv(path: Path, column: str) -> int | None:
    """Return a distinct count, or None when the source is unavailable."""
    try:
        return _count_unique_csv(str(path.resolve()), column, path.stat().st_mtime_ns)
    except (OSError, ValueError, UnicodeError, pd.errors.ParserError):
        return None
