"""Overview dashboard with selected EDA and model highlights."""

from __future__ import annotations

from html import escape

import pandas as pd
from components.csv_loader import count_unique_csv, load_csv, summarize_csv
from config import (
    EDA_STEP01_DIR,
    EDA_STEP02_DIR,
    EDA_STEP03_TABLES_DIR,
    MODEL_RESULTS_DIR,
)

import streamlit as st
from pages import analysis_feed
from pages.dataset_intro import DATASETS
from pages.model_analysis import render_performance_chart

HEADING_ICONS = {
    "folder": '<path d="M3 7V5a2 2 0 0 1 2-2h4l2 3h8a2 2 0 0 1 2 2v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7Z"/>',
    "chart": '<path d="M3 3v18h18M7 16v-5m5 5V7m5 9V4"/>',
    "chat": '<path d="M21 11a8 8 0 0 1-8 8H7l-5 3 2-6a8 8 0 0 1-1-5 8 8 0 0 1 8-8h2a8 8 0 0 1 8 8Z"/><path d="M7 9h10M7 13h7"/>',
    "brain": '<path d="M12 5a3 3 0 0 0-6-1 4 4 0 0 0-3 6 4 4 0 0 0 0 7 4 4 0 0 0 4 4 3 3 0 0 0 5-2V5Zm0 0a3 3 0 0 1 6-1 4 4 0 0 1 3 6 4 4 0 0 1 0 7 4 4 0 0 1-4 4 3 3 0 0 1-5-2"/><path d="M6 4v3m-3 3h3m-3 7h3m12-13v3m3 3h-3m3 7h-3M8 12l4 2 4-2"/>',
    "search": '<circle cx="10.5" cy="10.5" r="7.5"/><path d="m16 16 5 5"/>',
}


def _render_heading(title: str, icon: str, level: int = 3) -> None:
    """Render a semantic heading with a decorative theme-colored line icon."""
    svg = (
        '<svg class="overview-heading-icon" viewBox="0 0 24 24" '
        'fill="none" stroke="currentColor" stroke-width="1.8" '
        'stroke-linecap="round" stroke-linejoin="round" '
        f'aria-hidden="true" focusable="false">{HEADING_ICONS[icon]}</svg>'
    )
    st.markdown(
        f'<h{level} class="overview-section-heading">{svg}<span>{escape(title)}</span></h{level}>',
        unsafe_allow_html=True,
    )


def _load_sources() -> dict[str, pd.DataFrame | None]:
    files = {
        "step03_sample_flow": load_csv(
            EDA_STEP03_TABLES_DIR / "06_event_sample_flow.csv"
        ),
    }
    files["delta_lightgbm"] = load_csv(MODEL_RESULTS_DIR / "b_minus_a_lightgbm.csv")
    return files


def _number(frame: pd.DataFrame | None, column: str, mask=None) -> pd.Series:
    if frame is None or column not in frame:
        return pd.Series(dtype=float)
    values = frame[column] if mask is None else frame.loc[mask, column]
    return pd.to_numeric(values, errors="coerce").dropna()


def _lightgbm_sns_effect(frame: pd.DataFrame | None) -> tuple[float | None, str]:
    """Read LightGBM's SNS effect and its saved confidence interval."""
    columns = ["B−A", "CI하한", "CI상한"]
    if frame is None or len(frame) != 1 or not set(columns).issubset(frame.columns):
        return None, "비교 자료 부족"
    values = pd.to_numeric(frame.iloc[0][columns], errors="coerce")
    if values.isna().any() or values["CI하한"] > values["CI상한"]:
        return None, "비교 자료 부족"
    if values["CI하한"] > 0:
        status = "통계적으로 개선 확인"
    elif values["CI상한"] < 0:
        status = "통계적으로 성능 저하 확인"
    else:
        status = "통계적으로 뚜렷한 차이 없음"
    return float(values["B−A"]), status


def _render_result_image(title: str, description: str, path) -> None:
    st.markdown(f"#### {title}")
    st.caption(description)
    if not path.is_file():
        st.info("분석 결과 이미지를 찾을 수 없습니다.")
        return
    with st.container(key=f"overview_image_{path.stem}"):
        st.image(path, width="stretch")


def _render_detail_button(eda_page, post_id: str, slide_title: str) -> None:
    if st.button(
        "상세 보기 →", key=f"overview_detail_{post_id}_{slide_title}",
        width="stretch", disabled=eda_page is None,
    ):
        if not analysis_feed.select_result(post_id, slide_title):
            st.info("연결할 분석 결과를 찾을 수 없습니다.")
            return
        st.switch_page(eda_page)


def _render_dataset_cards() -> None:
    _render_heading("데이터 구성", "folder")
    st.markdown(
        '<p class="section-note">분석에 사용한 네 가지 원본 데이터의 규모와 수집 기간</p>',
        unsafe_allow_html=True,
    )
    cards = st.columns(4)
    for column, (name, metadata) in zip(cards, DATASETS.items()):
        with column.container(border=True):
            st.markdown(f"**{name}**")
            st.caption(metadata["description"])
            summary = summarize_csv(metadata["path"], metadata["date_column"])
            if summary is None:
                st.warning("원본 파일을 찾을 수 없습니다.", icon="⚠️")
                continue
            row_count, period = summary
            st.markdown(f"**{row_count:,}행**")
            st.caption(period)


def render(eda_page=None, data_page=None) -> None:
    files = _load_sources()
    change, sns_effect = _lightgbm_sns_effect(files["delta_lightgbm"])
    sns_metadata = DATASETS["SNS 언급량·감성"]
    stock_metadata = DATASETS["종목별 일별 주가"]
    keyword_count = count_unique_csv(
        sns_metadata["path"], sns_metadata["identity_column"]
    )
    stock_count = count_unique_csv(
        stock_metadata["path"], stock_metadata["identity_column"]
    )
    flow = files["step03_sample_flow"]
    event_count = None
    if flow is not None and {"구분", "사건 수"}.issubset(flow.columns):
        selected = _number(flow, "사건 수", flow["구분"].eq("선정한 상승 사건"))
        if not selected.empty:
            event_count = int(selected.iloc[0])
    with st.container(key="overview_kpis"):
        kpis = st.columns(4)
        kpis[0].metric(
            "🔎 분석 키워드",
            f"{keyword_count:,}개" if keyword_count is not None else "—",
        )
        kpis[1].metric(
            "🏢 분석 종목", f"{stock_count:,}개" if stock_count is not None else "—"
        )
        kpis[2].metric(
            "🚀 분석한 급등 사례",
            f"{event_count:,}건" if event_count is not None else "—",
        )
        kpis[3].metric(
            "🧠 LightGBM · SNS 추가 효과",
            f"{change:+.4f}" if change is not None else "—",
            delta=sns_effect,
            delta_color="off",
            delta_arrow="off",
            help="LightGBM에서 시장 정보(A)에 SNS 정보(B)를 추가했을 때의 PR-AUC 차이(B−A)입니다. 판정은 저장된 95% 신뢰구간을 기준으로 표시합니다.",
        )

    _render_dataset_cards()

    _render_heading("분석 결과 하이라이트", "chart")
    _render_heading("SNS와 주가의 움직임", "chat", level=4)
    gallery = [
        lambda: _render_result_image(
            "대표 사례 · 전력기기",
            "전력기기 화제 시작일 주변의 언급량과 종목별 흐름",
            EDA_STEP01_DIR / "figures" / "03_case_01.png",
        ),
        lambda: _render_result_image(
            "화제 전후 시장 반응",
            "실제 화제 시점과 30거래일 뒤 비교 시점의 거래량·변동폭",
            EDA_STEP02_DIR / "figures" / "01_volume_range_surge.png",
        ),
        lambda: _render_result_image(
            "화제 전후 구간별 CAR",
            "화제 전·직후·이후 구간의 누적 초과수익률",
            EDA_STEP01_DIR / "figures" / "04_car_windows.png",
        ),
        lambda: _render_result_image(
            "공시 유무에 따른 반응",
            "최근 공시가 있는 종목과 나머지 종목의 평균 누적 초과수익률",
            EDA_STEP01_DIR / "figures" / "05_disclosure_caar.png",
        ),
    ]
    targets = [
        ("step01", "대표 사례"),
        ("step02", "화제 전후 시장 반응"),
        ("step01", "화제 전후 구간별 CAR"),
        ("step01", "공시 유무에 따른 반응"),
    ]
    for index in range(0, len(gallery), 2):
        left, right = st.columns(2)
        with left.container(border=True, key=f"overview_gallery_{index}"):
            gallery[index]()
            _render_detail_button(eda_page, *targets[index])
        with right.container(border=True, key=f"overview_gallery_{index + 1}"):
            gallery[index + 1]()
            _render_detail_button(eda_page, *targets[index + 1])

    _render_heading("SNS를 더하면 예측이 달라질까?", "brain", level=4)
    with st.container(border=True, key="overview_gallery_model_performance"):
        render_performance_chart()
        _render_detail_button(eda_page, "model_performance", "시험 데이터 성능 비교")

    _render_heading("상세 페이지", "search")
    detail_links = st.columns(2)
    if data_page is not None:
        detail_links[0].page_link(
            data_page, label="데이터 구성 자세히 보기", icon="🗂️", width="stretch"
        )
    else:
        detail_links[0].caption("사이드바에서 데이터 소개를 선택하세요.")
    if eda_page is not None:
        detail_links[1].page_link(
            eda_page, label="분석 게시물 피드 보기", icon="🔎", width="stretch"
        )
    else:
        detail_links[1].caption("사이드바에서 통합 분석 피드를 선택하세요.")
