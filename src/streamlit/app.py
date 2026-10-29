"""Project dashboard entry point."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from components import csv_loader
from components.market_snapshot import render as render_market_snapshot
from components.page_layout import render_page_header

importlib.reload(csv_loader)

from pages import (
    analysis_feed,
    dashboard_overview,
    dataset_intro,
    home,
    model_intro,
    stock_charts,
)
from styles import apply_styles

# Streamlit reruns this entry point while imported page modules may remain cached.
dashboard_overview = importlib.reload(dashboard_overview)
analysis_feed = importlib.reload(analysis_feed)

import streamlit as st

st.set_page_config(
    page_title="SignalLab Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_styles()

st.sidebar.markdown("## ◈ 2조")
st.sidebar.caption("SOCIAL BUZZ × MARKET INTELLIGENCE")
st.sidebar.divider()
st.sidebar.caption("주식·SNS 분석 대시보드")


def render_analysis_feed() -> None:
    analysis_feed.render(model_intro_page=model_intro_page)


eda_page = st.Page(
    render_analysis_feed, title="분석 피드", icon="📸", url_path="analysis"
)
data_page = st.Page(
    dataset_intro.render, title="데이터 소개", icon="🗂️", url_path="data"
)


def render_model_intro() -> None:
    model_intro.render(analysis_page=eda_page)


model_intro_page = st.Page(
    render_model_intro, title="모델 소개", icon="🧠", url_path="model-intro"
)


def render_overview() -> None:
    """Render the overview with links to the registered detail pages."""
    dashboard_overview.render(eda_page=eda_page, data_page=data_page)


overview_page = st.Page(
    render_overview,
    title="종합 대시보드",
    icon="📊",
    url_path="overview",
)

pages = [
    st.Page(home.render, title="홈", icon="🏠", url_path="home", default=True),
    overview_page,
    st.Page(stock_charts.render, title="주식 차트", icon="📈", url_path="stocks"),
    data_page,
    model_intro_page,
    eda_page,
]
current_page = st.navigation(pages, position="sidebar")
render_market_snapshot()

render_page_header(
    "DASHBOARD",
    "SNS 화제성은 주가 급등의 선행 신호인가?",
)
current_page.run()
