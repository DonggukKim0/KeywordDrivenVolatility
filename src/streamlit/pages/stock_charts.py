"""Interactive stock candlestick and market benchmark page."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
from components.csv_loader import load_csv
from config import DATA_DIR

import streamlit as st

STOCK_FILE = DATA_DIR / "raw" / "theme_stock_daily.csv"
BLUE = "#2879D0"
TEAL = "#16A895"
RED = "#DE5962"
TEXT = "#2F455D"
GRID = "#EAF0F6"


def _load_stock_data() -> pd.DataFrame | None:
    frame = load_csv(STOCK_FILE)
    required = {
        "keyword",
        "ticker",
        "name",
        "date",
        "open",
        "high",
        "low",
        "close",
        "market",
        "market_index_close",
    }
    if frame is None or not required.issubset(frame.columns):
        return None

    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    for column in ("open", "high", "low", "close", "market_index_close"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["ticker"] = result["ticker"].astype(str).str.zfill(6)
    result["market"] = result["market"].astype(str).str.upper()
    return result.dropna(subset=["keyword", "ticker", "name", "date", "market"])


def _date_window(
    stock_rows: pd.DataFrame, keyword: str, ticker: str
) -> tuple[date, date] | None:
    dates = stock_rows.loc[stock_rows["ticker"].eq(ticker), "date"].dropna()
    if dates.empty:
        return None
    earliest = dates.min().date()
    latest = dates.max().date()
    default_start = max(earliest, latest - timedelta(days=180))
    selection = st.date_input(
        "조회 기간",
        value=(default_start, latest),
        min_value=earliest,
        max_value=latest,
        key=f"stock_date_range_{keyword}_{ticker}",
    )
    if isinstance(selection, tuple):
        selected_dates = [item for item in selection if isinstance(item, date)]
        if len(selected_dates) >= 2:
            return selected_dates[0], selected_dates[1]
        if len(selected_dates) == 1:
            return selected_dates[0], selected_dates[0]
    if isinstance(selection, date):
        return selection, selection
    return None


def _period_indices(frame: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    indices = frame.loc[
        frame["market"].isin(["KOSPI", "KOSDAQ"]) & frame["market_index_close"].gt(0),
        ["market", "date", "market_index_close"],
    ]
    indices = indices.sort_values("date").drop_duplicates(
        ["market", "date"], keep="first"
    )
    return indices.loc[indices["date"].dt.date.between(start, end)].copy()


def _period_return(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    values = values[values.gt(0)]
    if len(values) < 2:
        return None
    return float((values.iloc[-1] / values.iloc[0] - 1) * 100)


def _render_candlestick(frame: pd.DataFrame) -> None:
    trading_dates = frame["date"].dt.strftime("%Y-%m-%d")
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=trading_dates,
                open=frame["open"],
                high=frame["high"],
                low=frame["low"],
                close=frame["close"],
                name="주가",
                increasing=dict(line=dict(color=RED), fillcolor=RED),
                decreasing=dict(line=dict(color=BLUE), fillcolor=BLUE),
                whiskerwidth=0.55,
            )
        ]
    )
    fig.update_layout(
        height=570,
        margin=dict(l=18, r=18, t=14, b=18),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="white",
        font=dict(family="Pretendard, Malgun Gothic, sans-serif", color=TEXT),
        xaxis_title=None,
        yaxis_title="주가 (원)",
        xaxis_rangeslider_visible=False,
        showlegend=False,
    )
    fig.update_xaxes(type="category", gridcolor=GRID, nticks=9)
    fig.update_yaxes(gridcolor=GRID, tickformat=",.0f")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _render_index_panel(indices: pd.DataFrame, start: date, end: date) -> None:
    with st.container(border=True, key="period_index_panel"):
        st.markdown("#### 같은 기간 시장 지수")
        changes: dict[str, float | None] = {}
        chart = indices.copy()
        for market in ("KOSPI", "KOSDAQ"):
            series = chart.loc[chart["market"].eq(market)].sort_values("date")
            changes[market] = _period_return(series["market_index_close"])
            if not series.empty:
                first_close = series["market_index_close"].iloc[0]
                chart.loc[series.index, "변화율"] = (
                    series["market_index_close"] / first_close - 1
                ) * 100

        with st.container(key="period_index_metrics"):
            metric_cols = st.columns(2)
            for column, market in zip(metric_cols, ("KOSPI", "KOSDAQ")):
                value = changes[market]
                direction = "up" if value is not None and value > 0 else (
                    "down" if value is not None and value < 0 else "flat"
                )
                display_value = f"{value:+.2f}%" if value is not None else "—"
                with column.container(border=True, key=f"{market.lower()}_period_return"):
                    st.markdown(
                        f'<div class="market-period-label">{market} 기간 변화율</div>'
                        f'<div class="market-period-value {direction}">{display_value}</div>',
                        unsafe_allow_html=True,
                    )

        fig = go.Figure()
        for market, color in (("KOSPI", BLUE), ("KOSDAQ", TEAL)):
            series = chart.loc[chart["market"].eq(market)].sort_values("date")
            if not series.empty and "변화율" in series:
                fig.add_trace(
                    go.Scatter(
                        x=series["date"].dt.strftime("%Y-%m-%d"),
                        y=series["변화율"],
                        name=market,
                        mode="lines",
                        line=dict(color=color, width=2.5),
                    )
                )
        fig.update_layout(
            height=290,
            margin=dict(l=8, r=8, t=28, b=8),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="white",
            font=dict(family="Pretendard, Malgun Gothic, sans-serif", color=TEXT, size=11),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            hovermode="x unified",
        )
        fig.update_xaxes(type="category", gridcolor=GRID, nticks=5, tickangle=35)
        fig.update_yaxes(
            title="시작일 대비 변화율", ticksuffix="%", gridcolor=GRID, zeroline=True
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.caption(
            f"선택 기간: {start:%Y-%m-%d} ~ {end:%Y-%m-%d}. 각 지수의 첫 거래일 종가 대비 변화입니다."
        )


def render() -> None:
    st.subheader("키워드별 주가 차트")
    st.markdown(
        '<p class="section-note">키워드와 종목을 고르면 일별 OHLC 캔들과 같은 기간의 KOSPI·KOSDAQ 흐름을 비교합니다.</p>',
        unsafe_allow_html=True,
    )
    data = _load_stock_data()
    if data is None:
        st.warning(f"주가 데이터 파일 또는 필수 컬럼을 찾지 못했습니다: {STOCK_FILE}")
        return
    if data.empty:
        st.warning("주가 데이터에 표시할 행이 없습니다.")
        return

    filter_cols = st.columns([1, 1.2, 1.4])
    keywords = sorted(data["keyword"].dropna().astype(str).unique())
    with filter_cols[0]:
        keyword = st.selectbox("키워드", keywords, key="stock_keyword")

    keyword_rows = data[data["keyword"].eq(keyword)]
    stocks = (
        keyword_rows[["ticker", "name"]]
        .drop_duplicates()
        .sort_values(["name", "ticker"])
    )
    stock_labels = {
        row.ticker: f"{row.name} ({row.ticker})"
        for row in stocks.itertuples(index=False)
    }
    with filter_cols[1]:
        ticker = st.selectbox(
            "종목",
            list(stock_labels),
            format_func=lambda value: stock_labels[value],
            key=f"stock_ticker_{keyword}",
        )

    with filter_cols[2]:
        period = _date_window(keyword_rows, keyword, ticker)
    if period is None:
        st.info("조회 기간을 선택해 주세요.")
        return
    start, end = period
    if start > end:
        st.warning("시작일이 종료일보다 늦습니다. 조회 기간을 다시 선택해 주세요.")
        return

    selected_stock = keyword_rows.loc[keyword_rows["ticker"].eq(ticker)].copy()
    selected_stock = selected_stock[
        selected_stock["date"].dt.date.between(start, end)
    ].sort_values("date")
    selected_stock = selected_stock.dropna(subset=["open", "high", "low", "close"])
    selected_stock = selected_stock.drop_duplicates("date", keep="last")
    if selected_stock.empty:
        st.info("선택한 종목과 기간에 표시할 OHLC 데이터가 없습니다.")
        return

    market = str(selected_stock["market"].iloc[0])
    st.markdown(
        f"#### {stock_labels[ticker]} · {market} · {start:%Y-%m-%d} ~ {end:%Y-%m-%d}"
    )
    price_col, index_col = st.columns([2.2, 1], gap="medium")
    with price_col:
        _render_candlestick(selected_stock)
        st.caption(
            "일봉 기준 시가·고가·저가·종가 · 상승 봉은 빨강, 하락 봉은 파랑입니다."
        )
    with index_col:
        _render_index_panel(_period_indices(data, start, end), start, end)
