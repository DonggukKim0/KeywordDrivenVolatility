"""Interactive stock overlays for the existing three representative cases."""

import pandas as pd
import plotly.graph_objects as go
from config import DATA_DIR, EDA_STEP01_DIR, EDA_STEP01_TABLES_DIR
from plotly.subplots import make_subplots

import streamlit as st
from components.csv_loader import load_csv
from components.result_viewer import render_figure


@st.cache_data(show_spinner=False)
def _case_windows(path: str, modified_ns: int, cases: tuple) -> dict:
    """Cache only event windows, using the same calendar and CAR as STEP01."""
    del modified_ns
    frame = pd.read_csv(
        path,
        usecols=[
            "keyword",
            "ticker",
            "name",
            "date",
            "mention_z28",
            "excess_return_1d",
        ],
        dtype={"ticker": str},
        encoding="utf-8-sig",
        float_precision="round_trip",
    )
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    if frame.duplicated(["ticker", "date"]).any():
        raise ValueError("종목·날짜 중복 자료")
    dates = pd.DatetimeIndex(frame["date"].unique()).sort_values()
    result = {}
    for keyword, day0 in cases:
        day0 = pd.Timestamp(day0)
        if day0 not in dates:
            continue
        index = dates.get_loc(day0)
        if index < 10 or index + 20 >= len(dates):
            continue
        window = dates[index - 10 : index + 21]
        selected = frame.loc[frame["keyword"].eq(keyword)]
        stocks = (
            selected[["ticker", "name"]]
            .drop_duplicates("ticker")
            .sort_values(["name", "ticker"])
        )
        selected = selected.loc[selected["date"].isin(window)]
        if selected.empty:
            continue
        returns = selected.pivot_table(
            index="date", columns="ticker", values="excess_return_1d"
        )
        car = (
            returns.reindex(index=window, columns=stocks["ticker"]).fillna(0).cumsum()
            * 100
        )
        z28 = (
            selected.drop_duplicates("date")
            .set_index("date")["mention_z28"]
            .reindex(window)
        )
        result[keyword] = {"stocks": stocks, "car": car, "z28": z28}
    return result


def _reset_stocks() -> None:
    st.session_state["case_selected_tickers"] = []


def _overlay(case: dict, selected: list[str]) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    names = case["stocks"].set_index("ticker")["name"]
    days = list(range(-10, 21))
    dates = case["car"].index.strftime("%Y-%m-%d").tolist()
    for ticker in selected:
        fig.add_trace(
            go.Scatter(
                x=days,
                y=case["car"][ticker],
                mode="lines",
                name=f"{names[ticker]} ({ticker})",
                customdata=dates,
                hovertemplate="%{customdata} · Day %{x}<br>CAR %{y:.2f}%<extra>%{fullData.name}</extra>",
            ),
            secondary_y=False,
        )
    fig.add_trace(
        go.Scatter(
            x=days,
            y=case["z28"],
            mode="lines",
            name="언급 급증도(z28)",
            line=dict(color="#E8833A", dash="dash", width=2),
            customdata=dates,
            hovertemplate="%{customdata} · Day %{x}<br>언급 급증도 %{y:.2f}<extra></extra>",
        ),
        secondary_y=True,
    )
    fig.add_vline(
        x=0, line_dash="dash", line_color="#65788B", annotation_text="화제 시작일"
    )
    if selected:
        fig.add_hline(y=0, line_color="#DDE8F1", secondary_y=False)
    fig.update_xaxes(title_text="화제 시작일 기준 거래일 (Day)", range=[-10, 20])
    fig.update_yaxes(
        title_text="누적 초과수익률 CAR (%)", secondary_y=False, visible=bool(selected)
    )
    fig.update_yaxes(title_text="언급 급증도(z28)", secondary_y=True, showgrid=False)
    fig.update_layout(
        height=450,
        hovermode="x unified",
        margin=dict(l=30, r=30, t=45, b=30),
        legend=dict(orientation="h", y=-0.22),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="white",
        font=dict(color="#203B56"),
        colorway=["#168E84", "#2879D0", "#7B5EA7", "#D03B3B", "#B08C24"],
    )
    return fig


def render() -> None:
    path = EDA_STEP01_TABLES_DIR / "03_representative_cases.csv"
    try:
        cases = load_csv(path)
    except (OSError, ValueError, UnicodeError, pd.errors.ParserError):
        cases = None
    if cases is None or cases.empty or not {"키워드", "Day0"}.issubset(cases.columns):
        st.info("대표 사례 목록을 불러올 수 없습니다.")
        return
    cases = cases.head(3).reset_index(drop=True)
    keyword = st.selectbox(
        "대표 사례 키워드",
        cases["키워드"].tolist(),
        key="case_keyword",
        on_change=_reset_stocks,
    )
    position = cases.index[cases["키워드"].eq(keyword)][0]
    row = cases.iloc[position]
    st.caption(f"화제 시작일: {row['Day0']} · Day −10부터 +20까지")
    render_figure(EDA_STEP01_DIR / "figures" / f"03_case_{position + 1:02d}.png")

    st.markdown("#### 종목을 골라 언급 급증도와 함께 보기")
    source = DATA_DIR / "processed" / "model_dataset.csv"
    try:
        windows = _case_windows(
            str(source.resolve()),
            source.stat().st_mtime_ns,
            tuple(cases[["키워드", "Day0"]].itertuples(index=False, name=None)),
        )
        case = windows.get(keyword)
    except (OSError, ValueError, UnicodeError, pd.errors.ParserError):
        case = None
    if case is None:
        st.info("이 사례의 시계열 자료를 불러올 수 없습니다.")
    else:
        names = case["stocks"].set_index("ticker")["name"].to_dict()
        if "case_selected_tickers" in st.session_state:
            st.session_state["case_selected_tickers"] = [
                item
                for item in st.session_state["case_selected_tickers"]
                if item in names
            ]
        selected = st.multiselect(
            "함께 볼 종목",
            list(names),
            default=[],
            key="case_selected_tickers",
            format_func=lambda ticker: f"{names[ticker]} ({ticker})",
            placeholder="종목을 하나 이상 선택하세요",
        )
        if not selected:
            st.caption(
                "종목을 선택하면 언급 급증도 위에 해당 종목의 CAR 선이 함께 표시됩니다."
            )
        st.plotly_chart(
            _overlay(case, selected), width="stretch", config={"displayModeBar": False}
        )
        st.caption(
            "CAR은 Day −10부터 누적한 시장 대비 일별 수익률입니다. 기존 대표 사례와 동일하게 결측 수익률은 0으로 계산합니다."
        )
    with st.expander("선택한 대표 사례 결과"):
        st.dataframe(
            cases.loc[cases["키워드"].eq(keyword)], hide_index=True, width="stretch"
        )
