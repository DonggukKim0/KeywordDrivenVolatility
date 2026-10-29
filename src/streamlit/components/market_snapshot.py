"""Sidebar KOSPI/KOSDAQ quotes relative to the previous trading day's close."""

from datetime import datetime
import math
from zoneinfo import ZoneInfo

import requests
import streamlit as st

KST = ZoneInfo("Asia/Seoul")
QUOTE_URL = "https://polling.finance.naver.com/api/realtime"
INDICES = ("KOSPI", "KOSDAQ")


def _summarize_indices(payload: dict) -> dict:
    if payload.get("resultCode") != "success":
        raise ValueError("지수 조회 실패")
    result = payload["result"]
    as_of = datetime.fromtimestamp(float(result["time"]) / 1000, tz=KST)
    quotes = {}
    for area in result["areas"]:
        if area.get("name") != "SERVICE_INDEX":
            continue
        for item in area["datas"]:
            symbol = item["cd"]
            if symbol not in INDICES:
                continue
            # This endpoint supplies index points and point changes in hundredths.
            latest = float(item["nv"]) / 100
            magnitude = abs(float(item["cv"])) / 100
            rate = abs(float(item["cr"]))
            direction = {"1": 1, "2": 1, "3": 0, "4": -1, "5": -1}[str(item["rf"])]
            change = magnitude * direction
            if not all(math.isfinite(value) for value in (latest, magnitude, rate)):
                raise ValueError("유효하지 않은 지수 값")
            if latest <= 0 or latest - change <= 0:
                raise ValueError("유효하지 않은 지수 값")
            quotes[symbol] = {
                "close": latest,
                "change": change,
                "change_pct": rate * direction,
                "as_of": as_of,
            }
    if set(quotes) != set(INDICES):
        raise ValueError("지수 자료 부족")
    return quotes


@st.cache_data(ttl=60, show_spinner=False)
def _load_indices() -> dict | None:
    try:
        response = requests.get(
            QUOTE_URL,
            params={"query": "SERVICE_INDEX:KOSPI,KOSDAQ"},
            timeout=10,
        )
        response.raise_for_status()
        return _summarize_indices(response.json())
    except (requests.RequestException, ValueError, KeyError, TypeError, OverflowError):
        # Provider/network failures must not prevent access to analysis pages.
        return None


@st.fragment(run_every="60s")
def render() -> None:
    quotes = _load_indices()
    failed = quotes is None
    if quotes is not None:
        st.session_state["last_market_indices_snapshot"] = quotes
    else:
        quotes = st.session_state.get("last_market_indices_snapshot")

    with st.sidebar:
        with st.container(border=True, key="market_snapshot"):
            for position, name in enumerate(INDICES):
                quote = quotes[name] if quotes else None
                if position:
                    st.divider()
                st.markdown(f"**{name}**")
                if quote is None:
                    st.caption("지수 정보를 불러올 수 없습니다.")
                else:
                    change = quote["change"]
                    direction = "up" if change > 0 else "down" if change < 0 else "flat"
                    arrow = "▲" if change > 0 else "▼" if change < 0 else "―"
                    st.markdown(
                        f'<div class="market-index-value">{quote["close"]:,.2f}</div>'
                        f'<div class="market-index-change {direction}">{arrow} '
                        f'{change:+,.2f} ({quote["change_pct"]:+.2f}%) '
                        '<span class="market-index-label">(전일 종가 대비)</span></div>',
                        unsafe_allow_html=True,
                    )
                    if failed:
                        st.caption("최근 조회 실패 · 마지막으로 받은 값 표시 중")
            st.button(
                "시세 새로고침",
                key="refresh_indices",
                on_click=_load_indices.clear,
                width="stretch",
            )
