"""프로젝트에 있는 원본 데이터 소개와 미리보기."""

from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd
from components.csv_loader import load_csv
from config import DATA_DIR, PROJECT_ROOT

import streamlit as st

DATASETS = {
    "SNS 언급량·감성": {
        "path": DATA_DIR / "raw" / "sns_daily_mentions_sentiment.csv",
        "description": "키워드별 일일 언급량과 채널별 언급, 감성 지표를 담은 데이터입니다.",
        "date_column": "date",
        "identity_column": "keyword",
        "preview_columns": [
            "keyword",
            "date",
            "언급량 추이__전체",
            "감성__긍정",
            "감성__부정",
            "감성__중립",
        ],
        "column_descriptions": {
            "keyword": "언급량을 집계한 검색 키워드",
            "date": "언급량을 집계한 날짜",
            "언급량 추이__전체": "수집 채널 전체에서 집계한 언급량",
            "감성__긍정": "해당 날짜의 긍정 감성 지표",
            "감성__부정": "해당 날짜의 부정 감성 지표",
            "감성__중립": "해당 날짜의 중립 감성 지표",
        },
    },
    "종목별 일별 주가": {
        "path": DATA_DIR / "raw" / "theme_stock_daily.csv",
        "description": "키워드·테마와 연결된 종목의 일별 OHLCV 및 시장 지수 데이터입니다.",
        "date_column": "date",
        "identity_column": "ticker",
        "preview_columns": [
            "keyword",
            "ticker",
            "name",
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "market",
        ],
        "column_descriptions": {
            "keyword": "종목과 연결된 검색 키워드",
            "ticker": "종목 코드",
            "name": "종목명",
            "date": "주식 거래일",
            "open": "시가: 거래일의 첫 가격",
            "high": "고가: 거래일의 가장 높은 가격",
            "low": "저가: 거래일의 가장 낮은 가격",
            "close": "종가: 거래일의 마지막 가격",
            "volume": "거래량",
            "market": "시장 구분 (KOSPI 또는 KOSDAQ)",
        },
    },
    "기업 공시": {
        "path": DATA_DIR / "raw" / "opendart" / "opendart_disclosures.csv",
        "description": "OpenDART에서 수집한 기업별 공시 정보와 접수일, 공시 링크입니다.",
        "date_column": "rcept_dt",
        "identity_column": "name",
        "preview_columns": [
            "ticker",
            "name",
            "report_nm",
            "rcept_dt",
            "is_major",
            "disclosure_url",
        ],
        "column_descriptions": {
            "ticker": "종목 코드",
            "name": "종목명",
            "report_nm": "공시 보고서명",
            "rcept_dt": "공시 접수일",
            "is_major": "주요 공시 여부를 나타내는 값",
            "disclosure_url": "전자공시 원문 링크",
        },
    },
    "기업 재무정보": {
        "path": DATA_DIR / "raw" / "opendart" / "opendart_financials.csv",
        "description": "OpenDART 재무제표의 기업·사업연도·계정별 금액 정보입니다.",
        "date_column": "bsns_year",
        "identity_column": "ticker",
        "preview_columns": [
            "ticker",
            "bsns_year",
            "fs_div",
            "sj_div",
            "account_nm",
            "thstrm_amount",
            "currency",
        ],
        "column_descriptions": {
            "ticker": "종목 코드",
            "bsns_year": "재무제표 사업연도",
            "fs_div": "재무제표 구분 (연결 또는 별도)",
            "sj_div": "재무제표 종류 (재무상태표, 손익계산서 등)",
            "account_nm": "재무제표 계정명",
            "thstrm_amount": "당기 금액",
            "currency": "금액의 통화 단위",
        },
    },
}


def _relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


@st.cache_data(show_spinner=False)
def _keyword_tickers(path: str, modified_ns: int) -> pd.DataFrame:
    """Read only the keyword–stock mapping, preserving leading ticker zeros."""
    del modified_ns
    return (
        pd.read_csv(
            path,
            usecols=["keyword", "ticker"],
            dtype="string",
            encoding="utf-8-sig",
        )
        .dropna()
        .drop_duplicates()
    )


def _set_keyword_checks(dataset_name: str, keywords: list[str], checked: bool) -> None:
    for keyword in keywords:
        st.session_state[f"data_keyword_check_{dataset_name}_{keyword}"] = checked


def _keyword_rows(
    frame: pd.DataFrame,
    keywords: list[str],
    mapping: pd.DataFrame | None,
) -> pd.DataFrame:
    if mapping is None:
        return frame.loc[frame["keyword"].astype("string").isin(keywords)]
    tickers = mapping.loc[mapping["keyword"].isin(keywords), "ticker"]
    normalized = (
        frame["ticker"]
        .astype("string")
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(6)
    )
    return frame.loc[normalized.isin(tickers.str.zfill(6))]


def _filter_by_keyword(
    frame: pd.DataFrame,
    dataset_name: str,
) -> tuple[pd.DataFrame, list[str], pd.DataFrame | None]:
    mapping = None
    if "keyword" in frame.columns:
        keywords = sorted(frame["keyword"].dropna().astype(str).unique())
    elif "ticker" in frame.columns:
        stock_path = DATASETS["종목별 일별 주가"]["path"]
        try:
            mapping = _keyword_tickers(
                str(stock_path.resolve()), stock_path.stat().st_mtime_ns
            )
        except (OSError, ValueError, UnicodeError, pd.errors.ParserError):
            st.caption("키워드–종목 연결 자료를 불러올 수 없어 전체 데이터 표시")
            return frame, [], None
        keywords = sorted(mapping["keyword"].unique())
    else:
        return frame, [], None

    selected = []
    with st.expander("키워드 선택", expanded=True):
        select_all, clear_all = st.columns(2)
        select_all.button(
            "전체 선택",
            key=f"data_keywords_all_{dataset_name}",
            on_click=_set_keyword_checks,
            args=(dataset_name, keywords, True),
        )
        clear_all.button(
            "선택 해제",
            key=f"data_keywords_clear_{dataset_name}",
            on_click=_set_keyword_checks,
            args=(dataset_name, keywords, False),
        )
        columns = st.columns(3)
        for index, keyword in enumerate(keywords):
            if columns[index % 3].checkbox(
                keyword,
                value=True,
                key=f"data_keyword_check_{dataset_name}_{keyword}",
            ):
                selected.append(keyword)
    st.caption(
        f"키워드 {len(selected)}개 / {len(keywords)}개 선택 · 선택한 키워드 기준 요약 및 미리보기"
    )
    return _keyword_rows(frame, selected, mapping), selected, mapping


def _render_content() -> None:
    st.subheader("데이터 소개")
    st.markdown(
        '<p class="section-note">프로젝트에서 사용하는 원본 데이터의 구성과 일부 행을 확인할 수 있습니다. '
        "데이터 파일은 기존 저장 위치에서 읽습니다.</p>",
        unsafe_allow_html=True,
    )

    dataset_name = st.selectbox(
        "확인할 데이터", list(DATASETS), key="data_preview_dataset"
    )
    metadata = DATASETS[dataset_name]
    path = metadata["path"]
    st.markdown(f"#### {dataset_name}")
    st.write(metadata["description"])
    is_private_sns = dataset_name == "SNS 언급량·감성"
    if is_private_sns:
        st.markdown(
            '파일 위치: <span class="sns-private-blur" aria-hidden="true">'
            'data/raw/private_dataset.csv</span>'
            '<span class="sns-private-note"> (비공개)</span>',
            unsafe_allow_html=True,
        )
    else:
        st.caption(f"파일 위치: `{_relative_path(path)}`")

    if not path.is_file():
        if is_private_sns:
            st.info(
                "SNS 원자료는 공개 저장소에 포함되지 않습니다. 분석을 재현하려면 "
                "`docs/data_requirements.md`의 컬럼 명세에 맞는 CSV를 "
                "`data/raw/sns_daily_mentions_sentiment.csv`에 로컬로 준비해 주세요."
            )
            st.caption("이 저장소에는 실제 SNS 관측값이 포함되어 있지 않습니다.")
        else:
            st.warning(
                f"데이터 파일을 찾을 수 없습니다: `{_relative_path(path)}`. "
                "필요한 입력과 준비 방법은 `docs/data_requirements.md`를 확인해 주세요."
            )
        return

    frame = load_csv(path)
    if frame is None or frame.empty:
        st.warning("파일은 있지만 미리보기에 표시할 데이터가 없습니다.")
        return

    frame, selected_keywords, mapping = _filter_by_keyword(frame, dataset_name)
    if frame.empty:
        st.info(
            "선택한 키워드에 해당하는 데이터가 없습니다. 키워드를 하나 이상 선택해 주세요."
        )
        return

    date_column = metadata["date_column"]
    identity_column = metadata["identity_column"]
    date_values = (
        frame[date_column] if date_column in frame else pd.Series(dtype="object")
    )
    if date_column == "bsns_year":
        valid_dates = pd.to_numeric(date_values, errors="coerce").dropna()
        period_label = (
            f"{valid_dates.min():.0f} ~ {valid_dates.max():.0f}"
            if not valid_dates.empty
            else "—"
        )
    else:
        valid_dates = pd.to_datetime(date_values, errors="coerce").dropna()
        period_label = (
            f"{valid_dates.min():%y.%m.%d} ~ {valid_dates.max():%y.%m.%d}"
            if not valid_dates.empty
            else "—"
        )
    identities = (
        frame[identity_column].nunique(dropna=True)
        if identity_column in frame
        else None
    )

    summary_columns = st.columns(4)
    summary_columns[0].metric("행 수", f"{len(frame):,}")
    summary_columns[1].metric("컬럼 수", f"{len(frame.columns):,}")
    with summary_columns[2].container(key="dataset_period"):
        st.metric("기간", period_label)
    summary_columns[3].metric(
        "고유 항목 수",
        f"{identities:,}" if identities is not None else "—",
        help=f"{identity_column} 기준",
    )

    available_columns = [
        column for column in metadata["preview_columns"] if column in frame.columns
    ]
    with st.expander("컬럼 안내"):
        descriptions = metadata["column_descriptions"]
        guide_columns = [
            column for column in metadata["preview_columns"] if column in frame.columns
        ]
        st.dataframe(
            pd.DataFrame(
                {
                    "컬럼": guide_columns,
                    "설명": [descriptions[column] for column in guide_columns],
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("전체 컬럼명 확인"):
        st.write(", ".join(map(str, frame.columns)))

    preview_count = st.slider(
        "미리보기 행 수", min_value=1, max_value=20, value=10, step=1
    )
    preview_frame = frame
    if selected_keywords:
        preview_key = f"data_preview_active_keyword_{dataset_name}"
        if st.session_state.get(preview_key) not in selected_keywords:
            st.session_state[preview_key] = selected_keywords[0]
        preview_keyword = st.selectbox(
            "미리볼 키워드",
            selected_keywords,
            key=preview_key,
        )
        preview_frame = _keyword_rows(frame, [preview_keyword], mapping)
        st.caption(
            f"‘{preview_keyword}’ 데이터 {len(preview_frame):,}행 중 최대 {preview_count}행 표시"
        )
    if preview_frame.empty:
        st.info("이 키워드에 해당하는 데이터가 없습니다.")
    preview = (
        preview_frame[available_columns].head(preview_count)
        if available_columns
        else preview_frame.head(preview_count)
    )
    if is_private_sns:
        # Render placeholders only: CSS blur alone would expose the original values.
        headings = "".join(f"<th>{escape(str(column))}</th>" for column in preview.columns)
        rows = []
        for _, row in preview.iterrows():
            cells = []
            for column in preview.columns:
                if column in {"keyword", "date"}:
                    cells.append(f"<td>{escape(str(row[column]))}</td>")
                else:
                    cells.append(
                        '<td aria-label="비공개 수치"><span class="sns-private-blur" '
                        'aria-hidden="true">12345</span></td>'
                    )
            rows.append("<tr>" + "".join(cells) + "</tr>")
        st.caption("언급량·감성 수치는 비공개이며, 흐리게 표시된 값은 대체 값입니다.")
        st.markdown(
            '<div class="sns-private-preview"><table><thead><tr>'
            + headings + "</tr></thead><tbody>" + "".join(rows)
            + "</tbody></table></div>",
            unsafe_allow_html=True,
        )
    else:
        st.dataframe(preview, use_container_width=True, hide_index=True)


def render() -> None:
    with st.container(key="dataset_intro_page"):
        _render_content()
