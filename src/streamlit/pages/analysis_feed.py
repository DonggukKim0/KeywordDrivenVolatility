"""Instagram-inspired EDA feed and post viewer for STEP01 through STEP03."""

from __future__ import annotations

from pathlib import Path

from components import case_explorer
from components.result_viewer import render_figure, render_table
from config import (
    EDA_STEP01_DIR,
    EDA_STEP01_TABLES_DIR,
    EDA_STEP02_DIR,
    EDA_STEP02_TABLES_DIR,
    EDA_STEP03_DIR,
    EDA_STEP03_TABLES_DIR,
)

import streamlit as st
from pages import model_analysis

ASSET_DIR = Path(__file__).resolve().parents[1] / "assets"
PROFILE_IMAGE = ASSET_DIR / "profile_stock.svg"


def _table_slide(
    title: str,
    description: str,
    *tables: tuple[str, Path, bool],
    note: str = "",
    table_options: dict | None = None,
) -> dict:
    return {
        "title": title,
        "description": description,
        "tables": tables,
        "note": note,
        "table_options": table_options or {},
    }


POSTS = [
    {
        "id": "step01",
        "category": "데이터 분석",
        "step": "언급량과 수익률",
        "title": "화제가 커진 날, 수익률도 함께 움직였을까?",
        "description": "언급량과 수익률의 관계부터 화제 전후 CAR, 공시 비교와 다음 날 매수 결과까지 살펴봅니다.",
        "hashtags": ["SNS언급량", "초과수익률", "CAR검정", "공시비교"],
        "results_dir": EDA_STEP01_DIR,
        "cover": EDA_STEP01_DIR / "figures" / "01_keyword_correlations.png",
        "slides": [
            {
                "title": "언급량과 초과수익률의 관계",
                "description": "키워드와 종목별 상관을 나란히 비교합니다.",
                "figures": [
                    EDA_STEP01_DIR / "figures" / "01_keyword_correlations.png",
                    EDA_STEP01_DIR / "figures" / "02_stock_correlations.png",
                ],
            },
            _table_slide(
                "키워드별 상관 결과",
                "SNS 키워드 언급량과 초과수익률의 관계입니다.",
                (
                    "키워드 상관 결과",
                    EDA_STEP01_TABLES_DIR / "01_keyword_correlations.csv",
                    False,
                ),
                table_options={
                    "column_config": {
                        "최대z28": st.column_config.NumberColumn(
                            "최대 언급 급증도",
                            help="이전 28일 평균과 비교한 언급량 증가를 표준편차 단위로 나타낸 값의 최댓값. 클수록 평소보다 언급이 크게 늘어난 날이 있었음을 의미합니다.",
                        ),
                    }
                },
            ),
            _table_slide(
                "종목별 상관 및 이벤트 반응",
                "종목 단위의 상관 결과와 화제 시작일 주변 반응입니다.",
                (
                    "종목별 결과",
                    EDA_STEP01_TABLES_DIR / "02_stock_correlations_and_reactions.csv",
                    False,
                ),
                table_options={"hidden_columns": ("is_leader",)},
            ),
            {
                "title": "대표 사례",
                "description": "화제 시작일의 대표 사례와 종목별 흐름입니다.",
                "case_explorer": True,
                "note": "언급량과 수익률의 상관은 관측된 동행 관계이며, 인과효과나 예측력을 뜻하지 않습니다.",
            },
            {
                "title": "화제 전후 구간별 CAR",
                "description": "Day −5~+10의 누적 초과수익률입니다. 화제 전·직후·이후 구간을 나눠 이벤트 평균 CAR이 0과 다른지 t검정합니다.",
                "figures": [EDA_STEP01_DIR / "figures" / "04_car_windows.png"],
                "tables": [
                    (
                        "CAR 구간별 통계와 t검정",
                        EDA_STEP01_TABLES_DIR / "04_car_window_tests.csv",
                        False,
                    )
                ],
                "note": "오차막대는 이벤트 평균 CAR의 95% 신뢰구간입니다. 화제 전후의 관측 결과이며 SNS의 인과효과를 뜻하지 않습니다.",
            },
            {
                "title": "공시 유무에 따른 반응",
                "description": "화제 시작일의 최근 3거래일 공시 건수가 0보다 큰 종목과 나머지 종목의 평균 누적 초과수익률을 비교합니다.",
                "figures": [EDA_STEP01_DIR / "figures" / "05_disclosure_caar.png"],
                "tables": [
                    (
                        "공시 유무별 구간 CAR와 검정",
                        EDA_STEP01_TABLES_DIR / "06_disclosure_window_comparison.csv",
                        False,
                    ),
                    (
                        "일별 평균 누적 초과수익률",
                        EDA_STEP01_TABLES_DIR / "05_disclosure_caar.csv",
                        True,
                    ),
                ],
                "note": "차이 검정은 같은 이벤트 안에 두 공시 그룹이 모두 있는 경우의 '공시 없음−있음'을 비교합니다. 표의 그룹별 전체 평균 차이와는 집계 단위가 다릅니다.",
            },
            {
                "title": "화제 전후 일별 급등 비율",
                "description": "시장 대비 일일 초과수익률이 +5% 이상인 종목 비율을 전체 분석 기간의 평소 비율과 비교합니다.",
                "figures": [EDA_STEP01_DIR / "figures" / "06_daily_surge_rates.png"],
                "tables": [
                    (
                        "일별 급등 비율과 p값",
                        EDA_STEP01_TABLES_DIR / "07_daily_surge_rates.csv",
                        False,
                    )
                ],
                "note": "파란 막대는 평소보다 비율이 높고 이벤트 단위 t검정 p<0.05인 날입니다. 날짜별 검정에 다중비교 보정은 적용하지 않았습니다.",
            },
            _table_slide(
                "다음 날 시가에 매수했다면",
                "Day 0·1·2 밤에 화제를 알고 다음 날 시가에 매수해 1·2·3·5거래일 보유한 시장 대비 초과수익률입니다.",
                (
                    "매수 시점·보유 기간별 결과",
                    EDA_STEP01_TABLES_DIR / "08_next_open_holding_returns.csv",
                    False,
                ),
                note="수익률과 비율은 소수 단위입니다(0.01=1%). 평소는 전체 종목×거래일의 같은 보유수익률입니다. 지수 시가가 없어 시장수익률은 신호일 종가부터 매도일 종가까지로 계산하며, 거래비용은 반영하지 않습니다.",
            ),
        ],
    },
    {
        "id": "step02",
        "category": "데이터 분석",
        "step": "화제 전후 반응",
        "title": "화제 전후, 거래량과 변동폭은 어떻게 달라졌을까?",
        "description": "실제 화제 시점과 가짜 시점의 시장 반응을 비교합니다.",
        "hashtags": ["거래량", "변동폭", "이벤트분석"],
        "results_dir": EDA_STEP02_DIR,
        "cover": EDA_STEP02_DIR / "figures" / "01_volume_range_surge.png",
        "slides": [
            {
                "title": "화제 전후 시장 반응",
                "description": "Day −10~−1 평균을 기준으로 Day 0~+2 반응을 살펴봅니다.",
                "explanation": (
                    "**가짜 시점이란?**\n\n"
                    "실제 화제 시작일에서 **30거래일 뒤로 옮긴 비교용 날짜**입니다. "
                    "같은 종목의 거래량과 변동폭을 비교해, 실제 화제 시점의 반응이 더 큰지 살펴봅니다."
                ),
                "figure_layout": "vertical",
                "figures": [
                    EDA_STEP02_DIR / "figures" / "01_volume_range_surge.png",
                    EDA_STEP02_DIR / "figures" / "02_keyword_volume_ratios.png",
                ],
            },
            _table_slide(
                "실제 화제 시점과 가짜 시점",
                "실제 시점과 30거래일 뒤 가짜 시점의 요약 결과입니다.",
                (
                    "시점 비교 요약",
                    EDA_STEP02_TABLES_DIR / "02_volume_range_summary.csv",
                    False,
                ),
            ),
            _table_slide(
                "키워드별 거래량 배수",
                "화제 전 기준 대비 키워드별 거래량 변화를 확인합니다.",
                (
                    "키워드별 결과",
                    EDA_STEP02_TABLES_DIR / "03_keyword_volume_ratios.csv",
                    False,
                ),
            ),
            _table_slide(
                "종목·이벤트별 상세 배수",
                "종목과 이벤트 단위의 거래량·변동폭 상세 결과입니다.",
                (
                    "상세 결과",
                    EDA_STEP02_TABLES_DIR / "01_stock_volume_range_ratios.csv",
                    False,
                ),
                note="실제·가짜 시점 비교는 이벤트 주변의 기술통계이며, 인과효과를 뜻하지 않습니다.",
            ),
        ],
    },
    {
        "id": "step03",
        "category": "데이터 분석",
        "step": "급등 이후 신호",
        "title": "급등 이후, SNS와 시장에는 어떤 신호가 남았을까?",
        "description": "급등 사건과 붕괴, 감성, 뉴스·SNS 동행을 살펴봅니다.",
        "hashtags": ["급등사건", "감성분석", "뉴스SNS"],
        "results_dir": EDA_STEP03_DIR,
        "cover": EDA_STEP03_DIR / "figures" / "01_event_frequency.png",
        "slides": [
            {
                "title": "급등 사건 빈도와 흐름",
                "description": "테마별 상승 사건의 빈도를 살펴봅니다.",
                "figures": [EDA_STEP03_DIR / "figures" / "01_event_frequency.png"],
            },
            _table_slide(
                "사건 정의와 붕괴 시점",
                "하락 기준 민감도, 표본 흐름, 테마별 붕괴 비율과 첫 붕괴 시점입니다.",
                (
                    "하락 기준 민감도",
                    EDA_STEP03_TABLES_DIR / "15_definition_sensitivity.csv",
                    False,
                ),
                (
                    "표본 흐름",
                    EDA_STEP03_TABLES_DIR / "06_event_sample_flow.csv",
                    False,
                ),
                (
                    "테마별 붕괴 비율",
                    EDA_STEP03_TABLES_DIR / "07_theme_collapse_rates.csv",
                    False,
                ),
                (
                    "첫 붕괴 시점",
                    EDA_STEP03_TABLES_DIR / "08_collapse_timing.csv",
                    False,
                ),
                ("상승 사건 전체 목록", EDA_STEP03_TABLES_DIR / "events.csv", True),
            ),
            {
                "title": "SNS 동시 신호 그룹",
                "description": "급등 사례에서 관찰된 SNS 신호 그룹의 결과입니다.",
                "figures": [EDA_STEP03_DIR / "figures" / "03_signal_groups.png"],
                "tables": [
                    (
                        "사례 선정 결과",
                        EDA_STEP03_TABLES_DIR / "09_case_selection.csv",
                        False,
                    ),
                    (
                        "시점별 신호 그룹",
                        EDA_STEP03_TABLES_DIR / "10_landmark_signal_rates.csv",
                        False,
                    ),
                    (
                        "가격 상태별 신호 그룹",
                        EDA_STEP03_TABLES_DIR / "11_price_state_signal_rates.csv",
                        False,
                    ),
                ],
                "note": "신호별 그룹 차이는 테마·시기·현재 가격 상태가 섞인 관측 결과이며 예측 성능 검증이 아닙니다.",
            },
            *[
                {
                    "title": f"대표 사례 · {code}",
                    "description": "급등 사례의 주가와 SNS 신호 변화를 확인합니다.",
                    "figures": [EDA_STEP03_DIR / "figures" / f"02_case_{code}.png"],
                }
                for code in ("394280", "095340", "049480")
            ],
            _table_slide(
                "사건×거래일 관찰 자료",
                "신호 그룹 분석에 사용된 사건별 일별 관찰 자료입니다.",
                ("관찰 자료", EDA_STEP03_TABLES_DIR / "event_panel.csv.gz", True),
                table_options={
                    "hidden_columns": (
                        "mentions",
                        "news_mentions",
                        "social_mentions",
                        "sentiment_total",
                        "sns_file__log_mentions",
                    ),
                },
                note="관찰된 그룹 차이는 예측 성능 검증 결과가 아닙니다.",
            ),
            {
                "title": "부정 감성 변화와 이후 수익률",
                "description": "감성 변화 구간과 이후 수익률의 분포를 살펴봅니다.",
                "figures": [
                    EDA_STEP03_DIR / "figures" / "04_negative_change_boxplot.png"
                ],
                "tables": [
                    (
                        "변화 구간별 표본 요약",
                        EDA_STEP03_TABLES_DIR / "12_negative_change_boxplot.csv",
                        False,
                    )
                ],
            },
            {
                "title": "종목별 감성·수익률 상관",
                "description": "종목별 상관 결과와 감성 변화 구간을 비교합니다.",
                "figures": [EDA_STEP03_DIR / "figures" / "05_stock_correlations.png"],
                "tables": [
                    (
                        "종목별 상관",
                        EDA_STEP03_TABLES_DIR / "13_stock_correlations.csv",
                        False,
                    )
                ],
                "note": "미래 수익률은 관측일 이후의 기술통계이며 독립된 기간의 예측력 평가는 아닙니다.",
            },
            _table_slide(
                "미래 수익률 포함 일별 결과",
                "감성 관측일 이후의 일별 수익률 결과 자료입니다.",
                (
                    "일별 결과 자료",
                    EDA_STEP03_TABLES_DIR / "daily_outcomes.csv.gz",
                    True,
                ),
                note="관측일 이후의 기술통계이며 독립된 기간의 예측력 평가는 아닙니다.",
            ),
            {
                "title": "뉴스·SNS와 당일 주가 동행",
                "description": "테마별 뉴스·SNS 언급량과 주가 간 동행을 살펴봅니다.",
                "figures": [EDA_STEP03_DIR / "figures" / "06_theme_same_day.png"],
                "tables": [
                    (
                        "테마별 상관",
                        EDA_STEP03_TABLES_DIR / "14_theme_same_day_correlations.csv",
                        False,
                    )
                ],
                "note": "당일 동행은 오전·장중 순서를 구분하지 않으므로 선행 관계나 인과효과를 뜻하지 않습니다.",
            },
        ],
    },
]
POSTS.extend(model_analysis.MODEL_FEED_POSTS)


def _post_by_id(post_id: str) -> dict:
    return next(post for post in POSTS if post["id"] == post_id)


def _like_key(post_id: str) -> str:
    return f"eda_like_count_{post_id}"


def _comments_key(post_id: str) -> str:
    return f"eda_comments_{post_id}"


def _render_post_header(post: dict, view: str) -> None:
    avatar, account, menu = st.columns([0.2, 1, 0.28])
    with avatar:
        st.image(PROFILE_IMAGE, width=42)
    with account:
        st.markdown('<div class="eda-account-name">2조</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="eda-post-meta">{post["step"]} · 분석 게시물</div>',
            unsafe_allow_html=True,
        )
    with menu:
        st.button(
            "⋯",
            help="게시물 메뉴 열기",
            key=f"eda_menu_{view}_{post['id']}",
            on_click=_toggle_post_menu,
            args=(f"eda_post_menu_open_{view}_{post['id']}",),
        )
    menu_state_key = f"eda_post_menu_open_{view}_{post['id']}"
    if st.session_state.get(menu_state_key, False):
        with st.container(border=True):
            st.markdown("**게시물 정보**")
            st.write(post["title"])
            st.caption(f"{post['step']} · 분석 자료 {len(post['slides'])}장")
            st.button(
                "메뉴 닫기",
                key=f"eda_close_menu_{view}_{post['id']}",
                on_click=_close_post_menu,
                args=(menu_state_key,),
            )


def _render_hashtags(post: dict) -> None:
    tags = " ".join(f"<span>#{tag}</span>" for tag in post["hashtags"])
    st.markdown(f'<div class="eda-hashtags">{tags}</div>', unsafe_allow_html=True)


def select_result(post_id: str, slide_title: str) -> bool:
    """Select a result by stable post ID and slide title before page navigation."""
    post = next((post for post in POSTS if post["id"] == post_id), None)
    if post is None:
        return False
    index = next((i for i, slide in enumerate(post["slides"]) if slide["title"] == slide_title), None)
    if index is None:
        return False
    st.session_state["eda_selected_post"] = post_id
    st.session_state["eda_slide_index"] = index
    st.session_state["analysis_feed_category"] = post["category"]
    if post_id == "step01" and slide_title == "대표 사례":
        st.session_state["case_keyword"] = "전력기기"
        st.session_state["case_selected_tickers"] = []
    return True


def _open_post(post_id: str) -> None:
    st.session_state["eda_selected_post"] = post_id
    st.session_state["eda_slide_index"] = 0


def _toggle_post_menu(menu_state_key: str) -> None:
    was_open = st.session_state.get(menu_state_key, False)
    for key in list(st.session_state.keys()):
        if key.startswith("eda_post_menu_open_"):
            st.session_state[key] = False
    st.session_state[menu_state_key] = not was_open


def _close_post_menu(menu_state_key: str) -> None:
    st.session_state[menu_state_key] = False


def _close_post() -> None:
    st.session_state["eda_selected_post"] = None
    st.session_state["eda_slide_index"] = 0


def _increment_like(post_id: str) -> None:
    key = _like_key(post_id)
    st.session_state[key] = st.session_state.get(key, 0) + 1


def _move_slide(delta: int, slide_count: int) -> None:
    index = st.session_state.get("eda_slide_index", 0)
    st.session_state["eda_slide_index"] = max(0, min(slide_count - 1, index + delta))


def _submit_comment(post_id: str) -> None:
    draft_key = f"eda_comment_draft_{post_id}"
    comment = st.session_state.get(draft_key, "").strip()
    if not comment:
        st.session_state[f"eda_comment_notice_{post_id}"] = "댓글 내용을 입력해 주세요."
        return
    comments = list(st.session_state.get(_comments_key(post_id), []))
    comments.append(comment)
    st.session_state[_comments_key(post_id)] = comments
    st.session_state[draft_key] = ""
    st.session_state[f"eda_comment_notice_{post_id}"] = ""


def _render_feed() -> None:
    avatar_column, profile_column = st.columns([1, 3], gap="large")
    with avatar_column:
        st.image(PROFILE_IMAGE, width=132)
    with profile_column:
        st.markdown('<div class="eda-profile-name">2조</div>', unsafe_allow_html=True)
        post_count, follower_count, following_count = st.columns(3)
        for column, count, label in (
            (post_count, len(POSTS), "posts"),
            (follower_count, 19, "followers"),
            (following_count, 0, "following"),
        ):
            with column:
                st.markdown(
                    f'<div class="eda-profile-stat"><strong>{count}</strong>'
                    f"<span>{label}</span></div>",
                    unsafe_allow_html=True,
                )
        st.markdown(
            '<div class="eda-profile-bio">주식 데이터와 SNS 화제성 분석<br>'
            "데이터·모델 분석 결과를 게시물로 소개합니다.</div>",
            unsafe_allow_html=True,
        )
    st.divider()
    category = (
        st.pills(
            "게시물 분류",
            ["전체", "데이터 분석", "모델 분석"],
            default="전체",
            key="analysis_feed_category",
        )
        or "전체"
    )
    visible_posts = [
        post for post in POSTS if category == "전체" or post["category"] == category
    ]
    st.markdown("### 게시물")
    for start in range(0, len(visible_posts), 3):
        cards = st.columns(3, gap="large")
        for column, post in zip(cards, visible_posts[start : start + 3]):
            with column:
                with st.container(
                    height=625,
                    border=True,
                    key=f"eda_feed_post_{post['id']}",
                ):
                    _render_post_header(post, "feed")
                    with st.container(key=f"eda_feed_cover_{post['id']}"):
                        if "cover" in post:
                            render_figure(post["cover"])
                        else:
                            model_analysis.render_post_cover(post["cover_type"])
                    st.markdown(
                        f'<div class="eda-card-title">{post["title"]}</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f'<div class="eda-card-description">{post["description"]}</div>',
                        unsafe_allow_html=True,
                    )
                    _render_hashtags(post)
                    like_count = st.session_state.get(_like_key(post["id"]), 0)
                    comment_count = len(
                        st.session_state.get(_comments_key(post["id"]), [])
                    )
                    st.markdown(
                        f'<div class="eda-post-stats"><span><b class="eda-heart">♥</b> {like_count:,}</span>'
                        f"<span>댓글 {comment_count:,}</span>"
                        f"<span>자료 {len(post['slides'])}장</span></div>",
                        unsafe_allow_html=True,
                    )
                    st.button(
                        "게시물 보기  →",
                        key=f"eda_open_{post['id']}",
                        type="primary",
                        use_container_width=True,
                        on_click=_open_post,
                        args=(post["id"],),
                    )


def _render_slide(slide: dict) -> None:
    st.markdown(f"### {slide['title']}")
    if slide.get("description"):
        st.markdown(
            f'<p class="eda-slide-description">{slide["description"]}</p>',
            unsafe_allow_html=True,
        )
    if slide.get("explanation"):
        st.info(slide["explanation"])

    if slide.get("case_explorer"):
        case_explorer.render()
    elif slide.get("model_section"):
        model_analysis.render_post(slide["model_section"])
        return

    figures = slide.get("figures", [])
    if len(figures) == 2 and slide.get("figure_layout") != "vertical":
        left, right = st.columns(2, gap="medium")
        with left:
            render_figure(figures[0])
        with right:
            render_figure(figures[1])
    else:
        for figure in figures:
            render_figure(figure)

    for label, path, collapsed in slide.get("tables", []):
        if collapsed:
            with st.expander(label):
                render_table(path, preview_rows=1000, **slide.get("table_options", {}))
        else:
            st.markdown(f"#### {label}")
            render_table(path, preview_rows=2000, **slide.get("table_options", {}))

    if slide.get("note"):
        st.info(slide["note"])


def _render_post_detail(post: dict) -> None:
    post_id = post["id"]
    slides = post["slides"]
    slide_index = st.session_state.get("eda_slide_index", 0)
    slide = slides[slide_index]

    st.button("← 피드로 돌아가기", key="eda_back_to_feed", on_click=_close_post)
    with st.container(border=True):
        _render_post_header(post, "detail")
        with st.container(key="analysis_slide_progress"):
            st.progress((slide_index + 1) / len(slides))
        st.caption(f"{slide_index + 1} / {len(slides)}")
        _render_slide(slide)

        previous, center, following = st.columns([1, 2, 1])
        with previous:
            st.button(
                "← 이전",
                key="eda_previous_slide",
                disabled=slide_index == 0,
                on_click=_move_slide,
                args=(-1, len(slides)),
                use_container_width=True,
            )
        with center:
            st.markdown(
                f'<div class="eda-slide-position">{slide_index + 1} / {len(slides)}</div>',
                unsafe_allow_html=True,
            )
        with following:
            st.button(
                "다음 →",
                key="eda_next_slide",
                disabled=slide_index == len(slides) - 1,
                on_click=_move_slide,
                args=(1, len(slides)),
                use_container_width=True,
            )

        st.divider()
        like_count = st.session_state.get(_like_key(post_id), 0)
        st.button(
            f"❤️ 좋아요  {like_count:,}",
            key=f"eda_like_button_{post_id}",
            on_click=_increment_like,
            args=(post_id,),
        )
        st.markdown(f"**{post['title']}**  {post['description']}")
        _render_hashtags(post)

        comments = st.session_state.get(_comments_key(post_id), [])
        st.markdown(f"**댓글 {len(comments):,}개**")
        for comment in comments:
            with st.container():
                st.markdown("**방문자**")
                st.write(comment)

        with st.form(f"eda_comment_form_{post_id}", clear_on_submit=False):
            st.text_input(
                "댓글 남기기",
                placeholder="분석에 대한 의견을 남겨보세요.",
                key=f"eda_comment_draft_{post_id}",
                label_visibility="collapsed",
            )
            st.form_submit_button(
                "댓글 등록",
                on_click=_submit_comment,
                args=(post_id,),
                use_container_width=True,
            )
        notice = st.session_state.get(f"eda_comment_notice_{post_id}")
        if notice:
            st.caption(notice)


def render(model_intro_page=None) -> None:
    st.markdown('<h2 class="asacgram-wordmark">Asacgram</h2>', unsafe_allow_html=True)
    selected_post_id = st.session_state.get("eda_selected_post")
    if selected_post_id:
        post = _post_by_id(selected_post_id)
        if post["category"] == "모델 분석" and model_intro_page is not None:
            st.page_link(
                model_intro_page, label="모델·용어가 궁금하다면 모델 소개", icon="🧠"
            )
        _render_post_detail(post)
    else:
        _render_feed()
