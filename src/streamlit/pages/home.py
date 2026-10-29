"""Home page: project overview, analysis roadmap, team and usage guide."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

ASSET_DIR = Path(__file__).resolve().parents[1] / "assets"

TEAM_MEMBERS = [
    {
        "name": "김동국",
        "character": "⚽",
        "role": "조장",
        "intro": "포지션은 달라도 골대는 하나. 2조가 같은 방향으로 끝까지 뛸 수 있도록 패스를 연결합니다.",
    },
    {
        "name": "이해인",
        "character": "⚾",
        "role": "모델링, 발표",
        "intro": "모델은 숫자로, 발표는 말로 설득합니다. 과적합은 줄이고 전달력은 높이는 중입니다.",
    },
    {
        "name": "이가은",
        "character": "🦈",
        "role": "데이터 수집, 발표 자료 정리",
        "intro": "데이터 바다에서 쓸 만한 근거를 발견하면 놓치지 않습니다. 수집한 재료는 발표 자료로 야무지게 담아냅니다.",
    },
    {
        "name": "장기열",
        "character": "🚗",
        "role": "EDA, 대시보드 구현",
        "intro": "데이터 속 패턴을 찾아 대시보드 화면까지 데려옵니다. 분석부터 구현까지, 제가 운전합니다.",
    },
    {
        "name": "윤지석",
        "character": "😜",
        "role": "데이터 정제, EDA",
        "intro": "장난기는 있어도 결측치는 그냥 못 넘깁니다. 데이터를 다듬고, 그럴듯한 그래프에도 한 번 더 ‘왜?’를 묻습니다.",
    },
]


@st.dialog("팀원 소개")
def show_member_profile(member: dict[str, str]) -> None:
    st.markdown(
        f'<div class="profile-avatar">{member["character"]}</div>',
        unsafe_allow_html=True,
    )
    st.subheader(member["name"])
    st.markdown(f"**{member['role']}**")
    st.write(member["intro"])


def render() -> None:
    """Render the app landing page with project context and analysis roadmap."""
    st.markdown("### SNS 화제성과 주가 반응")
    sns_col, arrow_col, stock_col = st.columns([1, 0.24, 1])

    def topic_card(
        column, image_name: str, title: str, accent: str
    ) -> None:
        with column.container(border=True):
            image_col, text_col = st.columns([1.1, 1.4], vertical_alignment="center")
            with image_col:
                svg = (ASSET_DIR / image_name).read_text(encoding="utf-8")
                st.image(svg, width=155)
            with text_col:
                st.markdown(
                    f'<div class="topic-card-heading {accent}"><span>{title}</span></div>',
                    unsafe_allow_html=True,
                )

    topic_card(
        sns_col,
        "home_sns_mentions.svg",
        "SNS 언급량 증가",
        "topic-card-heading-sns",
    )
    with arrow_col:
        st.markdown('<div class="topic-arrow">→</div>', unsafe_allow_html=True)
    topic_card(
        stock_col,
        "home_stock_surge.svg",
        "주가 급등",
        "topic-card-heading-stock",
    )

    st.markdown("### 분석 구성")
    analysis_steps = [
        ("언급량과 수익률", "키워드·종목별 SNS 언급량과 초과수익률의 관계"),
        ("화제 전후 반응", "실제·가짜 화제 시점의 거래량과 변동폭 비교"),
        ("급등 이후 신호", "급등 이후 붕괴와 SNS 감성·뉴스 동행 분석"),
        (
            "모델링 · 예측 성능 검증",
            "시장 변수에 SNS를 더한 예측 성능 비교",
        ),
    ]
    for col, (title, description) in zip(st.columns(4), analysis_steps):
        with col.container(border=True, height=110):
            st.markdown(f"**{title}**")
            st.caption(description)

    st.markdown("### 연구팀 소개")
    st.markdown(
        '<p class="section-note">캐릭터를 눌러 팀원 소개를 확인하세요.</p>',
        unsafe_allow_html=True,
    )
    member_cols = st.columns(5)
    for index, (col, member) in enumerate(zip(member_cols, TEAM_MEMBERS)):
        with col:
            if st.button(
                member["character"],
                key=f"member_avatar_{index}",
                help=f"{member['name']} 자기소개 보기",
                use_container_width=True,
            ):
                show_member_profile(member)
            st.markdown(
                f'<div class="team-card"><div class="team-name">{member["name"]}</div>'
                f'<div class="team-role">{member["role"]}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("### 이용 안내")
    st.markdown(
        "왼쪽 메뉴에서 원하는 페이지 선택\n\n"
        "- **종합 대시보드** · 데이터 규모와 주요 분석 결과를 한눈에 확인\n"
        "- **주식 차트** · 종목별 가격과 거래량 흐름 탐색\n"
        "- **데이터 소개** · 원본 데이터의 수집 기간, 컬럼 설명과 일부 행 확인\n"
        "- **모델 소개** · 예측 목표, 모델별 학습 설정과 평가 용어 이해\n"
        "- **분석 피드** · EDA·모델 게시물을 열어 차트와 상세 결과 확인"
    )
    st.caption(
        "분석 피드의 게시물 보기로 상세 분석 탐색"
    )
    st.info(
        "분석 결과는 연구 데이터에 기반한 참고 정보이며, 상관관계가 인과관계나 미래 수익을 보장하지 않습니다. 🚨⚠️"
    )
