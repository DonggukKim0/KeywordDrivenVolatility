"""Beginner guide to the prediction task, models, settings and evaluation."""

from ast import literal_eval

import pandas as pd
from components.csv_loader import load_csv
from config import MODEL_OUTPUTS_DIR

import streamlit as st

# Descriptive settings from src/model/models.py; no model libraries are imported.
MODELS = {
    "logistic": (
        "로지스틱 회귀",
        "여러 정보에 가중치를 주어 조건에 해당할 가능성을 계산하는 모델",
    ),
    "rf": ("랜덤 포레스트", "여러 판단 나무의 의견을 모아 예측하는 모델"),
    "lightgbm": ("LightGBM", "앞선 판단의 실수를 보완하는 나무를 차례로 학습하는 모델"),
    "lstm": ("LSTM", "여러 날의 정보가 이어지는 흐름을 학습하는 모델"),
}
SETTINGS = {
    "logistic": [
        ("규제 조절값", "C", "작을수록 복잡한 판단을 더 제한", "0.001, 0.01, 0.1, 1.0"),
    ],
    "rf": [
        (
            "잎의 최소 표본 수",
            "min_samples_leaf",
            "하나의 판단 그룹에 필요한 최소 사례 수",
            "50, 200",
        ),
        (
            "분기에 사용할 변수 수",
            "max_features",
            "판단을 나눌 때 검토할 변수의 수 또는 비율",
            "전체 변수 수의 제곱근(sqrt), 50%",
        ),
    ],
    "lightgbm": [
        ("잎 수", "num_leaves", "나무 하나가 나눌 수 있는 판단 그룹 수", "15, 31"),
        ("나무 수", "n_estimators", "보완 학습에 사용할 나무의 개수", "200, 500"),
        (
            "잎의 최소 표본 수",
            "min_child_samples",
            "하나의 판단 그룹에 필요한 최소 사례 수",
            "100, 400",
        ),
    ],
    "lstm": [
        ("관찰 기간", "window", "한 번의 예측에 살펴보는 연속 거래일", "10거래일"),
        ("은닉 크기", "hidden", "흐름을 기억하는 내부 표현의 크기", "32"),
        (
            "드롭아웃",
            "dropout",
            "학습 중 일부 연결을 쉬게 해 과도한 암기를 줄이는 비율",
            "0.2",
        ),
        ("학습률", "learning_rate", "한 번에 판단을 수정하는 보폭", "0.001"),
        ("배치 크기", "batch_size", "한 번에 학습하는 사례 수", "1,024"),
        (
            "최대 학습 횟수",
            "max_epochs",
            "학습 자료를 반복해서 살펴보는 횟수의 상한",
            "20",
        ),
        (
            "조기 종료",
            "patience_limit",
            "검증 점수가 연속으로 개선되지 않으면 학습 종료",
            "3회",
        ),
    ],
}
FIXED = {
    "logistic": "최대 반복 2,000회 · 클래스 가중치 balanced · 입력 변수 표준화",
    "rf": "나무 200개 · 클래스 가중치 balanced_subsample · 난수 시드 0",
    "lightgbm": "학습률 0.03 · 행 표본 비율 0.8(매 반복 적용) · 변수 표본 비율 0.8 · 난수 시드 0 · 양성 가중치는 학습 자료의 비율로 계산",
    "lstm": "Adam 최적화 · 클래스 비율을 반영한 손실함수 · 시드 0·1·2로 반복 · 위 설정값은 고정, 검증 점수로 최적 가중치 선택",
}
LABELS = {
    "C": "규제 조절값",
    "num_leaves": "잎 수",
    "n_estimators": "나무 수",
    "min_child_samples": "잎의 최소 표본 수",
    "min_samples_leaf": "잎의 최소 표본 수",
    "max_features": "분기에 사용할 변수 수",
    "epochs": "실행한 학습 횟수",
    "valid_pr_auc": "최고 검증 PR-AUC",
}


def _training_records(frame: pd.DataFrame | None, model: str) -> pd.DataFrame:
    """Decode saved settings safely; malformed records remain visible as notices."""
    rows = []
    if frame is None or not {"모델", "변수", "설정"}.issubset(frame.columns):
        return pd.DataFrame()
    for _, row in frame.loc[frame["모델"].eq(model)].iterrows():
        record = {"변수 구성": row["변수"]}
        if pd.notna(row.get("시드")):
            record["시드"] = row["시드"]
        try:
            settings = literal_eval(str(row["설정"]))
            if (
                model == "lightgbm"
                and isinstance(settings, tuple)
                and len(settings) == 3
            ):
                settings = dict(
                    zip(("num_leaves", "n_estimators", "min_child_samples"), settings)
                )
            if not isinstance(settings, dict):
                raise ValueError("Unsupported settings")
            record.update(
                {LABELS.get(key, key): value for key, value in settings.items()}
            )
        except (ValueError, SyntaxError, TypeError, RecursionError):
            record["안내"] = "이 기록의 설정을 해석할 수 없습니다."
        rows.append(record)
    return pd.DataFrame(rows)


def _prepare_model_feed() -> None:
    st.session_state["eda_selected_post"] = None
    st.session_state["eda_slide_index"] = 0
    st.session_state["analysis_feed_category"] = "모델 분석"


def render(analysis_page=None) -> None:
    st.subheader("모델 소개")
    st.caption("예측 목표부터 학습 설정과 결과 읽는 법까지")
    st.markdown("### 무엇을 예측할까?")
    st.markdown(
        "**SNS 키워드의 언급 급증은 향후 2거래일의 종목 급등을 예측하는 데 도움이 될까?**"
    )
    st.write(
        "키워드가 평소보다 많이 언급되는 현상을 급등의 예측 신호로 활용할 수 있는지 살펴봅니다. 현재 모델은 언급 급증일만 따로 고르지 않고, 전체 종목·날짜에서 언급 급증도 등 SNS 정보를 이용해 예측합니다."
    )
    st.write(
        "이 연구에서 모델이 예측하는 급등은 다음 거래일 시가부터 두 번째 거래일 종가까지의 종목 수익률이 시장지수 수익률보다 5%포인트 이상 높은 경우입니다. ‘2거래일 뒤 하루 동안 오른 정도’를 뜻하는 것은 아닙니다."
    )
    st.write(
        "여기서 시장은 해당 종목이 속한 KOSPI 또는 KOSDAQ 지수를 뜻합니다. 시장과 비교한 성과이므로, 조건을 충족했다고 반드시 주가가 상승한 것은 아닙니다."
    )
    with st.expander("정답 계산의 정확한 시점"):
        st.write(
            "예측 기준일을 t라고 할 때, 종목은 t+1 시가부터 t+2 종가까지의 수익률을 사용합니다. 시장지수는 시가 대신 종가 자료를 사용하므로 t 종가부터 t+2 종가까지의 수익률을 비교 기준으로 사용합니다. t+1·t+2는 거래일 기준입니다."
        )
        st.write(
            "실제 관측 결과가 이 조건을 충족하면 학습 자료에 1, 충족하지 않으면 0으로 기록합니다. 이는 모델의 예측값이 아니라 나중에 확인한 결과입니다."
        )
    st.markdown("**예를 들어 시장지수가 2% 올랐다면?**")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "가상 사례": "종목 A",
                    "종목 수익률": "+8%",
                    "시장 수익률": "+2%",
                    "시장보다 높은 정도": "6%포인트",
                    "찾으려는 조건 충족": "예 · 차이가 5%포인트 이상",
                },
                {
                    "가상 사례": "종목 B",
                    "종목 수익률": "+4%",
                    "시장 수익률": "+2%",
                    "시장보다 높은 정도": "2%포인트",
                    "찾으려는 조건 충족": "아니요 · 차이가 5%포인트 미만",
                },
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "8%에서 2%를 뺀 차이를 ‘6%포인트(6%p)’라고 부릅니다. 종목 A처럼 시장보다 충분히 좋은 성과를 낸 사례를 미리 구분할 수 있는지 평가합니다."
    )
    st.caption(
        "모델의 출력은 이 조건에 해당할 예측 확률입니다. 확률 0.7이 수익률 70%를 뜻하지는 않습니다."
    )

    st.markdown("### 어떤 정보를 비교할까?")
    left, right = st.columns(2)
    with left.container(border=True):
        st.markdown("**A · 시장 정보**")
        st.write("최근 수익률, 거래량, 변동성, 공시")
    with right.container(border=True):
        st.markdown("**B · 시장 정보 + SNS**")
        st.write("A에 언급 급증, 관심 변화, 감성, 채널 확산 정보를 추가")
    st.markdown("**핵심 질문: SNS 정보를 더하면 예측에 도움이 될까?**")
    with st.expander("세부 입력 변수와 C 구성"):
        st.write("C는 SNS 정보만 사용하는 추가 비교 구성입니다.")
        st.markdown(
            "**시장 정보**: 시장 대비 1·3·20거래일 수익률, 직전 5거래일 대비 거래량, 20거래일 변동성, 최근 3거래일 공시 건수·공급계약 공시 여부"
        )
        st.markdown(
            "**SNS 정보**: 언급 급증도(직전 28일 대비), 언급량 변화(최근 3일/직전 14일), 로그 언급량, 화제 시작 후 일수, 긍정 감성 비율·직전 7일 대비 변화, 채널 확산 다양성 변화, 화제 기간 여부, 감성어 10건 미만 여부"
        )

    st.markdown("### 각각의 모델은 어떻게 다를까?")
    for column, (name, description) in zip(st.columns(4), MODELS.values()):
        with column.container(border=True, height=150):
            st.markdown(f"**{name}**")
            st.write(description)
    st.caption(
        "모델의 우열은 이름이나 복잡함이 아니라, 같은 시험 자료의 성능으로 비교합니다."
    )

    st.markdown("### 모델의 학습 설정")
    st.write(
        "하이퍼파라미터는 학습 전에 정하는 모델의 설정값입니다. 예를 들어 나무의 수나 학습 속도를 정합니다. 데이터에서 학습되는 가중치와는 구분됩니다."
    )
    summary = load_csv(MODEL_OUTPUTS_DIR / "models" / "train_summary.csv")
    for tab, (key, (name, _)) in zip(
        st.tabs([item[0] for item in MODELS.values()]), MODELS.items()
    ):
        with tab:
            if key == "lstm":
                st.write(
                    "10거래일 흐름을 보며 학습하고, 검증 점수가 개선되지 않으면 일찍 멈춥니다."
                )
            else:
                st.write("학습 기간 내부의 검증 PR-AUC가 가장 좋은 후보를 선택합니다.")
            st.markdown(
                "**저장된 학습 기록**" if key == "lstm" else "**저장된 최종 선택값**"
            )
            records = _training_records(summary, key)
            if records.empty:
                st.info("이 모델의 학습 기록을 찾을 수 없습니다.")
            else:
                st.dataframe(records, hide_index=True, width="stretch")
            if key == "lstm":
                st.caption(
                    "실행한 학습 횟수는 조기 종료까지 수행한 횟수이며, 최고 검증 점수의 가중치를 보관합니다. 검증 점수와 실행 횟수는 학습 결과 기록입니다."
                )
            with st.expander("설정의 의미와 코드 기준 후보·고정값"):
                st.dataframe(
                    pd.DataFrame(
                        SETTINGS[key],
                        columns=[
                            "쉬운 이름",
                            "코드 이름",
                            "의미",
                            "고정값" if key == "lstm" else "검토한 후보",
                        ],
                    ),
                    hide_index=True,
                    width="stretch",
                )
                st.caption(FIXED[key])
    st.caption(
        "후보·고정값은 현재 학습 코드 기준, 최종 선택값은 저장된 학습 기록 기준입니다. 시험 데이터는 설정 선택에 사용하지 않습니다."
    )

    st.markdown("### 결과의 숫자는 어떻게 읽을까?")
    metrics = [
        (
            "PR-AUC",
            "대상을 잘 찾아내면서 잘못 지목하는 사례를 줄이는 성능. 클수록 좋으며 정확도 백분율이 아닌 기준선과 비교할 점수",
        ),
        ("기준선", "시험 자료에서 정답 1이 차지하는 비율. PR-AUC 해석의 비교 기준"),
        (
            "상위 10% 정밀도",
            "예측 확률이 높은 상위 10% 중 실제 정답 1의 비율. 가상 예시: 100건 중 10건을 골라 3건이 맞으면 30%",
        ),
        (
            "ROC-AUC",
            "정답 1을 정답 0보다 높은 순위에 놓는 구분 성능. 0.5는 무작위 수준",
        ),
        ("B−A", "SNS를 추가한 B의 PR-AUC에서 A의 PR-AUC를 뺀 값. 양수면 B가 높음"),
        (
            "95% 신뢰구간",
            "추정 차이의 불확실성을 나타내는 범위. 0이 포함되면 차이가 뚜렷하지 않으며, 효과가 없다고 확정하는 의미는 아님",
        ),
    ]
    for title, explanation in metrics:
        st.markdown(f"**{title}** · {explanation}")
    with st.expander("추가 용어 사전"):
        for title, explanation in [
            (
                "시드",
                "무작위 학습의 출발값. LSTM은 0·1·2로 반복해 결과의 흔들림을 살펴봄",
            ),
            (
                "강건성·민감도",
                "기간·정답 기준·변수 구성 등을 바꿔도 결론이 유지되는지 확인",
            ),
            (
                "순열 중요도",
                "변수 값을 섞었을 때 성능이 얼마나 떨어지는지 측정. 인과관계의 증명은 아님",
            ),
            (
                "표준화 계수",
                "단위를 맞춘 입력 변수에 부여한 가중치. 크기와 부호는 다른 변수와 함께 해석",
            ),
            (
                "예측 확률",
                "모델이 추정한 조건 충족 가능성. 관측 빈도와 일치하도록 보정된 확률인지도 구분 필요",
            ),
        ]:
            st.markdown(f"**{title}** · {explanation}")

    st.markdown("### 어떻게 평가했을까?")
    st.markdown(
        '<div class="model-evaluation-flow"><span>과거 자료로 학습</span><b>→</b><span>학습 기간 내부에서 설정 선택</span><b>→</b><span>이후 자료로 시험</span></div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "학습: 2024-11-01 – 2025-12-26 · 시험: 2026-01-02 – 2026-10-23 · 경계 2025-12-29·30은 정답 기간이 시험 구간에 걸쳐 제외"
    )
    with st.expander("설정 선택에 사용한 검증 방식"):
        st.write(
            "LightGBM은 학습 기간 안에서 시간순 3겹 검증을 사용하며 경계 2거래일을 제외합니다. 나머지 모델은 학습 구간(2025-10-23까지)과 검증 구간(2025-10-28 – 2025-12-26)을 나누어 설정 또는 가중치를 선택합니다."
        )
    st.caption(
        "분석 피드에는 저장된 시험 결과를 표시합니다. 실시간 예측 서비스는 아닙니다."
    )
    st.markdown(
        "**추천 읽기 순서**: 시험 성능 → A/B 차이 → 강건성 → 변수 영향 → 종목별 예측"
    )
    if analysis_page is not None:
        if st.button(
            "모델 결과 보러 가기 →",
            on_click=_prepare_model_feed,
            key="intro_open_model_feed",
        ):
            st.switch_page(analysis_page)
