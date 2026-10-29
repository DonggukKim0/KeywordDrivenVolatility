"""Detailed model performance and robustness results."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from components.csv_loader import load_csv
from components.result_viewer import render_table
from config import MODEL_RESULTS_DIR

import streamlit as st

MODEL_FILES = {
    "LightGBM": "lightgbm",
    "로지스틱 회귀": "logistic",
    "랜덤 포레스트": "rf",
    "LSTM": "lstm",
}
MODEL_LABELS = {
    "lightgbm": "LightGBM",
    "logistic": "로지스틱 회귀",
    "rf": "랜덤 포레스트",
    "lstm": "LSTM",
}
COLORS = {"시장 변수 A": "#168E84", "시장+SNS 변수 B": "#2879D0", "기준선": "#E8A33B"}
TEXT = "#2F455D"
GRID = "#EAF0F6"

MODEL_FEED_POSTS = [
    {
        "id": "model_performance",
        "category": "모델 분석",
        "step": "MODEL 01",
        "title": "SNS 변수를 추가하면 모델 성능이 좋아질까?",
        "description": "시장 변수 A와 시장+SNS 변수 B의 시험 데이터 성능을 비교합니다.",
        "hashtags": ["모델성능", "PRAUC", "SNS변수"],
        "cover_type": "performance",
        "model_section": "performance",
        "download_files": [
            MODEL_RESULTS_DIR / f"metrics_{key}.csv" for key in MODEL_FILES.values()
        ],
        "slides": [
            {
                "title": "시험 데이터 성능 비교",
                "description": "기준선, 변수 구성, 모델별 주요 지표를 확인합니다.",
                "model_section": "performance",
            }
        ],
    },
    {
        "id": "model_ab_difference",
        "category": "모델 분석",
        "step": "MODEL 02",
        "title": "SNS 변수 효과는 모델마다 어떻게 달랐을까?",
        "description": "시장 변수 A와 SNS를 포함한 B의 PR-AUC 차이와 신뢰구간을 살펴봅니다.",
        "hashtags": ["AB비교", "PRAUC차이", "신뢰구간"],
        "cover_type": "differences",
        "model_section": "differences",
        "download_files": [
            MODEL_RESULTS_DIR / f"b_minus_a_{key}.csv" for key in MODEL_LABELS
        ],
        "slides": [
            {
                "title": "A/B 성능 차이",
                "description": "모델별 PR-AUC 차이와 판정 결과입니다.",
                "model_section": "differences",
            }
        ],
    },
    {
        "id": "model_robustness",
        "category": "모델 분석",
        "step": "MODEL 03",
        "title": "분석 기준을 바꿔도 결과가 유지될까?",
        "description": "기간·상승 기준과 변수 구성을 바꿔 결과의 민감도를 확인합니다.",
        "hashtags": ["강건성", "민감도분석", "변수구성"],
        "cover_type": "robustness",
        "model_section": "robustness",
        "download_files": [
            MODEL_RESULTS_DIR / "lgbm_민감도.csv",
            MODEL_RESULTS_DIR / "lgbm_변수구성변경.csv",
        ],
        "slides": [
            {
                "title": "LightGBM 강건성 분석",
                "description": "기준과 변수 구성을 변경한 결과입니다.",
                "model_section": "robustness",
            }
        ],
    },
    {
        "id": "model_features",
        "category": "모델 분석",
        "step": "MODEL 04",
        "title": "모델은 어떤 변수를 중요하게 볼까?",
        "description": "순열 중요도와 회귀 계수로 변수 영향 결과를 살펴봅니다.",
        "hashtags": ["변수중요도", "순열중요도", "회귀계수"],
        "cover_type": "features",
        "model_section": "features",
        "download_files": [
            MODEL_RESULTS_DIR / "lgbm_순열중요도_A.csv",
            MODEL_RESULTS_DIR / "lgbm_순열중요도_B.csv",
            MODEL_RESULTS_DIR / "logistic_B_표준화계수.csv",
            MODEL_RESULTS_DIR / "same_day_OLS계수.csv",
            MODEL_RESULTS_DIR / "same_day_z28구간별.csv",
        ],
        "slides": [
            {
                "title": "변수 영향과 중요도",
                "description": "LightGBM 순열 중요도와 회귀 계수를 확인합니다.",
                "model_section": "features",
            }
        ],
    },
    {
        "id": "model_predictions",
        "category": "모델 분석",
        "step": "MODEL 05",
        "title": "시험 데이터에서 종목별 예측은 어떻게 나왔을까?",
        "description": "모델별 예측 확률과 저장된 시험 데이터 결과를 확인합니다.",
        "hashtags": ["예측확률", "시험데이터", "종목분석"],
        "cover_type": "predictions",
        "model_section": "predictions",
        "download_files": [
            MODEL_RESULTS_DIR / f"predictions_{key}.csv" for key in MODEL_FILES.values()
        ],
        "slides": [
            {
                "title": "시험 데이터 예측 결과",
                "description": "모델을 선택해 예측 결과 일부와 전체 파일을 확인합니다.",
                "model_section": "predictions",
            }
        ],
    },
]


def _metrics_by_model() -> dict[str, pd.DataFrame | None]:
    return {
        label: load_csv(MODEL_RESULTS_DIR / f"metrics_{key}.csv")
        for label, key in MODEL_FILES.items()
    }


def _performance_table(files: dict[str, pd.DataFrame | None]) -> pd.DataFrame:
    frames = []
    for label, frame in files.items():
        if frame is None or not {"모델", "변수", "PR-AUC"}.issubset(frame.columns):
            continue
        part = frame.copy()
        part["모델 표시"] = label
        for column in ("PR-AUC", "ROC-AUC", "상위10%정밀도", "평가행수"):
            if column in part:
                part[column] = pd.to_numeric(part[column], errors="coerce")
        frames.append(part)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _comparison_table(performance: pd.DataFrame) -> pd.DataFrame:
    if performance.empty:
        return pd.DataFrame()
    scored = performance[
        performance["변수"].isin(["A", "B"]) & performance["PR-AUC"].notna()
    ]
    if scored.empty:
        return pd.DataFrame()
    grouped = scored.groupby(["모델 표시", "변수"], as_index=False)["PR-AUC"].mean()
    wide = grouped.pivot(
        index="모델 표시", columns="변수", values="PR-AUC"
    ).reset_index()
    wide = wide.rename(
        columns={"모델 표시": "모델", "A": "시장 변수 A", "B": "시장+SNS 변수 B"}
    )
    for column in ("시장 변수 A", "시장+SNS 변수 B"):
        if column not in wide:
            wide[column] = pd.NA
    wide["B−A"] = wide["시장+SNS 변수 B"] - wide["시장 변수 A"]
    return wide[["모델", "시장 변수 A", "시장+SNS 변수 B", "B−A"]]


def _render_model_baseline(performance: pd.DataFrame) -> None:
    if performance.empty:
        return
    baseline = pd.to_numeric(
        performance.loc[performance["변수"].eq("기준선"), "PR-AUC"], errors="coerce"
    ).dropna()
    st.metric("PR-AUC 기준선", f"{baseline.iloc[0]:.4f}" if not baseline.empty else "—")


def _base_layout(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=18, r=18, t=24, b=18),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="white",
        font=dict(family="Pretendard, Malgun Gothic, sans-serif", color=TEXT, size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return fig


def render_performance_chart(performance: pd.DataFrame | None = None) -> None:
    """Render the same test-performance chart in the feed and overview."""
    if performance is None:
        performance = _performance_table(_metrics_by_model())
    st.markdown("#### 시험 데이터 성능")
    if performance.empty:
        st.info("모델 성능 파일을 찾지 못했습니다.")
        return
    scored = performance[
        performance["변수"].isin(["A", "B"]) & performance["PR-AUC"].notna()
    ]
    if not scored.empty:
        chart = scored.groupby(["모델 표시", "변수"], as_index=False)["PR-AUC"].mean()
        chart["변수 구성"] = chart["변수"].map(
            {"A": "시장 변수 A", "B": "시장+SNS 변수 B"}
        )
        fig = px.bar(
            chart,
            x="모델 표시",
            y="PR-AUC",
            color="변수 구성",
            barmode="group",
            color_discrete_map=COLORS,
        )
        baseline = pd.to_numeric(
            performance.loc[performance["변수"].eq("기준선"), "PR-AUC"], errors="coerce"
        ).dropna()
        if not baseline.empty:
            fig.add_hline(
                y=float(baseline.iloc[0]),
                line_dash="dash",
                line_color=COLORS["기준선"],
                annotation_text=f"기준선 {baseline.iloc[0]:.1%}",
                annotation_position="top left",
            )
        fig.update_yaxes(title="PR-AUC", tickformat=".0%")
        fig.update_xaxes(title=None)
        st.plotly_chart(
            _base_layout(fig), width="stretch", config={"displayModeBar": False}
        )


def _render_performance(performance: pd.DataFrame, comparison: pd.DataFrame) -> None:
    render_performance_chart(performance)
    if performance.empty:
        return
    st.markdown("##### 모델·변수별 전체 지표")
    display_columns = [
        column
        for column in [
            "모델 표시",
            "변수",
            "시드",
            "PR-AUC",
            "ROC-AUC",
            "상위10%정밀도",
            "평가행수",
            "설정",
        ]
        if column in performance
    ]
    st.dataframe(performance[display_columns], width="stretch", hide_index=True)
    if not comparison.empty:
        st.markdown("##### A/B 성능 비교")
        display = comparison.copy()
        for column in ("시장 변수 A", "시장+SNS 변수 B", "B−A"):
            display[column] = pd.to_numeric(display[column], errors="coerce").map(
                lambda value: f"{value:.4f}" if pd.notna(value) else "—"
            )
        st.dataframe(display, width="stretch", hide_index=True)
    st.caption(
        "A는 시장 변수, B는 시장+SNS 변수 구성입니다. LSTM 성능은 시드별 원자료와 평균을 함께 제공합니다."
    )


def _load_differences() -> pd.DataFrame:
    parts = []
    for key, label in MODEL_LABELS.items():
        frame = load_csv(MODEL_RESULTS_DIR / f"b_minus_a_{key}.csv")
        if frame is None:
            continue
        frame = frame.copy()
        frame["모델 표시"] = label
        for column in ("B−A", "CI하한", "CI상한", "시드"):
            if column in frame:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        parts.append(frame)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _render_differences(frame: pd.DataFrame) -> None:
    st.markdown("#### 시장 변수 A 대비 SNS 포함 B의 PR-AUC 차이")
    if frame.empty or not {"모델 표시", "B−A"}.issubset(frame.columns):
        st.info("A/B 차이 결과 파일을 찾지 못했습니다.")
        return
    chart = frame.dropna(subset=["B−A"]).copy()
    fig = go.Figure()
    if {"CI하한", "CI상한"}.issubset(chart.columns):
        chart["오차 하한"] = (chart["B−A"] - chart["CI하한"]).clip(lower=0)
        chart["오차 상한"] = (chart["CI상한"] - chart["B−A"]).clip(lower=0)
        if "시드" in chart:
            seed_suffix = chart["시드"].map(
                lambda value: f" · 시드 {int(value)}" if pd.notna(value) else ""
            )
        else:
            seed_suffix = ""
        fig.add_trace(
            go.Scatter(
                x=chart["B−A"],
                y=chart["모델 표시"] + seed_suffix,
                mode="markers",
                marker=dict(size=10, color="#168E84"),
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=chart["오차 상한"],
                    arrayminus=chart["오차 하한"],
                ),
                hovertemplate="%{y}<br>B−A: %{x:.4f}<extra></extra>",
            )
        )
    else:
        fig.add_trace(
            go.Bar(
                x=chart["B−A"],
                y=chart["모델 표시"],
                orientation="h",
                marker_color="#168E84",
            )
        )
    fig.add_vline(x=0, line_dash="dash", line_color="#788A9C")
    fig.update_layout(showlegend=False)
    fig.update_xaxes(title="PR-AUC 차이 (B−A)")
    fig.update_yaxes(title=None)
    st.plotly_chart(
        _base_layout(fig, height=max(300, 90 + 42 * len(chart))),
        width="stretch",
        config={"displayModeBar": False},
    )
    st.dataframe(frame, width="stretch", hide_index=True)
    st.caption(
        "신뢰구간이 0을 포함하는지와 판정 열을 함께 확인하세요. 모델 간 결과는 시드 구성에 따라 달라질 수 있습니다."
    )


def _render_robustness() -> None:
    st.markdown("#### LightGBM 민감도 분석")
    sensitivity = load_csv(MODEL_RESULTS_DIR / "lgbm_민감도.csv")
    if sensitivity is not None and {"기준", "기준선", "A", "B"}.issubset(
        sensitivity.columns
    ):
        chart = sensitivity.melt(
            id_vars=[column for column in ["기간", "기준"] if column in sensitivity],
            value_vars=["기준선", "A", "B"],
            var_name="변수 구성",
            value_name="PR-AUC",
        )
        chart["PR-AUC"] = pd.to_numeric(chart["PR-AUC"], errors="coerce")
        chart = chart.dropna(subset=["PR-AUC"])
        if not chart.empty:
            chart["조건"] = (
                chart["기간"].astype(str) + " · " + chart["기준"].astype(str)
            )
            fig = px.line(
                chart,
                x="조건",
                y="PR-AUC",
                color="변수 구성",
                markers=True,
                color_discrete_map={
                    "기준선": COLORS["기준선"],
                    "A": COLORS["시장 변수 A"],
                    "B": COLORS["시장+SNS 변수 B"],
                },
            )
            fig.update_yaxes(title="PR-AUC", tickformat=".0%")
            fig.update_xaxes(title=None, tickangle=35)
            st.plotly_chart(
                _base_layout(fig, 420),
                width="stretch",
                config={"displayModeBar": False},
            )
    render_table(MODEL_RESULTS_DIR / "lgbm_민감도.csv")
    st.markdown("#### 변수 구성 변경")
    render_table(MODEL_RESULTS_DIR / "lgbm_변수구성변경.csv")
    st.caption(
        "기준·변수 구성을 바꾸어도 성능 차이가 재현되는지 확인하는 강건성 결과입니다."
    )


def _render_feature_effects() -> None:
    st.markdown("#### LightGBM 순열 중요도")
    importance_frames = []
    for variant in ("A", "B"):
        frame = load_csv(MODEL_RESULTS_DIR / f"lgbm_순열중요도_{variant}.csv")
        if frame is None or not {"Unnamed: 0", "떨어진정도"}.issubset(frame.columns):
            continue
        part = frame.rename(columns={"Unnamed: 0": "변수"}).copy()
        part["변수 구성"] = variant
        part["떨어진정도"] = pd.to_numeric(part["떨어진정도"], errors="coerce")
        importance_frames.append(part)
    if importance_frames:
        importance = pd.concat(importance_frames, ignore_index=True).dropna(
            subset=["떨어진정도"]
        )
        fig = px.bar(
            importance,
            x="떨어진정도",
            y="변수",
            color="변수 구성",
            orientation="h",
            barmode="group",
            color_discrete_map={
                "A": COLORS["시장 변수 A"],
                "B": COLORS["시장+SNS 변수 B"],
            },
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        fig.update_xaxes(title="변수 섞기 후 PR-AUC 감소량")
        fig.update_yaxes(title=None)
        st.plotly_chart(
            _base_layout(fig, max(360, 24 * importance["변수"].nunique())),
            width="stretch",
            config={"displayModeBar": False},
        )
        st.dataframe(importance, width="stretch", hide_index=True)
    else:
        st.info("순열 중요도 결과를 찾지 못했습니다.")

    with st.expander("로지스틱 회귀 표준화 계수"):
        render_table(MODEL_RESULTS_DIR / "logistic_B_표준화계수.csv")
    with st.expander("같은 날 회귀 계수"):
        render_table(MODEL_RESULTS_DIR / "same_day_OLS계수.csv")
    with st.expander("언급량 급증 구간별 같은 날 수익률"):
        render_table(MODEL_RESULTS_DIR / "same_day_z28구간별.csv")


def _render_predictions() -> None:
    st.markdown("#### 시험 데이터 예측 결과")
    st.markdown(
        "모델을 선택해 예측 확률 일부를 확인하고 전체 결과 파일을 내려받을 수 있습니다."
    )
    label = st.selectbox("모델", list(MODEL_FILES), key="model_prediction_choice")
    key = MODEL_FILES[label]
    render_table(
        MODEL_RESULTS_DIR / f"predictions_{key}.csv",
        key=f"download_predictions_{key}",
        preview_rows=1000,
        height=520,
    )
    st.caption(
        "예측 확률은 저장된 시험 데이터 결과입니다. 행 순서는 종목·날짜 기준 원본과 동일합니다."
    )


def render_post(section: str) -> None:
    """Render one model-analysis topic for its unified feed post."""
    metric_files = _metrics_by_model()
    performance = _performance_table(metric_files)
    comparison = _comparison_table(performance)

    if section == "performance":
        _render_model_baseline(performance)
        _render_performance(performance, comparison)
    elif section == "differences":
        _render_differences(_load_differences())
    elif section == "robustness":
        _render_robustness()
    elif section == "features":
        _render_feature_effects()
    elif section == "predictions":
        _render_predictions()
    else:
        st.info("지원하지 않는 모델 분석 게시물입니다.")


def _render_preview(fig: go.Figure, post_id: str) -> None:
    _base_layout(fig, height=260)
    fig.update_layout(margin=dict(l=8, r=8, t=12, b=10), font=dict(size=9))
    fig.update_xaxes(title=None, tickfont=dict(size=8))
    fig.update_yaxes(title=None, tickfont=dict(size=8))
    st.plotly_chart(
        fig,
        width="stretch",
        key=f"model_cover_{post_id}",
        config={"displayModeBar": False},
    )


def render_post_cover(cover_type: str) -> None:
    """Render a compact data-backed chart preview for a model feed post."""
    if cover_type == "performance":
        performance = _performance_table(_metrics_by_model())
        chart = (
            performance[
                performance["변수"].isin(["A", "B"]) & performance["PR-AUC"].notna()
            ]
            if not performance.empty
            else pd.DataFrame()
        )
        if chart.empty:
            st.info("모델 성능 미리보기가 없습니다.")
            return
        chart = chart.groupby(["모델 표시", "변수"], as_index=False)["PR-AUC"].mean()
        chart["변수 구성"] = chart["변수"].map(
            {"A": "시장 변수 A", "B": "시장+SNS 변수 B"}
        )
        fig = px.bar(
            chart,
            x="모델 표시",
            y="PR-AUC",
            color="변수 구성",
            barmode="group",
            color_discrete_map=COLORS,
        )
        fig.update_yaxes(tickformat=".0%")
        fig.update_layout(showlegend=False)
        _render_preview(fig, "performance")
        return

    if cover_type == "differences":
        chart = _load_differences()
        if chart.empty or "B−A" not in chart:
            st.info("A/B 비교 미리보기가 없습니다.")
            return
        chart = chart.dropna(subset=["B−A"])
        fig = px.scatter(
            chart,
            x="B−A",
            y="모델 표시",
            color_discrete_sequence=[COLORS["시장+SNS 변수 B"]],
        )
        fig.add_vline(x=0, line_dash="dash", line_color="#788A9C")
        fig.update_layout(showlegend=False)
        _render_preview(fig, "differences")
        return

    if cover_type == "robustness":
        sensitivity = load_csv(MODEL_RESULTS_DIR / "lgbm_민감도.csv")
        if sensitivity is None or not {"기준", "기준선", "A", "B"}.issubset(
            sensitivity.columns
        ):
            st.info("강건성 미리보기가 없습니다.")
            return
        chart = sensitivity.melt(
            id_vars=[column for column in ["기간", "기준"] if column in sensitivity],
            value_vars=["기준선", "A", "B"],
            var_name="변수 구성",
            value_name="PR-AUC",
        )
        chart["PR-AUC"] = pd.to_numeric(chart["PR-AUC"], errors="coerce")
        chart = chart.dropna(subset=["PR-AUC"])
        chart["조건"] = chart["기간"].astype(str) + " · " + chart["기준"].astype(str)
        fig = px.line(
            chart,
            x="조건",
            y="PR-AUC",
            color="변수 구성",
            markers=True,
            color_discrete_map={
                "기준선": COLORS["기준선"],
                "A": COLORS["시장 변수 A"],
                "B": COLORS["시장+SNS 변수 B"],
            },
        )
        fig.update_layout(showlegend=False)
        _render_preview(fig, "robustness")
        return

    if cover_type == "features":
        frames = []
        for variant in ("A", "B"):
            frame = load_csv(MODEL_RESULTS_DIR / f"lgbm_순열중요도_{variant}.csv")
            if frame is not None and {"Unnamed: 0", "떨어진정도"}.issubset(
                frame.columns
            ):
                part = frame.rename(columns={"Unnamed: 0": "변수"}).copy()
                part["변수 구성"] = variant
                part["떨어진정도"] = pd.to_numeric(part["떨어진정도"], errors="coerce")
                frames.append(part)
        if not frames:
            st.info("변수 중요도 미리보기가 없습니다.")
            return
        chart = pd.concat(frames, ignore_index=True).dropna(subset=["떨어진정도"])
        chart = (
            chart.groupby("변수", as_index=False)["떨어진정도"]
            .mean()
            .nlargest(5, "떨어진정도")
        )
        fig = px.bar(
            chart,
            x="떨어진정도",
            y="변수",
            orientation="h",
            color_discrete_sequence=[COLORS["시장+SNS 변수 B"]],
        )
        fig.update_layout(showlegend=False, yaxis={"categoryorder": "total ascending"})
        _render_preview(fig, "features")
        return

    if cover_type == "predictions":
        predictions = load_csv(MODEL_RESULTS_DIR / "predictions_lightgbm.csv")
        if predictions is None:
            st.info("예측 결과 미리보기가 없습니다.")
            return
        probability = next(
            (
                column
                for column in ("prob_B", "prob_A", "prob_C")
                if column in predictions
            ),
            None,
        )
        if probability is None:
            st.info("예측 확률 열을 찾지 못했습니다.")
            return
        values = pd.to_numeric(predictions[probability], errors="coerce").dropna()
        if len(values) > 1200:
            values = values.sample(1200, random_state=42)
        fig = px.histogram(
            values,
            x=probability,
            nbins=20,
            color_discrete_sequence=[COLORS["시장+SNS 변수 B"]],
        )
        fig.update_layout(showlegend=False)
        _render_preview(fig, "predictions")
        return

    st.info("모델 분석 미리보기가 없습니다.")
