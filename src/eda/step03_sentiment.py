"""STEP03: 부정 감성 변화와 이후 5거래일 수익률을 분석한다.

실행: python3 src/eda/step03_sentiment.py
입력: data/processed/eda/step03/stock_daily.csv.gz
출력: eda-outputs/step03/의 tables/와 figures/.
준비 데이터의 SNS는 관측 거래일 전날까지의 정보이며, 미래 수익률은 별도 표에만 저장한다.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from step03_eda_common import (
    BLUE,
    DEFAULT_INPUT,
    DEFAULT_OUTPUT,
    GRAY,
    RED,
    SEED,
    Report,
    build_metadata,
    configure_plots,
    load_stock_daily,
    log,
    plt,
    rank_correlation,
)
from step03_prepare_data import file_hash, unique_key


def build_daily_outcomes(base):
    # 미래 결과는 base에 넣지 않고 별도 DataFrame에서 계산합니다.
    daily_outcomes = base[
        [
            "ticker",
            "name",
            "keyword",
            "date",
            "close",
            "volume",
            "market_index_close",
            "negative_change_3",
            "negative_change_7",
            "sentiment_min_3",
            "sentiment_min_7",
        ]
    ].copy()
    group = daily_outcomes.groupby("ticker", sort=False)
    daily_outcomes["future_date_5"] = group.date.shift(-5)
    daily_outcomes["return_5_pct"] = (
        group.close.shift(-5) / daily_outcomes.close - 1
    ) * 100
    daily_outcomes["market_return_5_pct"] = (
        group.market_index_close.shift(-5) / daily_outcomes.market_index_close - 1
    ) * 100
    daily_outcomes["excess_return_5_pp"] = (
        daily_outcomes.return_5_pct - daily_outcomes.market_return_5_pct
    )
    daily_outcomes["negative_change_3_pp"] = daily_outcomes.negative_change_3 * 100
    return daily_outcomes


def plot_negative_change(daily_outcomes, report):
    negative_return_sample = daily_outcomes.dropna(
        subset=["negative_change_3_pp", "return_5_pct"]
    ).copy()
    if negative_return_sample.empty:
        raise ValueError("부정 감성 변화와 미래 수익률을 비교할 유효 표본이 없습니다.")
    BIN_LABELS = [
        "5%p 이상 감소",
        "1~5%p 감소",
        "−1~+1%p 변화",
        "1~5%p 증가",
        "5%p 초과 증가",
    ]
    BOUNDARIES = ["x ≤ −5", "−5 < x ≤ −1", "−1 < x ≤ 1", "1 < x ≤ 5", "5 < x"]
    negative_return_sample["change_bin"] = pd.cut(
        negative_return_sample.negative_change_3_pp,
        [-np.inf, -5, -1, 1, 5, np.inf],
        labels=BIN_LABELS,
        right=True,
    )
    box_rows, arrays = [], []
    for name in BIN_LABELS:
        g = negative_return_sample[negative_return_sample.change_bin.eq(name)]
        values = g.return_5_pct
        arrays.append(values.to_numpy())
        box_rows.append(
            {
                "group": name,
                "rows": len(g),
                "stocks": g.ticker.nunique(),
                "theme_dates": len(g[["keyword", "date"]].drop_duplicates()),
                "q25": values.quantile(0.25),
                "median": values.median(),
                "q75": values.quantile(0.75),
                "min": values.min(),
                "max": values.max(),
            }
        )
    box_summary = pd.DataFrame(box_rows)
    fig, ax = plt.subplots(figsize=(13, 5))
    box = ax.boxplot(
        arrays,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#243746", "linewidth": 2},
    )
    for patch, color in zip(
        box["boxes"], ["#A3BCD1", "#C0D0DF", "#D1D7DE", "#DBB7A6", "#CF9981"]
    ):
        patch.set_facecolor(color)
    low, high = ax.get_ylim()
    ax.set_ylim(low, high + (high - low) * 0.15)
    ax.axhline(0, color=GRAY, linestyle="--", linewidth=1)
    ax.set_xticks(
        range(1, 6),
        [
            f"{label}\n{bound}\nn={len(a):,}"
            for label, bound, a in zip(BIN_LABELS, BOUNDARIES, arrays)
        ],
    )
    for i, row in enumerate(box_summary.itertuples(), start=1):
        ax.text(
            i,
            0.98,
            f"중앙값 {row.median:+.2f}%",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=10,
        )
    ax.set(
        xlabel="부정 감성어 비율의 3일 변화 (%p)",
        ylabel="이후 5거래일 종가 수익률 (%)",
        title="부정 비율 변화 구간별 이후 수익률",
    )
    ax.grid(axis="y", alpha=0.15)
    report.show_figure(
        fig,
        "04_negative_change_boxplot",
        f"**읽기:** {len(negative_return_sample):,}개 종목×날짜를 비교했습니다. "
        f"그룹 중앙값은 {box_summary['median'].min():+.2f}%~{box_summary['median'].max():+.2f}%입니다. "
        "**해석:** 상자 겹침만으로 관계가 없다고 단정할 수 없습니다. 같은 테마의 SNS 값이 반복되고 미래 5일 구간도 겹칩니다.",
    )
    report.display(box_summary)
    report.save_table(daily_outcomes, "daily_outcomes.csv.gz")
    report.save_table(box_summary, "12_negative_change_boxplot.csv")
    return box_summary


def plot_stock_correlations(daily_outcomes, report):
    corr_rows = []
    for ticker, g in daily_outcomes.groupby("ticker", sort=True):
        for outcome in ["return_5_pct", "excess_return_5_pp"]:
            n, rho = rank_correlation(g.negative_change_3, g[outcome])
            corr_rows.append(
                {
                    "ticker": ticker,
                    "name": g.name.iloc[0],
                    "keyword": g.keyword.iloc[0],
                    "outcome": outcome,
                    "rows": n,
                    "spearman_rho": rho,
                }
            )
    stock_correlations = pd.DataFrame(corr_rows)
    rho = stock_correlations.loc[
        stock_correlations.outcome.eq("return_5_pct"), "spearman_rho"
    ].dropna()
    rng = np.random.default_rng(SEED)
    y = rng.uniform(-0.8, 0.8, len(rho))
    colors = np.where(rho < 0, BLUE, RED)
    fig, axes = plt.subplots(
        2, 1, figsize=(11, 6), gridspec_kw={"height_ratios": [3, 1]}
    )
    zoom = max(0.30, float(np.ceil(rho.abs().max() * 10) / 10)) if len(rho) else 0.30
    for ax, limit in zip(axes, [zoom, 1]):
        ax.scatter(rho, y, c=colors, s=17, alpha=0.7, edgecolors="none")
        ax.axvline(0, color=GRAY, linestyle="--")
        ax.set_xlim(-limit, limit)
        ax.set_yticks([])
        ax.set_xlabel("스피어만 상관계수 (0에 가까울수록 단조로운 관계가 약함)")
    axes[0].set_title(f"종목별 부정 변화와 이후 수익률의 관계 · {len(rho)}종목")
    axes[1].set_title("상관계수 전체 범위", fontsize=10)
    report.show_figure(
        fig,
        "05_stock_correlations",
        f"**읽기:** 상관계수 중앙값은 {rho.median():+.3f}이며 음수 {rho.lt(0).sum()}종목, 양수 {rho.gt(0).sum()}종목입니다. "
        "**해석:** 방향별 개수는 통계적으로 유의한 종목 수가 아닙니다. 비선형 관계나 일부 시기에만 나타나는 관계까지 부정하지 않습니다.",
    )
    report.save_table(stock_correlations, "13_stock_correlations.csv")
    return stock_correlations


def check_outcomes(outcomes):
    unique_key(outcomes, ["ticker", "date"], "미래 수익률 표본")
    tail = outcomes.groupby("ticker").tail(5)
    if (
        tail[
            [
                "future_date_5",
                "return_5_pct",
                "market_return_5_pct",
                "excess_return_5_pp",
            ]
        ]
        .notna()
        .any()
        .any()
    ):
        raise ValueError("종목별 마지막 5행의 미래 수익률이 결측이 아닙니다.")
    valid = outcomes.future_date_5.notna()
    if not outcomes.loc[valid, "future_date_5"].gt(outcomes.loc[valid, "date"]).all():
        raise ValueError("미래 수익률 날짜가 관측일보다 이후가 아닙니다.")
    if not np.allclose(
        outcomes.excess_return_5_pp,
        outcomes.return_5_pct - outcomes.market_return_5_pct,
        equal_nan=True,
    ):
        raise ValueError("시장 대비 수익률 계산이 일치하지 않습니다.")
    log("종목별 미래 수익률 날짜·말단 결측·시장 대비 수익률 점검 통과")


def main(input_path=DEFAULT_INPUT, output_dir=DEFAULT_OUTPUT):
    started = time.monotonic()
    input_path, output_dir = Path(input_path).resolve(), Path(output_dir).resolve()
    hashes = {"stock_daily": file_hash(input_path)}
    log("부정 감성 변화·이후 수익률 분석: 준비 데이터 읽기")
    base = load_stock_daily(
        input_path,
        [
            "market_index_close",
            "negative_change_3",
            "negative_change_7",
            "sentiment_min_3",
            "sentiment_min_7",
        ],
    )
    log("이후 5거래일 수익률 생성")
    outcomes = build_daily_outcomes(base)
    check_outcomes(outcomes)
    configure_plots()
    report = Report(output_dir)
    log("부정 비율 변화 구간별 수익률 분포")
    summary = plot_negative_change(outcomes, report)
    log("종목별 스피어만 순위 상관")
    correlations = plot_stock_correlations(outcomes, report)
    rho = correlations.loc[
        correlations.outcome.eq("return_5_pct"), "spearman_rho"
    ].dropna()
    metadata = build_metadata(
        "sentiment",
        {"stock_daily": input_path},
        hashes,
        {
            "input_rows": len(base),
            "valid_return_sample_rows": int(summary.rows.sum()),
            "stocks_with_valid_correlation": len(rho),
            "stock_correlation_rows": len(correlations),
            "median_stock_rho": float(rho.median()) if len(rho) else None,
            "figures": report.figures_written,
        },
        {
            "sns_cutoff": "previous_calendar_day",
            "negative_change_calendar_days": 3,
            "future_return_trading_days": 5,
            "min_correlation_pairs": 20,
            "boxplot_outliers_hidden_only": True,
            "correlation_jitter_seed": SEED,
        },
        time.monotonic() - started,
    )
    log(f"완료: {output_dir} ({time.monotonic() - started:.1f}초)")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="STEP03 부정 감성 변화와 이후 수익률 EDA"
    )
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT, help="준비된 stock_daily.csv.gz"
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT, help="STEP03 분석 결과 폴더"
    )
    args = parser.parse_args()
    main(args.input, args.output)
