"""STEP03: 뉴스·SNS 언급량 변화와 당일 테마 수익률의 동행을 분석한다.

실행: python3 src/eda/step03_comovement.py
입력: 준비된 stock_daily.csv.gz와 sns_daily.csv.gz
출력: eda-outputs/step03/의 tables/와 figures/.
뉴스·SNS는 당일 값을 사용하며, 선행 관계나 예측력 검증으로 해석하지 않는다.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from step03_eda_common import (
    DEFAULT_INPUT,
    DEFAULT_OUTPUT,
    DEFAULT_SNS_INPUT,
    GOLD,
    GRAY,
    GREEN,
    Report,
    build_metadata,
    configure_plots,
    load_sns_daily,
    load_stock_daily,
    log,
    plt,
    rank_correlation,
)
from step03_prepare_data import file_hash, unique_key


def build_theme_data(prices, sns_daily, report):
    calendar = pd.DatetimeIndex(prices.date.unique()).sort_values()
    previous_volume = prices.groupby("ticker", sort=False).volume.shift(1)
    theme_eligible = (
        prices.volume.gt(0)
        & previous_volume.gt(0)
        & prices.return_1.abs().le(0.31)
        & prices.return_1.notna()
        & prices.market_return_1.notna()
    )
    report.save_table(
        prices.loc[
            ~theme_eligible, ["ticker", "keyword", "date", "volume", "return_1"]
        ],
        "14_theme_excluded_stock_days.csv",
    )
    theme_prices = prices[theme_eligible].copy()
    theme_prices["excess_return_1"] = (
        theme_prices.return_1 - theme_prices.market_return_1
    )
    theme_aggregated = (
        theme_prices.groupby(["keyword", "date"])
        .agg(
            theme_excess_return=("excess_return_1", "mean"),
            valid_stocks=("ticker", "nunique"),
        )
        .reset_index()
    )
    theme_aggregated.loc[theme_aggregated.valid_stocks.lt(5), "theme_excess_return"] = (
        np.nan
    )
    common_dates = calendar[
        (calendar >= sns_daily.date.min()) & (calendar <= sns_daily.date.max())
    ]
    if len(common_dates) == 0:
        raise ValueError("주가와 SNS의 공통 관측 기간이 없습니다.")
    theme_daily = pd.MultiIndex.from_product(
        [sorted(prices.keyword.unique()), common_dates], names=["keyword", "date"]
    ).to_frame(index=False)
    theme_daily = theme_daily.merge(
        theme_aggregated, on=["keyword", "date"], how="left", validate="one_to_one"
    )
    theme_daily = theme_daily.merge(
        sns_daily[["keyword", "date", "news_mentions", "social_mentions"]],
        on=["keyword", "date"],
        how="left",
        validate="one_to_one",
    )
    theme_daily = theme_daily.sort_values(["keyword", "date"]).reset_index(drop=True)
    for source, count in [("news", "news_mentions"), ("sns", "social_mentions")]:
        theme_daily[f"{source}_change"] = (
            np.log1p(theme_daily[count]).groupby(theme_daily.keyword).diff()
        )
    theme_corr_rows = []
    for keyword, g in theme_daily.groupby("keyword", sort=True):
        # 두 채널을 동일한 날짜 표본으로 비교합니다.
        common = g.dropna(subset=["theme_excess_return", "news_change", "sns_change"])
        row = {"keyword": keyword}
        for source in ["news", "sns"]:
            n, rho_value = rank_correlation(
                common[f"{source}_change"], common.theme_excess_return, min_pairs=40
            )
            row[f"{source}_rho"], row[f"{source}_days"] = rho_value, n
        theme_corr_rows.append(row)
    theme_correlations = pd.DataFrame(theme_corr_rows).sort_values("sns_rho")
    return theme_daily, theme_correlations, int((~theme_eligible).sum())


def plot_comovement(theme_daily, theme_correlations, report):
    fig, ax = plt.subplots(figsize=(10, 11))
    positions = np.arange(len(theme_correlations))
    ax.hlines(
        positions,
        theme_correlations.news_rho,
        theme_correlations.sns_rho,
        color="#D1DCDD",
        linewidth=1.5,
    )
    ax.scatter(theme_correlations.news_rho, positions, color=GOLD, s=40, label="뉴스")
    ax.scatter(
        theme_correlations.sns_rho,
        positions,
        color=GREEN,
        s=40,
        label="SNS (뉴스 제외)",
    )
    ax.axvline(0, color=GRAY, linewidth=1)
    ax.set_yticks(positions, theme_correlations.keyword, fontsize=10)
    ax.set(
        xlabel="같은 날 언급량 변화와 시장 대비 수익률의 순위 상관",
        title="테마별 뉴스·SNS의 당일 동행 관계",
    )
    ax.legend(loc="best")
    ax.grid(axis="x", alpha=0.15)
    report.show_figure(
        fig,
        "06_theme_same_day",
        f"**읽기:** 테마별 상관의 중앙값은 뉴스 {theme_correlations.news_rho.median():+.3f}, "
        f"SNS {theme_correlations.sns_rho.median():+.3f}입니다. "
        "**해석:** 일별 자료로는 오전 SNS와 장중 주가 중 어느 쪽이 먼저였는지 알 수 없습니다. 동행을 선행·예측·원인으로 해석하지 않습니다.",
    )
    report.save_table(theme_daily, "theme_same_day_base.csv.gz")
    report.save_table(theme_correlations, "14_theme_same_day_correlations.csv")
    return None


def check_theme_data(daily, correlations):
    unique_key(daily, ["keyword", "date"], "테마×거래일 동행 표본")
    too_few = daily.valid_stocks.lt(5) | daily.valid_stocks.isna()
    if daily.loc[too_few, "theme_excess_return"].notna().any():
        raise ValueError("유효 종목이 5개 미만인 테마의 수익률이 결측이 아닙니다.")
    if not correlations.news_days.eq(correlations.sns_days).all():
        raise ValueError("뉴스·SNS의 상관 비교 날짜 수가 다릅니다.")
    for col in ["news_rho", "sns_rho"]:
        if not correlations[col].dropna().between(-1, 1).all():
            raise ValueError(f"상관계수가 범위 밖입니다: {col}")
    log("테마 키·유효 종목 수·동일 날짜 비교·상관 범위 점검 통과")


def main(
    input_path=DEFAULT_INPUT, sns_path=DEFAULT_SNS_INPUT, output_dir=DEFAULT_OUTPUT
):
    started = time.monotonic()
    paths = {
        "stock_daily": Path(input_path).resolve(),
        "sns_daily": Path(sns_path).resolve(),
    }
    hashes = {key: file_hash(path) for key, path in paths.items()}
    output_dir = Path(output_dir).resolve()
    log("뉴스·SNS 당일 동행 분석: 준비 데이터 2개 읽기")
    prices = load_stock_daily(paths["stock_daily"], ["return_1", "market_return_1"])
    sns_daily = load_sns_daily(paths["sns_daily"])
    if set(prices.keyword) != set(sns_daily.keyword):
        raise ValueError("주가와 SNS의 키워드 집합이 다릅니다.")
    configure_plots()
    report = Report(output_dir)
    log("거래량·수익률 조건 필터 및 테마별 집계")
    daily, correlations, excluded = build_theme_data(prices, sns_daily, report)
    check_theme_data(daily, correlations)
    log("테마별 뉴스·SNS 동행 관계 그림")
    plot_comovement(daily, correlations, report)
    report.display(correlations)
    metadata = build_metadata(
        "comovement",
        paths,
        hashes,
        {
            "stock_rows": len(prices),
            "excluded_stock_days": excluded,
            "theme_daily_rows": len(daily),
            "themes": len(correlations),
            "figures": report.figures_written,
        },
        {
            "sns_cutoff": "same_calendar_day",
            "news_channels": 3,
            "social_channels": 7,
            "max_abs_stock_return": 0.31,
            "min_valid_stocks": 5,
            "min_correlation_pairs": 40,
            "weekend_mentions_aggregated": False,
            "common_news_sns_dates": True,
        },
        time.monotonic() - started,
    )
    log(f"완료: {output_dir} ({time.monotonic() - started:.1f}초)")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="STEP03 뉴스·SNS와 당일 주가 동행 EDA")
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT, help="준비된 stock_daily.csv.gz"
    )
    parser.add_argument(
        "--sns", type=Path, default=DEFAULT_SNS_INPUT, help="준비된 sns_daily.csv.gz"
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT, help="STEP03 분석 결과 폴더"
    )
    args = parser.parse_args()
    main(args.input, args.sns, args.output)
