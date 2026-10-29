"""STEP02: 화제 시작 전후 거래량·일중 변동폭을 실제·가짜 시점으로 비교한다.

실행: python3 src/eda/step02_volume_response.py
입력: data/processed/model_dataset.csv, data/raw/theme_stock_daily.csv
출력: eda-outputs/step02/tables/와 figures/.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from eda_reporting import Report, configure_plots, log, plt
from scipy.stats import ttest_rel

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data/processed/model_dataset.csv"
DEFAULT_STOCK = ROOT / "data/raw/theme_stock_daily.csv"
DEFAULT_OUTPUT = ROOT / "eda-outputs/step02"
REQUIRED_DATA = ["keyword", "ticker", "date", "in_event", "mention_z28"]
WINDOW = (-10, 10)
PLACEBO_SHIFT = 30
VOLUME_SURGE, RANGE_SURGE = 2.0, 1.5
BLUE_CASE, INK, GRID = "#2a78d6", "#333333", "#e6e6e2"


def load_inputs(dataset_path, stock_path):
    for path in (dataset_path, stock_path):
        if not Path(path).is_file():
            raise FileNotFoundError(f"입력 파일이 없습니다: {path}")
    data = pd.read_csv(
        dataset_path,
        encoding="utf-8-sig",
        dtype={"ticker": str},
        float_precision="round_trip",
        low_memory=False,
    )
    missing = set(REQUIRED_DATA) - set(data.columns)
    if missing:
        raise ValueError(f"STEP02 분석 데이터 필수 컬럼 누락: {sorted(missing)}")
    data["date"] = pd.to_datetime(data.date, format="%Y-%m-%d", errors="raise")
    if (
        data[["ticker", "date"]].isna().any().any()
        or data.duplicated(["ticker", "date"]).any()
    ):
        raise ValueError(
            "STEP02 분석 데이터: ticker+date 키에 결측 또는 중복이 있습니다."
        )
    stock = pd.read_csv(
        stock_path,
        encoding="utf-8-sig",
        dtype={"ticker": str},
        usecols=["ticker", "date", "high", "low", "close", "volume"],
        float_precision="round_trip",
        low_memory=False,
    )
    stock["date"] = pd.to_datetime(
        stock.date.astype(str).str.replace(".", "-", regex=False),
        format="%Y-%m-%d",
        errors="raise",
    )
    stock = stock.drop_duplicates(["ticker", "date"]).sort_values(["ticker", "date"])
    if stock[["ticker", "date", "high", "low", "close", "volume"]].isna().any().any():
        raise ValueError("주가 원자료의 STEP02 필수 열에 결측이 있습니다.")
    stock["range"] = (stock.high - stock.low) / stock.groupby("ticker").close.shift(1)
    matched = (
        data[["ticker", "date"]]
        .drop_duplicates()
        .merge(stock, on=["ticker", "date"], how="left", validate="one_to_one")
    )
    if matched.volume.isna().any():
        raise ValueError(
            "분석 데이터 중 주가 원자료와 연결되지 않은 종목·날짜가 있습니다."
        )
    return data.sort_values(["keyword", "ticker", "date"]).reset_index(drop=True), stock


def all_events(data):
    dates = np.sort(data.date.unique())
    position = {date: i for i, date in enumerate(dates)}
    days = data.drop_duplicates(["keyword", "date"])[
        ["keyword", "date", "in_event", "mention_z28"]
    ]
    days = days.sort_values(["keyword", "date"])
    previous = days.groupby("keyword").in_event.shift(fill_value=0)
    events = days[(days.in_event == 1) & (previous == 0)][
        ["keyword", "date", "mention_z28"]
    ]
    return events.reset_index(drop=True), dates, position


def run_analysis(data, stock, report):
    stock = (
        stock.drop_duplicates(["ticker", "date"]).sort_values(["ticker", "date"]).copy()
    )
    stock["range"] = (stock.high - stock.low) / stock.groupby("ticker").close.shift(1)
    matched = data[["ticker", "date"]].merge(
        stock, on=["ticker", "date"], how="left", validate="one_to_one"
    )
    volume = matched.pivot(index="date", columns="ticker", values="volume")
    price_range = matched.pivot(index="date", columns="ticker", values="range")
    events, dates, position = all_events(data)
    tickers = data.groupby("keyword").ticker.unique()
    low, high = WINDOW
    base = slice(0, -low)
    react = slice(-low, -low + 3)
    rows, paths = [], {"실제": ([], []), "가짜 시점": ([], [])}
    skipped = 0
    for event_id, (keyword, day0) in enumerate(
        events[["keyword", "date"]].itertuples(index=False, name=None)
    ):
        i = position[day0]
        if i + low < 0 or i + PLACEBO_SHIFT + high >= len(dates):
            skipped += 1
            continue
        for kind, start in (("실제", i), ("가짜 시점", i + PLACEBO_SHIFT)):
            window = dates[start + low : start + high + 1]
            vol = volume.reindex(window)[list(tickers[keyword])].to_numpy(dtype=float)
            rng = price_range.reindex(window)[list(tickers[keyword])].to_numpy(
                dtype=float
            )
            vol_base, rng_base = (
                np.nanmean(vol[base], axis=0),
                np.nanmean(rng[base], axis=0),
            )
            ok = (
                (vol_base > 0)
                & (rng_base > 0)
                & np.isfinite(vol_base)
                & np.isfinite(rng_base)
            )
            if not ok.any():
                continue
            vol_ratio, rng_ratio = vol[:, ok] / vol_base[ok], rng[:, ok] / rng_base[ok]
            paths[kind][0].append(vol_ratio.T)
            paths[kind][1].append(rng_ratio.T)
            theme = (
                np.nansum(vol[react], axis=1).mean()
                / np.nansum(vol[base], axis=1).mean()
            )
            for ticker, v, r in zip(
                np.asarray(tickers[keyword])[ok],
                np.nanmean(vol_ratio[react], axis=0),
                np.nanmean(rng_ratio[react], axis=0),
            ):
                rows.append(
                    {
                        "event": event_id,
                        "keyword": keyword,
                        "ticker": ticker,
                        "구분": kind,
                        "거래량배수": v,
                        "변동폭배수": r,
                        "테마거래량배수": theme,
                    }
                )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("비교 가능한 실제·가짜 시점 표본이 없습니다.")
    report.save_table(frame, "01_stock_volume_range_ratios.csv")
    summary = frame.groupby("구분").agg(
        종목이벤트수=("거래량배수", "size"),
        거래량배수_중앙값=("거래량배수", "median"),
        거래량2배이상비율=("거래량배수", lambda x: (x >= VOLUME_SURGE).mean()),
        변동폭배수_중앙값=("변동폭배수", "median"),
        변동폭1_5배이상비율=("변동폭배수", lambda x: (x >= RANGE_SURGE).mean()),
    )
    theme = (
        frame.drop_duplicates(["event", "구분"])
        .groupby("구분")
        .테마거래량배수.agg(["mean", "median"])
    )
    summary["테마거래량배수_평균"], summary["테마거래량배수_중앙값"] = (
        theme["mean"],
        theme["median"],
    )
    summary = summary.reindex(["실제", "가짜 시점"])
    report.save_table(summary.reset_index(names="구분"), "02_volume_range_summary.csv")
    logs = frame.assign(
        lv=np.log(frame.거래량배수.clip(lower=1e-3)),
        lr=np.log(frame.변동폭배수.clip(lower=1e-3)),
    )
    by_event = logs.groupby(["event", "구분"])[["lv", "lr"]].mean().unstack().dropna()
    tests = {
        name: ttest_rel(by_event[col]["실제"], by_event[col]["가짜 시점"]).pvalue
        for name, col in (("거래량", "lv"), ("변동폭", "lr"))
    }
    by_keyword = (
        frame[frame.구분.eq("실제")]
        .groupby("keyword")
        .거래량배수.median()
        .sort_values()
    )
    report.save_table(
        by_keyword.rename("거래량배수").reset_index(), "03_keyword_volume_ratios.csv"
    )

    days = np.arange(-5, high + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    for ax, k, threshold, label in (
        (axes[0], 0, VOLUME_SURGE, "거래량 2배 이상"),
        (axes[1], 1, RANGE_SURGE, "일중 변동폭 1.5배 이상"),
    ):
        for kind, color, style in (
            ("실제", BLUE_CASE, "-"),
            ("가짜 시점", "#9a9a94", "--"),
        ):
            if not paths[kind][k]:
                continue
            ratio = np.vstack(paths[kind][k])[:, -low - 5 :]
            ax.plot(
                days,
                (ratio >= threshold).mean(axis=0) * 100,
                color=color,
                linestyle=style,
                marker="o",
                markersize=3.5,
                linewidth=2.2,
                label=f"{kind} ({len(ratio):,}건)",
            )
        ax.axvspan(0, 2, color=GRID, alpha=0.6, zorder=0)
        ax.axvline(0, color=INK, linestyle=":", linewidth=1)
        ax.set_xticks(days)
        ax.set_xlabel("화제 시작일(Day 0) 기준 거래일")
        ax.set_ylabel(f"{label}인 종목 비율 (%)")
        ax.set_title(f"{label} (기준 = Day −10 ~ −1 평균)")
        ax.legend(loc="upper right")
        ax.grid(axis="y", color=GRID)
        ax.set_axisbelow(True)
    report.show_figure(
        fig,
        "01_volume_range_surge",
        f"**읽기:** 화제 Day 0~+2의 거래량·일중 변동폭 급증 비율을 실제 시점과 30거래일 뒤 가짜 시점에서 비교합니다. "
        "**해석:** 실제·가짜 시점 차이는 사건 주변의 기술통계이며, 인과 효과를 뜻하지 않습니다.",
    )

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.barh(
        by_keyword.index,
        by_keyword.values,
        height=0.7,
        color=[BLUE_CASE if value > 1 else "#9a9a94" for value in by_keyword.values],
    )
    ax.axvline(1, color=INK, linewidth=0.8)
    ax.set_xlabel("화제 직후(Day 0 ~ +2) 거래량 ÷ 평소(Day −10 ~ −1) — 종목 중앙값")
    ax.set_title("키워드별 거래량 반응 (1보다 크면 평소보다 증가)")
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    report.show_figure(
        fig,
        "02_keyword_volume_ratios",
        f"**읽기:** {len(by_keyword)}개 키워드 중 {int(by_keyword.gt(1).sum())}개에서 종목별 중앙 거래량 배수가 1을 넘습니다. "
        "**해석:** 분모는 화제 전 거래량이며, 키워드별 효과 추정치는 아닙니다.",
    )
    report.display(summary.reset_index(names="구분"))
    log(
        f"이벤트 {len(by_event):,}개, 기간 끝 경계로 제외 {skipped:,}개, "
        f"대응 t-검정 p값: 거래량 {tests['거래량']:.3g}, 변동폭 {tests['변동폭']:.3g}"
    )
    return summary, by_keyword, tests, len(by_event), skipped


def main(
    dataset_path=DEFAULT_INPUT, stock_path=DEFAULT_STOCK, output_dir=DEFAULT_OUTPUT
):
    started = time.monotonic()
    dataset_path, stock_path = Path(dataset_path).resolve(), Path(stock_path).resolve()
    output_dir = Path(output_dir).resolve()
    log(f"분석 데이터 읽기: {dataset_path}")
    data, stock = load_inputs(dataset_path, stock_path)
    configure_plots()
    report = Report(output_dir)
    result = run_analysis(data, stock, report)
    log(f"완료: {output_dir} ({time.monotonic() - started:.1f}초)")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="STEP02 화제 시작 전후 거래량·변동폭 반응"
    )
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT, help="RQ1 분석용 종목×거래일 CSV"
    )
    parser.add_argument(
        "--stock",
        type=Path,
        default=DEFAULT_STOCK,
        help="거래량·고가·저가가 있는 주가 원자료",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT, help="STEP02 결과 폴더"
    )
    args = parser.parse_args()
    main(args.input, args.stock, args.output)
