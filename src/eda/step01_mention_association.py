"""STEP01: SNS 언급량과 종목 초과수익률의 연관 및 대표 사례를 분석한다.

실행: python3 src/eda/step01_mention_association.py
입력: data/processed/model_dataset.csv
출력: eda-outputs/step01/tables/와 figures/.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from eda_reporting import Report, configure_plots, log, plt
from matplotlib import font_manager
from scipy.stats import spearmanr, ttest_1samp, ttest_rel
from scipy.stats import t as t_dist

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data/processed/model_dataset.csv"
DEFAULT_OUTPUT = ROOT / "eda-outputs/step01"
DEFAULT_STOCK = ROOT / "data/raw/theme_stock_daily.csv"
REQUIRED = [
    "keyword",
    "ticker",
    "name",
    "is_leader",
    "date",
    "mention_z28",
    "excess_return_1d",
    "in_event",
]
FEATURE_EVENT_WINDOW = (-5, 10)
CASE_WINDOW = (-10, 20)
SURGE = 0.05
BLUE_CASE, INK, GRID = "#2a78d6", "#333333", "#e6e6e2"
STOCK_COLORS = ["#d03b3b", "#1f9e89", "#7b5ea7"]
CAR_WINDOWS = {
    "[-5,-1] 화제 전": (-5, -1),
    "[0,+2] 화제 직후": (0, 2),
    "[+3,+10] 그 이후": (3, 10),
    "[0,+10] 전체": (0, 10),
}
HOLD_DAYS = (1, 2, 3, 5)
EVENT_WINDOW = FEATURE_EVENT_WINDOW
COLOR_Y0, COLOR_Y1 = "#9a9a94", "#2a78d6"
SPAN_GRAY, SPAN_RED = "#f2f2f2", "#fde3df"


def load_dataset(path):
    if not Path(path).is_file():
        raise FileNotFoundError(f"분석 입력 파일이 없습니다: {path}")
    data = pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype={"ticker": str},
        float_precision="round_trip",
        low_memory=False,
    )
    missing = set(REQUIRED) - set(data.columns)
    if missing:
        raise ValueError(f"분석 입력 필수 컬럼 누락: {sorted(missing)}")
    data["date"] = pd.to_datetime(data.date, format="%Y-%m-%d", errors="raise")
    if (
        data[["ticker", "date"]].isna().any().any()
        or data.duplicated(["ticker", "date"]).any()
    ):
        raise ValueError("STEP01 입력: ticker+date 키에 결측 또는 중복이 있습니다.")
    return data.sort_values(["keyword", "ticker", "date"]).reset_index(drop=True)


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


def find_events(data, window=None):
    events, dates, position = all_events(data)
    low, high = window or FEATURE_EVENT_WINDOW
    valid = events.date.map(position).between(-low, len(dates) - 1 - high)
    return (
        events.loc[valid, ["keyword", "date"]].reset_index(drop=True),
        dates,
        position,
    )


def event_ar_matrix(data, return_tickers=False):
    events, dates, position = find_events(data)
    ar = data.pivot_table(index="date", columns="ticker", values="excess_return_1d")
    tickers = data.groupby("keyword").ticker.unique()
    low, high = FEATURE_EVENT_WINDOW
    rows, event_ids, row_tickers, row_day0 = [], [], [], []
    for event_id, (keyword, day0) in enumerate(events.itertuples(index=False)):
        i = position[day0]
        window = ar.loc[dates[i + low : i + high + 1], tickers[keyword]].values.T
        rows.append(window)
        event_ids.extend([event_id] * len(window))
        row_tickers.extend(tickers[keyword])
        row_day0.extend([day0] * len(window))
    result = (np.vstack(rows), np.array(event_ids), events)
    return (
        result + (np.array(row_tickers), np.array(row_day0))
        if return_tickers
        else result
    )


def mention_association(data):
    daily = (
        data.groupby(["keyword", "date"])
        .agg(mention_z28=("mention_z28", "first"), ar=("excess_return_1d", "mean"))
        .dropna()
    )
    keyword_table = (
        daily.groupby("keyword")
        .apply(
            lambda g: pd.Series(
                {
                    "상관": spearmanr(g.mention_z28, g.ar)[0],
                    "최대z28": g.mention_z28.max(),
                    "일수": len(g),
                }
            )
        )
        .sort_values("상관", ascending=False)
    )
    valid = data.dropna(subset=["mention_z28", "excess_return_1d"])
    stock_table = (
        valid.groupby(["keyword", "ticker", "name", "is_leader"])
        .apply(lambda g: spearmanr(g.mention_z28, g.excess_return_1d)[0])
        .rename("상관")
        .reset_index()
        .sort_values("상관", ascending=False)
    )
    return keyword_table, stock_table


def stock_event_reaction(data):
    matrix, event_ids, _, tickers, _ = event_ar_matrix(data, return_tickers=True)
    low = FEATURE_EVENT_WINDOW[0]
    frame = pd.DataFrame(
        {
            "ticker": tickers,
            "event": event_ids,
            "day0": matrix[:, -low] * 100,
            "car02": np.nansum(matrix[:, -low : -low + 3], axis=1) * 100,
        }
    )
    return frame.groupby("ticker").agg(
        이벤트수=("event", "size"),
        Day0평균=("day0", "mean"),
        CAR02평균=("car02", "mean"),
        Day0급등비율=("day0", lambda x: (x >= SURGE * 100).mean()),
    )


def plot_style(ax, axis="x"):
    ax.grid(axis=axis, color=GRID)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def run_analysis(data, report, n_cases=3, n_stocks=3):
    keyword_table, stock_table = mention_association(data)
    reaction = stock_event_reaction(data)
    stock_table = stock_table.join(reaction, on="ticker")
    report.save_table(keyword_table.reset_index(), "01_keyword_correlations.csv")
    report.save_table(stock_table, "02_stock_correlations_and_reactions.csv")
    top = list(keyword_table.index[:n_cases])

    fig, ax = plt.subplots(figsize=(9, 9))
    order = keyword_table.iloc[::-1]
    ax.barh(
        order.index,
        order["상관"],
        height=0.7,
        color=[BLUE_CASE if keyword in top else "#9a9a94" for keyword in order.index],
    )
    ax.axvline(0, color=INK, linewidth=0.8)
    ax.set_xlabel("같은 날 언급량 급증(z28)과 10종목 평균 초과수익률의 순위 상관")
    ax.set_title("키워드별 언급량–주가 연관 (파란색 = 대표 사례 키워드)")
    plot_style(ax)
    report.show_figure(
        fig,
        "01_keyword_correlations",
        f"**읽기:** {len(keyword_table)}개 키워드의 상관 중앙값은 {keyword_table.상관.median():+.2f}입니다. "
        "**해석:** 같은 날짜의 언급량과 초과수익률 간 순위 상관이며, 선행·인과 효과를 뜻하지 않습니다.",
    )

    fig, ax = plt.subplots(figsize=(10, 10))
    for row, keyword in enumerate(order.index):
        stocks = stock_table[stock_table.keyword.eq(keyword)].sort_values("상관")
        ax.scatter(
            stocks.상관, [row] * len(stocks), s=28, color="#9a9a94", alpha=0.8, zorder=2
        )
        ax.plot(
            [stocks.상관.min(), stocks.상관.max()],
            [row, row],
            color=GRID,
            linewidth=4,
            zorder=1,
        )
        best = stocks.iloc[-1]
        ax.scatter(best.상관, row, s=40, color=BLUE_CASE, zorder=3)
        ax.annotate(
            best["name"],
            (best.상관, row),
            xytext=(5, -3),
            textcoords="offset points",
            fontsize=7,
        )
    ax.set_yticks(range(len(order)), order.index)
    ax.axvline(0, color=INK, linewidth=0.8)
    ax.set_xlabel("언급량 급증과 종목 초과수익률의 순위 상관 (파란 점 = 키워드 내 1위)")
    ax.set_title("같은 키워드 안에서도 종목별 연관이 다르다")
    plot_style(ax)
    report.show_figure(
        fig,
        "02_stock_correlations",
        f"**읽기:** {len(stock_table)}개 종목의 상관 중앙값은 {stock_table.상관.median():+.2f}입니다. "
        "**해석:** 키워드 내 종목별 차이를 요약하며, 개별 종목의 예측 성능을 평가하지 않습니다.",
    )

    events, dates, position = all_events(data)
    low, high = CASE_WINDOW
    events = events[events.date.map(position).between(-low, len(dates) - 1 - high)]
    ar = data.pivot_table(index="date", columns="ticker", values="excess_return_1d")
    names = data.drop_duplicates("ticker").set_index("ticker")["name"]
    z28 = (
        data.drop_duplicates(["keyword", "date"])
        .set_index(["keyword", "date"])
        .mention_z28
    )
    rows = []
    for keyword in top:
        candidates = events[events.keyword.eq(keyword)].sort_values(
            "mention_z28", ascending=False
        )
        if candidates.empty:
            report.say(f"대표 사례를 찾을 수 없음: {keyword}")
            continue
        event = candidates.iloc[0]
        i = position[event.date]
        window = dates[i + low : i + high + 1]
        stocks = stock_table[stock_table.keyword.eq(keyword)].set_index("ticker")
        tickers = list(stocks.index)
        highlight, worst = tickers[:n_stocks], tickers[-1]
        car = ar.loc[window, tickers].fillna(0).cumsum() * 100
        days = np.arange(low, high + 1)
        post = ar.loc[dates[i : i + 3], tickers].sum() * 100
        fig, ax = plt.subplots(figsize=(11, 5))
        for ticker in tickers:
            if ticker not in highlight and ticker != worst:
                ax.plot(days, car[ticker], color="#9a9a94", alpha=0.35, linewidth=1)
        for ticker, color in zip(highlight, STOCK_COLORS):
            ax.plot(
                days,
                car[ticker],
                color=color,
                linewidth=1.8,
                label=f"{names[ticker]} (상관 {stocks.loc[ticker, '상관']:.2f})",
            )
        ax.plot(
            days,
            car[worst],
            color=INK,
            linestyle=":",
            linewidth=1.8,
            label=f"{names[worst]} (상관 {stocks.loc[worst, '상관']:.2f}, 최하위)",
        )
        ax.plot(
            days,
            car.median(axis=1),
            color=BLUE_CASE,
            linewidth=3,
            label="10종목 중앙값",
        )
        ax.axvline(0, color=INK, linestyle="--", linewidth=1)
        ax.axhline(0, color=INK, linewidth=0.6)
        ax.set_xlabel("화제 시작일(Day 0) 기준 거래일")
        ax.set_ylabel("누적 초과수익률 CAR (%, Day −10부터)")
        ax.set_title(
            f"{keyword} | Day 0 {event.date:%Y-%m-%d} | 언급량 급증 z28 {event.mention_z28:.1f}"
        )
        twin = ax.twinx()
        twin.plot(
            days,
            [z28.get((keyword, date), np.nan) for date in window],
            color="#e8833a",
            linestyle="--",
            linewidth=1.6,
            label="언급량 급증 z28",
        )
        twin.set_ylabel("언급량 급증 z28")
        handles, labels = ax.get_legend_handles_labels()
        extra_handles, extra_labels = twin.get_legend_handles_labels()
        ax.legend(
            handles + extra_handles, labels + extra_labels, loc="upper left", fontsize=8
        )
        ax.grid(color=GRID)
        ax.set_axisbelow(True)
        report.show_figure(
            fig,
            f"03_case_{len(rows) + 1:02d}",
            f"**읽기:** {keyword}의 대표 Day 0는 {event.date:%Y-%m-%d}, z28은 {event.mention_z28:.1f}입니다. "
            "**해석:** 사례의 누적 수익률 경로를 보여주며 일반적인 효과나 인과를 뜻하지 않습니다.",
        )
        rows.append(
            {
                "키워드": keyword,
                "Day0": event.date,
                "Day0_z28": event.mention_z28,
                "CAR[0,+2]_10종목평균": post.mean(),
                "CAR[0,+2]_중앙값": post.median(),
                "CAR[0,+2]_상관상위3": post[highlight].mean(),
                "CAR[0,+2]_상관최하위": post[worst],
                "상관상위3": ", ".join(names[t] for t in highlight),
                "상관최하위": names[worst],
            }
        )
    cases = pd.DataFrame(rows)
    report.save_table(cases, "03_representative_cases.csv")
    report.display(cases)
    return keyword_table, stock_table, cases


def set_korean_font():
    """한글 폰트를 지정한다 (Noto Sans CJK → 맑은 고딕 → 애플고딕 순서로 찾음)."""
    paths = [
        path for path in font_manager.findSystemFonts() if "NotoSansCJK-Regular" in path
    ]
    if paths:
        font_manager.fontManager.addfont(paths[0])
        plt.rcParams["font.family"] = font_manager.FontProperties(
            fname=paths[0]
        ).get_name()
    else:
        installed = {font.name for font in font_manager.fontManager.ttflist}
        for name in ("Malgun Gothic", "AppleGothic", "NanumGothic"):
            if name in installed:
                plt.rcParams["font.family"] = name
                break
    plt.rcParams["axes.unicode_minus"] = False


def forward_returns(stock_path):
    """주가 원자료로 "t+1 시가에 사서 t+k 종가에 판" 시장 대비 초과수익률(fx_k)을 계산한다.

    y와 같은 방식이다 (fx2 ≥ 5% 이면 y = 1). gap_next = 다음 날 시가 ÷ 오늘 종가 − 1 (종목만, 지수 시가는 없음).
    """
    stock = pd.read_csv(
        stock_path,
        dtype={"ticker": str},
        encoding="utf-8-sig",
        usecols=["ticker", "date", "open", "close", "market_index_close"],
    )
    stock = stock.drop_duplicates(["ticker", "date"]).sort_values(["ticker", "date"])
    stock["date"] = pd.to_datetime(
        stock["date"].astype(str).str.replace(".", "-", regex=False)
    ).dt.strftime("%Y-%m-%d")
    group = stock.groupby("ticker")
    next_open = group["open"].shift(-1)
    next_open = next_open.where(next_open > 0)
    for days in HOLD_DAYS:
        stock[f"fx{days}"] = (group["close"].shift(-days) / next_open - 1) - (
            group["market_index_close"].shift(-days) / stock["market_index_close"] - 1
        )
    stock["gap_next"] = next_open / stock["close"] - 1
    return stock[["ticker", "date", "gap_next"] + [f"fx{days}" for days in HOLD_DAYS]]


PLOT_SPANS = [
    ((-5, -1), SPAN_GRAY),
    ((0, 2), SPAN_RED),
    ((3, 10), SPAN_GRAY),
]  # 그림 배경: 화제 전 / 직후 / 이후


def shade_windows(ax):
    """EDA 3·4 그림 배경에 세 구간(화제 전, 직후 0~2일, 이후)을 색으로 표시한다."""
    for (start, end), color in PLOT_SPANS:
        ax.axvspan(start - 0.5, end + 0.5, color=color, zorder=0)


def run_eda3(data, output_dir):
    """구간별 CAR을 종목 기준으로 요약하고, 이벤트 평균 CAR로 t-검정한다 (표본 = 이벤트 수).

    그림: (가) Day −5부터 누적한 평균 흐름, (나) 세 구간 CAR 막대 + 95% 신뢰구간 + p값
    """
    matrix, event_ids, events = event_ar_matrix(data)
    low, high = EVENT_WINDOW
    days = np.arange(low, high + 1)
    rows, bars = [], []
    for name, (start, end) in CAR_WINDOWS.items():
        car = np.nansum(matrix[:, start - low : end - low + 1], axis=1) * 100
        event_car = pd.Series(car).groupby(event_ids).mean()
        test = ttest_1samp(event_car, 0)
        half = (
            t_dist.ppf(0.975, len(event_car) - 1)
            * event_car.std(ddof=1)
            / np.sqrt(len(event_car))
        )
        rows.append(
            {
                "구간": name,
                "평균CAR(%)": car.mean(),
                "종목 중앙값(%)": np.median(car),
                "오른 종목 비율(%)": (car > 0).mean() * 100,
                "t값": test.statistic,
                "p값": test.pvalue,
                "이벤트수": len(event_car),
            }
        )
        if name != "[0,+10] 전체":
            bars.append(
                (
                    name.replace(" ", "\n", 1),
                    event_car.mean(),
                    half,
                    test.pvalue,
                    start == 0,
                )
            )
    table = pd.DataFrame(rows)
    table.to_csv(
        output_dir / "tables/04_car_window_tests.csv", index=False, encoding="utf-8-sig"
    )

    fig, (ax, bx) = plt.subplots(
        1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1.6, 1]}
    )
    shade_windows(ax)
    curve = np.nanmean(np.nancumsum(matrix, axis=1), axis=0) * 100
    ax.plot(days, curve, color=COLOR_Y1, linewidth=2.4, marker="o", markersize=4)
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.set_xticks(days)
    ax.set_xlabel("화제 시작일(Day 0) 기준 거래일")
    ax.set_ylabel("평균 누적 초과수익률 (%, Day −5부터)")
    ax.set_title("(가) 화제 전후 누적 흐름")

    labels, values, halves, pvalues, focus = zip(*bars)
    positions = np.arange(len(bars))
    bx.bar(
        positions,
        values,
        yerr=halves,
        capsize=6,
        width=0.55,
        color=[COLOR_Y1 if f else "#b8c2cc" for f in focus],
    )
    for x, value, half, p in zip(positions, values, halves, pvalues):
        bx.text(
            x,
            value + half + 0.12,
            f"{value:+.1f}%\n" + ("p<0.001" if p < 0.001 else f"p={p:.2f}"),
            ha="center",
            fontsize=10,
        )
    bx.axhline(0, color=INK, linewidth=0.8)
    bx.set_xticks(positions, labels)
    bx.set_ylim(min(0, min(values) - 1), max(values) + max(halves) + 1)
    bx.set_ylabel("구간 누적 초과수익률 (%)")
    bx.set_title("(나) 구간별 CAR과 0 대비 t-검정 (오차막대 95%)")
    for axis in (ax, bx):
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
    fig.suptitle(
        f"화제 시작 전후 누적 초과수익률 (이벤트 {len(events)}개 × 10종목)", fontsize=13
    )
    fig.tight_layout()
    fig.savefig(output_dir / "figures/04_car_windows.png", dpi=130)
    plt.close(fig)

    print(f"\n[EDA 3] 이벤트 {len(events)}개 × 10종목 (단위 %)")
    print(table.round(3).to_string(index=False))
    return table


# =============================================================================
# EDA 4  공시 유무에 따른 반응 비교
#    목적: 회사 공시 없이도 올랐나 = "이름만 같아서 오른다"는 문제 정의 확인
# =============================================================================
def run_eda4(data, output_dir):
    """Day 0의 disclosure_count_3d(최근 3거래일 공시 건수) > 0이면 '공시 있음'으로 나눠 CAR을 비교한다.

    - 그룹별: 이벤트 안 그룹 평균 CAR이 0과 다른지 t-검정 (표본 = 이벤트 수)
    - 그룹 차이: 두 그룹이 모두 있는 이벤트에서 (공시 없음 − 공시 있음) 평균 차이를 대응 t-검정
    """
    matrix, event_ids, events, tickers, day0 = event_ar_matrix(
        data, return_tickers=True
    )
    disclosure = data.set_index(["ticker", "date"])["disclosure_count_3d"]
    has = np.array([disclosure.get((t, d), np.nan) for t, d in zip(tickers, day0)]) > 0
    low, high = EVENT_WINDOW
    days = np.arange(low, high + 1)
    car = np.nancumsum(matrix, axis=1) * 100
    curve = pd.DataFrame(
        {
            "Day": days,
            "공시 있음": car[has].mean(axis=0),
            "공시 없음": car[~has].mean(axis=0),
        }
    )
    curve.to_csv(
        output_dir / "tables/05_disclosure_caar.csv", index=False, encoding="utf-8-sig"
    )

    rows = []
    for name, (start, end) in CAR_WINDOWS.items():
        window = np.nansum(matrix[:, start - low : end - low + 1], axis=1) * 100
        frame = pd.DataFrame({"event": event_ids, "has": has, "car": window})
        by_event = frame.groupby(["event", "has"])["car"].mean().unstack()
        row = {"구간": name}
        for flag, label in ((True, "공시 있음"), (False, "공시 없음")):
            values = by_event[flag].dropna()
            row[f"{label}_평균"] = frame.loc[frame["has"] == flag, "car"].mean()
            row[f"{label}_중앙값"] = frame.loc[frame["has"] == flag, "car"].median()
            row[f"{label}_p값"] = ttest_1samp(values, 0).pvalue
        paired = by_event.dropna()
        diff = paired[False] - paired[True]
        row["차이(없음−있음)"] = diff.mean()
        row["차이_p값"] = ttest_1samp(diff, 0).pvalue
        row["비교이벤트수"] = len(diff)
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(
        output_dir / "tables/06_disclosure_window_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    shade_windows(ax)
    ax.plot(
        days,
        curve["공시 있음"],
        color=COLOR_Y0,
        linewidth=2.2,
        marker="o",
        markersize=3,
        label=f"공시 있음 ({has.sum():,}건)",
    )
    ax.plot(
        days,
        curve["공시 없음"],
        color=COLOR_Y1,
        linewidth=2.2,
        marker="o",
        markersize=3,
        label=f"공시 없음 ({(~has).sum():,}건)",
    )
    ax.axvline(0, color=INK, linestyle="--", linewidth=1)
    ax.axhline(0, color=INK, linewidth=0.6)
    ax.set_xticks(days)
    ax.set_xlabel("화제 시작일(Day 0) 기준 거래일")
    ax.set_ylabel("평균 누적 초과수익률 (%, Day −5부터)")
    ax.set_title("공시 유무별 평균 누적 초과수익률")
    ax.legend(loc="upper left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / "figures/05_disclosure_caar.png", dpi=130)
    plt.close(fig)

    print(
        f"\n[EDA 4] 공시 있음 {has.sum():,}건 ({has.mean():.1%}) / 공시 없음 {(~has).sum():,}건"
    )
    print(table.round(4).to_string(index=False))
    return table


# =============================================================================
# EDA 5  화제 시작일 전후 일별 반응 + 다음 날 시가 매수
#    목적: 화제 효과가 며칠 동안 나타나나, 화제를 보고 다음 날 시가에 사면 남는 상승이 있나
# =============================================================================
def run_eda5(data, output_dir):
    """Day별 급등 비율, 평균 초과수익률(AAR), 누적(CAAR)을 평소 수준과 비교한다."""
    matrix, event_ids, events = event_ar_matrix(data)
    low, high = EVENT_WINDOW
    days = np.arange(low, high + 1)
    base_rate = (data["excess_return_1d"] >= SURGE).mean()
    rate = (matrix >= SURGE).mean(axis=0)
    # 이벤트별 급등 비율이 평소와 다른지 t-검정 (표본 = 이벤트 수, EDA 3과 같은 방식)
    by_event = pd.DataFrame((matrix >= SURGE).astype(float)).groupby(event_ids).mean()
    pvalues = np.array(
        [ttest_1samp(by_event[i], base_rate).pvalue for i in range(len(days))]
    )
    table = pd.DataFrame(
        {
            "Day": days,
            "급등비율(%)": rate * 100,
            "평소 급등비율(%)": base_rate * 100,
            "평소 대비(%p)": (rate - base_rate) * 100,
            "p값": pvalues,
        }
    )
    table.to_csv(
        output_dir / "tables/07_daily_surge_rates.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(f"\n[EDA 5] 평소 급등 비율 {base_rate:.2%} (단위 %)")
    print(table.round(2).to_string(index=False))

    fig, ax = plt.subplots(figsize=(10, 4.5))
    # 평소보다 유의하게(p<0.05) 높은 날만 강조
    colors = [
        COLOR_Y1 if r > base_rate and p < 0.05 else "#b8c2cc"
        for r, p in zip(rate, pvalues)
    ]
    ax.bar(days, rate * 100, color=colors, width=0.6, edgecolor="white", linewidth=2)
    ax.axhline(
        base_rate * 100,
        color=COLOR_Y0,
        linestyle="--",
        linewidth=1.5,
        label=f"평소 {base_rate:.1%}",
    )
    ax.legend(loc="upper right")
    ax.set_xticks(days)
    ax.set_xlabel("화제 시작일 기준 거래일 (Day)")
    ax.set_ylabel("시장보다 +5% 이상 오른 종목 비율 (%)")
    ax.set_title(
        "화제 시작일 전후 급등 비율 (파란색 = 평소보다 유의하게 높은 날, p<0.05)"
    )
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / "figures/06_daily_surge_rates.png", dpi=130)
    plt.close(fig)
    return table


def run_eda5_hold(data, forward, output_dir):
    """Day d 밤에 화제를 알고 다음 날 시가에 샀다면, k거래일 보유 동안 평소보다 더 올랐나.

    평소 = 분석 기간 전체 종목 × 거래일의 같은 보유 수익률. 이벤트 값이 평소와 같으면 살 시점에는 효과가 남아 있지 않다는 뜻.
    """
    events, dates, position = find_events(data)
    tickers = data.groupby("keyword")["ticker"].unique()
    pairs = [
        (ticker, dates[position[day0] + d], d)
        for keyword, day0 in events.itertuples(index=False)
        for d in (0, 1, 2)
        for ticker in tickers[keyword]
    ]
    frame = pd.DataFrame(pairs, columns=["ticker", "date", "Day"]).merge(
        forward, on=["ticker", "date"], how="left"
    )
    base = data[["ticker", "date"]].merge(forward, on=["ticker", "date"], how="left")
    rows = []
    for d, group in frame.groupby("Day"):
        for days in HOLD_DAYS:
            event, normal = group[f"fx{days}"].dropna(), base[f"fx{days}"].dropna()
            rows.append(
                {
                    "매수": f"Day {d} 밤 → 다음 날 시가",
                    "보유일": days,
                    "평균": event.mean(),
                    "평소평균": normal.mean(),
                    "중앙값": event.median(),
                    "+5%이상": (event >= SURGE).mean(),
                    "평소+5%이상": (normal >= SURGE).mean(),
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(
        output_dir / "tables/08_next_open_holding_returns.csv",
        index=False,
        encoding="utf-8-sig",
    )
    gap_event = frame.loc[frame["Day"] == 0, "gap_next"].dropna()
    gap_base = base["gap_next"].dropna()
    print("\n[EDA 5 추가] 화제를 보고 다음 날 시가에 샀다면 (시장 대비 초과수익률)")
    print(table.round(4).to_string(index=False))
    print(
        f"Day 0 다음 날 시가 갭(종목): 평균 {gap_event.mean():+.2%}, 중앙값 {gap_event.median():+.2%} "
        f"(평소 평균 {gap_base.mean():+.2%}, 중앙값 {gap_base.median():+.2%})"
    )
    return table


def run_added_eda(data, report, stock_path):
    # Match the reference script's default Matplotlib style and font setup.
    with plt.rc_context():
        plt.rcdefaults()
        set_korean_font()
        reference_data = data.copy()
        reference_data["date"] = reference_data.date.dt.strftime("%Y-%m-%d")
        run_eda3(reference_data, report.root)
        run_eda4(reference_data, report.root)
        run_eda5(reference_data, report.root)
        run_eda5_hold(reference_data, forward_returns(stock_path), report.root)
    report.figures_written += 3


def main(input_path=DEFAULT_INPUT, output_dir=DEFAULT_OUTPUT, stock_path=DEFAULT_STOCK):
    started = time.monotonic()
    input_path, output_dir = Path(input_path).resolve(), Path(output_dir).resolve()
    log(f"입력 읽기: {input_path}")
    data = load_dataset(input_path)
    configure_plots()
    report = Report(output_dir)
    log(f"언급량·수익률 연관 및 대표 사례 분석: {len(data):,}행")
    keyword_table, stock_table, cases = run_analysis(data, report)
    run_added_eda(data, report, stock_path)
    report.display(keyword_table.head(10).reset_index())
    report.display(
        stock_table[
            [
                "keyword",
                "name",
                "상관",
                "이벤트수",
                "Day0평균",
                "CAR02평균",
                "Day0급등비율",
            ]
        ].head(10)
    )
    log(f"완료: {output_dir} ({time.monotonic() - started:.1f}초)")
    return {
        "keywords": len(keyword_table),
        "stocks": len(stock_table),
        "cases": len(cases),
        "figures": report.figures_written,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="STEP01 언급량과 수익률 연관 분석")
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT, help="RQ1 분석용 종목×거래일 CSV"
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT, help="STEP01 결과 폴더"
    )
    parser.add_argument(
        "--stock", type=Path, default=DEFAULT_STOCK, help="OHLCV 원자료"
    )
    args = parser.parse_args()
    main(args.input, args.output, args.stock)
