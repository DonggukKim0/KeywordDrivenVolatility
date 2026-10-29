"""STEP03: 급등·붕괴 사건과 SNS 신호의 관계를 탐색한다.

입력: step03_prepare_data.py의 stock_daily.csv.gz
출력: eda-outputs/step03/tables, figures

실행:
    python3 src/eda/step03_events.py
    python3 src/eda/step03_events.py --output /tmp/step03

Day 0은 10거래일 상승률 20% 이상 조건에 진입한 날이다.
Day -10부터 현재까지의 최고 종가 대비 20% 이상 하락한 첫날을 붕괴로 정의한다.
사건 비율은 확인 전 붕괴를 제외하고 20거래일을 모두 관찰한 사건으로 계산한다.
"""

import argparse
import platform
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from step03_eda_common import Report, configure_plots, load_stock_daily, plt
from step03_prepare_data import file_hash, unique_key

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data/processed/eda/step03/stock_daily.csv.gz"
DEFAULT_OUTPUT = ROOT / "eda-outputs/step03"
RISE_WINDOW, RISE_THRESHOLD = 10, 0.20
FOLLOWUP, DROP_THRESHOLD = 20, 0.20
BLUE, RED, GREEN, GOLD, GRAY = "#2864A0", "#BF513F", "#167F89", "#B67A20", "#8794A2"
SIGNAL_NAMES = {False: "동시 신호 없음", True: "동시 신호 있음"}


def log(message):
    print(f"[STEP03 events {time.strftime('%H:%M:%S')}] {message}", flush=True)


def load_base(path):
    base = load_stock_daily(
        path,
        [
            "common_eligible",
            f"rise_{RISE_WINDOW}",
            f"prior_sd_{RISE_WINDOW}",
            "attention_decrease_7",
            "negative_change_7",
            "mean_mentions_7",
            "negative_share_7",
        ],
    )
    if (
        not pd.api.types.is_bool_dtype(base.common_eligible.dtype)
        or base.common_eligible.isna().any()
    ):
        raise ValueError("common_eligible은 결측 없는 bool이어야 합니다.")
    return base


def select_event_starts(stocks):
    starts = []
    for ticker, g in stocks.items():
        condition = (
            g.common_eligible & g[f"rise_{RISE_WINDOW}"].ge(RISE_THRESHOLD)
        ).to_numpy()
        entries = np.flatnonzero(condition & ~np.r_[False, condition[:-1]])
        locked_until = -1
        for t0 in entries:
            if t0 <= locked_until:
                continue
            locked_until = t0 + FOLLOWUP
            starts.append(
                {
                    "event_id": f"fixed_{RISE_WINDOW}_{RISE_THRESHOLD * 100:g}__{ticker}__{g.date.iloc[t0]:%Y%m%d}",
                    "ticker": ticker,
                    "name": g.name.iloc[0],
                    "keyword": g.keyword.iloc[0],
                    "confirmation_index": int(t0),
                    "confirmation_date": g.date.iloc[t0],
                    "rise_start_index": int(t0 - RISE_WINDOW),
                    "planned_end_index": int(t0 + FOLLOWUP),
                    "rise_return": g[f"rise_{RISE_WINDOW}"].iloc[t0],
                    "prior_volatility": g[f"prior_sd_{RISE_WINDOW}"].iloc[t0],
                }
            )
    event_starts = pd.DataFrame(starts)
    if event_starts.empty:
        raise ValueError(
            "현재 기준에 해당하는 상승 사건이 없습니다. 입력 기간과 설정을 확인하세요."
        )
    return event_starts


def build_events(stocks, event_starts, base, drop=DROP_THRESHOLD):
    outcomes, daily = [], []
    for e in event_starts.to_dict("records"):
        g = stocks[e["ticker"]]
        close = g.close.to_numpy()
        t0, start = e["confirmation_index"], e["rise_start_index"]
        end = min(t0 + FOLLOWUP, len(g) - 1)
        peaks = np.maximum.accumulate(close[start : end + 1])
        initial_peak = close[start : t0 + 1].max()
        last_peak = start + np.flatnonzero(close[start : t0 + 1] == initial_peak)[-1]
        preexisting = bool(
            np.any(close[last_peak : t0 + 1] / initial_peak - 1 <= -drop + 1e-12)
        )
        hits = (
            np.flatnonzero(
                close[t0 + 1 : end + 1] / peaks[t0 + 1 - start :] - 1 <= -drop + 1e-12
            )
            + t0
            + 1
        )
        hit = int(hits[0]) if len(hits) and not preexisting else None
        full = t0 + FOLLOWUP < len(g)
        label = (
            np.nan
            if preexisting
            else 1.0
            if hit is not None
            else 0.0
            if full
            else np.nan
        )
        outcomes.append(
            {
                **e,
                "drop_threshold": drop,
                "preexisting": preexisting,
                "full_followup": full,
                "collapse_index": hit,
                "collapse_day": hit - t0 if hit is not None else np.nan,
                "first_collapse_date": g.date.iloc[hit] if hit is not None else pd.NaT,
                "event_label": label,
                "event_group": "확인 전 붕괴"
                if preexisting
                else "붕괴"
                if hit is not None
                else "비붕괴"
                if full
                else "관측 부족",
            }
        )
        for j in range(t0, end + 1):
            known_peak = close[start : j + 1].max()
            peak_index = start + np.flatnonzero(close[start : j + 1] == known_peak)[-1]
            at_risk = not preexisting and j < t0 + FOLLOWUP and (hit is None or j < hit)
            daily.append(
                {
                    "event_id": e["event_id"],
                    "ticker": e["ticker"],
                    "date": g.date.iloc[j],
                    "age_day": j - t0,
                    "observation_index": j,
                    "current_drawdown": close[j] / known_peak - 1,
                    "known_peak": known_peak,
                    "days_since_known_peak": j - peak_index,
                    "at_risk": at_risk,
                    "next_day_collapse": float(hit == j + 1)
                    if at_risk and j + 1 < len(g)
                    else np.nan,
                }
            )
    panel = pd.DataFrame(daily).merge(
        base, on=["ticker", "date"], how="left", validate="many_to_one"
    )
    panel["signal_observed"] = (
        panel[["attention_decrease_7", "negative_change_7"]].notna().all(axis=1)
    )
    panel["joint_signal"] = (
        panel.attention_decrease_7.gt(0) & panel.negative_change_7.gt(0)
    ).astype("boolean")
    panel.loc[~panel.signal_observed, "joint_signal"] = pd.NA
    panel["row_id"] = panel.event_id + "__D" + panel.age_day.astype(str).str.zfill(2)
    return pd.DataFrame(outcomes), panel


def plot_event_frequency(complete_events, report):
    theme_rates = complete_events.groupby("keyword").agg(
        events=("event_id", "size"), collapses=("event_label", "sum")
    )
    theme_rates["rate"] = theme_rates.collapses / theme_rates.events
    theme_rates = theme_rates.sort_values("rate")
    timing = complete_events.loc[
        complete_events.event_label.eq(1), "collapse_day"
    ].astype(int)
    fig, axes = plt.subplots(
        1, 2, figsize=(14, 9), gridspec_kw={"width_ratios": [1.15, 1]}
    )
    axes[0].barh(theme_rates.index, theme_rates.rate * 100, color=BLUE)
    for y, row in enumerate(theme_rates.itertuples()):
        axes[0].text(
            row.rate * 100 + 1,
            y,
            f"{int(row.collapses)}/{row.events}",
            va="center",
            fontsize=9,
        )
    axes[0].set(
        xlabel="20거래일 안 붕괴한 사건 (%)", title="테마별 붕괴 비율", xlim=(0, 100)
    )
    axes[0].tick_params(axis="y", labelsize=9)
    counts = timing.value_counts().reindex(range(1, FOLLOWUP + 1), fill_value=0)
    axes[1].bar(counts.index, counts, color=RED)
    axes[1].set(
        xlabel="상승 확인 후 첫 붕괴까지 거래일",
        ylabel="붕괴 사건 수",
        title="첫 붕괴 시점",
        xticks=[1, 5, 10, 15, 20],
    )
    report.show_figure(
        fig,
        "01_event_frequency",
        f"**읽기:** {len(complete_events):,}사건 중 {len(timing):,}건({complete_events.event_label.mean():.1%})이 붕괴했습니다. "
        + (
            f"가장 많은 발생일은 {', '.join(map(str, counts[counts.eq(counts.max())].index))}거래일째, 중앙 시점은 {timing.median():g}거래일째입니다. "
            if len(timing)
            else "붕괴 사건이 없습니다. "
        )
        + "**해석:** 이는 이미 급등한 사건의 사후 빈도입니다. 변동성이 큰 모든 종목의 일반적인 위험이나 예측 시점으로 해석하지 않습니다.",
    )
    report.save_table(theme_rates.reset_index(), "07_theme_collapse_rates.csv")
    report.save_table(
        counts.rename_axis("collapse_day").rename("events").reset_index(),
        "08_collapse_timing.csv",
    )
    return theme_rates.reset_index(), counts


def plot_cases(events, event_panel, stocks, report):
    case_requests = [
        ("394280", "2025-02-17"),
        ("095340", "2025-02-17"),
        ("049480", "2026-08-12"),
    ]
    case_rows = []
    for ticker, confirmation in case_requests:
        found = events[
            events.ticker.eq(ticker)
            & events.confirmation_date.eq(pd.Timestamp(confirmation))
        ]
        if len(found) != 1:
            report.say(
                f"사례 {ticker} / {confirmation}: 현재 사건 정의에서 찾을 수 없습니다."
            )
            continue
        e = found.iloc[0]
        g = stocks[ticker]
        t0, start = int(e.confirmation_index), int(e.rise_start_index)
        end = min(t0 + FOLLOWUP, len(g) - 1)
        path = g.iloc[start : end + 1]
        x = np.arange(start, end + 1) - t0
        close = path.close.to_numpy()
        price_index = close / g.close.iloc[t0] * 100
        threshold = (
            np.maximum.accumulate(close) / g.close.iloc[t0] * 100 * (1 - DROP_THRESHOLD)
        )
        landmark = event_panel[
            event_panel.event_id.eq(e.event_id) & event_panel.age_day.eq(5)
        ]
        signal = landmark.joint_signal.iloc[0] if len(landmark) else pd.NA
        signal_text = "SNS 결측" if pd.isna(signal) else SIGNAL_NAMES[bool(signal)]
        after5 = pd.notna(e.collapse_day) and e.collapse_day > 5
        outcome = (
            f"Day {int(e.collapse_day)} 붕괴"
            if pd.notna(e.collapse_day)
            else e.event_group
        )
        fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
        axes[0].plot(x, price_index, color=BLUE, label="종가 (Day 0=100)")
        axes[0].plot(x, threshold, "--", color=RED, label="그날까지 고점의 80%")
        axes[0].set(
            ylabel="가격 지수",
            title=f"{e['name']} ({ticker}) · {e.keyword} · {confirmation}\n{signal_text} / {outcome}",
        )
        axes[1].plot(x, path.mean_mentions_7, color=GREEN, label="최근 7일 평균 언급량")
        axes[1].set_ylabel("언급량 (건/일)")
        axes[2].plot(
            x, path.negative_share_7 * 100, color=RED, label="최근 7일 부정 감성어 비율"
        )
        axes[2].set(
            ylabel="부정 감성어 비율 (%)", xlabel="상승 확인일 기준 거래일 (Day 0)"
        )
        for ax in axes:
            ax.axvline(0, color=GRAY, linestyle=":", linewidth=1)
            ax.axvline(5, color=GREEN, linestyle="--", linewidth=1)
            ax.axvspan(5, FOLLOWUP, color=GRAY, alpha=0.07)
            if pd.notna(e.collapse_day):
                ax.axvline(e.collapse_day, color=RED, linestyle=":", linewidth=1)
            ax.legend(loc="best", fontsize=9)
            ax.grid(axis="y", alpha=0.15)
        if len(landmark) and landmark.signal_observed.iloc[0]:
            row = landmark.iloc[0]
            prior_mentions = np.expm1(
                np.log1p(row.mean_mentions_7) + row.attention_decrease_7
            )
            prior_negative = row.negative_share_7 - row.negative_change_7
            reading = (
                f"Day 5에서 전날까지 최근 7일과 그 앞 7일을 비교하면, "
                f"평균 언급량은 {prior_mentions:,.0f}건에서 {row.mean_mentions_7:,.0f}건으로, "
                f"부정 비율은 {prior_negative:.1%}에서 {row.negative_share_7:.1%}로 바뀌었습니다. "
                + (
                    "두 SNS 조건을 모두 충족했습니다. "
                    if bool(signal)
                    else "두 SNS 조건을 동시에 충족하지 않았습니다. "
                )
            )
        else:
            reading = "Day 5의 SNS 상태를 확인할 수 없습니다. "
        result_text = (
            f"첫 붕괴는 상승 확인일로부터 {int(e.collapse_day)}거래일 뒤, "
            f"Day 5 점검으로부터 {int(e.collapse_day - 5)}거래일 뒤입니다."
            if after5
            else "이 사건의 관측 결과는 " + outcome + "입니다."
        )
        report.show_figure(
            fig,
            f"02_case_{ticker}",
            "**읽기:** "
            + reading
            + result_text
            + " **해석:** 사례의 시간 순서를 보여주며, 신호가 붕괴를 예측했다는 검증은 아닙니다.",
        )
        case_rows.append(
            {
                "ticker": ticker,
                "confirmation_date": e.confirmation_date,
                "event_id": e.event_id,
                "day5_signal": signal_text,
                "collapse_day": e.collapse_day,
                "event_group": e.event_group,
            }
        )
    report.save_table(pd.DataFrame(case_rows), "09_case_selection.csv")
    return pd.DataFrame(case_rows)


def plot_signal_groups(event_panel, complete_events, report):
    landmarks = []
    for day in (0, 5, 10):
        z = event_panel[
            event_panel.age_day.eq(day)
            & event_panel.at_risk
            & event_panel.signal_observed
            & event_panel.volume.gt(0)
        ]
        z = z.merge(
            complete_events[["event_id", "event_label"]],
            on="event_id",
            validate="one_to_one",
        )
        for signal in (False, True):
            g = z[z.joint_signal.eq(signal)]
            landmarks.append(
                {
                    "day": day,
                    "signal": signal,
                    "events": len(g),
                    "collapses": g.event_label.sum(),
                    "rate": g.event_label.mean(),
                    "outcome_window": f"Day {day + 1}~{FOLLOWUP}",
                }
            )
    landmark_summary = pd.DataFrame(landmarks)
    daily_risk = event_panel[
        event_panel.at_risk
        & event_panel.signal_observed
        & event_panel.volume.gt(0)
        & event_panel.next_day_collapse.notna()
    ].copy()
    DD_LABELS = ["0~5%", "5~10%", "10~15%", "15~20%"]
    daily_risk["drawdown_bin"] = pd.cut(
        -daily_risk.current_drawdown,
        [0, 0.05, 0.10, 0.15, 0.20],
        labels=DD_LABELS,
        include_lowest=True,
        right=False,
    )
    state_summary = (
        daily_risk.groupby(["drawdown_bin", "joint_signal"], observed=True)
        .agg(
            rows=("row_id", "size"),
            collapses=("next_day_collapse", "sum"),
            rate=("next_day_collapse", "mean"),
        )
        .reset_index()
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for signal, offset, color in [(False, -0.18, BLUE), (True, 0.18, RED)]:
        g = (
            landmark_summary[landmark_summary.signal.eq(signal)]
            .set_index("day")
            .reindex([0, 5, 10])
        )
        bars = axes[0].bar(
            np.arange(3) + offset,
            g.rate * 100,
            0.34,
            color=color,
            label=SIGNAL_NAMES[signal],
        )
        for bar, n in zip(bars, g.events):
            if np.isfinite(bar.get_height()):
                axes[0].text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1,
                    f"n={int(n)}",
                    ha="center",
                    fontsize=8,
                )
        q = (
            state_summary[state_summary.joint_signal.eq(signal)]
            .set_index("drawdown_bin")
            .reindex(DD_LABELS)
        )
        axes[1].plot(
            DD_LABELS, q.rate * 100, "o-", color=color, label=SIGNAL_NAMES[signal]
        )
    axes[0].set(
        xticks=np.arange(3),
        xticklabels=["확인일", "5일째", "10일째"],
        ylabel="남은 기간 붕괴 비율 (%)",
        title="같은 시점의 사건끼리 비교",
    )
    axes[0].set_ylim(0, max(50, landmark_summary.rate.max() * 100 + 10))
    axes[1].set(
        xlabel="현재 고점 대비 하락폭",
        ylabel="다음 거래일 첫 붕괴 비율 (%)",
        title="현재 가격 상태별 비교",
    )
    for ax in axes:
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.15)
    day0 = landmark_summary[landmark_summary.day.eq(0)].set_index("signal")
    near_peak_rate = daily_risk.loc[
        daily_risk.drawdown_bin.eq(DD_LABELS[0]), "next_day_collapse"
    ].mean()
    near_line_rate = daily_risk.loc[
        daily_risk.drawdown_bin.eq(DD_LABELS[-1]), "next_day_collapse"
    ].mean()
    report.show_figure(
        fig,
        "03_signal_groups",
        f"**읽기:** 확인일 기준 붕괴 비율은 동시 신호 없음 {day0.loc[False, 'rate']:.1%}, "
        f"있음 {day0.loc[True, 'rate']:.1%}입니다. Day 5·10에서는 그날까지 살아남은 사건만 남습니다. "
        f"오른쪽의 두 그룹을 합한 다음날 빈도는 현재 하락폭 0~5%에서 {near_peak_rate:.2%}, 15~20%에서 {near_line_rate:.1%}입니다. "
        "**해석:** 그룹 차이에는 테마·시기·현재 하락폭 차이가 섞일 수 있습니다. 독립적인 SNS 효과나 예측 성능을 뜻하지 않습니다.",
    )
    report.display(landmark_summary)
    report.display(state_summary)
    report.save_table(landmark_summary, "10_landmark_signal_rates.csv")
    report.save_table(state_summary, "11_price_state_signal_rates.csv")
    return landmark_summary, state_summary


def definition_sensitivity(stocks, event_starts, base, events, report):
    sensitivity_rows = []
    for drop in (0.15, 0.20, 0.25):
        event_set = (
            events
            if np.isclose(drop, DROP_THRESHOLD)
            else build_events(stocks, event_starts, base, drop)[0]
        )
        full = event_set[event_set.full_followup & ~event_set.preexisting]
        sensitivity_rows.append(
            {
                "하락 기준": drop,
                "전체 상승 사건": len(event_set),
                "확인 전 붕괴": int(event_set.preexisting.sum()),
                "20일 관찰 사건": len(full),
                "붕괴 사건": int(full.event_label.sum()),
                "붕괴 비율": full.event_label.mean(),
            }
        )
    definition_sensitivity = pd.DataFrame(sensitivity_rows)
    report.display(definition_sensitivity)
    report.save_table(definition_sensitivity, "15_definition_sensitivity.csv")
    return definition_sensitivity


def check_events(events, panel):
    unique_key(events, ["event_id"], "상승 사건")
    unique_key(panel, ["row_id"], "사건×거래일")
    unique_key(panel, ["event_id", "age_day"], "사건 경과일")
    if not panel.groupby("event_id").next_day_collapse.sum().le(1).all():
        raise ValueError("한 사건에 첫 붕괴가 여러 번 기록되었습니다.")
    complete = events[events.full_followup & ~events.preexisting]
    if complete.empty:
        raise ValueError("20거래일 관찰을 완료한 분석 대상 사건이 없습니다.")
    if complete.event_label.isna().any():
        raise ValueError("관찰 완료 사건의 결과가 결측입니다.")
    censored = ~events.full_followup & ~events.preexisting & events.collapse_day.isna()
    if events.loc[censored, "event_label"].notna().any():
        raise ValueError("관측 부족 사건을 비붕괴로 처리했습니다.")
    if events.loc[events.preexisting, "event_label"].notna().any():
        raise ValueError("확인 전 붕괴 사건에 결과가 기록되었습니다.")
    hits = events.collapse_day.dropna()
    if not hits.between(1, FOLLOWUP).all():
        raise ValueError("첫 붕괴 시점이 관찰 기간 밖에 있습니다.")
    if panel.loc[~panel.signal_observed, "joint_signal"].notna().any():
        raise ValueError("SNS 결측을 신호 없음으로 처리했습니다.")
    log("사건 중복·붕괴 시점·관측 부족·SNS 결측 점검 통과")
    return complete.copy()


def main(input_path=DEFAULT_INPUT, output_dir=DEFAULT_OUTPUT):
    started = time.monotonic()
    input_path, output_dir = Path(input_path).resolve(), Path(output_dir).resolve()
    log(f"준비 데이터 읽기: {input_path}")
    base = load_base(input_path)
    input_hash = file_hash(input_path)
    stocks = {
        ticker: g.reset_index(drop=True)
        for ticker, g in base.groupby("ticker", sort=False)
    }
    log(f"상승 사건 선정: {len(base):,}행, {len(stocks):,}종목")
    event_starts = select_event_starts(stocks)
    log(f"사건 및 일별 관찰 결과 생성: {len(event_starts):,}사건")
    events, panel = build_events(stocks, event_starts, base)
    complete = check_events(events, panel)
    configure_plots()
    report = Report(output_dir)
    flow = pd.DataFrame(
        [
            ["선정한 상승 사건", len(events)],
            ["확인 전 붕괴 사건", int(events.preexisting.sum())],
            [
                "20일 미완료·확인 전 붕괴 제외",
                int((~events.full_followup & ~events.preexisting).sum()),
            ],
            ["20일 동일 관찰 사건", len(complete)],
            ["동일 관찰 사건 중 붕괴", int(complete.event_label.sum())],
        ],
        columns=["구분", "사건 수"],
    )
    report.display(flow)
    report.save_table(events, "events.csv")
    report.save_table(panel, "event_panel.csv.gz")
    report.save_table(flow, "06_event_sample_flow.csv")
    log("붕괴 빈도·시점 분석")
    plot_event_frequency(complete, report)
    log("원본 노트북의 사례 3개 분석")
    cases = plot_cases(events, panel, stocks, report)
    log("동시 SNS 신호 비교: 관심 감소 및 부정 비율 증가")
    plot_signal_groups(panel, complete, report)
    log("하락 기준 15%·20%·25% 민감도 분석")
    sensitivity = definition_sensitivity(stocks, event_starts, base, events, report)
    if file_hash(input_path) != input_hash:
        raise ValueError("실행 중 입력 데이터가 변경되었습니다.")
    metadata = {
        "step": "STEP03",
        "analysis": "events",
        "source_notebook": "RQ3_raw_EDA.ipynb",
        "input": {"path": str(input_path), "sha256": input_hash},
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "rules": {
            "rise_window": RISE_WINDOW,
            "rise_threshold": RISE_THRESHOLD,
            "followup": FOLLOWUP,
            "drop_threshold": DROP_THRESHOLD,
            "sns_cutoff": "previous_calendar_day",
            "new_event_lockout_trading_days": FOLLOWUP,
            "collapse_peak_start": "confirmation_minus_rise_window",
        },
        "counts": {
            "input_rows": len(base),
            "events": len(events),
            "event_panel_rows": len(panel),
            "complete_events": len(complete),
            "collapses": int(complete.event_label.sum()),
            "collapse_rate": float(complete.event_label.mean()),
            "cases_found": len(cases),
            "figures": report.figures_written,
        },
        "definition_sensitivity": sensitivity.astype(object)
        .where(sensitivity.notna(), None)
        .to_dict(orient="records"),
        "checks": {"event_validation_passed": True, "input_unchanged": True},
        "model_training_or_validation": False,
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }
    log(f"완료: {output_dir} ({time.monotonic() - started:.1f}초)")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="STEP03 급등·붕괴 사건 EDA")
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT, help="준비된 stock_daily.csv.gz"
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT, help="STEP03 분석 결과 폴더"
    )
    args = parser.parse_args()
    main(args.input, args.output)
