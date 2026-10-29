"""시장변수 정의서의 13개 feature를 종목·키워드 단위 CSV로 생성한다."""

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from preprocess_sns_data import (
    MENTION_TOTAL,
    calculate_event_features,
    calculate_mention_features,
)

ROOT = Path(__file__).resolve().parents[2]
META = ["category", "keyword", "theme"]
PRICE_FEATURES = [
    "excess_return_20d",
    "volume_ratio_5d",
    "volatility_20d",
    "drawdown_20d",
    "excess_return_1d",
    "excess_return_3d",
]
THEME_FEATURES = ["theme_up_ratio", "leader_return", "days_since_leader"]
DISCLOSURE_WINDOWS = {
    "disclosure_count_3d": 3,
    "contract_disclosure_3d": 3,
    "dilution_disclosure_20d": 20,
    "rumor_clarification_5d": 5,
}
FEATURES = (
    PRICE_FEATURES[:4] + THEME_FEATURES + list(DISCLOSURE_WINDOWS) + PRICE_FEATURES[4:]
)
OHLCV = ["open", "high", "low", "close", "volume"]
SNS_INPUT = ROOT / "data/raw/sns_daily_mentions_sentiment.csv"
KEYWORD_ALIASES = {"딥시크 (DeepSeek)": "딥시크"}
THEME_UP_THRESHOLD = 0.05


def require(frame, columns, label):
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{label} 필수 컬럼 누락: {', '.join(sorted(missing))}")


def read_csv(path):
    return pd.read_csv(path, dtype=str, encoding="utf-8-sig", keep_default_na=False)


def validate_tickers(frame):
    if not frame.ticker.str.fullmatch(r"[0-9]{6}").all():
        raise ValueError("종목코드는 6자리 숫자여야 합니다.")


def load_stock(path):
    frame = read_csv(path)
    required = (
        META + ["ticker", "name", "date"] + OHLCV + ["market", "market_index_close"]
    )
    require(frame, required, "주가")
    if frame.empty:
        raise ValueError("주가 입력이 비어 있습니다.")
    for col in META + ["ticker", "name", "market"]:
        frame[col] = frame[col].str.strip()
        if frame[col].eq("").any():
            raise ValueError(f"빈 식별 정보: {col}")
    validate_tickers(frame)
    frame["date"] = pd.to_datetime(frame.date, format="%Y-%m-%d", errors="raise")
    if frame.date.isna().any() or frame.duplicated(["keyword", "ticker", "date"]).any():
        raise ValueError("주가 날짜 누락 또는 keyword+ticker+date 중복")
    if not frame.market.isin(["KOSPI", "KOSDAQ"]).all():
        raise ValueError("market은 KOSPI 또는 KOSDAQ이어야 합니다.")
    if not frame.groupby("keyword").ticker.nunique().eq(10).all():
        raise ValueError("키워드는 고정된 10종목이어야 합니다.")
    if frame.groupby("keyword")[META].nunique().gt(1).any().any():
        raise ValueError("키워드 메타데이터가 일관되지 않습니다.")
    if frame.groupby("ticker").market.nunique().gt(1).any():
        raise ValueError(
            "동일 종목의 시장이 변경되었습니다. 시장 이력 처리가 필요합니다."
        )
    for col in (
        OHLCV
        + ["market_index_close"]
        + (["market_return_1d"] if "market_return_1d" in frame else [])
    ):
        frame[col] = pd.to_numeric(frame[col].replace("", np.nan), errors="raise")
        if np.isinf(frame[col]).any():
            raise ValueError(f"무한대 입력: {col}")
    frame[OHLCV[:4] + ["market_index_close"]] = frame[
        OHLCV[:4] + ["market_index_close"]
    ].where(frame[OHLCV[:4] + ["market_index_close"]] > 0)
    frame["volume"] = frame.volume.where(frame.volume >= 0)
    if (
        frame[["ticker", "date"] + OHLCV]
        .drop_duplicates()
        .duplicated(["ticker", "date"])
        .any()
    ):
        raise ValueError("동일 종목·날짜의 주가 불일치")
    for _, group in frame.groupby(["market", "date"]):
        if group.market_index_close.nunique(dropna=False) != 1:
            raise ValueError("동일 시장·날짜의 지수 불일치")
    return frame


def classify_title(title):
    title = re.sub(r"\s+", "", str(title))
    title = re.sub(r"^(?:\[[^\]]*\])+", "", title)
    return {
        "contract_disclosure_3d": bool(
            re.search(r"단일판매[ㆍ·・/]?공급계약체결", title)
        )
        and "해지" not in title,
        "dilution_disclosure_20d": bool(
            re.search(r"유상증자|유무상증자|전환사채|신주인수권부사채|교환사채", title)
        ),
        "rumor_clarification_5d": bool(re.search(r"풍문|보도|소문", title))
        and bool(re.search(r"해명|조회공시.*답변", title)),
    }


def load_disclosures(path):
    frame = read_csv(path)
    require(frame, ["ticker", "rcept_no", "rcept_dt", "report_nm", "corp_cls"], "공시")
    validate_tickers(frame)
    frame["rcept_dt"] = pd.to_datetime(
        frame.rcept_dt, format="%Y-%m-%d", errors="raise"
    )
    if (
        frame.rcept_dt.isna().any()
        or frame.rcept_no.eq("").any()
        or frame.duplicated(["ticker", "rcept_no"]).any()
    ):
        raise ValueError("공시 접수일·접수번호 누락 또는 중복")
    return frame


def validate_financials(path):
    frame = read_csv(path)
    require(
        frame,
        [
            "ticker",
            "rcept_no",
            "rcept_dt",
            "bsns_year",
            "reprt_code",
            "fs_div",
            "sj_div",
            "account_id",
            "account_nm",
            "thstrm_amount",
        ],
        "재무",
    )
    validate_tickers(frame)
    dates = pd.to_datetime(frame.rcept_dt, format="%Y-%m-%d", errors="raise")
    if dates.isna().any():
        raise ValueError("재무 접수일 누락")


def load_sns_events(path):
    """SNS 원본의 전체 언급량으로 기존 SNS 전처리와 동일한 이벤트 기간을 계산한다."""
    frame = read_csv(path)
    require(frame, ["keyword", "date", MENTION_TOTAL], "SNS")
    frame = frame[["keyword", "date", MENTION_TOTAL]].copy()
    if frame.empty or frame.keyword.str.strip().eq("").any():
        raise ValueError("SNS 데이터 또는 키워드가 비어 있습니다.")
    frame["keyword"] = frame.keyword.str.strip().replace(KEYWORD_ALIASES)
    frame["date"] = pd.to_datetime(
        frame.date.str.strip().str.replace(".", "-", regex=False),
        format="%Y-%m-%d",
        errors="raise",
    )
    if frame.date.isna().any() or frame.duplicated(["keyword", "date"]).any():
        raise ValueError("SNS 날짜 누락 또는 키워드 별칭 정규화 후 keyword+date 중복")
    frame["mentions"] = pd.to_numeric(
        frame[MENTION_TOTAL].str.replace(",", "", regex=False), errors="raise"
    )
    if (
        frame.mentions.isna().any()
        or not np.isfinite(frame.mentions).all()
        or frame.mentions.lt(0).any()
    ):
        raise ValueError("SNS 전체 언급량은 유한한 0 이상 숫자여야 합니다.")
    results = []
    for keyword, group in frame.groupby("keyword", sort=True):
        group = group.sort_values("date").reset_index(drop=True)
        expected = pd.date_range(group.date.min(), group.date.max(), freq="D")
        if len(group) != len(expected):
            raise ValueError(f"SNS 달력일이 누락되었습니다: {keyword}")
        group = calculate_event_features(calculate_mention_features(group))
        group["event_start"] = pd.NaT
        active = group.days_since_day0.notna()
        group.loc[active, "event_start"] = group.loc[active, "date"] - pd.to_timedelta(
            group.loc[active, "days_since_day0"], unit="D"
        )
        results.append(group[["keyword", "date", "event_start"]])
    return pd.concat(results, ignore_index=True)


def calculate_theme_features(stock_features, events):
    """이벤트 안에서 관측된 일별 초과수익률만으로 확산·대장주 변수를 계산한다."""
    missing = sorted(set(stock_features.keyword) - set(events.keyword))
    if missing:
        raise ValueError(f"SNS에 대응하는 키워드가 없습니다: {', '.join(missing)}")
    calendar = pd.DatetimeIndex(stock_features.date.unique()).sort_values()
    event_groups = {
        key: group.set_index("date").event_start
        for key, group in events.groupby("keyword")
    }
    records = []
    for keyword, group in stock_features.groupby("keyword", sort=True):
        tickers = sorted(group.ticker.unique())
        if len(tickers) != 10 or group.duplicated(["ticker", "date"]).any():
            raise ValueError(
                f"{keyword}: 테마 변수는 중복 없이 고정된 10종목이 필요합니다."
            )
        excess = group.pivot(
            index="date", columns="ticker", values="excess_return_1d"
        ).reindex(calendar)
        event_starts = event_groups[keyword].reindex(calendar)
        current_event = None
        leader = leader_position = None
        leader_uncertain = False
        ever_up = {}
        for position, (day, event_start) in enumerate(event_starts.items()):
            values = dict.fromkeys(THEME_FEATURES, np.nan)
            if pd.isna(event_start):
                current_event = None
            else:
                if current_event is None or event_start != current_event:
                    current_event = event_start
                    leader = leader_position = None
                    leader_uncertain = False
                    ever_up = dict.fromkeys(tickers, False)
                returns = excess.loc[day]
                hits = []
                for ticker in tickers:
                    value = returns[ticker]
                    if pd.isna(value):
                        if ever_up[ticker] is False:
                            ever_up[ticker] = None
                    elif value >= THEME_UP_THRESHOLD - 1e-12:
                        ever_up[ticker] = True
                        hits.append((ticker, value))
                if all(value is not None for value in ever_up.values()):
                    values["theme_up_ratio"] = sum(ever_up.values()) / len(tickers)
                if leader is None:
                    # 최초 상승 이전/당일 누락이 있으면 누가 먼저였는지 확정할 수 없다.
                    leader_uncertain = leader_uncertain or returns.isna().any()
                    if hits and not leader_uncertain:
                        leader = sorted(hits, key=lambda item: (-item[1], item[0]))[0][
                            0
                        ]
                        leader_position = position
                if leader is not None:
                    values["leader_return"] = returns[leader]
                    values["days_since_leader"] = position - leader_position
            records.append({"keyword": keyword, "date": day, **values})
    result = pd.DataFrame(records, columns=["keyword", "date"] + THEME_FEATURES)
    result[THEME_FEATURES] = result[THEME_FEATURES].astype(float)
    return result


def coverage(path, raw, start, end):
    manifest = Path(path).parent / "opendart_collection.json"
    stocks = set(raw.ticker)
    if manifest.exists():
        meta = json.loads(manifest.read_text(encoding="utf-8"))
        stocks = {stock["ticker"] for stock in meta["stocks"]}
        actual_start = pd.Timestamp(meta["disclosure_start"])
        actual_end = pd.Timestamp(meta["disclosure_end"])
    else:
        actual_start = pd.Timestamp("2024-09-04")
        actual_end = raw.rcept_dt.max()
    begin = max(pd.Timestamp("2024-09-04"), actual_start, pd.Timestamp(start))
    finish = pd.Timestamp(end) if end else actual_end
    if pd.isna(finish) or finish < begin or finish > actual_end:
        raise ValueError(
            "공시 가용 종료일을 확인하세요. 실제 수집 종료일을 초과할 수 없습니다."
        )
    return begin, finish, stocks


def disclosure_features(raw, calendar, begin, end, available=True):
    result = pd.DataFrame(np.nan, index=calendar, columns=list(DISCLOSURE_WINDOWS))
    if not available:
        return result
    events = pd.DataFrame(0.0, index=calendar, columns=list(DISCLOSURE_WINDOWS))
    if not raw.empty:
        flags = pd.DataFrame(
            [classify_title(title) for title in raw.report_nm], index=raw.index
        )
        flags["disclosure_count_3d"] = 1
        positions = calendar.searchsorted(raw.rcept_dt)
        for pos, values in zip(
            positions, flags[list(DISCLOSURE_WINDOWS)].to_numpy(dtype=float)
        ):
            if pos < len(calendar):
                events.iloc[pos] += values
    for col, window in DISCLOSURE_WINDOWS.items():
        values = events[col].rolling(window, min_periods=window).sum()
        # (직전 N거래일의 마지막 날짜, t] 구간에 포함된 휴장일 공시도 센다.
        valid = pd.Series(False, index=calendar)
        if len(calendar) > window:
            valid.iloc[window:] = (
                calendar[:-window] >= begin - pd.Timedelta(days=1)
            ) & (calendar[window:] <= end)
        values = values.where(valid)
        result[col] = (
            values
            if col == "disclosure_count_3d"
            else (values > 0).astype(float).where(values.notna())
        )
    return result


def build_datasets(stock, raw, begin, end, covered, sns_events=None):
    calendar = pd.DatetimeIndex(stock.date.unique()).sort_values()
    market_returns = {}
    for market, group in stock.groupby("market"):
        close = (
            group.drop_duplicates("date")
            .set_index("date")
            .market_index_close.reindex(calendar)
        )
        returns = {n: close.pct_change(n, fill_method=None) for n in [1, 3, 20]}
        if "market_return_1d" in group:
            supplied = group.set_index("date").market_return_1d
            expected = returns[1].reindex(supplied.index)
            valid = supplied.notna() & expected.notna()
            if not np.isclose(
                supplied[valid], expected[valid], rtol=1e-5, atol=1e-8
            ).all():
                raise ValueError("market_return_1d가 지수로 계산한 수익률과 다릅니다.")
        market_returns[market] = returns
    parts = []
    for ticker, group in stock.groupby("ticker", sort=True):
        market = group.market.iloc[0]
        classes = set(raw.loc[raw.ticker.eq(ticker), "corp_cls"])
        if classes and classes != {"Y" if market == "KOSPI" else "K"}:
            raise ValueError(f"{ticker}: 공시 corp_cls와 주가 market 불일치")
        prices = (
            group.drop_duplicates("date").set_index("date")[OHLCV].reindex(calendar)
        )
        invalid = prices.high.lt(prices.low)
        prices.loc[invalid, OHLCV] = np.nan
        result = pd.DataFrame(index=calendar)
        daily_return = prices.close.pct_change(fill_method=None)
        for n in [1, 3, 20]:
            result[f"excess_return_{n}d"] = (
                prices.close.pct_change(n, fill_method=None) - market_returns[market][n]
            )
        previous = prices.volume.shift(1).rolling(5, min_periods=5).mean()
        result["volume_ratio_5d"] = prices.volume / previous.where(previous > 0)
        result["volatility_20d"] = daily_return.rolling(20, min_periods=20).std(ddof=0)
        result["drawdown_20d"] = (
            prices.close / prices.close.rolling(20, min_periods=20).max() - 1
        )
        result = result.join(
            disclosure_features(
                raw[raw.ticker.eq(ticker)], calendar, begin, end, ticker in covered
            )
        )
        result[THEME_FEATURES] = np.nan
        result.index.name = "date"
        result = result.reset_index()
        result.insert(0, "ticker", ticker)
        parts.append(result)
    features = pd.concat(parts, ignore_index=True)
    linked = stock[META + ["ticker", "name", "date"]].merge(
        features, on=["ticker", "date"], how="left", validate="many_to_one"
    )
    if sns_events is not None:
        theme = calculate_theme_features(linked, sns_events)
        linked = linked.drop(columns=THEME_FEATURES).merge(
            theme, on=["keyword", "date"], how="left", validate="many_to_one"
        )
    linked = (
        linked[META + ["ticker", "name", "date"] + FEATURES]
        .sort_values(["keyword", "ticker", "date"])
        .reset_index(drop=True)
    )
    records = []
    for keys, group in linked.groupby(META + ["date"], sort=True):
        row = dict(zip(META + ["date"], keys))
        for col in FEATURES:
            if len(group) != 10 or group[col].isna().any():
                row[col] = np.nan
            elif col in DISCLOSURE_WINDOWS:
                row[col] = int(group[col].gt(0).sum())
            else:
                row[col] = group[col].mean()
        records.append(row)
    return linked, pd.DataFrame(records, columns=META + ["date"] + FEATURES)


def save_csv(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=".tmp", delete=False
    ) as file:
        temporary = Path(file.name)
    try:
        frame.to_csv(
            temporary, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d"
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stock", type=Path, default=ROOT / "data/raw/theme_stock_daily.csv"
    )
    parser.add_argument(
        "--disclosures",
        type=Path,
        default=ROOT / "data/raw/opendart/opendart_disclosures.csv",
    )
    parser.add_argument(
        "--financials",
        type=Path,
        default=ROOT / "data/raw/opendart/opendart_financials.csv",
    )
    parser.add_argument(
        "--sns", type=Path, default=SNS_INPUT, help="SNS 키워드별 일별 언급량 raw CSV"
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--disclosure-start", default="2024-09-04")
    parser.add_argument(
        "--disclosure-end", help="manifest가 없으면 마지막 공시일까지만 가용하다고 간주"
    )
    args = parser.parse_args(argv)
    try:
        stock = load_stock(args.stock)
        raw = load_disclosures(args.disclosures)
        validate_financials(args.financials)
        sns_events = load_sns_events(args.sns)
        begin, end, covered = coverage(
            args.disclosures, raw, args.disclosure_start, args.disclosure_end
        )
        outputs = build_datasets(stock, raw, begin, end, covered, sns_events)
        for name, frame in zip(
            ["market_stock_features.csv", "market_keyword_features.csv"], outputs
        ):
            save_csv(frame, args.output_dir / name)
            print(
                f"{name}: {len(frame):,}행, {frame.date.min():%Y-%m-%d} ~ {frame.date.max():%Y-%m-%d}"
            )
            print("결측 건수:", frame[FEATURES].isna().sum().to_dict())
        print(
            "테마 변수 3개는 SNS 화제 기간과 일별 초과수익률로 계산했습니다. 이벤트 밖·대장주 미확정 구간은 NA입니다."
        )
        print(
            "대장주는 이벤트별 선행 상승 종목이며, 후보 CSV의 고정 is_leader 표시는 사용하지 않습니다."
        )
        print("재무 CSV는 검증에만 사용했습니다.")
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"시장 변수 생성 실패: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
