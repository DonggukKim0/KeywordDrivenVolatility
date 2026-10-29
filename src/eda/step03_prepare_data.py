"""STEP03 EDA의 종목·거래일 및 키워드·달력일 데이터를 준비한다.

팀원이 제공한 노트북의 계산식과 시점 기준을 유지한다. 입력은 읽기만 하며,
미래 수익률·붕괴 결과·사건 표본·그림은 이후 EDA에서 생성한다.

실행 (프로젝트 루트 이외의 위치에서도 실행 가능):
    python3 src/eda/step03_prepare_data.py
    python3 src/eda/step03_prepare_data.py --output /tmp/step03

출력: stock_daily.csv.gz, sns_daily.csv.gz, data_dictionary.csv,
      feature_comparison.csv, preparation_metadata.json
"""

import argparse
import hashlib
import json
import platform
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "data/processed/eda/step03"
DEFAULT_INPUTS = {
    "market_features": ROOT / "data/processed/market_stock_features.csv",
    "sns_features": ROOT / "data/processed/sns_keyword_features.csv",
    "stock_raw": ROOT / "data/raw/theme_stock_daily.csv",
    "sns_raw": ROOT / "data/raw/sns_daily_mentions_sentiment.csv",
    "disclosures": ROOT / "data/raw/opendart/opendart_disclosures.csv",
    "financials": ROOT / "data/raw/opendart/opendart_financials.csv",
}
SNS_WINDOWS = (3, 7)  # SNS는 주말을 포함한 달력일, 가격은 거래일
CHANNELS = [
    "커뮤니티",
    "네이버 카페",
    "다음 카페",
    "인스타그램",
    "유튜브",
    "블로그",
    "티스토리",
    "네이버 뉴스",
    "다음 뉴스",
    "언론사 뉴스",
]
CHANNEL_COLS = [f"언급량 추이__{name}" for name in CHANNELS]
NEWS_COLS = [f"언급량 추이__{name}" for name in CHANNELS[-3:]]
SENTIMENT_COLS = ["감성__긍정", "감성__부정", "감성__중립"]
META = ["category", "keyword", "theme", "ticker", "name", "date"]
PRICE_FILE_REBUILD = [
    "excess_return_1d",
    "excess_return_3d",
    "excess_return_20d",
    "volume_ratio_5d",
    "volatility_20d",
    "drawdown_20d",
]
SNS_DERIVED = [
    "attention_decrease_3",
    "attention_decrease_7",
    "negative_change_3",
    "negative_change_7",
    "negative_share_3",
    "negative_share_7",
    "sentiment_min_3",
    "sentiment_min_7",
    "mentions",
    "news_mentions",
    "social_mentions",
    "sentiment_total",
    "mean_mentions_7",
    "log_mean_mentions_7",
    "nonnews_decrease_7",
]


def log(message):
    print(f"[STEP03 prepare {time.strftime('%H:%M:%S')}] {message}", flush=True)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_input(path):
    frame = pd.read_csv(
        path,
        encoding="utf-8-sig",
        low_memory=False,
        dtype={"ticker": str, "rcept_no": str, "corp_code": str},
    )
    if frame.empty:
        raise ValueError(f"빈 입력 파일: {path}")
    for column in ("date", "rcept_dt"):
        if column in frame:
            text = (
                frame[column].astype(str).str.strip().str.replace(".", "-", regex=False)
            )
            frame[column] = pd.to_datetime(text, format="%Y-%m-%d", errors="raise")
            if frame[column].isna().any():
                raise ValueError(f"날짜 결측: {path}, {column}")
    if "keyword" in frame:
        frame["keyword"] = (
            frame.keyword.str.normalize("NFC")
            .str.strip()
            .replace({"딥시크 (DeepSeek)": "딥시크"})
        )
    if "ticker" in frame and not frame.ticker.str.fullmatch(r"\d{6}", na=False).all():
        raise ValueError(f"종목코드는 6자리 문자열이어야 합니다: {path}")
    if "ticker" in frame and "date" in frame:
        frame = frame.sort_values(["ticker", "date"]).reset_index(drop=True)
    elif "keyword" in frame and "date" in frame:
        frame = frame.sort_values(["keyword", "date"]).reset_index(drop=True)
    return frame


def require(frame, columns, label):
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{label} 필수 열 누락: {sorted(missing)}")


def unique_key(frame, keys, label):
    if frame[keys].isna().any().any() or frame.duplicated(keys).any():
        raise ValueError(f"{label}: {keys}에 결측 또는 중복이 있습니다.")


def build_price_features(raw):
    frame = raw.copy()
    group = frame.groupby("ticker", sort=False)
    frame["trade_index"] = group.cumcount()
    frame["return_1"] = group.close.pct_change(fill_method=None)
    frame["return_3"] = group.close.pct_change(3, fill_method=None)
    frame["log_return_1"] = np.log(frame.close / group.close.shift(1))
    frame["volatility_20"] = frame.groupby("ticker").log_return_1.transform(
        lambda x: x.rolling(20).std()
    )
    frame["market_return_1"] = group.market_index_close.pct_change(fill_method=None)
    frame["market_return_5"] = group.market_index_close.pct_change(5, fill_method=None)
    recent = group.volume.transform(lambda x: x.rolling(5).mean())
    prior = group.volume.transform(lambda x: x.shift(5).rolling(20).mean())
    frame["volume_log_ratio_5"] = np.log1p(recent) - np.log1p(prior)
    # 기존 RQ3와 같은 사건 모집단: 5/10/20일 후보의 사전 60일 변동성이 모두 계산 가능한 구간.
    frame["common_eligible"] = True
    for window in (5, 10, 20):
        frame[f"rise_{window}"] = group.close.pct_change(window, fill_method=None)
        frame[f"prior_sd_{window}"] = frame.groupby("ticker").log_return_1.transform(
            lambda x: x.shift(window).rolling(60).std()
        )
        frame["common_eligible"] &= frame[f"rise_{window}"].notna() & frame[
            f"prior_sd_{window}"
        ].gt(0)
    return frame


def build_sns_features(raw):
    frame = raw.copy()
    frame["mentions"] = frame["언급량 추이__전체"].astype(float)
    frame["news_mentions"] = frame[NEWS_COLS].sum(axis=1)
    frame["social_mentions"] = frame.mentions - frame.news_mentions
    frame["negative_count"] = frame["감성__부정"].astype(float)
    frame["sentiment_total"] = frame[SENTIMENT_COLS].sum(axis=1)
    group = frame.groupby("keyword", sort=False)
    for window in SNS_WINDOWS:
        current = group.mentions.transform(lambda x: x.rolling(window).mean())
        previous = group.mentions.transform(
            lambda x: x.shift(window).rolling(window).mean()
        )
        frame[f"attention_decrease_{window}"] = np.log1p(previous) - np.log1p(current)
        nr = group.negative_count.transform(lambda x: x.rolling(window).sum())
        npast = group.negative_count.transform(
            lambda x: x.shift(window).rolling(window).sum()
        )
        dr = group.sentiment_total.transform(lambda x: x.rolling(window).sum())
        dp = group.sentiment_total.transform(
            lambda x: x.shift(window).rolling(window).sum()
        )
        frame[f"negative_share_{window}"] = nr / dr.where(dr.gt(0))
        frame[f"negative_change_{window}"] = frame[
            f"negative_share_{window}"
        ] - npast / dp.where(dp.gt(0))
        frame[f"sentiment_min_{window}"] = np.minimum(dr, dp)
    frame["mean_mentions_7"] = group.mentions.transform(lambda x: x.rolling(7).mean())
    frame["log_mean_mentions_7"] = np.log1p(frame.mean_mentions_7)
    frame["nonnews_decrease_7"] = np.log1p(
        group.social_mentions.transform(lambda x: x.shift(7).rolling(7).mean())
    ) - np.log1p(group.social_mentions.transform(lambda x: x.rolling(7).mean()))
    return frame


def rebuild_supplied_sns(raw, feature_columns):
    parts = []
    for _, source in raw.groupby("keyword", sort=False):
        g = source.copy()
        m = g["언급량 추이__전체"].astype(float)
        total = g[SENTIMENT_COLS].sum(axis=1)
        previous28 = m.shift(1).rolling(28)
        g["mention_ratio28"] = m / previous28.mean().where(previous28.mean().gt(0))
        g["mention_z28"] = (m - previous28.mean()) / previous28.std(ddof=0).where(
            previous28.std(ddof=0).gt(0)
        )
        previous14 = m.shift(1).rolling(14).mean()
        g["mention_momentum_3_14"] = m.rolling(3).mean() / previous14.where(
            previous14.gt(0)
        )
        g["log_mentions"] = np.log1p(m)
        for name, col in [("negative", "감성__부정"), ("positive", "감성__긍정")]:
            g[f"{name}_share"] = g[col] / total.where(total.gt(0))
            g[f"{name}_share_vs_prior7"] = (
                g[f"{name}_share"] - g[f"{name}_share"].shift(1).rolling(7).mean()
            )
        shares = g[CHANNEL_COLS].div(m.where(m.gt(0)), axis=0)
        terms = shares * np.log(shares.where(shares.gt(0)))
        entropy = (-terms.fillna(0).sum(axis=1) / np.log(len(CHANNELS))).where(m.gt(0))
        g["entropy_vs_prior7"] = entropy - entropy.shift(1).rolling(7).mean()
        hot = (g.mention_ratio28.ge(3) & m.ge(20)).to_numpy()
        start = None
        cold_days, peak_index, peak_value = 0, 0, 0.0
        age, peak_age = [], []
        for i, value in enumerate(m.to_numpy()):
            if start is None and hot[i]:
                start, cold_days, peak_index, peak_value = i, 0, i, value
            elif start is not None:
                cold_days = 0 if hot[i] else cold_days + 1
                if value >= peak_value:
                    peak_index, peak_value = i, value
                if cold_days >= 7:
                    start = None
            age.append(np.nan if start is None else i - start)
            peak_age.append(np.nan if start is None else i - peak_index)
        g["days_since_day0"], g["days_since_peak"] = age, peak_age
        g["in_event"] = g.days_since_day0.notna().astype(int)
        g["low_sentiment"] = total.lt(10).astype(int)
        parts.append(g[["keyword", "date"] + feature_columns])
    return pd.concat(parts, ignore_index=True)


def compare_columns(given, rebuilt, keys, columns, label):
    joined = given[keys + columns].merge(
        rebuilt[keys + columns],
        on=keys,
        suffixes=("_given", "_rebuilt"),
        validate="one_to_one",
        how="left",
        indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        raise ValueError(f"{label}: 원자료에 없는 제공 피처 행이 있습니다.")
    rows = []
    for col in columns:
        a, b = joined[f"{col}_given"], joined[f"{col}_rebuilt"]
        same = np.isclose(a, b, rtol=1e-7, atol=1e-9, equal_nan=True)
        rows.append(
            {
                "자료": label,
                "변수": col,
                "비교 행": len(joined),
                "불일치 행": int((~same).sum()),
                "최대 절대차": (a - b).abs().max(),
            }
        )
    return pd.DataFrame(rows)


def build_filing_context(prices, disclosures, financials):
    title = (
        disclosures.report_nm.fillna("").astype(str).str.replace(r"\s+", "", regex=True)
    )
    flags = pd.DataFrame(
        {
            "dart_seen_count_3d": np.ones(len(disclosures)),
            "dart_seen_contract_3d": (
                title.str.contains(r"단일판매[ㆍ·・/]?공급계약체결")
                & ~title.str.contains("해지")
            ).astype(float),
            "dart_seen_dilution_20d": title.str.contains(
                "유상증자|유무상증자|전환사채|신주인수권부사채|교환사채"
            ).astype(float),
            "dart_seen_clarification_5d": (
                title.str.contains("풍문|보도|소문")
                & title.str.contains("해명|조회공시.*답변")
            ).astype(float),
        }
    )
    windows = dict(zip(flags.columns, [3, 3, 20, 5]))
    d = pd.concat([disclosures[["ticker", "rcept_dt"]], flags], axis=1)
    financial_receipts = financials[
        ["ticker", "rcept_no", "rcept_dt"]
    ].drop_duplicates()
    unique_key(financial_receipts, ["ticker", "rcept_no"], "재무 보고서 접수일")
    parts = []
    for ticker, rows in prices.groupby("ticker", sort=False):
        dates = pd.DatetimeIndex(rows.date)
        g = pd.DataFrame({"ticker": ticker, "date": dates})
        raw = d[d.ticker.eq(ticker)]
        positions = dates.searchsorted(raw.rcept_dt, side="right")
        valid = positions < len(dates)
        for col, window in windows.items():
            daily = np.zeros(len(dates))
            np.add.at(daily, positions[valid], raw[col].to_numpy()[valid])
            rolled = pd.Series(daily).rolling(window, min_periods=window).sum()
            # 첫 거래일 이전 접수는 위치 0에 모이므로, 위치 0을 포함한 불완전한 창은 사용하지 않습니다.
            g[col] = rolled.where(np.arange(len(dates)) >= window)
        received = pd.DatetimeIndex(
            financial_receipts.loc[
                financial_receipts.ticker.eq(ticker), "rcept_dt"
            ].unique()
        ).sort_values()
        prior = received.searchsorted(dates, side="left") - 1
        g["financial_last_receipt_date"] = [
            received[i] if i >= 0 else pd.NaT for i in prior
        ]
        g["financial_days_since_receipt"] = (
            g.date - g.financial_last_receipt_date
        ).dt.days
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def validate_inputs(inputs):
    market, sns = inputs["market_features"], inputs["sns_features"]
    stock, raw = inputs["stock_raw"], inputs["sns_raw"]
    disclosures, financials = inputs["disclosures"], inputs["financials"]
    require(
        stock, META + ["close", "volume", "market", "market_index_close"], "주가 원자료"
    )
    require(market, ["ticker", "date", "keyword"] + PRICE_FILE_REBUILD, "시장 피처")
    require(sns, ["keyword", "date"], "SNS 피처")
    require(
        raw,
        ["keyword_id", "keyword", "date", "언급량 추이__전체"]
        + CHANNEL_COLS
        + SENTIMENT_COLS,
        "SNS 원자료",
    )
    require(disclosures, ["ticker", "rcept_no", "rcept_dt", "report_nm"], "공시 원자료")
    require(
        financials,
        ["ticker", "rcept_no", "rcept_dt", "bsns_year", "fs_div", "sj_div"],
        "재무 원자료",
    )
    for label, frame, keys in [
        ("주가", stock, ["ticker", "date"]),
        ("시장 피처", market, ["ticker", "date"]),
        ("SNS 원자료", raw, ["keyword", "date"]),
        ("SNS 피처", sns, ["keyword", "date"]),
        ("공시", disclosures, ["ticker", "rcept_no"]),
    ]:
        unique_key(frame, keys, label)
    if not stock.groupby("ticker").keyword.nunique().eq(1).all():
        raise ValueError(
            "종목별 테마가 여러 개입니다. 사건 중복 방지 규칙을 먼저 정해야 합니다."
        )
    if set(stock.keyword) != set(raw.keyword) or set(stock.keyword) != set(sns.keyword):
        raise ValueError("주가와 SNS의 테마 집합이 다릅니다. 별칭표를 확인하세요.")
    calendar = pd.DatetimeIndex(stock.date.unique()).sort_values()
    for ticker, group in stock.groupby("ticker"):
        if not pd.DatetimeIndex(group.date).equals(calendar):
            raise ValueError(
                f"{ticker}: 거래일 누락이 있습니다. 누락을 채우지 않습니다."
            )
    for keyword, group in raw.groupby("keyword"):
        expected = pd.date_range(group.date.min(), group.date.max(), freq="D")
        if not pd.DatetimeIndex(group.date).equals(expected):
            raise ValueError(f"{keyword}: SNS 달력일이 누락되었습니다.")
    market_columns = [c for c in market if c not in META]
    sns_columns = [c for c in sns if c not in ["keyword_id", "keyword", "date"]]
    for frame, columns in [
        (stock, ["close", "volume", "market_index_close"]),
        (raw, CHANNEL_COLS + SENTIMENT_COLS + ["언급량 추이__전체"]),
        (market, market_columns),
        (sns, sns_columns),
    ]:
        for col in columns:
            frame[col] = pd.to_numeric(frame[col], errors="raise")
            if np.isinf(frame[col]).any():
                raise ValueError(f"무한대 입력: {col}")
    if not (
        stock.close.gt(0) & stock.market_index_close.gt(0) & stock.volume.ge(0)
    ).all():
        raise ValueError("종가·지수는 양수, 거래량은 0 이상이어야 합니다.")
    counts = raw[CHANNEL_COLS + SENTIMENT_COLS + ["언급량 추이__전체"]]
    if counts.isna().any().any() or counts.lt(0).any().any():
        raise ValueError("SNS 건수에 결측 또는 음수가 있습니다.")
    if not raw[CHANNEL_COLS].sum(axis=1).eq(raw["언급량 추이__전체"]).all():
        raise ValueError("SNS 채널별 합계와 전체 언급량이 다릅니다.")
    return market_columns, sns_columns


def audit_features(inputs, prices, sns_columns):
    rebuilt_sns = rebuild_supplied_sns(inputs["sns_raw"], sns_columns)
    rebuilt_price = prices[["ticker", "date"]].copy()
    group = prices.groupby("ticker", sort=False)
    for window in (1, 3, 20):
        rebuilt_price[f"excess_return_{window}d"] = group.close.pct_change(
            window, fill_method=None
        ) - group.market_index_close.pct_change(window, fill_method=None)
    prior_volume = group.volume.transform(lambda x: x.shift(1).rolling(5).mean())
    rebuilt_price["volume_ratio_5d"] = prices.volume / prior_volume.where(
        prior_volume.gt(0)
    )
    rebuilt_price["volatility_20d"] = group.return_1.transform(
        lambda x: x.rolling(20).std(ddof=0)
    )
    rebuilt_price["drawdown_20d"] = (
        prices.close / group.close.transform(lambda x: x.rolling(20).max()) - 1
    )
    return pd.concat(
        [
            compare_columns(
                inputs["sns_features"],
                rebuilt_sns,
                ["keyword", "date"],
                sns_columns,
                "SNS 피처",
            ),
            compare_columns(
                inputs["market_features"],
                rebuilt_price,
                ["ticker", "date"],
                PRICE_FILE_REBUILD,
                "시장 피처",
            ),
        ],
        ignore_index=True,
    )


def join_stock_daily(
    prices, sns_daily, inputs, filing_context, market_columns, sns_columns
):
    market_join = inputs["market_features"][
        ["ticker", "date", "keyword"] + market_columns
    ].rename(
        columns={
            "keyword": "market_keyword",
            **{c: f"market_file__{c}" for c in market_columns},
        }
    )
    base = prices.merge(
        market_join,
        on=["ticker", "date"],
        how="left",
        validate="one_to_one",
        indicator="market_join",
    )
    if (
        not base.market_join.eq("both").all()
        or not base.keyword.eq(base.market_keyword).all()
    ):
        raise ValueError(
            "시장 피처의 종목·날짜 또는 테마가 원자료와 일치하지 않습니다."
        )
    base["has_market_features"] = base.market_join.eq("both")
    base = base.drop(columns=["market_join", "market_keyword"])
    base["sns_cutoff_date"] = base.date - pd.Timedelta(days=1)
    raw_join = sns_daily[["keyword", "date"] + SNS_DERIVED].rename(
        columns={"date": "sns_cutoff_date"}
    )
    raw_join["sns_source_date"] = raw_join.sns_cutoff_date
    base = base.merge(
        raw_join, on=["keyword", "sns_cutoff_date"], how="left", validate="many_to_one"
    )
    file_join = inputs["sns_features"][["keyword", "date"] + sns_columns].rename(
        columns={
            "date": "sns_cutoff_date",
            **{c: f"sns_file__{c}" for c in sns_columns},
        }
    )
    file_join["sns_feature_source_date"] = file_join.sns_cutoff_date
    base = base.merge(
        file_join, on=["keyword", "sns_cutoff_date"], how="left", validate="many_to_one"
    )
    base = base.merge(
        filing_context, on=["ticker", "date"], how="left", validate="one_to_one"
    )
    base["has_sns_raw"] = base.sns_source_date.notna()
    base["has_sns_features"] = base.sns_feature_source_date.notna()
    return base.sort_values(["ticker", "date"]).reset_index(drop=True)


def check_prepared(base, sns_daily, inputs):
    unique_key(base, ["ticker", "date"], "STEP03 종목 기본 데이터")
    unique_key(sns_daily, ["keyword", "date"], "STEP03 SNS 일별 데이터")
    if len(base) != len(inputs["stock_raw"]):
        raise ValueError("기본 데이터 결합 후 주가 행 수가 달라졌습니다.")
    if any(c.startswith(("future_", "next_day_", "event_label")) for c in base):
        raise ValueError("기본 데이터에 미래 결과 열이 포함되어 있습니다.")
    for col in ["sns_source_date", "sns_feature_source_date"]:
        observed = base[col].notna()
        if not base.loc[observed, col].eq(base.loc[observed, "sns_cutoff_date"]).all():
            raise ValueError(f"SNS 전날 연결 오류: {col}")
    known_receipt = base.financial_last_receipt_date.notna()
    if (
        not base.loc[known_receipt, "financial_last_receipt_date"]
        .lt(base.loc[known_receipt, "date"])
        .all()
    ):
        raise ValueError("관측일 당일 또는 이후의 재무 접수일이 연결되었습니다.")
    checked = 0
    valid = base.dropna(subset=["negative_change_7", "attention_decrease_7"])
    for row in valid.sample(min(40, len(valid)), random_state=20261027).itertuples():
        raw = inputs["sns_raw"]
        history = raw[
            raw.keyword.eq(row.keyword) & raw.date.le(row.sns_cutoff_date)
        ].tail(14)
        if len(history) != 14:
            raise ValueError("SNS 7일 비교 표본에 14일 관측이 없습니다.")
        previous, recent = history.iloc[:7], history.iloc[7:]
        attention = np.log1p(previous["언급량 추이__전체"].mean()) - np.log1p(
            recent["언급량 추이__전체"].mean()
        )
        negative = (
            recent["감성__부정"].sum() / recent[SENTIMENT_COLS].to_numpy().sum()
            - previous["감성__부정"].sum() / previous[SENTIMENT_COLS].to_numpy().sum()
        )
        if not (
            np.isclose(row.attention_decrease_7, attention)
            and np.isclose(row.negative_change_7, negative)
        ):
            raise ValueError(f"SNS 독립 재계산 불일치: {row.keyword}, {row.date}")
        checked += 1
    return checked


def source_profile(inputs):
    rows = []
    for key, frame in inputs.items():
        date_col = "date" if "date" in frame else "rcept_dt"
        rows.append(
            {
                "source": key,
                "rows": len(frame),
                "columns": len(frame.columns),
                "start": frame[date_col].min().strftime("%Y-%m-%d"),
                "end": frame[date_col].max().strftime("%Y-%m-%d"),
                "stocks": int(frame.ticker.nunique()) if "ticker" in frame else None,
                "keywords": int(frame.keyword.nunique())
                if "keyword" in frame
                else None,
            }
        )
    return rows


def variable_definitions(filing_columns):
    definitions = {
        "trade_index": "종목 내 거래일 순서, 0부터 시작",
        "return_1": "종가 / 전 거래일 종가 - 1",
        "return_3": "종가 / 3거래일 전 종가 - 1",
        "log_return_1": "log(종가 / 전 거래일 종가)",
        "volatility_20": "최근 20거래일 로그수익률 표준편차, ddof=1",
        "market_return_1": "시장지수 / 전 거래일 지수 - 1",
        "market_return_5": "시장지수 / 5거래일 전 지수 - 1",
        "volume_log_ratio_5": "log1p(최근 5거래일 평균 거래량) - log1p(그 앞 20거래일 평균 거래량)",
        "common_eligible": "5/10/20일 상승률과 각각 이전 60일 변동성 계산 가능, 변동성 > 0",
        "mentions": "뉴스 포함 전체 언급량",
        "news_mentions": "뉴스 3채널 언급량 합",
        "social_mentions": "뉴스를 제외한 7채널 언급량 합",
        "sentiment_total": "긍정+부정+중립 감성어 건수",
        "mean_mentions_7": "최근 7달력일 평균 언급량",
        "log_mean_mentions_7": "log1p(최근 7일 평균 언급량)",
        "nonnews_decrease_7": "뉴스 제외 앞선 7일 평균의 log1p - 최근 7일 평균의 log1p",
        "sns_cutoff_date": "주가 관측일의 전날 달력일",
        "sns_source_date": "실제로 연결된 SNS 원자료 날짜",
        "sns_feature_source_date": "실제로 연결된 제공 SNS 피처 날짜",
        "financial_last_receipt_date": "관측일보다 이른 최근 재무 보고서 접수일",
        "financial_days_since_receipt": "관측일 - 최근 재무 보고서 접수일, 달력일",
    }
    for window in (5, 10, 20):
        definitions[f"rise_{window}"] = f"종가 / {window}거래일 전 종가 - 1"
        definitions[f"prior_sd_{window}"] = (
            f"{window}거래일 전까지 직전 60개 로그수익률 표준편차, ddof=1"
        )
    for window in SNS_WINDOWS:
        definitions[f"attention_decrease_{window}"] = (
            f"앞선 {window}일 평균 언급량 log1p - 최근 {window}일 평균 log1p"
        )
        definitions[f"negative_share_{window}"] = (
            f"최근 {window}일 부정 건수 합 / 전체 감성어 건수 합"
        )
        definitions[f"negative_change_{window}"] = (
            f"최근 {window}일 부정 비율 - 앞선 {window}일 부정 비율"
        )
        definitions[f"sentiment_min_{window}"] = (
            f"최근/앞선 {window}일 전체 감성어 건수 합 중 작은 값"
        )
    for col in ["has_market_features", "has_sns_raw", "has_sns_features"]:
        definitions[col] = (
            "해당 원본 행이 연결되었는지 여부. 피처 값이 모두 유효하다는 뜻은 아님"
        )
    for col in filing_columns:
        if col.startswith("dart_seen_"):
            definitions[col] = (
                "접수일 다음 거래일부터 포함한 제목 분류 건수의 최근 거래일 합. 열명 끝의 d는 거래일 창"
            )
    definitions["negative_count"] = "부정 감성어 건수"
    return definitions


def build_dictionary(base, sns_daily, inputs, paths, filing_columns):
    definitions = variable_definitions(filing_columns)
    rows = []
    for dataset, frame, raw_key in [
        ("stock_daily", base, "stock_raw"),
        ("sns_daily", sns_daily, "sns_raw"),
    ]:
        for col in frame:
            if col.startswith("market_file__"):
                source = str(paths["market_features"])
                definition, basis = (
                    "제공 시장 피처 원값. 접두사 뒤가 원래 열명",
                    "관측 거래일",
                )
            elif col.startswith("sns_file__"):
                source = str(paths["sns_features"])
                definition, basis = (
                    "제공 SNS 피처 원값. 접두사 뒤가 원래 열명",
                    "전날 달력일",
                )
            elif col in inputs[raw_key].columns:
                source = str(paths[raw_key])
                definition, basis = (
                    "원자료 열 그대로",
                    "관측 거래일" if dataset == "stock_daily" else "해당 달력일",
                )
            else:
                source = "step03_prepare_data.py에서 계산"
                definition = definitions.get(col)
                if definition is None:
                    raise ValueError(f"변수 정의 누락: {dataset}.{col}")
                basis = "열 정의 참조"
                if col in SNS_DERIVED or col == "negative_count":
                    basis = (
                        "전날 달력일까지"
                        if dataset == "stock_daily"
                        else "해당 달력일까지"
                    )
            rows.append(
                {
                    "dataset": dataset,
                    "column": col,
                    "dtype": str(frame[col].dtype),
                    "source": source,
                    "definition": definition,
                    "time_basis": basis,
                    "missing_rows": int(frame[col].isna().sum()),
                }
            )
    return pd.DataFrame(rows)


def main(output_dir=DEFAULT_OUTPUT, **input_overrides):
    started = time.monotonic()
    unknown = set(input_overrides) - set(DEFAULT_INPUTS)
    if unknown:
        raise ValueError(f"알 수 없는 입력 이름: {sorted(unknown)}")
    paths = {
        key: Path(input_overrides.get(key, path)).resolve()
        for key, path in DEFAULT_INPUTS.items()
    }
    log("입력 CSV 6개 읽기 및 SHA-256 기록 시작")
    hashes = {key: file_hash(path) for key, path in paths.items()}
    inputs = {key: read_input(path) for key, path in paths.items()}
    log("입력 키·날짜·수치·거래일·SNS 건수 점검")
    market_columns, sns_columns = validate_inputs(inputs)
    log("가격 및 SNS 분석 변수 생성")
    prices = build_price_features(inputs["stock_raw"])
    sns_daily = build_sns_features(inputs["sns_raw"])
    log("제공 피처와 원자료 재계산 값 비교")
    comparison = audit_features(inputs, prices, sns_columns)
    mismatches = int(comparison["불일치 행"].sum())
    if mismatches:
        warnings.warn(
            f"제공 피처 불일치 합계 {mismatches:,}건. feature_comparison.csv를 확인하세요. 원값은 보존합니다."
        )
    log("공시·재무 접수일 계산 및 종목×거래일 데이터 결합")
    filings = build_filing_context(prices, inputs["disclosures"], inputs["financials"])
    base = join_stock_daily(
        prices, sns_daily, inputs, filings, market_columns, sns_columns
    )
    log("날짜 연결 및 SNS 표본 독립 재계산 점검")
    spot_checks = check_prepared(base, sns_daily, inputs)
    dictionary = build_dictionary(base, sns_daily, inputs, paths, filings.columns)
    for key, path in paths.items():
        if file_hash(path) != hashes[key]:
            raise ValueError(f"실행 중 입력 파일이 변경되었습니다: {path}")
    missingness = (
        pd.concat(
            [
                inputs["market_features"][market_columns]
                .isna()
                .mean()
                .rename("missing_fraction")
                .to_frame()
                .assign(source="market_features"),
                inputs["sns_features"][sns_columns]
                .isna()
                .mean()
                .rename("missing_fraction")
                .to_frame()
                .assign(source="sns_features"),
            ]
        )
        .rename_axis("column")
        .reset_index()
    )
    metadata = {
        "step": "STEP03",
        "source_notebook": "RQ3_raw_EDA.ipynb",
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "inputs": {
            key: {"path": str(path), "sha256": hashes[key]}
            for key, path in paths.items()
        },
        "source_profile": source_profile(inputs),
        "rules": {
            "sns_calendar_windows": list(SNS_WINDOWS),
            "price_trading_windows": [5, 10, 20],
            "prior_volatility_window": 60,
            "stock_sns_lag_calendar_days": 1,
            "financial_receipt_strictly_before_observation": True,
            "missing_values_filled": False,
            "future_outcomes_included": False,
        },
        "outputs": {
            "stock_daily": {"rows": len(base), "columns": len(base.columns)},
            "sns_daily": {"rows": len(sns_daily), "columns": len(sns_daily.columns)},
        },
        "checks": {
            "input_validation_passed": True,
            "prepared_validation_passed": True,
            "input_hashes_unchanged": True,
            "independent_sns_spot_checks": spot_checks,
            "supplied_feature_mismatches": mismatches,
        },
        "joins": {
            col: int(base[col].sum())
            for col in ["has_market_features", "has_sns_raw", "has_sns_features"]
        },
        "feature_missingness": missingness.to_dict(orient="records"),
    }
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for frame, filename in [
        (base, "stock_daily.csv.gz"),
        (sns_daily, "sns_daily.csv.gz"),
        (dictionary, "data_dictionary.csv"),
        (comparison, "feature_comparison.csv"),
    ]:
        log(f"저장: {filename} ({len(frame):,}행 × {len(frame.columns)}열)")
        frame.to_csv(
            output_dir / filename,
            index=False,
            encoding="utf-8-sig",
            date_format="%Y-%m-%d",
        )
    metadata["elapsed_seconds"] = round(time.monotonic() - started, 2)
    (output_dir / "preparation_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    log(f"완료: {output_dir} ({time.monotonic() - started:.1f}초)")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="STEP03 EDA 기본 데이터 준비")
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT, help="출력 디렉토리"
    )
    for key, default in DEFAULT_INPUTS.items():
        parser.add_argument(
            "--" + key.replace("_", "-"),
            type=Path,
            default=default,
            help=f"{key} 입력 CSV (기본: {default.relative_to(ROOT)})",
        )
    args = vars(parser.parse_args())
    main(output_dir=args.pop("output"), **args)
