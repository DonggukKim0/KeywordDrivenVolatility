"""후보 종목의 NAVER 일별 주가와 소속 시장의 KOSPI/KOSDAQ 지수를 수집한다."""

import argparse
import sys
import tempfile
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import FinanceDataReader as fdr
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CANDIDATES = ROOT / "data/raw/theme_candidates.csv"
OUTPUT = ROOT / "data/raw/theme_stock_daily.csv"
CANDIDATE_COLUMNS = ["category", "keyword", "theme", "ticker", "name"]
PRICE_COLUMNS = ["open", "high", "low", "close", "volume"]
MARKET_COLUMNS = ["market", "market_index_close", "market_return_1d"]
MARKET_SYMBOLS = {"KOSPI": "KS11", "KOSDAQ": "KQ11"}
OUTPUT_COLUMNS = CANDIDATE_COLUMNS + ["date"] + PRICE_COLUMNS + MARKET_COLUMNS


def load_candidates(path=CANDIDATES):
    """후보 CSV의 필수 값·중복을 검증한다. 테마와 종목 수는 제한하지 않는다."""
    candidates = pd.read_csv(
        path, dtype=str, encoding="utf-8-sig", keep_default_na=False
    )
    missing = set(CANDIDATE_COLUMNS) - set(candidates.columns)
    if missing:
        raise ValueError(f"후보 CSV 필수 컬럼 누락: {', '.join(sorted(missing))}")
    candidates = candidates[CANDIDATE_COLUMNS].apply(lambda col: col.str.strip())
    if candidates.empty or candidates.eq("").any().any():
        raise ValueError("후보 CSV가 비어 있거나 필수 값이 없습니다.")
    if not candidates["ticker"].str.fullmatch(r"[0-9]{6}").all():
        raise ValueError("종목코드는 6자리 숫자 문자열이어야 합니다.")
    if candidates.duplicated(["keyword", "ticker"]).any():
        raise ValueError("후보 CSV에 keyword+ticker 중복이 있습니다.")
    if candidates.groupby("keyword")[["category", "theme"]].nunique().gt(1).any().any():
        raise ValueError("키워드마다 대분류와 테마는 하나여야 합니다.")
    if candidates.groupby("ticker")["name"].nunique().gt(1).any():
        raise ValueError("동일 종목코드에 서로 다른 종목명이 있습니다.")
    return candidates


def fetch_market_mapping(tickers):
    """KRX 현재 상장 목록으로 종목의 소속 시장을 확인한다. 추정으로 채우지 않는다."""
    for attempt in range(1, 4):
        try:
            listing = fdr.StockListing("KRX")
            if not {"Code", "Market"}.issubset(listing.columns):
                raise ValueError("KRX 상장 목록에 Code 또는 Market 열이 없습니다.")
            mapping = listing[["Code", "Market"]].copy()
            mapping.columns = ["ticker", "market"]
            mapping["ticker"] = mapping.ticker.astype(str).str.strip().str.zfill(6)
            mapping["market"] = mapping.market.astype(str).str.strip().str.upper()
            # KOSDAQ GLOBAL은 코스닥 내 구분이며 기준 지수는 KOSDAQ이다.
            mapping.loc[mapping.market.eq("KOSDAQ GLOBAL"), "market"] = "KOSDAQ"
            mapping = mapping[mapping.ticker.isin(tickers)].drop_duplicates()
            if mapping.ticker.duplicated().any():
                raise ValueError("동일 종목에 여러 시장이 연결되어 있습니다.")
            missing = sorted(set(tickers) - set(mapping.ticker))
            if missing:
                raise ValueError(f"상장 목록에 없는 종목: {', '.join(missing)}")
            unsupported = mapping.loc[
                ~mapping.market.isin(MARKET_SYMBOLS), "ticker"
            ].tolist()
            if unsupported:
                raise ValueError(f"KOSPI/KOSDAQ 외 시장 종목: {', '.join(unsupported)}")
            return mapping.reset_index(drop=True)
        except Exception as exc:
            if attempt == 3:
                raise ValueError(f"시장 구분 조회 실패: {exc}") from exc
            time.sleep(1)


def validate_market_index(frame, market, start, end):
    """직전 거래일을 포함한 지수 종가로 수익률을 계산하고 요청 기간만 반환한다."""
    if frame.empty or "Close" not in frame.columns:
        raise ValueError(f"{market} 지수 종가가 없습니다.")
    dates = pd.to_datetime(frame.index, errors="raise")
    if not isinstance(dates, pd.DatetimeIndex) or dates.isna().any():
        raise ValueError(f"{market} 지수의 날짜가 유효하지 않습니다.")
    if dates.tz is not None:
        dates = dates.tz_convert("Asia/Seoul").tz_localize(None)
    close = pd.to_numeric(frame["Close"], errors="raise").copy()
    close.index = dates.normalize()
    if close.index.duplicated().any():
        raise ValueError(f"{market} 지수 날짜가 중복되었습니다.")
    close = close.sort_index().loc[: pd.Timestamp(end)]
    if (
        close.isna().any()
        or close.isin([float("inf"), float("-inf")]).any()
        or close.le(0).any()
    ):
        raise ValueError(f"{market} 지수 종가에 결측치·무한대·0 이하 값이 있습니다.")
    result = pd.DataFrame(
        {
            "market_index_close": close,
            "market_return_1d": close.pct_change(fill_method=None),
        }
    )
    result = result.loc[pd.Timestamp(start) : pd.Timestamp(end)].copy()
    if result.empty or result.market_return_1d.isna().any():
        raise ValueError(f"{market} 지수 또는 직전 거래일 종가가 부족합니다.")
    result.index.name = "date"
    result = result.reset_index()
    result.insert(0, "market", market)
    return result


def fetch_market_indices(markets, start, end):
    """필요한 시장별 지수를 한 번씩 수집하며 실패 시 최대 3회 시도한다."""
    collected = []
    warmup_start = start - timedelta(days=30)
    for market in sorted(set(markets)):
        for attempt in range(1, 4):
            try:
                frame = fdr.DataReader(
                    MARKET_SYMBOLS[market], warmup_start.isoformat(), end.isoformat()
                )
                collected.append(validate_market_index(frame, market, start, end))
                break
            except Exception as exc:
                if attempt == 3:
                    raise ValueError(f"{market} 지수 수집 실패: {exc}") from exc
                time.sleep(1)
    return pd.concat(collected, ignore_index=True)


def attach_market_data(prices, mapping, indices):
    """종목·날짜를 보존해 시장지수를 연결하며 누락은 임의 채우지 않는다."""
    result = prices.merge(mapping, on="ticker", how="left", validate="many_to_one")
    result = result.merge(
        indices, on=["market", "date"], how="left", validate="many_to_one"
    )
    missing = result[MARKET_COLUMNS].isna().any(axis=1)
    if missing.any():
        sample = result.loc[missing, ["ticker", "date"]].head(5).to_dict("records")
        raise ValueError(f"주가 날짜에 대응하는 시장지수가 누락되었습니다: {sample}")
    if len(result) != len(prices):
        raise ValueError("시장지수 연결 후 주가 행 수가 변경되었습니다.")
    return result


def validate_prices(frame, ticker, start, end):
    """조회한 주가의 날짜·컬럼·값을 검증하고 저장할 형태로 정리한다."""
    if frame.empty:
        raise ValueError("조회 결과가 없습니다.")
    required = [column.capitalize() for column in PRICE_COLUMNS]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"주가 필수 컬럼 누락: {', '.join(sorted(missing))}")
    prices = frame[required].copy()
    prices.columns = PRICE_COLUMNS
    dates = pd.to_datetime(frame.index, errors="raise")
    if not isinstance(dates, pd.DatetimeIndex) or dates.isna().any():
        raise ValueError("유효한 날짜 인덱스가 필요합니다.")
    if dates.tz is not None:
        dates = dates.tz_convert("Asia/Seoul").tz_localize(None)
    prices.index = dates.normalize()
    prices.index.name = "date"
    prices = prices.loc[
        (prices.index >= pd.Timestamp(start)) & (prices.index <= pd.Timestamp(end))
    ]
    if prices.empty:
        raise ValueError("요청 기간에 해당하는 데이터가 없습니다.")
    if prices.index.duplicated().any():
        raise ValueError("같은 날짜의 주가가 중복되어 있습니다.")
    for column in PRICE_COLUMNS:
        prices[column] = pd.to_numeric(prices[column], errors="raise")
    if (
        prices.isna().any().any()
        or prices.isin([float("inf"), float("-inf")]).any().any()
    ):
        raise ValueError("주가 또는 거래량에 결측치/무한대가 있습니다.")
    if prices.lt(0).any().any():
        raise ValueError("가격과 거래량은 음수일 수 없습니다.")
    prices = prices.sort_index().reset_index()
    prices.insert(1, "ticker", ticker)
    return prices


def fetch_stock_prices(candidates, start, end, limit=None):
    """중복 종목을 한 번씩 조회하고, 최대 3회 시도 후 성공 데이터와 실패 목록을 반환한다."""
    tickers = candidates["ticker"].drop_duplicates().tolist()
    if limit is not None:
        tickers = tickers[:limit]
    collected, failures = [], []
    attempted = False
    for position, ticker in enumerate(tickers, 1):
        for attempt in range(1, 4):
            if attempted:
                time.sleep(1)
            attempted = True
            try:
                frame = fdr.DataReader(
                    f"NAVER:{ticker}", start.isoformat(), end.isoformat()
                )
                prices = validate_prices(frame, ticker, start, end)
            except Exception as exc:
                print(
                    f"[{position}/{len(tickers)}] {ticker} 시도 {attempt}/3 실패: {exc}",
                    flush=True,
                )
                if attempt == 3:
                    failures.append((ticker, str(exc)))
                continue
            collected.append(prices)
            print(
                f"[{position}/{len(tickers)}] {ticker}: {len(prices)}행 "
                f"({prices['date'].min():%Y-%m-%d} ~ {prices['date'].max():%Y-%m-%d})",
                flush=True,
            )
            break
    prices = pd.concat(collected, ignore_index=True) if collected else pd.DataFrame()
    return prices, failures


def merge_keywords(candidates, prices):
    """종목코드로 주가와 키워드 정보를 연결하고 중복·행 수를 확인한 뒤 정렬한다."""
    if prices.duplicated(["ticker", "date"]).any():
        raise ValueError("수집 데이터에 ticker+date 중복이 있습니다.")
    result = candidates.merge(prices, on="ticker", how="inner")[OUTPUT_COLUMNS]
    if result.duplicated(["keyword", "ticker", "date"]).any():
        raise ValueError("분석 데이터에 keyword+ticker+date 중복이 있습니다.")
    links = candidates.groupby("ticker").size()
    expected = int(prices["ticker"].map(links).sum())
    if len(result) != expected:
        raise ValueError("키워드 연결 후 행 수가 예상과 다릅니다.")
    return result.sort_values(["keyword", "ticker", "date"]).reset_index(drop=True)


def save_output(result, path=OUTPUT):
    """임시 파일에 CSV 저장을 마친 뒤 결과 파일을 교체해 기존 데이터 손상을 방지한다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, suffix=".tmp", delete=False
        ) as file:
            temporary = Path(file.name)
        result.to_csv(
            temporary, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d"
        )
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def parse_date(value):
    """명령행의 YYYY-MM-DD 문자열을 날짜로 변환하고 형식 오류를 알린다."""
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "날짜는 YYYY-MM-DD 형식이어야 합니다."
        ) from exc


def positive_int(value):
    """명령행의 종목 수 제한 값을 1 이상의 정수로 검증한다."""
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("1 이상의 정수를 입력하세요.")
    return number


def main(argv=None):
    """실행 옵션에 따라 수집부터 CSV 저장까지 수행하고 결과에 맞는 종료 코드를 반환한다."""
    yesterday = datetime.now(ZoneInfo("Asia/Seoul")).date() - timedelta(days=1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--start",
        type=parse_date,
        default=date(2024, 10, 1),
        help="수집 시작일 (YYYY-MM-DD, 기본: 2024-10-01)",
    )
    parser.add_argument(
        "--end",
        type=parse_date,
        default=date(2026, 10, 24),
        help="수집 종료일 (YYYY-MM-DD, 기본: 2026-10-24)",
    )
    parser.add_argument("--limit", type=positive_int, help="테스트용 최대 고유 종목 수")
    parser.add_argument(
        "--input", type=Path, default=CANDIDATES, help="테마별 후보 CSV"
    )
    parser.add_argument("--output", type=Path, default=OUTPUT, help="수집 결과 CSV")
    args = parser.parse_args(argv)
    if args.start > args.end:
        parser.error("시작일은 종료일보다 늦을 수 없습니다.")
    if args.end > yesterday:
        parser.error("종료일은 한국 시간 기준 전날까지 지정할 수 있습니다.")
    try:
        candidates = load_candidates(args.input)
        print(f"출처: NAVER | 요청 기간: {args.start} ~ {args.end}", flush=True)
        if args.limit:
            print(
                f"테스트 모드: 고유 종목 최대 {args.limit}개; 기존 결과 CSV를 교체합니다.",
                flush=True,
            )
        prices, failures = fetch_stock_prices(
            candidates, args.start, args.end, args.limit
        )
        if prices.empty:
            print("전체 수집 실패: 기존 결과 파일을 유지합니다.", file=sys.stderr)
            return 1
        mapping = fetch_market_mapping(prices["ticker"].unique())
        indices = fetch_market_indices(mapping["market"].unique(), args.start, args.end)
        prices = attach_market_data(prices, mapping, indices)
        result = merge_keywords(candidates, prices)
        save_output(result, args.output)
        print(
            f"저장: {args.output} ({len(result):,}행, 고유 종목 {prices['ticker'].nunique()}개)"
        )
        print(
            f"실제 수집 범위: {prices['date'].min():%Y-%m-%d} ~ {prices['date'].max():%Y-%m-%d}"
        )
        print(f"성공 {prices['ticker'].nunique()}종목 / 실패 {len(failures)}종목")
        if failures:
            print("일부 종목이 누락된 결과입니다. 실패 종목:", file=sys.stderr)
            for ticker, error in failures:
                print(f"  {ticker}: {error}", file=sys.stderr)
        return 1 if failures else 0
    except (OSError, ValueError, KeyError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
