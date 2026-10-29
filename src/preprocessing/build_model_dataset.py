"""모델 분석용 데이터(종목 × 거래일)를 만든다. 정답(y)도 함께 계산한다.

입력
    theme_candidates.csv          키워드-종목 매칭 (30개 키워드 × 10종목, is_leader 포함)
    market_stock_features.csv     시장 변수 (종목 × 거래일)
    theme_stock_daily.csv         주가 원자료 (market, open, close, market_index_close 사용)
    sns_keyword_features.csv      SNS 피쳐 (키워드 × 달력일, preprocess_sns_data.py 결과)

출력
    model_dataset.csv               2024-11-01 이후 종목 × 거래일, 식별자 8 + 시장 변수 7 + SNS 변수 9 + 정답 2

정답(y)
    y = 다음 2거래일(t+1 시가 → t+2 종가) 동안 종목 수익률 − 같은 기간 시장지수 수익률이 +5% 이상이면 1
    - 지수는 원자료에 종가만 있어 t 종가 → t+2 종가 수익률로 뺀다 (지수의 밤사이 변동은 작음).
    - 마지막 2거래일, t+1 시가가 0 이하인 날은 계산할 수 없어 NA.
    - future_excess_2d는 y 검산용이다. 미래 값이므로 절대 모델 입력에 넣지 않는다.

결합 규칙
    - 매칭 파일 기준으로 시장 변수를 붙인다 (keyword + ticker).
    - SNS 피쳐는 keyword + date로 붙인다. 같은 키워드의 10종목에 같은 값이 들어간다.
    - 주가가 있는 거래일만 남는다. 주말 SNS 값은 쓰지 않는다.
    - 키워드 이름이 다른 경우(SNS "딥시크 (DeepSeek)" ↔ 주가 "딥시크")는 KEYWORD_ALIASES로 맞춘다.
"""

import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
THEME_INPUT = ROOT / "data/raw/theme_candidates.csv"
MARKET_INPUT = ROOT / "data/processed/market_stock_features.csv"
STOCK_INPUT = ROOT / "data/raw/theme_stock_daily.csv"
SNS_INPUT = ROOT / "data/processed/sns_keyword_features.csv"
OUTPUT = ROOT / "data/processed/model_dataset.csv"

START_DATE = "2024-11-01"  # SNS 피쳐 분석 시작일
TARGET_DAYS = 2  # 정답 기간: t+1 시가 → t+2 종가
TARGET_THRESHOLD = 0.05  # 시장보다 +5% 이상 더 오르면 1
KEYWORD_ALIASES = {"딥시크 (DeepSeek)": "딥시크"}  # SNS 키워드명 → 주가 키워드명

IDENTIFIERS = [
    "category",
    "keyword",
    "theme",
    "ticker",
    "name",
    "is_leader",
    "market",
    "date",
]
MARKET_FEATURES = [
    "excess_return_1d",  # M27 오늘 시장보다 더 오른 만큼
    "excess_return_3d",  # M28 최근 3거래일 시장보다 더 오른 만큼
    "excess_return_20d",  # M3  최근 20거래일 시장보다 더 오른 만큼
    "volume_ratio_5d",  # M6  오늘 거래량 ÷ 직전 5거래일 평균
    "volatility_20d",  # M9  최근 20거래일 수익률의 흔들림
    "disclosure_count_3d",  # M15 최근 3거래일 공시 건수
    "contract_disclosure_3d",  # M17 최근 3거래일 공급계약 공시 여부
]
SNS_FEATURES = [
    "mention_z28",  # 평소 대비 급증 (표준편차 단위)
    "mention_momentum_3_14",  # 최근 3일 ÷ 직전 14일 평균 언급량
    "log_mentions",  # ln(1 + 언급량)
    "days_since_day0",  # 화제 시작일부터 지난 일수 (이벤트 밖 NA)
    "positive_share",  # 긍정 감성어 비율
    "positive_share_vs_prior7",  # 긍정 비율의 직전 7일 대비 변화
    "entropy_vs_prior7",  # 채널 확산 다양성의 직전 7일 대비 변화 (후보)
    "in_event",  # 이벤트 기간이면 1
    "low_sentiment",  # 감성어 10건 미만이면 1
]
TARGET_COLUMNS = ["future_excess_2d", "y"]  # future_excess_2d는 검산용 (모델 입력 금지)
OUTPUT_COLUMNS = IDENTIFIERS + MARKET_FEATURES + SNS_FEATURES + TARGET_COLUMNS


def read_csv(path, **kwargs):
    """CSV를 읽고 비어 있으면 오류를 낸다."""
    frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False, **kwargs)
    if frame.empty:
        raise ValueError(f"빈 파일입니다: {path}")
    return frame


def require(frame, columns, name):
    """필수 컬럼이 모두 있는지 확인한다."""
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} 필수 컬럼 누락: {', '.join(sorted(missing))}")


def load_inputs(theme_path, market_path, stock_path, sns_path):
    """네 입력 파일을 읽고 키 중복과 매칭 일치를 검증한다."""
    theme = read_csv(theme_path, dtype={"ticker": str})
    market = read_csv(market_path, dtype={"ticker": str})
    stock = read_csv(
        stock_path,
        dtype={"ticker": str},
        usecols=["ticker", "date", "market", "open", "close", "market_index_close"],
    )
    sns = read_csv(sns_path)

    require(
        theme,
        ["category", "keyword", "theme", "ticker", "name", "is_leader"],
        "매칭 파일",
    )
    require(market, ["keyword", "ticker", "date"] + MARKET_FEATURES, "시장 변수 파일")
    require(sns, ["keyword", "date"] + SNS_FEATURES, "SNS 피쳐 파일")

    if theme.duplicated(["keyword", "ticker"]).any():
        raise ValueError("매칭 파일에 keyword+ticker 중복이 있습니다.")
    if market.duplicated(["keyword", "ticker", "date"]).any():
        raise ValueError("시장 변수 파일에 keyword+ticker+date 중복이 있습니다.")

    # 매칭 파일과 시장 변수 파일의 키워드-종목 쌍이 같아야 한다.
    theme_pairs = set(zip(theme["keyword"], theme["ticker"]))
    market_pairs = set(zip(market["keyword"], market["ticker"]))
    if theme_pairs != market_pairs:
        raise ValueError(
            f"키워드-종목 쌍 불일치: 매칭에만 {len(theme_pairs - market_pairs)}개, "
            f"시장 변수에만 {len(market_pairs - theme_pairs)}개"
        )

    # SNS 키워드명을 주가 키워드명으로 맞춘다.
    sns["keyword"] = sns["keyword"].replace(KEYWORD_ALIASES)
    missing_keywords = set(theme["keyword"]) - set(sns["keyword"])
    if missing_keywords:
        raise ValueError(
            f"SNS 피쳐에 없는 키워드: {', '.join(sorted(missing_keywords))}"
        )
    if sns.duplicated(["keyword", "date"]).any():
        raise ValueError("SNS 피쳐에 keyword+date 중복이 있습니다.")

    stock = stock.drop_duplicates(["ticker", "date"])
    for frame in (market, stock, sns):
        frame["date"] = pd.to_datetime(
            frame["date"].astype(str).str.replace(".", "-", regex=False)
        )
    return theme, market, stock, sns


def calculate_target(stock):
    """종목별로 다음 TARGET_DAYS 거래일의 시장 대비 초과수익률과 y를 계산한다."""
    stock = stock.sort_values(["ticker", "date"]).copy()
    group = stock.groupby("ticker")
    next_open = group["open"].shift(-1)
    next_open = next_open.where(next_open > 0)  # 시가가 0 이하이면 수익률 계산 불가
    stock_return = group["close"].shift(-TARGET_DAYS) / next_open - 1
    index_return = (
        group["market_index_close"].shift(-TARGET_DAYS) / stock["market_index_close"]
        - 1
    )
    stock["future_excess_2d"] = stock_return - index_return
    stock["y"] = (stock["future_excess_2d"] >= TARGET_THRESHOLD).astype("Int64")
    stock.loc[stock["future_excess_2d"].isna(), "y"] = pd.NA
    return stock[["ticker", "date", "market"] + TARGET_COLUMNS]


def build_dataset(theme, market, stock, sns):
    """매칭 → 시장 변수 → 시장 구분 → SNS 순서로 붙이고 분석 기간만 남긴다."""
    data = theme.merge(
        market[["keyword", "ticker", "date"] + MARKET_FEATURES],
        on=["keyword", "ticker"],
        how="inner",
        validate="one_to_many",
    )
    data = data.merge(
        calculate_target(stock),
        on=["ticker", "date"],
        how="left",
        validate="many_to_one",
    )
    data = data.merge(
        sns[["keyword", "date"] + SNS_FEATURES],
        on=["keyword", "date"],
        how="left",
        validate="many_to_one",
    )
    data = data[data["date"] >= START_DATE]

    if (
        data[SNS_FEATURES[-2]].isna().any()
    ):  # in_event가 비어 있으면 SNS가 안 붙은 날짜가 있다는 뜻
        raise ValueError("SNS 피쳐가 붙지 않은 행이 있습니다.")
    if data.duplicated(["ticker", "date"]).any():
        raise ValueError("결과에 ticker+date 중복이 있습니다.")
    return (
        data[OUTPUT_COLUMNS]
        .sort_values(["keyword", "ticker", "date"])
        .reset_index(drop=True)
    )


def save_dataset(result, path=OUTPUT):
    """비어 있지 않은 결과를 UTF-8 BOM CSV로 저장한 뒤 기존 파일을 교체한다."""
    if result.empty:
        raise ValueError("빈 결과는 저장하지 않습니다.")
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


def main(
    theme_path=THEME_INPUT,
    market_path=MARKET_INPUT,
    stock_path=STOCK_INPUT,
    sns_path=SNS_INPUT,
    output_path=OUTPUT,
):
    """입력 검증부터 모델 데이터 저장까지 실행하고 요약을 출력한다."""
    try:
        theme, market, stock, sns = load_inputs(
            theme_path, market_path, stock_path, sns_path
        )
        result = build_dataset(theme, market, stock, sns)
        save_dataset(result, output_path)
        print(f"모델 데이터 {len(result):,}행 × {len(result.columns)}열")
        print(
            f"기간: {result['date'].min():%Y-%m-%d} ~ {result['date'].max():%Y-%m-%d}, "
            f"키워드 {result['keyword'].nunique()}개, 종목 {result['ticker'].nunique()}개"
        )
        print("결측(NA) 수:")
        for column, count in result.isna().sum().items():
            if count:
                print(f"  {column}: {count:,}")
        print(f"저장: {output_path}")
        valid = result["y"].notna()
        print(
            f"y: {int(valid.sum()):,}행 계산, 양성 비율 {result.loc[valid, 'y'].astype(float).mean():.2%}"
        )
        print(
            "시장 변수는 모델 A·B, SNS 변수는 모델 B에만 넣으세요. future_excess_2d는 검산용이니 모델 입력에서 빼세요."
        )
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"모델 데이터 생성 실패: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    # 사용법: python build_model_dataset.py [매칭] [시장 변수] [주가 원자료] [SNS 피쳐] [출력]
    arguments = [Path(argument) for argument in sys.argv[1:6]]
    sys.exit(main(*arguments))
