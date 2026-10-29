"""썸트렌드 키워드별 일별 언급량·감성 원자료로 SNS feature 10개와 보조 변수를 생성한다.

변수 번호(NO.)는 'SNS 피쳐 변수정의서'의 우선순위와 같다.
    0  mention_ratio28           Day 0 기준값 (모델 입력 아님)
    1  mention_z28               이전 28일 대비 언급 급증 (표준편차 단위)
    2  negative_share_vs_prior7  이전 7일 대비 부정 비율 변화
    3  days_since_day0           이번 화제 시작일(Day 0)부터 지난 일수
    4  mention_momentum_3_14     최근 3일 평균 ÷ 이전 14일 평균
    5  days_since_peak           이번 화제 안에서 최고 언급일로부터 지난 일수
    6  log_mentions              ln(1 + 언급량)
    7  negative_share            부정 감성어 비율
    8  positive_share_vs_prior7  이전 7일 대비 긍정 비율 변화
    9  positive_share            긍정 감성어 비율
    10 entropy_vs_prior7         이전 7일 대비 채널 확산 다양성 변화
    보조 in_event                이벤트 기간이면 1
    보조 low_sentiment           감성어 합계가 10건 미만이면 1

공통 규칙
    - 모든 계산은 t일과 그 이전 날짜만 사용한다 (미래 정보 없음).
    - 이동 구간은 거래일이 아니라 달력일 기준이다.
    - 계산할 수 없으면 NA로 둔다. 0 채우기·앞뒤 값 채우기는 하지 않는다.
    - 스케일링·극단값 자르기는 하지 않는다. 학습/시험 분할 후 모델 담당자가 처리한다.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data/raw/sns_daily_mentions_sentiment.csv"
OUTPUT = ROOT / "data/processed/sns_keyword_features.csv"

TARGET_START = "2024-11-01"  # 앞의 28일(2024-10-04 ~ 10-31)은 이동 계산용 워밍업
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
MENTION_TOTAL = "언급량 추이__전체"
MENTION_CHANNELS = [f"언급량 추이__{channel}" for channel in CHANNELS]
SENTIMENT_COLUMNS = ["감성__긍정", "감성__부정", "감성__중립"]

DAY0_RATIO = 3  # 평소(이전 28일 평균) 대비 3배 이상
DAY0_MIN_MENTIONS = 20  # 그날 언급량 최소 20건
EVENT_END_DAYS = 7  # 7일 연속 Day 0 조건 미충족이면 이벤트 종료
LOW_SENTIMENT = 10  # 감성어 합계 10건 미만이면 감성 비율을 믿기 어려움

IDENTIFIERS = ["keyword_id", "keyword", "date"]
OUTPUT_COLUMNS = IDENTIFIERS + [
    "mention_ratio28",  # 0
    "mention_z28",  # 1
    "negative_share_vs_prior7",  # 2
    "days_since_day0",  # 3
    "mention_momentum_3_14",  # 4
    "days_since_peak",  # 5
    "log_mentions",  # 6
    "negative_share",  # 7
    "positive_share_vs_prior7",  # 8
    "positive_share",  # 9
    "entropy_vs_prior7",  # 10
    "in_event",
    "low_sentiment",
]


def load_sns_data(path=INPUT):
    """원본 CSV를 읽고 필수 컬럼, 날짜, 키워드×날짜 중복, 빠진 날짜를 검증한다."""
    frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    required = (
        ["keyword_id", "keyword", "date", MENTION_TOTAL]
        + MENTION_CHANNELS
        + SENTIMENT_COLUMNS
    )
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"필수 컬럼 누락: {', '.join(sorted(missing))}")
    frame = frame[required].copy()
    if frame.empty:
        raise ValueError("원본 데이터가 비어 있습니다.")
    # 원본 날짜는 2024.11.1 또는 2024-11-01 형식이 섞일 수 있다.
    frame["date"] = pd.to_datetime(
        frame["date"].astype(str).str.strip().str.replace(".", "-", regex=False),
        format="%Y-%m-%d",
        errors="raise",
    )
    if frame.duplicated(["keyword_id", "date"]).any():
        raise ValueError("keyword_id+date 중복이 있습니다.")
    for column in [MENTION_TOTAL] + MENTION_CHANNELS + SENTIMENT_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if frame[column].isna().any() or (frame[column] < 0).any():
            raise ValueError(f"건수가 비어 있거나 음수입니다: {column}")
    # 이동 구간이 달력일 기준이므로 키워드마다 날짜가 하루도 빠지면 안 된다.
    for keyword_id, group in frame.groupby("keyword_id"):
        expected = pd.date_range(group["date"].min(), group["date"].max(), freq="D")
        if len(group) != len(expected):
            raise ValueError(f"빠진 날짜가 있습니다: {keyword_id}")
    return frame.sort_values(["keyword_id", "date"]).reset_index(drop=True)


def prior_window(series, window, func="mean"):
    """t일을 제외한 직전 window일(t-window ~ t-1)의 통계량. 구간에 NA가 하나라도 있으면 NA."""
    shifted = series.shift(1).rolling(window, min_periods=window)
    if func == "std":
        return shifted.std(ddof=0)  # 모집단 표준편차
    return shifted.mean()


def calculate_base(frame):
    """원자료를 기본 값(언급량, 감성 비율, 채널 엔트로피)으로 바꾼다."""
    frame = frame.copy()
    frame["mentions"] = frame[MENTION_TOTAL].astype(float)
    positive, negative, neutral = (frame[column] for column in SENTIMENT_COLUMNS)
    frame["sentiment_count"] = positive + negative + neutral
    has_sentiment = frame["sentiment_count"] > 0

    # 9 positive_share = 긍정 ÷ (긍정 + 부정 + 중립). 감성어 0건이면 NA
    frame["positive_share"] = (positive / frame["sentiment_count"]).where(has_sentiment)
    # 7 negative_share = 부정 ÷ (긍정 + 부정 + 중립). 감성어 0건이면 NA
    frame["negative_share"] = (negative / frame["sentiment_count"]).where(has_sentiment)

    # 채널 확산 다양성 E = -Σ p·ln(p) ÷ ln(10), p = 채널 언급량 ÷ 전체 언급량. 0·ln(0) = 0
    has_mentions = frame["mentions"] > 0
    shares = frame[MENTION_CHANNELS].div(frame["mentions"].where(has_mentions), axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(shares > 0, shares * np.log(shares), 0.0)
    frame["channel_entropy_norm"] = pd.Series(
        -terms.sum(axis=1) / np.log(len(CHANNELS)), index=frame.index
    ).where(has_mentions)
    return frame


def calculate_mention_features(group):
    """한 키워드의 언급량 변수(0, 1, 4, 6번)를 계산한다."""
    mentions = group["mentions"]
    mean28 = prior_window(mentions, 28)
    std28 = prior_window(mentions, 28, "std")

    # 0 mention_ratio28 = 오늘 ÷ 이전 28일 평균
    group["mention_ratio28"] = mentions / mean28.where(mean28 > 0)
    # 1 mention_z28 = (오늘 − 이전 28일 평균) ÷ 이전 28일 표준편차
    group["mention_z28"] = (mentions - mean28) / std28.where(std28 > 0)
    # 4 mention_momentum_3_14 = 최근 3일(t-2 ~ t) 평균 ÷ 이전 14일(t-14 ~ t-1) 평균
    recent3 = mentions.rolling(3, min_periods=3).mean()
    mean14 = prior_window(mentions, 14)
    group["mention_momentum_3_14"] = recent3 / mean14.where(mean14 > 0)
    # 6 log_mentions = ln(1 + 오늘 언급량)
    group["log_mentions"] = np.log1p(mentions)
    return group


def calculate_change_features(group):
    """한 키워드의 '이전 7일 대비 변화' 변수(2, 8, 10번)를 계산한다.

    오늘 값 − 이전 7일(t-7 ~ t-1) 일별 값의 단순 평균. 8일 중 하루라도 NA면 NA.
    """
    group["negative_share_vs_prior7"] = group["negative_share"] - prior_window(
        group["negative_share"], 7
    )  # 2
    group["positive_share_vs_prior7"] = group["positive_share"] - prior_window(
        group["positive_share"], 7
    )  # 8
    group["entropy_vs_prior7"] = group["channel_entropy_norm"] - prior_window(
        group["channel_entropy_norm"], 7
    )  # 10
    return group


def calculate_event_features(group):
    """한 키워드의 이벤트 변수(3, 5번)를 하루씩 앞으로 진행하며 계산한다.

    Day 0   : mention_ratio28 ≥ 3 이고 그날 언급량 ≥ 20건인 날 (이벤트가 없을 때)
    종료    : 7일 연속 Day 0 조건을 만족하지 못하면 종료. 이후 조건을 만족하면 새 Day 0
    정점    : Day 0 ~ t 중 언급량이 가장 많은 날. 같은 값이면 더 최근 날
    t일 값을 정할 때 t 이후 날짜는 보지 않는다.
    """
    hot = (
        (group["mention_ratio28"] >= DAY0_RATIO)
        & (group["mentions"] >= DAY0_MIN_MENTIONS)
    ).to_numpy()
    mentions = group["mentions"].to_numpy()
    since_day0, since_peak = [], []
    elapsed = None  # 이벤트 밖이면 None
    cold_days = peak_value = peak_index = 0
    for index, is_hot in enumerate(hot):
        if elapsed is None:
            if is_hot:  # 새 이벤트 시작
                elapsed, cold_days = 0, 0
                peak_value, peak_index = mentions[index], index
        else:
            elapsed += 1
            cold_days = 0 if is_hot else cold_days + 1
            if mentions[index] >= peak_value:
                peak_value, peak_index = mentions[index], index
            if cold_days >= EVENT_END_DAYS:  # 이벤트 종료: 오늘부터 이벤트 밖
                elapsed = None
        since_day0.append(np.nan if elapsed is None else elapsed)
        since_peak.append(np.nan if elapsed is None else index - peak_index)
    group["days_since_day0"] = since_day0  # 3
    group["days_since_peak"] = since_peak  # 5
    return group


def build_feature_dataset(frame):
    """키워드별로 변수를 계산하고 워밍업 기간을 뺀 결과를 만든다."""
    base = calculate_base(frame)
    results = []
    for _, group in base.groupby("keyword_id", sort=True):
        group = group.copy()
        group = calculate_mention_features(group)
        group = calculate_change_features(group)
        group = calculate_event_features(group)
        results.append(group)
    result = pd.concat(results, ignore_index=True)
    result["in_event"] = result["days_since_day0"].notna().astype(int)
    result["low_sentiment"] = (result["sentiment_count"] < LOW_SENTIMENT).astype(int)
    result = result.replace([np.inf, -np.inf], np.nan)
    result = result[result["date"] >= TARGET_START][OUTPUT_COLUMNS]
    if result.empty:
        raise ValueError("분석 기간에 해당하는 행이 없습니다.")
    return result.sort_values(["keyword_id", "date"]).reset_index(drop=True)


def save_features(result, path=OUTPUT):
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


def main(input_path=INPUT, output_path=OUTPUT):
    """원본 검증부터 feature CSV 저장까지 실행하고 결측·이벤트 요약을 출력한다."""
    try:
        frame = load_sns_data(input_path)
        result = build_feature_dataset(frame)
        save_features(result, output_path)
        print(
            f"원본 {len(frame):,}행 → SNS feature {len(result):,}행 × {len(result.columns)}열"
        )
        print(
            f"기간: {result['date'].min():%Y-%m-%d} ~ {result['date'].max():%Y-%m-%d}, 키워드 {result['keyword_id'].nunique()}개"
        )
        print(
            f"이벤트 {int((result['days_since_day0'] == 0).sum())}개, 이벤트 기간 {int(result['in_event'].sum()):,}행"
        )
        print(f"감성어 {LOW_SENTIMENT}건 미만 {int(result['low_sentiment'].sum()):,}행")
        print("결측(NA) 수:")
        for column, count in result.isna().sum().items():
            if count:
                print(f"  {column}: {count:,}")
        print(f"저장: {output_path}")
        print("mention_ratio28은 Day 0 기준값이므로 모델 입력에서 제외하세요.")
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"전처리 실패: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    # 사용법: python preprocess_sns_data.py [입력 CSV] [출력 CSV]  (생략하면 기본 경로)
    arguments = sys.argv[1:]
    sys.exit(main(*(Path(argument) for argument in arguments[:2])))
