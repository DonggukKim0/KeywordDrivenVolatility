"""공통 모델 설정과 데이터 처리·평가 함수.

경로, 변수 목록, 기간, 데이터 읽기, 이벤트 탐색, 평가 지표를 제공합니다.

프로젝트 내 기본 경로는 data/processed, data/raw, model-outputs 아래다.
"""

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from sklearn.metrics import average_precision_score

warnings.filterwarnings("ignore")

# =============================================================================
# 경로
# =============================================================================
HERE = Path(__file__).resolve().parent
PROJECT = (
    HERE.parents[1]
    if len(HERE.parents) > 1 and (HERE.parents[1] / "data").exists()
    else None
)


def _pick(*candidates):
    """후보 경로 중 실제로 있는 첫 파일, 없으면 첫 후보."""
    return next((path for path in candidates if path.exists()), candidates[0])


if PROJECT is not None:
    DATA_DIR, RAW_DIR = PROJECT / "data/processed", PROJECT / "data/raw"
    MODEL_DIR, OUTPUT_DIR = (
        PROJECT / "model-outputs/models",
        PROJECT / "model-outputs/results",
    )
else:
    DATA_DIR = RAW_DIR = HERE
    MODEL_DIR, OUTPUT_DIR = HERE / "models", HERE / "results"

DATASET = DATA_DIR / "model_dataset.csv"
TRAIN_FILE, TEST_FILE = DATA_DIR / "train.csv", DATA_DIR / "test.csv"
STOCK_RAW = RAW_DIR / "theme_stock_daily.csv"
SNS_RAW = DATA_DIR / "sns_keyword_features.csv"

# =============================================================================
# 변수와 기간
# =============================================================================
MARKET_FEATURES = [
    "excess_return_1d",  # 전날 대비 시장보다 더 오른 비율
    "excess_return_3d",  # 3거래일 전 대비 시장보다 더 오른 비율
    "excess_return_20d",  # 20거래일 전 대비 시장보다 더 오른 비율
    "volume_ratio_5d",  # 오늘 거래량 ÷ 직전 5거래일 평균
    "volatility_20d",  # 최근 20거래일 하루 수익률의 흔들림 (표준편차)
    "disclosure_count_3d",  # 최근 3거래일 공시 건수
    "contract_disclosure_3d",  # 최근 3거래일 공급계약 공시 여부
]
SNS_FEATURES = [
    "mention_z28",  # (오늘 언급량 − 직전 28일 평균) ÷ 직전 28일 표준편차
    "mention_momentum_3_14",  # 최근 3일 평균 언급량 ÷ 직전 14일 평균
    "log_mentions",  # ln(1 + 오늘 언급량)
    "days_since_day0",  # 화제 시작일부터 지난 일수 (화제 기간 밖 NA)
    "positive_share",  # 긍정 감성어 비율
    "positive_share_vs_prior7",  # 긍정 비율 − 직전 7일 평균
    "entropy_vs_prior7",  # 채널 확산 다양성 − 직전 7일 평균
    "in_event",  # SNS 화제 기간이면 1
    "low_sentiment",  # 감성어 10건 미만이면 1
]
FEATURES = MARKET_FEATURES + SNS_FEATURES
MODEL_FEATURES = {"A": MARKET_FEATURES, "B": FEATURES, "C": SNS_FEATURES}
BINARY_FEATURES = ("in_event", "low_sentiment", "contract_disclosure_3d")

TRAIN_END, TEST_START = (
    "2025-12-26",
    "2026-01-01",
)  # 2025-12-29, 30은 y 구간이 시험 기간에 걸쳐 제외
FIT_END, VALID_START = (
    "2025-10-23",
    "2025-10-28",
)  # 학습 기간 안 검증 구간 (로지스틱·랜덤포레스트·LSTM 설정 고르기)
EVENT_WINDOW = (-5, 10)  # 이벤트 스터디 구간 (Day −5 ~ +10, 거래일)
SURGE = 0.05  # "급등"으로 세는 초과수익률 기준
HOLD_DAYS = (1, 2, 3, 5)  # EDA 4 보유 기간, 2-5 민감도 기간
SEED = 0
KEYWORD_ALIASES = {"딥시크 (DeepSeek)": "딥시크"}

COLOR_Y0, COLOR_Y1 = "#9a9a94", "#2a78d6"
INK, GRID = "#333333", "#e6e6e2"


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


def load_dataset(path=DATASET):
    """모델 데이터(또는 rq1_train/test.csv)를 읽는다. y가 없는 행도 남긴다 (EDA·LSTM 시퀀스용)."""
    data = pd.read_csv(path, dtype={"ticker": str}, encoding="utf-8-sig")
    missing = set(FEATURES + ["y", "keyword", "ticker", "date"]) - set(data.columns)
    if missing:
        raise ValueError(f"필수 컬럼 누락: {', '.join(sorted(missing))}")
    return data.sort_values(["keyword", "ticker", "date"]).reset_index(drop=True)


def labeled(data, target="y"):
    """y가 있는 행만 남기고 y를 정수로 바꾼다."""
    result = data[data[target].notna()].copy()
    result[target] = result[target].astype(int)
    return result


def split(data):
    """날짜 기준으로 학습·시험을 나눈다 (경계 2거래일 제외)."""
    return data[data["date"] <= TRAIN_END], data[data["date"] >= TEST_START]


def all_events(data):
    """키워드별 화제 시작일(Day 0) 전체 목록.

    Day 0 = 키워드별로 in_event가 0에서 1로 바뀐 첫 거래일 (주말 시작 이벤트는 다음 거래일).
    """
    dates = np.sort(data["date"].unique())
    position = {date: i for i, date in enumerate(dates)}
    days = data.drop_duplicates(["keyword", "date"])[
        ["keyword", "date", "in_event", "mention_z28"]
    ]
    days = days.sort_values(["keyword", "date"])
    previous = days.groupby("keyword")["in_event"].shift(fill_value=0)
    events = days[(days["in_event"] == 1) & (previous == 0)][
        ["keyword", "date", "mention_z28"]
    ]
    return events.reset_index(drop=True), dates, position


def find_events(data, window=None):
    """전후 구간(기본 Day −5 ~ +10)이 모두 있는 이벤트만 남긴다."""
    events, dates, position = all_events(data)
    low, high = window or EVENT_WINDOW
    inside = events["date"].map(position).between(-low, len(dates) - 1 - high)
    return (
        events.loc[inside, ["keyword", "date"]].reset_index(drop=True),
        dates,
        position,
    )


def event_ar_matrix(data, return_tickers=False):
    """이벤트 × 종목마다 Day −5 ~ +10의 일별 초과수익률(AR) 행렬을 만든다.

    return_tickers=True면 각 행의 종목코드와 Day 0 날짜도 돌려준다 (EDA 3 공시 구분용).
    """
    events, dates, position = find_events(data)
    ar = data.pivot_table(index="date", columns="ticker", values="excess_return_1d")
    tickers = data.groupby("keyword")["ticker"].unique()
    low, high = EVENT_WINDOW
    rows, event_ids, row_tickers, row_day0 = [], [], [], []
    for event_id, (keyword, day0) in enumerate(events.itertuples(index=False)):
        i = position[day0]
        window = ar.loc[dates[i + low : i + high + 1], tickers[keyword]].values.T
        rows.append(window)
        event_ids += [event_id] * len(window)
        row_tickers += list(tickers[keyword])
        row_day0 += [day0] * len(window)
    result = (np.vstack(rows), np.array(event_ids), events)
    if return_tickers:
        return result + (np.array(row_tickers), np.array(row_day0))
    return result


def pr_auc(y, score):
    return average_precision_score(y, score)


def top_precision(y, score, share=0.10):
    """확률 상위 share 안에서 실제 y = 1 비율."""
    y = np.asarray(y)
    return y[score >= np.quantile(score, 1 - share)].mean()


def bootstrap_diff(y, score_a, score_b, dates, n=300):
    """날짜 단위로 다시 뽑아 PR-AUC(B) − PR-AUC(A)의 95% 신뢰구간을 구한다."""
    rng = np.random.default_rng(SEED)
    y = np.asarray(y)
    unique = np.unique(dates)
    index = {date: np.where(dates == date)[0] for date in unique}
    diffs = []
    for _ in range(n):
        sample = np.concatenate(
            [index[date] for date in rng.choice(unique, len(unique))]
        )
        diffs.append(
            pr_auc(y[sample], score_b[sample]) - pr_auc(y[sample], score_a[sample])
        )
    return np.percentile(diffs, [2.5, 97.5])


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
