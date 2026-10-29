"""STEP03 EDA의 데이터 읽기, 그림 설정, 결과 저장 공통 기능."""

import platform
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from step03_prepare_data import file_hash, require, unique_key

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data/processed/eda/step03/stock_daily.csv.gz"
DEFAULT_SNS_INPUT = ROOT / "data/processed/eda/step03/sns_daily.csv.gz"
DEFAULT_OUTPUT = ROOT / "eda-outputs/step03"
BLUE, RED, GREEN, GOLD, GRAY = "#2864A0", "#BF513F", "#167F89", "#B67A20", "#8794A2"
SEED = 20261027


from eda_reporting import Report, configure_plots, log


def load_stock_daily(path, required_columns=()):
    if not Path(path).is_file():
        raise FileNotFoundError(
            f"{path}가 없습니다. 먼저 python3 src/eda/step03_prepare_data.py를 실행하세요."
        )
    dates = [
        "date",
        "sns_cutoff_date",
        "sns_source_date",
        "sns_feature_source_date",
        "financial_last_receipt_date",
    ]
    # 소수점 반올림으로 사건 선정 경계가 바뀌지 않도록 CSV의 float를 복원한다.
    base = pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype={"ticker": str},
        float_precision="round_trip",
        low_memory=False,
    )
    require(
        base,
        ["ticker", "name", "keyword", "date", "close", "volume", "trade_index"]
        + list(required_columns),
        "STEP03 기본 데이터",
    )
    for col in dates:
        if col in base:
            base[col] = pd.to_datetime(base[col], format="%Y-%m-%d", errors="raise")
    unique_key(base, ["ticker", "date"], "STEP03 기본 데이터")
    if not base.ticker.str.fullmatch(r"\d{6}", na=False).all():
        raise ValueError("종목코드는 6자리 문자열이어야 합니다.")
    if not base.close.gt(0).all():
        raise ValueError("종가는 결측 없는 양수여야 합니다.")
    base = base.sort_values(["ticker", "date"]).reset_index(drop=True)
    for ticker, group in base.groupby("ticker", sort=False):
        if not np.array_equal(group.trade_index.to_numpy(), np.arange(len(group))):
            raise ValueError(f"{ticker}: 거래일 인덱스가 연속되지 않습니다.")
    return base


def load_sns_daily(path):
    if not Path(path).is_file():
        raise FileNotFoundError(
            f"{path}가 없습니다. 먼저 step03_prepare_data.py를 실행하세요."
        )
    frame = pd.read_csv(
        path, encoding="utf-8-sig", float_precision="round_trip", low_memory=False
    )
    require(
        frame,
        ["keyword", "date", "news_mentions", "social_mentions"],
        "STEP03 SNS 일별 데이터",
    )
    frame["date"] = pd.to_datetime(frame.date, format="%Y-%m-%d", errors="raise")
    unique_key(frame, ["keyword", "date"], "STEP03 SNS 일별 데이터")
    if (
        frame[["news_mentions", "social_mentions"]].isna().any().any()
        or frame[["news_mentions", "social_mentions"]].lt(0).any().any()
    ):
        raise ValueError("뉴스·SNS 언급량에 결측 또는 음수가 있습니다.")
    return frame.sort_values(["keyword", "date"]).reset_index(drop=True)


def rank_correlation(x, y, min_pairs=20):
    paired = pd.DataFrame({"x": np.asarray(x), "y": np.asarray(y)}).dropna()
    if len(paired) < min_pairs or paired.x.nunique() < 2 or paired.y.nunique() < 2:
        return len(paired), np.nan
    return len(paired), float(
        paired.x.rank(method="average").corr(paired.y.rank(method="average"))
    )


def build_metadata(analysis, paths, hashes, counts, rules, elapsed):
    """입력 변경을 점검하고 호출자에게 실행 정보를 반환한다. 파일은 저장하지 않는다."""
    for key, path in paths.items():
        if file_hash(path) != hashes[key]:
            raise ValueError(f"실행 중 입력 데이터가 변경되었습니다: {path}")
    metadata = {
        "step": "STEP03",
        "analysis": analysis,
        "source_notebook": "RQ3_raw_EDA.ipynb",
        "inputs": {
            key: {"path": str(Path(path).resolve()), "sha256": hashes[key]}
            for key, path in paths.items()
        },
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "counts": counts,
        "rules": rules,
        "checks": {"input_unchanged": True, "analysis_validation_passed": True},
        "model_training_or_validation": False,
        "elapsed_seconds": round(elapsed, 2),
    }
    return metadata
