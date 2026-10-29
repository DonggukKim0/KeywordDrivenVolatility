"""강건성 검증과 참고 분석으로 모델 결과가 특정 조건에 좌우되는지 확인한다.

    2-2  시간 이동 검증 (3개 기간)       특정 시험 기간 때문에 나온 결과가 아닌가
    2-4  같은 날 관계 (OLS)              예측은 못 해도 SNS 급증이 그날 주가와 함께 움직이나 (예측이 아닌 관계 분석)
    2-5  민감도 (기간 4 × 기준 4)        결론이 y 정의에 좌우되나
    2-6  (참고) 화제 직후(0~2일)만       표본이 작아 참고용
    2-7  (참고) SNS 변수 구성 변경       변수를 늘리거나 줄여도 결론이 같은가

LightGBM 설정은 train.py가 저장한 lightgbm_A.pkl·lightgbm_B.pkl에서 읽는다 (없으면 학습 데이터로 새로 고름).
여러 기간·조건으로 모델을 새로 학습하므로 전체 데이터(model_dataset.csv)를 쓴다.

입력: model_dataset.csv, theme_stock_daily.csv (2-5), SNS 피쳐 원본 (2-7)
사용법: python3 src/model/robustness.py [모델 데이터] [주가 원자료] [출력 폴더] [SNS 피쳐 원본]
"""

import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from common import (
    DATASET,
    FEATURES,
    HOLD_DAYS,
    KEYWORD_ALIASES,
    MARKET_FEATURES,
    MODEL_DIR,
    MODEL_FEATURES,
    OUTPUT_DIR,
    SNS_FEATURES,
    SNS_RAW,
    STOCK_RAW,
    TEST_START,
    TRAIN_END,
    bootstrap_diff,
    forward_returns,
    labeled,
    load_dataset,
    pr_auc,
    split,
)
from models import fit_predict_lgbm, tune_lgbm

SENS_THRESHOLDS = (0.03, 0.05, 0.07, 0.10)
ROLLING = [
    ("2025-06-30", "2025-12-26"),
    ("2025-12-26", "2026-06-30"),
    ("2026-06-30", "2026-10-23"),
]  # (학습 끝, 시험 끝)


def load_params(data, model_dir=MODEL_DIR):
    """train.py가 저장한 LightGBM A·B 설정(잎 수, 나무 수, 잎 최소 표본). 없으면 학습 데이터로 새로 고른다."""
    params = {}
    for key in ("A", "B"):
        path = Path(model_dir) / f"lightgbm_{key}.pkl"
        if path.exists():
            params[key] = tuple(joblib.load(path)["params"])
        else:
            print(f"{path.name}가 없어 학습 데이터로 설정을 새로 고릅니다 (약 1분)")
            params[key] = tune_lgbm(split(labeled(data))[0], MODEL_FEATURES[key])
    print(f"LightGBM 설정 (잎, 나무, 잎 최소 표본): A {params['A']}, B {params['B']}")
    return params


def run_rolling(data, params):
    """2-2 시간 이동 검증. 확인: 특정 시험 기간 때문에 나온 결과가 아닌가.

    학습 끝을 옮겨 가며 3번 학습·시험한다 (각 시험은 학습 끝 다음 3번째 거래일부터, 경계 2거래일 제외).
    """
    frame = labeled(data)
    dates = np.sort(frame["date"].unique())
    rows = []
    for train_end, test_end in ROLLING:
        test_start = dates[np.searchsorted(dates, train_end, side="right") + 2]
        train = frame[frame["date"] <= train_end]
        test = frame[(frame["date"] >= test_start) & (frame["date"] <= test_end)]
        scores = {
            name: fit_predict_lgbm(train, test, MODEL_FEATURES[name], params[name])[1]
            for name in ("A", "B")
        }
        low, high = bootstrap_diff(
            test["y"], scores["A"], scores["B"], test["date"].values
        )
        a, b = pr_auc(test["y"], scores["A"]), pr_auc(test["y"], scores["B"])
        rows.append(
            {
                "학습": f"~{train_end}",
                "시험": f"{test_start}~{test_end}",
                "기준선": test["y"].mean(),
                "A": a,
                "B": b,
                "B−A": b - a,
                "CI하한": low,
                "CI상한": high,
            }
        )
    table = pd.DataFrame(rows)
    print("\n[2-2 시간 이동 검증]")
    print(table.round(4).to_string(index=False))
    return table


def run_same_day(data, output_dir):
    """2-4 같은 날 관계 (OLS, 날짜 단위 군집 표준오차).

    확인: SNS 급증은 그날 주가와 함께 움직이나. 예측이 아니라 관계를 재는 분석.
    종속변수 = 그날 초과수익률(%), 통제변수 = 전날 기준 시장 변수.
    """
    frame = data.sort_values(["ticker", "date"]).copy()
    group = frame.groupby("ticker")
    for column in [
        "excess_return_3d",
        "excess_return_20d",
        "volatility_20d",
        "volume_ratio_5d",
    ]:
        frame[f"{column}_lag1"] = group[column].shift(1)
    frame = frame.dropna(
        subset=[
            "excess_return_1d",
            "mention_z28",
            "excess_return_3d_lag1",
            "excess_return_20d_lag1",
            "volatility_20d_lag1",
            "volume_ratio_5d_lag1",
        ]
    )
    frame["ret_pct"] = frame["excess_return_1d"] * 100
    frame["z28_clip"] = frame["mention_z28"].clip(
        -3, 10
    )  # 극단값(최대 약 2,000)이 계수를 지배하지 않게
    formula = (
        "ret_pct ~ z28_clip + in_event + excess_return_3d_lag1 + excess_return_20d_lag1"
        " + volatility_20d_lag1 + volume_ratio_5d_lag1"
    )
    fitted = smf.ols(formula, frame).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(frame["date"])[0]}
    )
    coefficients = pd.DataFrame({"계수": fitted.params, "p값": fitted.pvalues})
    coefficients.to_csv(output_dir / "same_day_OLS계수.csv", encoding="utf-8-sig")
    bins = pd.cut(
        frame["mention_z28"],
        [-np.inf, 0, 1, 2, 3, 5, np.inf],
        labels=["<0", "0~1", "1~2", "2~3", "3~5", "5+"],
    )
    by_bin = frame.groupby(bins, observed=True)["ret_pct"].agg(["mean", "size"])
    by_bin.to_csv(output_dir / "same_day_z28구간별.csv", encoding="utf-8-sig")
    print(f"\n[2-4 같은 날 관계] {int(fitted.nobs):,}행")
    print(coefficients.round(4).to_string())
    table = by_bin.rename(columns={"mean": "그날 평균 초과수익률(%)", "size": "행수"})
    table.index.name = "mention_z28 구간"
    print("\nmention_z28 구간별 그날 평균 초과수익률\n" + table.round(2).to_string())


def run_sensitivity(data, params, forward, output_dir):
    """2-5 민감도. 확인: y 기간(1/2/3/5일) × 기준(+3/5/7/10%)을 바꿔도 B ≈ A인가.

    어떤 y가 맞는지 고르는 분석이 아니라, y를 어떻게 정해도 결론이 같은지 보는 분석.
    """
    merged = data.merge(forward, on=["ticker", "date"], how="left")
    rows = []
    for days in HOLD_DAYS:
        for threshold in SENS_THRESHOLDS:
            future = merged[f"fx{days}"]
            merged["y_alt"] = np.where(
                future.notna(), (future >= threshold).astype(float), np.nan
            )
            train, test = split(labeled(merged, "y_alt"))
            scores = {
                name: fit_predict_lgbm(
                    train, test, MODEL_FEATURES[name], params[name], "y_alt"
                )[1]
                for name in ("A", "B")
            }
            low, high = bootstrap_diff(
                test["y_alt"], scores["A"], scores["B"], test["date"].values
            )
            a, b = (
                pr_auc(test["y_alt"], scores["A"]),
                pr_auc(test["y_alt"], scores["B"]),
            )
            rows.append(
                {
                    "기간": f"{days}일",
                    "기준": f"+{threshold:.0%}",
                    "기준선": test["y_alt"].mean(),
                    "A": a,
                    "B": b,
                    "B−A": b - a,
                    "CI하한": low,
                    "CI상한": high,
                    "판정": "B 낮음"
                    if high < 0
                    else ("B 높음" if low > 0 else "차이 없음"),
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "lgbm_민감도.csv", index=False, encoding="utf-8-sig")
    print("\n[2-5 민감도]")
    print(table.round(4).to_string(index=False))
    return table


def run_early_event(data, params):
    """2-6 (참고) 화제 직후(시작 0~2일)만 학습·평가. 확인: SNS 효과가 몰릴 구간에서는 도움이 되나.

    화제 기간이 전체의 약 11%라 효과가 묻혔을 가능성에 대한 답변용. 표본이 작아 참고로만 쓴다.
    """
    subset = data[data["days_since_day0"] <= 2]
    train, test = split(labeled(subset))
    scores = {
        name: fit_predict_lgbm(train, test, MODEL_FEATURES[name], params[name])[1]
        for name in ("A", "B")
    }
    low, high = bootstrap_diff(test["y"], scores["A"], scores["B"], test["date"].values)
    a, b = pr_auc(test["y"], scores["A"]), pr_auc(test["y"], scores["B"])
    print(
        f"\n[2-6 (참고) 화제 직후 0~2일] 학습 {len(train):,} / 시험 {len(test):,}행, 기준선 {test['y'].mean():.4f}"
    )
    print(
        f"A {a:.4f}  B {b:.4f}  B − A {b - a:+.4f} (95% 신뢰구간 {low:+.4f} ~ {high:+.4f})"
    )


# 2-7 (참고) SNS 변수 구성 변경
UNUSED_SNS = [
    "mention_ratio28",  # 오늘 언급량 ÷ 직전 28일 평균
    "negative_share",  # 부정 감성어 비율
    "negative_share_vs_prior7",  # 부정 비율 − 직전 7일 평균
    "days_since_peak",  # 언급량 정점 이후 지난 일수
]
SENTIMENT_SNS = ["positive_share", "positive_share_vs_prior7", "low_sentiment"]


def run_feature_variants(data, sns_path, output_dir):
    """2-7 (참고) SNS 변수를 늘리거나 줄여도 B ≈ A인가. 확인: 변수를 골라서 나온 결론이 아닌가.

    - 늘리기: 쓰지 않은 SNS 변수 4개를 더함 (13개), 4개를 하나씩 A에 더함
    - 줄이기: 감성어 변수 3개 제외 (6개), 언급량 변수 3개만, mention_z28 1개만
    주 분할 + 시간 이동 3개 기간에서 각각 A와 비교한다 (설정은 A·B 기본값 고정).
    """
    sns = pd.read_csv(
        sns_path, encoding="utf-8-sig", usecols=["keyword", "date"] + UNUSED_SNS
    )
    sns["keyword"] = sns["keyword"].replace(KEYWORD_ALIASES)
    sns["date"] = pd.to_datetime(
        sns["date"].astype(str).str.replace(".", "-", regex=False)
    ).dt.strftime("%Y-%m-%d")
    frame = labeled(data.merge(sns, on=["keyword", "date"], how="left"))

    no_sentiment = [column for column in SNS_FEATURES if column not in SENTIMENT_SNS]
    variants = {
        "B 기존 (SNS 9개)": FEATURES,
        "B + 미사용 4개 (SNS 13개)": FEATURES + UNUSED_SNS,
        "B − 감성어 3개 (SNS 6개)": MARKET_FEATURES + no_sentiment,
        "B 언급량 3개만": MARKET_FEATURES
        + ["mention_z28", "mention_momentum_3_14", "log_mentions"],
        "B mention_z28 1개만": MARKET_FEATURES + ["mention_z28"],
    }
    variants.update(
        {f"A + {column}": MARKET_FEATURES + [column] for column in UNUSED_SNS}
    )

    dates = np.sort(frame["date"].unique())
    periods = [("주 분할", TRAIN_END, TEST_START, dates[-1])]
    for train_end, test_end in ROLLING:
        periods.append(
            (
                f"~{train_end} 학습",
                train_end,
                dates[np.searchsorted(dates, train_end, side="right") + 2],
                test_end,
            )
        )

    rows = []
    for period, train_end, test_start, test_end in periods:
        train = frame[frame["date"] <= train_end]
        test = frame[(frame["date"] >= test_start) & (frame["date"] <= test_end)]
        score_a = fit_predict_lgbm(train, test, MARKET_FEATURES, (15, 200, 100))[1]
        base_a = pr_auc(test["y"], score_a)
        for name, features in variants.items():
            score = fit_predict_lgbm(train, test, features, (15, 200, 400))[1]
            low, high = bootstrap_diff(
                test["y"], score_a, score, test["date"].values, n=200
            )
            value = pr_auc(test["y"], score)
            rows.append(
                {
                    "기간": period,
                    "시험": f"{test_start}~{test_end}",
                    "변수 구성": name,
                    "A": base_a,
                    "모델": value,
                    "모델−A": value - base_a,
                    "CI하한": low,
                    "CI상한": high,
                    "판정": "A보다 낮음"
                    if high < 0
                    else ("A보다 높음" if low > 0 else "차이 없음"),
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(
        output_dir / "lgbm_변수구성변경.csv", index=False, encoding="utf-8-sig"
    )
    print("\n[2-7 (참고) SNS 변수 구성 변경] 값 = 모델 − A, 괄호 = 판정")
    pivot = table.assign(
        값=table["모델−A"].map("{:+.4f}".format) + " (" + table["판정"] + ")"
    )
    print(
        pivot.pivot(index="변수 구성", columns="기간", values="값")
        .reindex(variants)
        .to_string()
    )
    return table


def main(
    dataset_path=DATASET,
    stock_path=STOCK_RAW,
    output_dir=OUTPUT_DIR,
    sns_path=SNS_RAW,
    model_dir=MODEL_DIR,
):
    start = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_dataset(dataset_path)
    params = load_params(data, model_dir)
    run_rolling(data, params)
    run_same_day(data, output_dir)
    run_sensitivity(data, params, forward_returns(stock_path), output_dir)
    run_early_event(data, params)
    run_feature_variants(data, sns_path, output_dir)
    print(f"\n검증 결과 저장: {output_dir} ({time.time() - start:.0f}초)")


if __name__ == "__main__":
    main(*[Path(argument) for argument in sys.argv[1:5]])
