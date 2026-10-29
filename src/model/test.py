"""모델 평가: 저장된 모델로 시험 데이터(test.csv, 2026년)를 예측하고 성능을 비교한다.

    2-1  모델 A·B·C 성능 (PR-AUC, ROC-AUC, 상위 10% 정밀도), B − A와 95% 신뢰구간(날짜 단위 부트스트랩)
    2-3  순열 중요도 (lightgbm A·B): 변수 하나를 섞었을 때 시험 PR-AUC가 떨어지는 정도
    3-x  로지스틱(B 표준화 계수 포함)·랜덤포레스트·LSTM(시드별) 성능, B − A

평가 지표
    PR-AUC: 급등(양성)을 잡는 것이 목적이라 양성 기준 지표를 쓴다. 기준선 = 무작위로 찍었을 때 = 양성 비율
    상위 10% 정밀도: 급등 확률 상위 10% 중 실제 급등 비율

저장 (결과 폴더)
    predictions_{모델}.csv   종목 × 날짜별 급등 확률 (대시보드용)
    metrics_{모델}.csv       성능표
    b_minus_a_{모델}.csv     B − A와 신뢰구간
    lgbm_순열중요도_A.csv, lgbm_순열중요도_B.csv, logistic_B_표준화계수.csv

사용법
    python3 src/model/test.py --model lightgbm
    python3 src/model/test.py --model all
    주피터: import test; test.main(model="lightgbm")
"""

import argparse
import copy
import sys
from pathlib import Path

import joblib
import models as M
import numpy as np
import pandas as pd
from common import (
    MODEL_DIR,
    OUTPUT_DIR,
    SEED,
    TEST_FILE,
    TRAIN_FILE,
    bootstrap_diff,
    labeled,
    load_dataset,
    pr_auc,
    top_precision,
)
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

PREDICTORS = {
    "lightgbm": M.predict_lightgbm,
    "logistic": M.predict_logistic,
    "rf": M.predict_rf,
}


def load_bundles(name, model_dir, features):
    """저장된 모델 파일을 {(변수 세트, 시드): 모델} 로 읽는다."""
    bundles = {}
    pattern = f"{name}_*_seed*.pth" if name == "lstm" else f"{name}_*.pkl"
    for path in sorted(Path(model_dir).glob(pattern)):
        parts = path.stem.split("_")
        key = parts[1]
        if features and key not in features:
            continue
        if name == "lstm":
            M.require_torch()
            bundles[(key, int(parts[2].replace("seed", "")))] = M.torch.load(
                path, weights_only=False
            )
        else:
            bundles[(key, None)] = joblib.load(path)
    return bundles


def evaluate(name, bundles, test, history, output_dir):
    predictions = test[["keyword", "ticker", "name", "date", "y"]].copy()
    rows, diffs = (
        [{"모델": name, "변수": "기준선", "시드": None, "PR-AUC": test["y"].mean()}],
        [],
    )
    for (key, seed), bundle in sorted(
        bundles.items(), key=lambda item: (item[0][1] or 0, item[0][0])
    ):
        if name == "lstm":
            score = M.predict_lstm(bundle, history, test)
        else:
            score = PREDICTORS[name](bundle, test)
        column = f"prob_{key}" + ("" if seed is None else f"_seed{seed}")
        predictions[column] = score
        valid = ~np.isnan(score)
        rows.append(
            {
                "모델": name,
                "변수": key,
                "시드": seed,
                "PR-AUC": pr_auc(test["y"][valid], score[valid]),
                "ROC-AUC": roc_auc_score(test["y"][valid], score[valid]),
                "상위10%정밀도": top_precision(test["y"][valid], score[valid]),
                "평가행수": int(valid.sum()),
                "설정": bundle["params"],
            }
        )
    for seed in sorted({seed for _, seed in bundles}, key=lambda s: s or 0):
        suffix = "" if seed is None else f"_seed{seed}"
        a, b = "prob_A" + suffix, "prob_B" + suffix
        if a in predictions and b in predictions:
            frame = predictions.dropna(subset=[a, b])
            low, high = bootstrap_diff(
                frame["y"], frame[a].values, frame[b].values, frame["date"].values
            )
            diff = pr_auc(frame["y"], frame[b]) - pr_auc(frame["y"], frame[a])
            diffs.append(
                {
                    "모델": name,
                    "시드": seed,
                    "B−A": diff,
                    "CI하한": low,
                    "CI상한": high,
                    "판정": "B 낮음"
                    if high < 0
                    else ("B 높음" if low > 0 else "차이 없음"),
                }
            )

    metrics, diffs = pd.DataFrame(rows), pd.DataFrame(diffs)
    predictions.to_csv(
        output_dir / f"predictions_{name}.csv", index=False, encoding="utf-8-sig"
    )
    metrics.to_csv(
        output_dir / f"metrics_{name}.csv", index=False, encoding="utf-8-sig"
    )
    print(f"\n[{name}] 시험 {len(test):,}행, 기준선 {test['y'].mean():.4f}")
    print(metrics.drop(columns=["모델", "설정"]).round(4).to_string(index=False))
    if not diffs.empty:
        diffs.to_csv(
            output_dir / f"b_minus_a_{name}.csv", index=False, encoding="utf-8-sig"
        )
        print(diffs.drop(columns="모델").round(4).to_string(index=False))
    return metrics, diffs


def run_permutation(bundles, test, output_dir):
    """2-3 순열 중요도 (lightgbm A·B). 0이면 그 변수 없이도 예측이 같다는 뜻."""
    for key in ("A", "B"):
        if (key, None) not in bundles:
            continue
        bundle = bundles[(key, None)]
        model = bundle["model"]
        if sys.platform == "darwin":
            # 별도 작업 프로세스의 LightGBM/OpenMP 충돌을 피한다.
            # 저장된 모델과 원본 번들의 실행 설정은 바꾸지 않는다.
            model = copy.deepcopy(model)
            model.set_params(n_jobs=1)
        print(
            f"\n[2-3 순열 중요도 {key}] 시작: 변수 {len(bundle['features'])}개 × 5회, "
            f"작업 프로세스=1, LightGBM n_jobs={model.n_jobs}",
            flush=True,
        )
        # LightGBM 내부 병렬화와 joblib 프로세스 병렬화를 중첩하지 않는다.
        perm = permutation_importance(
            model,
            test[bundle["features"]],
            test["y"],
            scoring="average_precision",
            n_repeats=5,
            random_state=SEED,
            n_jobs=1,
        )
        table = pd.DataFrame(
            {"떨어진정도": perm.importances_mean, "표준편차": perm.importances_std},
            index=bundle["features"],
        ).sort_values("떨어진정도", ascending=False)
        table.to_csv(output_dir / f"lgbm_순열중요도_{key}.csv", encoding="utf-8-sig")
        print(f"\n[2-3 순열 중요도 {key}]")
        print(table.round(4).to_string())


def logistic_coefficients(bundles, output_dir):
    if ("B", None) not in bundles:
        return
    bundle = bundles[("B", None)]
    coefficients = pd.Series(
        bundle["model"].coef_[0], index=bundle["columns"]
    ).sort_values(key=abs, ascending=False)
    coefficients.to_csv(output_dir / "logistic_B_표준화계수.csv", encoding="utf-8-sig")
    print(
        "로지스틱 B 표준화 계수 상위 5: "
        + ", ".join(f"{k} {v:+.3f}" for k, v in coefficients.head(5).items())
    )


def main(
    model="all",
    features=None,
    train_path=TRAIN_FILE,
    test_path=TEST_FILE,
    model_dir=MODEL_DIR,
    output_dir=OUTPUT_DIR,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    test_all = load_dataset(test_path)
    test = labeled(test_all).sort_values(["ticker", "date"]).reset_index(drop=True)
    names = M.MODEL_NAMES if model == "all" else (model,)
    history = None
    results = {}
    for name in names:
        print(f"\n[{name}] 평가 시작", flush=True)
        bundles = load_bundles(name, model_dir, features)
        if not bundles:
            print(
                f"\n[{name}] 저장된 모델이 없어 건너뜀 (먼저 python3 src/model/train.py --model {name})"
            )
            continue
        if (
            name == "lstm" and history is None
        ):  # 시험 초반 시퀀스에 학습 기간 마지막 흐름이 필요
            history = pd.concat([load_dataset(train_path), test_all], ignore_index=True)
        results[name] = evaluate(name, bundles, test, history, output_dir)
        if name == "lightgbm":
            run_permutation(bundles, test, output_dir)
        if name == "logistic":
            logistic_coefficients(bundles, output_dir)
    print(f"\n결과 저장: {output_dir}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="모델 평가")
    parser.add_argument(
        "--model", default="all", help="lightgbm | logistic | rf | lstm | all"
    )
    parser.add_argument(
        "--features",
        nargs="+",
        default=None,
        help="A B C 중 여러 개 (생략하면 저장된 것 전부)",
    )
    parser.add_argument(
        "--train", default=TRAIN_FILE, type=Path, help="학습 데이터 (LSTM 시퀀스용)"
    )
    parser.add_argument(
        "--test", default=TEST_FILE, type=Path, help="시험 데이터 (test.csv)"
    )
    parser.add_argument(
        "--model-dir", default=MODEL_DIR, type=Path, help="저장된 모델 폴더"
    )
    parser.add_argument(
        "--output", default=OUTPUT_DIR, type=Path, help="결과 저장 폴더"
    )
    args = parser.parse_args()
    main(args.model, args.features, args.train, args.test, args.model_dir, args.output)
