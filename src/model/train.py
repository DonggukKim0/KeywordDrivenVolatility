"""모델 학습: 학습 데이터(train.csv)로 모델을 만들고 저장한다.

모델과 변수 세트를 골라 학습할 수 있다.
    --model     lightgbm | logistic | rf | lstm | all          (기본 all)
    --features  A B C 중 여러 개 (A = 시장 7, B = 시장 7 + SNS 9, C = SNS 9)
                생략하면 lightgbm은 A B C, 나머지는 A B

저장 (모델 폴더)
    lightgbm_A.pkl, logistic_B.pkl, rf_A.pkl ...      (joblib)
    lstm_A_seed0.pth, lstm_B_seed2.pth ...              (torch.save, 시드 0·1·2)
    train_summary.csv                                   (모델별 고른 설정)

사용법
    python3 src/model/train.py --model lightgbm --features A B
    python3 src/model/train.py --model all
    주피터: import train; train.main(model="lightgbm", features=["A", "B"])
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import joblib
import models as M
import pandas as pd
from common import MODEL_DIR, MODEL_FEATURES, TRAIN_FILE, labeled, load_dataset


def resolve(model, features):
    """선택한 모델·변수 세트를 (모델, 변수 세트) 목록으로 바꾼다."""
    names = M.MODEL_NAMES if model == "all" else (model,)
    for name in names:
        if name not in M.MODEL_NAMES:
            raise ValueError(
                f"모델 이름은 {', '.join(M.MODEL_NAMES)}, all 중 하나: {name}"
            )
    jobs = []
    for name in names:
        for key in features or M.DEFAULT_FEATURES[name]:
            if key not in MODEL_FEATURES:
                raise ValueError(f"변수 세트는 A, B, C 중 하나: {key}")
            jobs.append((name, key))
    return jobs


def main(model="all", features=None, train_path=TRAIN_FILE, model_dir=MODEL_DIR):
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    history = load_dataset(train_path)  # y가 없는 행 포함 (LSTM 시퀀스용)
    train = labeled(history)
    print(
        f"학습 데이터 {train['date'].min()} ~ {train['date'].max()}, {len(train):,}행, 양성 비율 {train['y'].mean():.2%}",
        flush=True,
    )

    rows = []
    for name, key in resolve(model, features):
        columns = MODEL_FEATURES[key]
        start = time.time()
        print(f"[{datetime.now():%H:%M:%S}] 시작: {name} {key}", flush=True)
        if name == "lstm":
            for seed in M.LSTM_SEEDS:
                print(
                    f"[{datetime.now():%H:%M:%S}] 시작: lstm {key}, 시드 {seed}",
                    flush=True,
                )
                bundle = M.train_lstm(history, columns, seed)
                path = model_dir / f"lstm_{key}_seed{seed}.pth"
                M.torch.save(bundle, path)
                rows.append(
                    {
                        "모델": name,
                        "변수": key,
                        "시드": seed,
                        "설정": bundle["params"],
                        "파일": path.name,
                    }
                )
                print(
                    f"[lstm {key} 시드 {seed}] 검증 PR-AUC {bundle['params']['valid_pr_auc']:.4f}, "
                    f"{bundle['params']['epochs']} epoch → {path.name}",
                    flush=True,
                )
            continue
        trainer = {
            "lightgbm": M.train_lightgbm,
            "logistic": M.train_logistic,
            "rf": M.train_rf,
        }[name]
        bundle = trainer(train, columns)
        path = model_dir / f"{name}_{key}.pkl"
        joblib.dump(bundle, path)
        rows.append(
            {
                "모델": name,
                "변수": key,
                "시드": None,
                "설정": bundle["params"],
                "파일": path.name,
            }
        )
        print(
            f"[{name} {key}] 설정 {bundle['params']} ({time.time() - start:.0f}초) → {path.name}",
            flush=True,
        )

    summary = pd.DataFrame(rows)
    summary_path = model_dir / "train_summary.csv"
    if summary_path.exists():  # 다른 모델을 따로 학습했을 때 기존 기록을 남기고 갱신
        old = pd.read_csv(summary_path, encoding="utf-8-sig")
        summary = pd.concat(
            [
                old[~old["파일"].isin(summary["파일"])],
                summary.astype({"시드": "object"}),
            ]
        )
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"\n모델 저장: {model_dir}", flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="모델 학습")
    parser.add_argument(
        "--model", default="all", help="lightgbm | logistic | rf | lstm | all"
    )
    parser.add_argument("--features", nargs="+", default=None, help="A B C 중 여러 개")
    parser.add_argument(
        "--train", default=TRAIN_FILE, type=Path, help="학습 데이터 (train.csv)"
    )
    parser.add_argument(
        "--model-dir", default=MODEL_DIR, type=Path, help="모델 저장 폴더"
    )
    args = parser.parse_args()
    main(args.model, args.features, args.train, args.model_dir)
