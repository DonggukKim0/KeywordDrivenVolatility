"""예측 모델 4종의 설정, 전처리, LSTM 클래스를 정의한다.

| 모델 | 구현 | 설정 고르는 법 |
|---|---|---|
| lightgbm | lightgbm.LGBMClassifier | 학습 기간 안 시간순 3겹 검증 (잎 수, 나무 수, 잎 최소 표본) |
| logistic | sklearn LogisticRegression | 학습 기간 안 검증 구간 (규제 강도 C) |
| rf       | sklearn RandomForestClassifier | 학습 기간 안 검증 구간 (잎 최소 표본, 변수 비율) |
| lstm     | LSTMNet (아래 직접 정의, PyTorch) | 학습 기간 안 검증 구간으로 조기 종료, 시드 3개 반복 |

변수 세트: A = 시장 7개, B = 시장 7 + SNS 9, C = SNS 9 (common.MODEL_FEATURES)
불균형 처리: LightGBM scale_pos_weight, 로지스틱·RF class_weight, LSTM pos_weight
"""

import itertools
import sys
import time

import lightgbm as lgb
import numpy as np
from common import BINARY_FEATURES, FIT_END, SEED, TRAIN_END, VALID_START, pr_auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

try:
    import torch
    from torch import nn
except ImportError:  # LSTM만 PyTorch가 필요하다
    torch = nn = None

MODEL_NAMES = ("lightgbm", "logistic", "rf", "lstm")
DEFAULT_FEATURES = {
    "lightgbm": ("A", "B", "C"),
    "logistic": ("A", "B"),
    "rf": ("A", "B"),
    "lstm": ("A", "B"),
}
LSTM_SEEDS = (0, 1, 2)
LSTM_WINDOW = 10  # 직전 10거래일 흐름


def split_fit_valid(train):
    """학습 기간 안에서 설정을 고를 때 쓰는 앞부분(fit)과 검증 구간(valid)."""
    return train[train["date"] <= FIT_END], train[train["date"] >= VALID_START]


# =============================================================================
# LightGBM
# =============================================================================
LGBM_GRID = list(
    itertools.product([15, 31], [200, 500], [100, 400])
)  # 잎 수, 나무 수, 잎 최소 표본


def lgbm(num_leaves, n_estimators, min_child_samples, pos_weight):
    return lgb.LGBMClassifier(
        num_leaves=num_leaves,
        learning_rate=0.03,
        n_estimators=n_estimators,
        min_child_samples=min_child_samples,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        verbose=-1,
        random_state=SEED,
    )


def tune_lgbm(train, features, target="y"):
    """학습 기간 안에서 시간순 3겹 검증(확장 창, 경계 2거래일 제외)으로 설정을 고른다."""
    dates = np.sort(train["date"].unique())
    n = len(dates)
    cuts = [int(n * 0.55), int(n * 0.70), int(n * 0.85), n]
    pos_weight = (1 - train[target].mean()) / train[target].mean()
    best = None
    for params in LGBM_GRID:
        scores = []
        for k in range(3):
            fit = train[train["date"] <= dates[cuts[k] - 3]]
            valid = train[
                (train["date"] >= dates[cuts[k]])
                & (train["date"] <= dates[cuts[k + 1] - 1])
            ]
            model = lgbm(*params, pos_weight).fit(fit[features], fit[target])
            scores.append(
                pr_auc(valid[target], model.predict_proba(valid[features])[:, 1])
            )
        if best is None or np.mean(scores) > best[0]:
            best = (np.mean(scores), params)
    return best[1]


def fit_predict_lgbm(train, test, features, params, target="y"):
    """정해진 설정으로 학습하고 시험 데이터 확률을 낸다 (robustness.py에서 반복 학습에 사용)."""
    pos_weight = (1 - train[target].mean()) / train[target].mean()
    model = lgbm(*params, pos_weight).fit(train[features], train[target])
    return model, model.predict_proba(test[features])[:, 1]


def train_lightgbm(train, features):
    params = tune_lgbm(train, features)
    pos_weight = (1 - train["y"].mean()) / train["y"].mean()
    model = lgbm(*params, pos_weight).fit(train[features], train["y"])
    return {"model": model, "features": features, "params": params}


def predict_lightgbm(bundle, frame):
    return bundle["model"].predict_proba(frame[bundle["features"]])[:, 1]


# =============================================================================
# 전처리 (로지스틱·LSTM)
# =============================================================================
def fit_scaler(fit, features):
    """기준값(중앙값, 상하위 1%, 평균·표준편차)을 fit 데이터에서만 구한다.

    - days_since_day0: 화제 밖 NA → 0 (in_event가 화제 여부를 구분)
    - 결측이 있는 변수: fit 중앙값으로 채우고 "_na" 표시 변수를 추가
    - 연속형: 상하위 1%로 자른 뒤 표준화
    """
    scaler = {"features": list(features), "columns": {}}
    for column in features:
        values = fit[column]
        if column == "days_since_day0":
            scaler["columns"][column] = {"fill": 0.0}
            continue
        entry = {}
        if values.isna().any():
            entry["fill"] = float(values.median())
            entry["flag"] = True
            values = values.fillna(entry["fill"])
        if column not in BINARY_FEATURES:
            low, high = values.quantile([0.01, 0.99])
            clipped = values.clip(low, high)
            entry.update(
                low=float(low),
                high=float(high),
                mean=float(clipped.mean()),
                std=float(clipped.std()),
            )
        scaler["columns"][column] = entry
    return scaler


def apply_scaler(frame, scaler):
    result = frame[scaler["features"]].copy()
    for column in scaler["features"]:
        entry = scaler["columns"][column]
        if entry.get("flag"):
            result[f"{column}_na"] = result[column].isna().astype(int)
        if "fill" in entry:
            result[column] = result[column].fillna(entry["fill"])
        if "mean" in entry:
            result[column] = (
                result[column].clip(entry["low"], entry["high"]) - entry["mean"]
            ) / entry["std"]
    return result


# =============================================================================
# 로지스틱 회귀
# =============================================================================
LOGISTIC_C = (0.001, 0.01, 0.1, 1.0)  # 규제 강도 후보


def train_logistic(train, features):
    fit, valid = split_fit_valid(train)
    best = None
    for c in LOGISTIC_C:
        scaler = fit_scaler(fit, features)
        model = LogisticRegression(C=c, class_weight="balanced", max_iter=2000).fit(
            apply_scaler(fit, scaler), fit["y"]
        )
        score = pr_auc(
            valid["y"], model.predict_proba(apply_scaler(valid, scaler))[:, 1]
        )
        best = max(best or (score, c), (score, c))
    scaler = fit_scaler(train, features)
    x_train = apply_scaler(train, scaler)
    model = LogisticRegression(C=best[1], class_weight="balanced", max_iter=2000).fit(
        x_train, train["y"]
    )
    return {
        "model": model,
        "features": features,
        "scaler": scaler,
        "columns": list(x_train.columns),
        "params": {"C": best[1]},
    }


def predict_logistic(bundle, frame):
    return bundle["model"].predict_proba(apply_scaler(frame, bundle["scaler"]))[:, 1]


# =============================================================================
# 랜덤포레스트 (결측값을 그대로 받음, scikit-learn 1.4 이상)
# =============================================================================
RF_GRID = list(
    itertools.product((50, 200), ("sqrt", 0.5))
)  # 잎 최소 표본, 나무마다 쓰는 변수 비율


def forest(leaf, max_features):
    return RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=leaf,
        max_features=max_features,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=SEED,
    )


def train_rf(train, features):
    fit, valid = split_fit_valid(train)
    best = None
    for leaf, max_features in RF_GRID:
        model = forest(leaf, max_features).fit(fit[features], fit["y"])
        score = pr_auc(valid["y"], model.predict_proba(valid[features])[:, 1])
        if best is None or score > best[0]:
            best = (score, leaf, max_features)
    model = forest(best[1], best[2]).fit(train[features], train["y"])
    return {
        "model": model,
        "features": features,
        "params": {"min_samples_leaf": best[1], "max_features": best[2]},
    }


def predict_rf(bundle, frame):
    return bundle["model"].predict_proba(frame[bundle["features"]])[:, 1]


# =============================================================================
# LSTM (직접 정의한 모델 클래스)
# =============================================================================
if nn is not None:

    class LSTMNet(nn.Module):
        """직전 10거래일의 변수 흐름 → 다음 2거래일 급등 확률(로짓).

        LSTM(은닉 32) → 마지막 시점 출력 → Dropout 0.2 → Linear(1)
        """

        def __init__(self, n_features, hidden=32, dropout=0.2):
            super().__init__()
            self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
            self.drop = nn.Dropout(dropout)
            self.out = nn.Linear(hidden, 1)

        def forward(self, x):
            hidden, _ = self.lstm(x)
            return self.out(self.drop(hidden[:, -1])).squeeze(-1)


def require_torch():
    if torch is None:
        raise ImportError("LSTM은 PyTorch가 필요합니다 (pip install torch)")
    # 평가에서도 학습과 동일하게 OpenMP 병렬 연산 충돌을 피한다.
    if sys.platform == "darwin":
        torch.set_num_threads(1)


def build_sequences(frame, scaler, window=LSTM_WINDOW):
    """종목별로 직전 window행을 묶어 시퀀스를 만든다. 반환: (시퀀스 배열, 각 시퀀스 마지막 행 정보).

    frame에는 y가 비어 있는 행도 넣는다 (흐름을 끊지 않기 위해). 학습·평가 대상은 호출하는 쪽에서 y로 거른다.
    """
    frame = frame.sort_values(["ticker", "date"]).reset_index(drop=True)
    prepared = apply_scaler(frame, scaler)[scaler["features"]].values.astype(np.float32)
    ends = []
    for _, index in frame.groupby("ticker").indices.items():
        ends += list(np.sort(index)[window - 1 :])
    ends = np.array(ends)
    return np.stack([prepared[end - window + 1 : end + 1] for end in ends]), frame.loc[
        ends
    ].reset_index(drop=True)


def predict_lstm_array(model, x):
    model.eval()
    with torch.no_grad():
        return np.concatenate(
            [
                torch.sigmoid(model(torch.from_numpy(x[i : i + 4096]))).numpy()
                for i in range(0, len(x), 4096)
            ]
        )


def train_lstm(train, features, seed, max_epochs=20, patience_limit=3):
    """학습 기간 앞부분(~2025-10-23)으로 학습, 검증 구간(2025-10-28 ~ 12-26) PR-AUC가 3번 연속 안 오르면 멈춘다."""
    started = time.monotonic()

    def log(message):
        elapsed = time.monotonic() - started
        print(f"[lstm seed={seed} +{elapsed:.1f}s] {message}", flush=True)

    require_torch()
    log("PyTorch 스레드 설정 시작")
    # macOS에서는 LightGBM의 Homebrew libomp와 PyTorch에 포함된 libomp가
    # 함께 로드될 수 있다. 병렬 randperm/인덱싱의 barrier에서 충돌하거나
    # 멈추는 문제를 피하기 위해 LSTM CPU 연산은 단일 스레드로 실행한다.
    threads = 1 if sys.platform == "darwin" else 2
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    np.random.seed(seed)
    log(f"PyTorch 설정 완료: CPU 스레드={torch.get_num_threads()}")
    if sys.platform == "darwin":
        log("macOS OpenMP 라이브러리 충돌 방지를 위해 CPU 1스레드 사용")
    log("전처리: 스케일러 계산 시작")
    scaler = fit_scaler(train[train["date"] <= FIT_END], features)
    log("전처리: 시퀀스 생성 시작")
    x, meta = build_sequences(train, scaler)
    has_y = meta["y"].notna().values
    x, meta = x[has_y], meta[has_y]
    y = meta["y"].values.astype(np.float32)
    fit = (meta["date"] <= FIT_END).values
    valid = ((meta["date"] >= VALID_START) & (meta["date"] <= TRAIN_END)).values
    log(
        f"전처리 완료: 전체 시퀀스={len(x):,}, 학습={int(fit.sum()):,}, 검증={int(valid.sum()):,}, 입력 모양={x.shape}"
    )

    log("모델·옵티마이저 초기화 및 학습 텐서 준비 시작")
    model = LSTMNet(x.shape[2])
    optimizer = torch.optim.Adam(model.parameters(), 1e-3)
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor((1 - y[fit].mean()) / y[fit].mean())
    )
    x_fit, y_fit = torch.from_numpy(x[fit]), torch.from_numpy(y[fit])
    log("모델·옵티마이저 및 학습 텐서 준비 완료")
    best, patience, epochs = (-1.0, None), 0, 0
    for _ in range(max_epochs):
        epochs += 1
        epoch_started = time.monotonic()
        model.train()
        log(f"에폭 {epochs}/{max_epochs} 시작")
        log(f"에폭 {epochs}: 학습 순서 셔플 시작")
        order = torch.randperm(len(x_fit))
        log(f"에폭 {epochs}: 학습 순서 셔플 완료")
        batch_count = (len(x_fit) + 1023) // 1024
        for batch_number, i in enumerate(range(0, len(x_fit), 1024), start=1):
            batch = order[i : i + 1024]
            log(f"에폭 {epochs} 배치 {batch_number}/{batch_count}: 텐서 인덱싱 시작")
            x_batch, y_batch = x_fit[batch], y_fit[batch]
            log(
                f"에폭 {epochs} 배치 {batch_number}/{batch_count}: 인덱싱 완료, 학습 연산 시작"
            )
            optimizer.zero_grad()
            loss_fn(model(x_batch), y_batch).backward()
            optimizer.step()
            log(f"에폭 {epochs} 배치 {batch_number}/{batch_count}: 완료")
        log(f"에폭 {epochs}: 검증 PR-AUC 계산 시작")
        score = pr_auc(y[valid], predict_lstm_array(model, x[valid]))
        log(
            f"에폭 {epochs}: 검증 PR-AUC={score:.4f} ({time.monotonic() - epoch_started:.1f}초)"
        )
        if score > best[0]:
            best, patience = (
                (score, {k: v.clone() for k, v in model.state_dict().items()}),
                0,
            )
        else:
            patience += 1
            if patience >= patience_limit:
                log(f"조기 종료: {patience_limit}회 연속 검증 점수 개선 없음")
                break
    log(f"학습 완료: {epochs} epoch")
    return {
        "state_dict": best[1],
        "features": features,
        "scaler": scaler,
        "n_features": x.shape[2],
        "seed": seed,
        "params": {"valid_pr_auc": best[0], "epochs": epochs},
    }


def load_lstm(bundle):
    require_torch()
    model = LSTMNet(bundle["n_features"])
    model.load_state_dict(bundle["state_dict"])
    return model


def predict_lstm(bundle, history, target_rows):
    """history(학습+시험 전체 행)로 시퀀스를 만든 뒤 target_rows(ticker, date)에 해당하는 예측만 돌려준다."""
    x, meta = build_sequences(history, bundle["scaler"])
    keys = meta[["ticker", "date"]].assign(row=np.arange(len(meta)))
    rows = target_rows[["ticker", "date"]].merge(
        keys, on=["ticker", "date"], how="left"
    )["row"]
    probability = np.full(len(rows), np.nan)
    found = rows.notna().values
    probability[found] = predict_lstm_array(
        load_lstm(bundle), x[rows[found].astype(int).values]
    )
    return probability
