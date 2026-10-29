"""모델 입력 데이터를 날짜 기준으로 학습·시험 파일로 나눈다.

    학습: 2024-11-01 ~ 2025-12-26   → train.csv
    시험: 2026-01-02 ~ 2026-10-23   → test.csv
    제외: 2025-12-29, 30 (y 구간이 시험 기간에 걸침)

- 무작위로 섞지 않는다. 같은 날 같은 테마 종목들이 함께 움직여서, 섞으면 같은 날 정보가 학습과 시험에 동시에 들어간다.
- y가 비어 있는 행(마지막 2거래일, 거래정지일)도 남긴다. LSTM이 직전 10거래일 흐름을 만들 때 필요하고,
  학습·평가할 때는 train.py·test.py가 y가 있는 행만 쓴다.
- 학습 기간 안의 검증 구간(설정 고르기용)과 시간 이동 검증(robustness.py)은 코드 안에서 날짜로 나눈다.

사용법: python3 src/model/data_split.py [모델 데이터] [저장 폴더]
"""

import sys
from pathlib import Path

from common import DATA_DIR, DATASET, TEST_START, TRAIN_END, labeled, load_dataset


def main(dataset_path=DATASET, output_dir=DATA_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = (
        load_dataset(dataset_path)
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )
    train = data[data["date"] <= TRAIN_END]
    test = data[data["date"] >= TEST_START]
    gap = sorted(set(data["date"]) - set(train["date"]) - set(test["date"]))

    train.to_csv(output_dir / "train.csv", index=False, encoding="utf-8-sig")
    test.to_csv(output_dir / "test.csv", index=False, encoding="utf-8-sig")
    for name, frame in (("학습", train), ("시험", test)):
        valid = labeled(frame)
        print(
            f"{name}: {frame['date'].min()} ~ {frame['date'].max()}, {len(frame):,}행 "
            f"(y 있음 {len(valid):,}행, 양성 비율 {valid['y'].mean():.2%})"
        )
    print(f"제외한 날짜: {', '.join(gap)}")
    print(f"저장: {output_dir / 'train.csv'}, {output_dir / 'test.csv'}")
    return train, test


if __name__ == "__main__":
    main(*[Path(argument) for argument in sys.argv[1:3]])
