# EDA 데이터 준비

EDA는 STEP01·STEP02·STEP03으로 정리합니다. STEP01·02는 RQ1의 언급량 연관과 거래량 반응을, STEP03은 급등 사건과 SNS 변화를 분석합니다.
STEP03은 기본 데이터 준비 후 사건 분석, 감성 변화와 이후 수익률, 뉴스·SNS 당일 동행 분석을 각각 실행합니다.

## 실행

프로젝트 루트에서 실행합니다. 기본 입력·출력 경로는 스크립트 위치를 기준으로 정하므로 작업 디렉토리가 달라도 같습니다.

```bash
python3 src/eda/step03_prepare_data.py
```

입력은 기존 프로젝트의 CSV 6개입니다.

| 입력 | 용도 |
| --- | --- |
| `data/raw/theme_stock_daily.csv` | 종가·거래량·시장지수 |
| `data/raw/sns_daily_mentions_sentiment.csv` | 일별 언급량·감성어·채널별 건수 |
| `data/processed/market_stock_features.csv` | 기존 시장 피처 연결·재계산 비교 |
| `data/processed/sns_keyword_features.csv` | 기존 SNS 피처 연결·재계산 비교 |
| `data/raw/opendart/opendart_disclosures.csv` | 공시 분류 및 최근 접수 건수 |
| `data/raw/opendart/opendart_financials.csv` | 이전 재무 보고서 접수일 |

입력 CSV와 전달받은 노트북·그림은 변경하지 않습니다. 같은 출력 폴더로 다시 실행하면 준비 결과 5개를 갱신합니다.

## 출력

기본 위치는 `data/processed/eda/step03/`입니다.

| 파일 | 내용 |
| --- | --- |
| `stock_daily.csv.gz` | 종목×거래일. 가격·기존 시장/SNS 피처·분석 변수·공시/재무 접수 정보 |
| `sns_daily.csv.gz` | 키워드×달력일. SNS 원자료와 뉴스/SNS 언급량·감성 변화 변수 |
| `data_dictionary.csv` | 두 데이터의 변수명·형식·출처·계산식·시점·결측 건수 |
| `feature_comparison.csv` | 제공 피처와 원자료 재계산 값의 비교. 불일치 시 경고하며 원값 보존 |
| `preparation_metadata.json` | 입력 경로·SHA-256·자료 범위·계산 설정·점검 결과·연결 행 수·결측률 |

종목코드의 앞자리 0을 보존해서 읽습니다.

```python
from pathlib import Path
import pandas as pd

prepared = Path("data/processed/eda/step03")
stock_daily = pd.read_csv(
    prepared / "stock_daily.csv.gz", encoding="utf-8-sig", dtype={"ticker": str},
    parse_dates=["date", "sns_cutoff_date", "sns_source_date",
                 "sns_feature_source_date", "financial_last_receipt_date"],
)
sns_daily = pd.read_csv(
    prepared / "sns_daily.csv.gz", encoding="utf-8-sig", parse_dates=["date"],
)
```

## EDA 분석 결과 위치

준비된 입력 데이터는 `data/processed/eda/step03/`에서 읽고, 후속 분석이 생성하는 표와 그림은 프로젝트 루트의 `eda-outputs/`에 저장합니다.
각 STEP에 같은 구조를 사용합니다. STEP03 사건 분석을 실행하면 아래 위치에 결과가 생성됩니다.

```text
eda-outputs/
├── step01/
│   ├── tables/
│   └── figures/
├── step02/
│   ├── tables/
│   └── figures/
└── step03/
    ├── tables/
    └── figures/
```

## STEP01 언급량과 수익률 연관 분석

```bash
python3 src/eda/step01_mention_association.py
```

기본 입력은 `data/processed/model_dataset.csv`입니다. 키워드·종목별 언급량 급증과 초과수익률의 순위 상관, 상관 상위 키워드의 대표 이벤트를 분석합니다.
표와 그림은 `eda-outputs/step01/tables/`, `eda-outputs/step01/figures/`에 저장합니다.
입력·출력 경로는 `--input`, `--output`으로 변경할 수 있습니다.
STEP01은 CAR 구간 검정, 공시 유무 비교, 일별 급등 비율과 다음 날 시가 매수 시뮬레이션도 생성합니다. 주가 원자료는 기본 `data/raw/theme_stock_daily.csv`를 사용하며 `--stock`으로 변경할 수 있습니다. 이벤트 창은 Day −5~+10, 매수 시점은 Day 0·1·2, 보유 기간은 1·2·3·5거래일입니다.

## STEP02 화제 전후 거래량·변동폭 반응

```bash
python3 src/eda/step02_volume_response.py
```

기본 입력은 `data/processed/model_dataset.csv`와 `data/raw/theme_stock_daily.csv`입니다. Day −10~−1을 기준으로 Day 0~+2 거래량·일중 변동폭을 계산하고, 30거래일 뒤 가짜 시점과 비교합니다.
표와 그림은 `eda-outputs/step02/tables/`, `eda-outputs/step02/figures/`에 저장합니다.
입력·출력 경로는 `--input`, `--stock`, `--output`으로 변경할 수 있습니다.

## STEP03 급등·붕괴 사건 분석

데이터 준비 후 실행합니다.

```bash
python3 src/eda/step03_events.py
```

`stock_daily.csv.gz`를 읽어 사건 선정, 붕괴 빈도·시점, 사례 3개, SNS 동시 신호 비교,
하락 기준 15%·20%·25% 민감도를 계산합니다. 표와 요약 수치는 터미널에도 출력합니다.

- Day 0: 최근 10거래일 종가 상승률 20% 이상 조건에 처음 진입한 날.
- 사건 재선정: 같은 종목은 Day 20까지 새 사건을 만들지 않으며, 상승 조건에 다시 진입해야 합니다.
- 붕괴: Day −10부터 해당일까지의 최고 종가 대비 20% 이상 낮아진 첫날. 관찰 기간은 Day 1~20입니다.
- 붕괴 비율: 확인 전 붕괴를 제외하고 20거래일을 모두 관찰한 사건을 분모로 사용합니다.
- SNS 동시 신호: 전날까지의 7일 관심 감소와 7일 부정 감성어 비율 증가를 모두 충족한 상태입니다. 결측은 별도로 유지합니다.

| 출력 위치 | 내용 |
| --- | --- |
| `eda-outputs/step03/tables/events.csv` | 상승 사건당 한 행, 붕괴·관측 완료 여부 |
| `eda-outputs/step03/tables/event_panel.csv.gz` | 사건×거래일별 가격 상태·SNS 신호·다음날 붕괴 결과 |
| `eda-outputs/step03/tables/06_*.csv` ~ `11_*.csv`, `15_*.csv` | 표본 흐름·빈도·시점·사례·신호 비교·민감도 결과표 |
| `eda-outputs/step03/figures/01_*.png` ~ `03_*.png` | 빈도·시점 그림, 사례 그림 3개, 신호 비교 그림: 총 5개 |

입력 파일은 변경하지 않으며, 같은 출력 경로로 다시 실행하면 해당 사건 분석 결과를 갱신합니다.
기간이 짧아 관찰 완료 사건이 없으면 오류를 표시합니다. 기본 사례가 다른 입력에 없으면 메시지를 출력하고 해당 사례를 건너뜁니다.
그룹 차이에는 테마·시기·가격 상태가 섞여 있으며, 이 분석은 SNS의 인과효과나 예측 성능 검증이 아닙니다.

경로를 바꾸려면 다음처럼 실행합니다.

```bash
python3 src/eda/step03_events.py --input data/processed/eda/step03/stock_daily.csv.gz --output /tmp/step03
```

## STEP03 부정 감성 변화와 이후 수익률

```bash
python3 src/eda/step03_sentiment.py
```

`stock_daily.csv.gz`에서 관측일 종가부터 5거래일 뒤 종가까지의 수익률을 계산합니다.
전날까지의 3일 부정 감성어 비율 변화로 나눈 수익률 상자그림과 종목별 스피어만 상관을 생성합니다.
상관은 수익률과 시장 대비 수익률 각각 계산하며, 유효 쌍이 20개 미만이거나 상수인 경우 결측으로 둡니다.
종목별 마지막 5행의 미래 결과도 결측으로 유지합니다.

- `tables/daily_outcomes.csv.gz`: 미래 수익률을 연결한 별도 표본.
- `tables/12_negative_change_boxplot.csv`: 변화 구간별 표본 수와 수익률 요약.
- `tables/13_stock_correlations.csv`: 종목별 수익률·시장 대비 수익률 상관.
- `figures/04_negative_change_boxplot.png`, `figures/05_stock_correlations.png`.

## STEP03 뉴스·SNS와 당일 주가 동행

```bash
python3 src/eda/step03_comovement.py
```

`stock_daily.csv.gz`의 가격과 `sns_daily.csv.gz`의 **당일 언급량**을 사용합니다.
뉴스 3채널과 뉴스 외 SNS 7채널의 전 거래일 대비 `log1p` 언급량 변화를 각각 계산합니다.
테마의 당일 시장 대비 수익률과 같은 날짜 표본으로 순위 상관을 비교합니다. 주말 언급량은 합산하지 않습니다.
유효 종목이 5개 미만인 테마·날짜는 수익률을 결측으로 두며, 상관 계산에는 최소 40쌍이 필요합니다.

- `tables/14_theme_excluded_stock_days.csv`: 거래량·수익률 조건에서 제외된 종목·날짜.
- `tables/theme_same_day_base.csv.gz`: 테마×거래일 분석 데이터.
- `tables/14_theme_same_day_correlations.csv`: 테마별 뉴스·SNS 상관과 동일 표본 일수.
- `figures/06_theme_same_day.png`.

위 경로는 모두 `eda-outputs/step03/` 기준입니다. 분석 스크립트는 결과표와 그림만 저장하며 설명·메타데이터용 `.md`·`.json`은 생성하지 않습니다. 수치와 해석은 터미널에 출력합니다.
각 스크립트는 `--input`·`--output`으로 경로를 바꿀 수 있고,
동행 분석은 `--sns`로 SNS 입력을 지정할 수 있습니다. 세 분석은 서로의 결과 파일에 의존하지 않으며,
표·그림 저장과 시각화 설정은 `src/eda/eda_reporting.py`를 공유하며, STEP03 입력 데이터 읽기와 상관 계산은 `src/eda/step03_eda_common.py`에서 제공합니다.

## 계산과 시점 기준

- 가격의 이동 기간은 거래일, SNS의 이동 기간은 주말을 포함한 달력일입니다.
- `stock_daily`의 관측일에는 전날까지의 SNS를 연결합니다. 실제 연결된 SNS 날짜도 저장합니다.
- 당일 동행 분석에서는 `sns_daily`를 당일 가격에 연결합니다. `stock_daily`의 전날 SNS 값을 재사용하지 않습니다.
- 관심 감소는 앞선 기간과 최근 기간의 평균 언급량에 각각 `log1p`를 적용한 차이입니다. 부정 변화는 두 기간의 부정 감성어 비율 차이입니다.
- 부정 비율의 분모는 긍정·부정·중립 감성어 건수이며, 분모가 0이면 결측으로 둡니다.
- 공시는 접수일 다음 거래일부터 건수에 포함하고, 재무 정보는 관측일보다 이전 접수일만 연결합니다.
- 기존 시장 피처는 `market_file__`, 기존 SNS 피처는 `sns_file__` 접두사로 보존합니다.
- 준비 데이터에는 미래 결과를 넣지 않습니다. 사건 결과는 `step03_events.py`, 이후 수익률은 `step03_sentiment.py`에서 별도로 생성합니다.

## 경로 변경

```bash
python3 src/eda/step03_prepare_data.py --output /tmp/step03
```

개별 입력도 `--stock-raw`, `--sns-raw`, `--market-features`, `--sns-features`,
`--disclosures`, `--financials`로 지정할 수 있습니다. 전체 옵션은 `--help`로 확인합니다.

프로젝트의 수집·전처리 과정은 [수집·전처리 스크립트 사용법](preprocessing_usage.md)을 참고하세요.
