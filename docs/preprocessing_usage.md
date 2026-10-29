# 전처리·수집 스크립트 사용법

아래 명령은 프로젝트 루트에서 실행합니다.

## 주가 수집

`collect_stock_data.py`는 후보 종목별 일별 주가와 시장지수 자료를 수집합니다. 기본으로 `theme_candidates.csv`를 읽고 `theme_stock_daily.csv`를 생성합니다.

```bash
python3 src/preprocessing/collect_stock_data.py
```

수집 기간, 입력 파일, 출력 경로를 지정할 수 있습니다. `--limit`은 테스트용 종목 수 제한입니다.

```bash
python3 src/preprocessing/collect_stock_data.py \
  --start 2024-10-01 \
  --end 2026-10-24 \
  --input data/raw/theme_candidates.csv \
  --output data/raw/theme_stock_daily.csv
```

## OpenDART 자료 수집

`collect_opendart_data.py`는 후보 종목의 공시 목록과 재무제표 자료를 수집합니다. API 키를 `.opendart`에 준비한 뒤 실행합니다. 기본 출력 디렉토리는 `data/raw/opendart/`입니다.

```bash
python3 src/preprocessing/collect_opendart_data.py
```

입력 후보 파일, API 키 파일, 조회 기간, 출력 디렉토리를 지정할 수 있습니다.

```bash
python3 src/preprocessing/collect_opendart_data.py \
  --input data/raw/theme_candidates.csv \
  --key-file .opendart \
  --start 2024-10-04 \
  --end 2026-10-24 \
  --raw-dir data/raw/opendart
```

## SNS 피쳐 생성

`preprocess_sns_data.py`는 SNS 일별 원자료를 읽어 `sns_keyword_features.csv`를 생성합니다.

```bash
python3 src/preprocessing/preprocess_sns_data.py
```

첫 번째 인자로 입력 CSV, 두 번째 인자로 출력 CSV 경로를 지정할 수 있습니다.

```bash
python3 src/preprocessing/preprocess_sns_data.py \
  data/raw/sns_daily_mentions_sentiment.csv \
  data/processed/sns_keyword_features.csv
```

## 시장 피쳐 생성

`preprocess_market_data.py`는 주가, SNS, 공시 자료를 연결해 종목별·키워드별 시장 피쳐 CSV를 생성합니다.

```bash
python3 src/preprocessing/preprocess_market_data.py \
  --stock data/raw/theme_stock_daily.csv \
  --disclosures data/raw/opendart/opendart_disclosures.csv \
  --financials data/raw/opendart/opendart_financials.csv \
  --sns data/raw/sns_daily_mentions_sentiment.csv \
  --output-dir data/processed
```

기본 출력 디렉토리는 `data/processed/`이며, `--disclosure-start`와 `--disclosure-end`로 공시 가용 기간을 지정할 수 있습니다.
