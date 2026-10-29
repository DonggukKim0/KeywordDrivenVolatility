"""후보 CSV의 고유 종목에 대해 공시 목록과 재무제표를 수집한다.

예: python3 src/preprocessing/collect_opendart_data.py --start 2024-10-04 --end 2026-10-24
키: DART_API_KEY 환경변수, .opendart 파일, 기존 접근 확인 스크립트의 입력 변수 순.
전체 수집이 성공한 경우에만 raw 파일을 교체한다. --limit 1로 시험 수집 가능.
--reuse-existing은 동일 분석 기간의 기존 raw를 재사용하고 신규 종목만 요청한다.
"""

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen
from zipfile import BadZipFile, ZipFile
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data/raw/opendart"
DISCLOSURE_COLUMNS = [
    "ticker",
    "name",
    "corp_code",
    "corp_name",
    "corp_cls",
    "report_nm",
    "rcept_no",
    "rcept_dt",
    "flr_nm",
    "rm",
    "is_major",
    "disclosure_url",
]
FINANCIAL_COLUMNS = [
    "ticker",
    "corp_code",
    "rcept_no",
    "rcept_dt",
    "bsns_year",
    "reprt_code",
    "fs_div",
    "sj_div",
    "account_id",
    "account_nm",
    "thstrm_amount",
    "thstrm_add_amount",
    "frmtrm_amount",
    "frmtrm_q_amount",
    "frmtrm_add_amount",
    "currency",
]


def parse_date(value):
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "날짜는 YYYY-MM-DD 형식이어야 합니다."
        ) from None


def load_candidates(path):
    frame = pd.read_csv(path, dtype=str, encoding="utf-8-sig", keep_default_na=False)
    required = ["category", "keyword", "theme", "ticker", "name"]
    if not set(required).issubset(frame.columns) or frame.empty:
        raise ValueError(
            "후보 CSV에는 category,keyword,theme,ticker,name과 1개 이상의 행이 필요합니다."
        )
    columns = required + (["is_leader"] if "is_leader" in frame.columns else [])
    frame = frame[columns].apply(lambda col: col.str.strip())
    if frame.ticker.str.fullmatch(r"[0-9]{1,6}").all():
        frame["ticker"] = frame.ticker.str.zfill(6)
    if frame.eq("").any().any() or not frame.ticker.str.fullmatch(r"\d{6}").all():
        raise ValueError("후보 CSV의 필수 값 또는 6자리 종목코드를 확인하세요.")
    aliases = frame.groupby("ticker").name.nunique()
    if aliases.gt(1).any():
        print(
            "동일 코드에 여러 종목명 표기가 있습니다. 코드를 기준으로 한 번만 수집하고 "
            f"첫 번째 이름을 사용합니다: {', '.join(aliases[aliases.gt(1)].index)}",
            flush=True,
        )
    if "is_leader" in frame.columns:
        if not frame.is_leader.isin(["0", "1"]).all():
            raise ValueError("is_leader는 0 또는 1이어야 합니다.")
        if frame.groupby(["keyword", "ticker"]).is_leader.nunique().gt(1).any():
            raise ValueError("같은 키워드·종목에 상충하는 is_leader 값이 있습니다.")
    # 테마·종목 수는 제한하지 않는다. 중복 연결은 한 번만 저장한다.
    frame = frame.drop_duplicates(["keyword", "ticker"]).reset_index(drop=True)
    if "is_leader" in frame.columns:
        if (
            not frame.assign(is_leader=frame.is_leader.astype(int))
            .groupby("keyword")
            .is_leader.sum()
            .eq(1)
            .all()
        ):
            raise ValueError("키워드마다 is_leader=1인 종목을 정확히 1개 지정하세요.")
    return frame


def resolve_key(path):
    key = os.environ.get("DART_API_KEY", "").strip()
    if not key:
        from check_opendart_access import OPENDART_API_KEY, load_api_key

        if Path(path).is_file():
            key = load_api_key(path)
        else:
            key = OPENDART_API_KEY.strip()
    if not re.fullmatch(r"[A-Za-z0-9]{40}", key):
        raise ValueError(
            "DART_API_KEY, .opendart 또는 접근 확인 스크립트에 API 키를 지정하세요."
        )
    return key


class DartClient:
    def __init__(self, key, interval=0.3):
        self.key = key
        self.interval = interval
        self.last_request = 0.0

    def request(self, endpoint, params=None, binary=False):
        query = urlencode({"crtfc_key": self.key, **(params or {})})
        url = f"https://opendart.fss.or.kr/api/{endpoint}?{query}"
        for attempt in range(3):
            time.sleep(max(0, self.interval - (time.monotonic() - self.last_request)))
            self.last_request = time.monotonic()
            try:
                with urlopen(url, timeout=30) as response:
                    body = response.read()
            except HTTPError as exc:
                if exc.code < 500:
                    raise ValueError(f"OpenDART HTTP 오류 {exc.code}") from None
            except (URLError, TimeoutError, OSError):
                pass  # 예외 URL에 포함될 수 있는 키는 절대 출력하지 않는다.
            else:
                if binary and body.startswith(b"PK"):
                    return body
                try:
                    payload = (
                        json.loads(body)
                        if not binary
                        else {node.tag: node.text for node in ET.fromstring(body)}
                    )
                except (ValueError, ET.ParseError, UnicodeError):
                    raise ValueError("OpenDART 응답 형식 오류") from None
                if not isinstance(payload, dict):
                    raise ValueError("OpenDART 응답은 객체여야 합니다.")
                status = payload.get("status")
                if status == "000":
                    if binary:
                        raise ValueError("고유번호 ZIP 응답이 없습니다.")
                    return payload
                if status == "013" and not binary:
                    return {"list": [], "total_page": 0}
                if status not in {"800", "900"}:
                    safe_status = (
                        status
                        if str(status).isdigit() and len(str(status)) == 3
                        else "unknown"
                    )
                    raise ValueError(
                        f"OpenDART 오류 코드 {safe_status} (인증·IP·요청 한도 등을 확인하세요.)"
                    )
            if attempt < 2:
                time.sleep(2**attempt)
        raise ValueError("OpenDART 연결 실패 또는 서버 오류: 3회 요청 실패")

    def disclosures(self, corp_code, start, end, major=False, disclosure_type=None):
        params = {
            "corp_code": corp_code,
            "bgn_de": start.strftime("%Y%m%d"),
            "end_de": end.strftime("%Y%m%d"),
            "last_reprt_at": "N",
            "sort": "date",
            "sort_mth": "asc",
            "page_count": 100,
        }
        if major or disclosure_type:
            params["pblntf_ty"] = "B" if major else disclosure_type
        records = []
        page, pages = 1, 1
        while page <= pages:
            payload = self.request("list.json", {**params, "page_no": page})
            rows = payload.get("list", [])
            pages = int(payload.get("total_page", 0))
            if not isinstance(rows, list) or (pages >= page and not rows):
                raise ValueError("공시 페이지 응답이 비어 있거나 잘못되었습니다.")
            records.extend(rows)
            page += 1
        return records


def corp_mapping(client):
    body = client.request("corpCode.xml", binary=True)
    try:
        with ZipFile(BytesIO(body)) as archive:
            tree = ET.fromstring(archive.read("CORPCODE.xml"))
    except (BadZipFile, KeyError, ET.ParseError):
        raise ValueError("고유번호 ZIP/XML을 읽을 수 없습니다.") from None
    mapping = {}
    for node in tree.findall("list"):
        ticker = (node.findtext("stock_code") or "").strip()
        code = (node.findtext("corp_code") or "").strip()
        if re.fullmatch(r"\d{6}", ticker) and re.fullmatch(r"\d{8}", code):
            if ticker in mapping and mapping[ticker] != code:
                raise ValueError(f"종목 {ticker}의 고유번호가 중복됩니다.")
            mapping[ticker] = code
    return mapping


def receipt_date(number, reported_date):
    if not re.fullmatch(r"\d{14}", str(number)):
        raise ValueError("API 응답의 접수번호가 유효하지 않습니다.")
    # 접수번호 앞자리와 실제 접수일은 다를 수 있다. 공식 접수일을 사용한다.
    return datetime.strptime(reported_date, "%Y%m%d").date().isoformat()


def collect(client, candidates, start, end):
    stocks = candidates[["ticker", "name"]].drop_duplicates("ticker")
    mapping = corp_mapping(client)
    missing = [t for t in stocks.ticker if t not in mapping]
    if missing:
        raise ValueError(f"DART 고유번호 미매핑 종목: {', '.join(missing)}")
    disclosures, financials, coverage = [], [], []
    # 30일 rolling 및 전 7일 대비 변화량을 계산하기 위한 여유 구간.
    query_start = start - timedelta(days=30)
    for index, stock in enumerate(stocks.itertuples(index=False), 1):
        ticker, name = stock.ticker, stock.name
        code = mapping[ticker]
        print(f"[{index}/{len(stocks)}] {ticker} {name} 수집", flush=True)
        rows = client.disclosures(code, query_start, end)
        major_ids = {
            r["rcept_no"]
            for r in client.disclosures(code, query_start, end, major=True)
        }
        if not major_ids.issubset({r["rcept_no"] for r in rows}):
            raise ValueError(
                "전체 목록과 주요사항보고 목록이 일치하지 않습니다. 다시 수집하세요."
            )
        for row in rows:
            filing_day = receipt_date(row["rcept_no"], row["rcept_dt"])
            if (
                row.get("corp_code") != code
                or not query_start.isoformat() <= filing_day <= end.isoformat()
            ):
                raise ValueError("공시 응답 종목 또는 날짜가 조회 범위와 다릅니다.")
            disclosures.append(
                {
                    **{c: row.get(c, "") for c in DISCLOSURE_COLUMNS},
                    "ticker": ticker,
                    "name": name,
                    "rcept_dt": filing_day,
                    "is_major": int(row["rcept_no"] in major_ids),
                    "disclosure_url": f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={row['rcept_no']}",
                }
            )
        report_dates = {
            r["rcept_no"]: receipt_date(r["rcept_no"], r["rcept_dt"])
            for r in client.disclosures(
                code, date(max(2015, start.year - 2), 1, 1), end, disclosure_type="A"
            )
        }
        available, unavailable = 0, 0
        # 시작 시점의 직전 공시와 전년 비교값을 확보할 수 있도록 과거 2개년 포함.
        for year in range(max(2015, start.year - 2), end.year + 1):
            for report in ["11013", "11012", "11014", "11011"]:
                params = {"corp_code": code, "bsns_year": year, "reprt_code": report}
                fs_div = "CFS"
                items = client.request(
                    "fnlttSinglAcntAll.json", {**params, "fs_div": fs_div}
                ).get("list", [])
                if not items:
                    fs_div = "OFS"
                    items = client.request(
                        "fnlttSinglAcntAll.json", {**params, "fs_div": fs_div}
                    ).get("list", [])
                if not items:
                    unavailable += 1
                    continue
                available += 1
                for item in items:
                    # 모든 BS/IS/CIS 계정 원문 값을 보존해 계정명 변형을 나중에 확인 가능.
                    if item.get("sj_div") not in {"BS", "IS", "CIS"}:
                        continue
                    filing_day = report_dates.get(item["rcept_no"])
                    if filing_day is None:
                        # 종료일 이후 제출된 최신 응답은 과거 feature에 사용하지 않는다.
                        if item["rcept_no"][:8] > end.strftime("%Y%m%d"):
                            continue
                        raise ValueError(
                            f"{ticker}: 재무 접수번호의 공식 접수일을 확인할 수 없습니다."
                        )
                    if item.get("corp_code") != code:
                        raise ValueError(
                            "재무 응답의 기업 고유번호가 일치하지 않습니다."
                        )
                    financials.append(
                        {
                            **{c: item.get(c, "") for c in FINANCIAL_COLUMNS},
                            "ticker": ticker,
                            "corp_code": code,
                            "rcept_dt": filing_day,
                            "bsns_year": year,
                            "reprt_code": report,
                            "fs_div": fs_div,
                        }
                    )
        coverage.append(
            {
                "ticker": ticker,
                "name": name,
                "corp_code": code,
                "disclosure_count": len(rows),
                "financial_reports": available,
                "financial_reports_not_available": unavailable,
            }
        )
    raw = pd.DataFrame(disclosures, columns=DISCLOSURE_COLUMNS)
    if raw.duplicated(["ticker", "rcept_no"]).any():
        raise ValueError("공시 페이지에 중복 접수번호가 있습니다. 다시 수집하세요.")
    finance = pd.DataFrame(financials, columns=FINANCIAL_COLUMNS).drop_duplicates()
    manifest = {
        "schema_version": 1,
        "feature_start": start.isoformat(),
        "feature_end": end.isoformat(),
        "disclosure_start": query_start.isoformat(),
        "disclosure_end": end.isoformat(),
        "collected_at": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "stocks": coverage,
        "candidates": candidates.to_dict("records"),
        "financial_history": "API snapshot; original versions before later corrections may be unavailable",
    }
    return raw, finance, manifest


def collect_reusing_existing(candidates, start, end, directory, key_file):
    """검증된 기존 raw의 종목은 재조회하지 않고 신규 종목만 병합한다."""
    from preprocess_opendart_data import load_raw

    old_raw, old_finance, old_meta = load_raw(directory)
    if (
        old_meta["feature_start"] != start.isoformat()
        or old_meta["feature_end"] != end.isoformat()
    ):
        raise ValueError(
            "재사용 시 --start와 --end는 기존 수집 기간과 같아야 합니다: "
            f"{old_meta['feature_start']} ~ {old_meta['feature_end']}"
        )
    old_stocks = {stock["ticker"]: stock for stock in old_meta["stocks"]}
    tickers = candidates.ticker.drop_duplicates().tolist()
    added = [ticker for ticker in tickers if ticker not in old_stocks]
    reused = [ticker for ticker in tickers if ticker in old_stocks]
    removed = sorted(set(old_stocks) - set(tickers))
    print(
        f"기존 재사용 {len(reused)}종목 / 신규 요청 {len(added)}종목 / 제외 {len(removed)}종목",
        flush=True,
    )
    if added:
        new_raw, new_finance, new_meta = collect(
            DartClient(resolve_key(key_file)),
            candidates[candidates.ticker.isin(added)].copy(),
            start,
            end,
        )
    else:
        new_raw, new_finance = old_raw.iloc[:0].copy(), old_finance.iloc[:0].copy()
        new_meta = {"stocks": []}
    raw = pd.concat([old_raw[old_raw.ticker.isin(tickers)], new_raw], ignore_index=True)
    finance = pd.concat(
        [old_finance[old_finance.ticker.isin(tickers)], new_finance], ignore_index=True
    )
    names = candidates.drop_duplicates("ticker").set_index("ticker").name.to_dict()
    raw["name"] = raw.ticker.map(names)
    order = {ticker: index for index, ticker in enumerate(tickers)}
    for frame in [raw, finance]:
        frame["rcept_dt"] = pd.to_datetime(frame.rcept_dt).dt.strftime("%Y-%m-%d")
        frame["_order"] = frame.ticker.map(order)
        frame.sort_values(["_order", "rcept_dt", "rcept_no"], inplace=True)
        frame.drop(columns="_order", inplace=True)
        frame.reset_index(drop=True, inplace=True)
    if raw.duplicated(["ticker", "rcept_no"]).any():
        raise ValueError("재사용 병합 결과에 공시 중복이 있습니다.")
    sources = {**old_stocks, **{stock["ticker"]: stock for stock in new_meta["stocks"]}}
    stocks = []
    for ticker in tickers:
        stock = {**sources[ticker], "name": names[ticker]}
        stock["source_collected_at"] = (
            new_meta["collected_at"]
            if ticker in added
            else stock.get("source_collected_at", old_meta["collected_at"])
        )
        stocks.append(stock)
    now = datetime.now(ZoneInfo("Asia/Seoul")).isoformat()
    manifest = {
        **old_meta,
        "collected_at": now,
        "updated_at": now,
        "stocks": stocks,
        "candidates": candidates.to_dict("records"),
        "update": {
            "reused_tickers": reused,
            "added_tickers": added,
            "removed_tickers": removed,
            "reused_snapshot_collected_at": old_meta["collected_at"],
        },
    }
    # 이전 후보 파일의 대장주 지정 정보는 새 후보 목록에 대해 재검증해야 한다.
    manifest.pop("leader_assignment", None)
    return raw, finance, manifest


def save_raw(raw, finance, manifest, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # manifest를 마지막에 교체하고 해시를 검증해 중단 시 혼합 파일을 처리하지 않는다.
    staged = []
    try:
        manifest["sha256"] = {}
        for filename, frame in [
            ("opendart_disclosures.csv", raw),
            ("opendart_financials.csv", finance),
        ]:
            with tempfile.NamedTemporaryFile(dir=directory, delete=False) as file:
                temp = Path(file.name)
            staged.append((temp, directory / filename))
            frame.to_csv(temp, index=False, encoding="utf-8-sig")
            manifest["sha256"][filename] = hashlib.sha256(temp.read_bytes()).hexdigest()
        with tempfile.NamedTemporaryFile(dir=directory, delete=False) as file:
            meta_temp = Path(file.name)
        staged.append((meta_temp, directory / "opendart_collection.json"))
        meta_temp.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        for temp, target in staged:
            temp.replace(target)
    finally:
        for temp, _ in staged:
            temp.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=ROOT / "data/raw/theme_candidates.csv"
    )
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--key-file", type=Path, default=ROOT / ".opendart")
    parser.add_argument("--start", type=parse_date, default=date(2024, 10, 4))
    parser.add_argument(
        "--end",
        type=parse_date,
        default=datetime.now(ZoneInfo("Asia/Seoul")).date() - timedelta(days=1),
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="입력 순서로 고유 종목 N개만 수집; 별도 --raw-dir 사용 권장",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="기존 raw의 동일 기간 데이터를 재사용하고 신규 종목만 수집",
    )
    args = parser.parse_args()
    if (
        args.start > args.end
        or args.start.year < 2015
        or (args.limit is not None and args.limit < 1)
    ):
        parser.error("2015년 이후의 올바른 날짜 범위와 양수 limit을 지정하세요.")
    if args.reuse_existing and args.limit is not None:
        parser.error("--reuse-existing과 --limit은 함께 사용할 수 없습니다.")
    try:
        candidates = load_candidates(args.input)
        if args.limit is not None:
            tickers = candidates.ticker.drop_duplicates().head(args.limit)
            candidates = candidates[candidates.ticker.isin(tickers)].copy()
        if args.reuse_existing:
            raw, finance, meta = collect_reusing_existing(
                candidates, args.start, args.end, args.raw_dir, args.key_file
            )
        else:
            raw, finance, meta = collect(
                DartClient(resolve_key(args.key_file)), candidates, args.start, args.end
            )
        save_raw(raw, finance, meta, args.raw_dir)
        print(
            f"저장 완료: 공시 {len(raw):,}건, 재무계정 {len(finance):,}행 / {args.raw_dir}"
        )
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        # request() 내부에서는 인증정보를 포함하는 원격 예외를 제거한다.
        print(f"수집 실패: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
