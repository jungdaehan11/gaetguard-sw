"""갯가드 관측소 실측·예측 조위 모으기 (국립해양조사원 조위관측소 실측·예측 조위, 공공데이터포털 15142507)

무엇을 하나
  - 관측소별로 날짜마다 실측 조위와 예측 조위를 받아 CSV 하나에 이어서 저장
  - 중간에 끊겨도 다시 돌리면 이미 받은 날은 건너뜀 (실패한 날만 다시 받음)
  - 503·시간 초과는 3·10·30·60초 기다렸다 다시 시도, 3일 연속 실패하면 5분 쉬고 계속
  - 이 CSV로 "실측 - 예측" 차이가 높이 문제인지 시각(지연) 문제인지 분석

인증키: gg_tide_fetch.py 와 같음 (같은 폴더의 datago_key.txt 또는 환경변수 DATA_GO_KR_KEY)

실행 (cmd, 이 폴더에서)
  python gg_station_fetch.py                                  # 영흥도·인천송도·인천, 어제까지 184일, 10분 간격
  python gg_station_fetch.py --days 30                        # 최근 30일만
  python gg_station_fetch.py --start 20260505 --end 20261104  # 기간 직접
  python gg_station_fetch.py --stations DT_0043               # 관측소 하나만 (영흥도)

결과
  tide\\station_<관측소코드>_<간격>m.csv   열: date, time, obs_cm(실측), pred_cm(예측), station
"""
import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta

URL = os.environ.get("SURVEY_URL",
                     "https://apis.data.go.kr/1192136/surveyTideLevel/GetSurveyTideLevelApiService")
HERE = os.path.dirname(os.path.abspath(__file__))
# 2026-11-05 실제 API로 확인한 코드 (돌려준 관측소 이름으로 확인)
STATIONS = {"DT_0043": "영흥도", "DT_0052": "인천송도", "DT_0001": "인천"}


def load_key():
    k = os.environ.get("DATA_GO_KR_KEY", "").strip()
    p = os.path.join(HERE, "datago_key.txt")
    if not k and os.path.exists(p):
        with open(p, encoding="utf-8-sig") as f:
            k = f.read().strip()
    if not k:
        sys.exit("인증키가 없습니다. 이 폴더에 datago_key.txt 를 만들고 키 한 줄만 넣어 주세요.")
    return urllib.parse.unquote(k) if "%" in k else k


def call(key, code, day, minute, page):
    q = dict(serviceKey=key, type="json", obsCode=code, reqDate=day, min=str(minute),
             pageNo=str(page), numOfRows="300")
    req = urllib.request.Request(URL + "?" + urllib.parse.urlencode(q), headers={"User-Agent": "gaetguard-station/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        raw = r.read().decode("utf-8", errors="replace")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        raise RuntimeError("JSON이 아닌 응답: " + " ".join(raw.split())[:200])
    if "response" in data:
        data = data["response"]
    head = data.get("header", {})
    rc = str(head.get("resultCode", ""))
    if rc not in ("", "0", "00", "200"):
        raise RuntimeError(f"API 오류 {rc}: {head.get('resultMsg')}")
    body = data.get("body", {})
    items = (body.get("items") or {}).get("item") or [] if isinstance(body.get("items"), dict) else (body.get("items") or [])
    if isinstance(items, dict):
        items = [items]
    return items, int(body.get("totalCount") or 0)


RETRY_WAIT = (3, 10, 30, 60)      # 503·시간 초과 때 기다렸다 다시 (초)
PAUSE = 0.6                        # 호출 사이 쉬는 시간 (초) — 너무 빨리 부르면 503이 늘어남 [추정]


def call_retry(key, code, day, minute, page):
    for k, wait in enumerate((0,) + RETRY_WAIT):
        if wait:
            print(f"      {day} {page}쪽 다시 시도 {k}/{len(RETRY_WAIT)} ({wait}초 뒤)", flush=True)
            time.sleep(wait)
        try:
            return call(key, code, day, minute, page)
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or k == len(RETRY_WAIT):
                raise
        except (urllib.error.URLError, TimeoutError, OSError):
            if k == len(RETRY_WAIT):
                raise
        except RuntimeError as e:
            if "오류 22" in str(e) or "LIMITED" in str(e) or k == len(RETRY_WAIT):
                raise
            if "오류 23" not in str(e) and "오류 99" not in str(e) and "오류 4" not in str(e):
                raise


def fetch_day(key, code, day, minute):
    rows, page = [], 1
    while page <= 10:
        items, total = call_retry(key, code, day, minute, page)
        rows += items
        if not items or len(rows) >= total:
            break
        page += 1
        time.sleep(PAUSE)
    return rows


def hhmm(s):
    s = str(s or "").strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y%m%d%H%M%S", "%Y%m%d%H%M", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).strftime("%H:%M")
        except ValueError:
            pass
    return s


def num(v):
    try:
        return "" if v is None or str(v).strip() in ("", "-") else f"{float(v):.1f}"
    except ValueError:
        return ""


def main():
    ap = argparse.ArgumentParser(description="관측소 실측·예측 조위 모으기")
    ap.add_argument("--stations", nargs="+", default=list(STATIONS))
    ap.add_argument("--start", help="yyyyMMdd")
    ap.add_argument("--end", help="yyyyMMdd (기본: 어제)")
    ap.add_argument("--days", type=int, default=184, help="--start 없을 때 끝날부터 거슬러 올라갈 날 수")
    ap.add_argument("--min", type=int, default=10, help="간격 분 (1~60)")
    a = ap.parse_args()

    end = datetime.strptime(a.end, "%Y%m%d").date() if a.end else date.today() - timedelta(days=1)
    start = datetime.strptime(a.start, "%Y%m%d").date() if a.start else end - timedelta(days=a.days - 1)
    days = [(start + timedelta(days=i)).strftime("%Y%m%d") for i in range((end - start).days + 1)]
    key = load_key()
    outdir = os.path.join(HERE, "tide")
    os.makedirs(outdir, exist_ok=True)
    print(f"{start} ~ {end} ({len(days)}일), {a.min}분 간격, 관측소 {', '.join(a.stations)}\n")

    for code in a.stations:
        path = os.path.join(outdir, f"station_{code}_{a.min}m.csv")
        done = set()
        if os.path.exists(path):
            with open(path, encoding="utf-8-sig") as f:
                done = {r["date"] for r in csv.DictReader(f)}
        todo = [d for d in days if d not in done]
        print(f"[{code} {STATIONS.get(code, '')}] 이미 받은 날 {len(done)}일, 이번에 받을 날 {len(todo)}일 → {path}")
        new = not os.path.exists(path)
        failed = []
        with open(path, "a", newline="", encoding="utf-8-sig" if new else "utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["date", "time", "obs_cm", "pred_cm", "station"])
            streak = 0
            for k, d in enumerate(todo, 1):
                try:
                    rows = fetch_day(key, code, d, a.min)
                    streak = 0
                except KeyboardInterrupt:
                    print(f"\n   멈춤. {d} 앞날까지 저장됨 → 같은 명령을 다시 실행하면 이어서 받음")
                    return
                except (urllib.error.URLError, RuntimeError, TimeoutError, OSError) as e:
                    failed.append(d)
                    streak += 1
                    print(f"   {d} 실패 (다시 시도 후에도): {e}")
                    if "오류 22" in str(e) or "LIMITED" in str(e):
                        print("   하루 호출 한도 초과 → 내일 다시 돌리면 이어서 받음")
                        break
                    if streak >= 3:
                        print("   3일 연속 실패 → 공공데이터포털이 불안정한 것 같음. 5분 쉬고 계속 (그만두려면 Ctrl+C)")
                        try:
                            time.sleep(300)
                        except KeyboardInterrupt:
                            print("   멈춤. 같은 명령을 나중에 다시 실행하면 빠진 날만 받음")
                            return
                        streak = 0
                    continue
                name = rows[0].get("obsvtrNm", "") if rows else ""
                for r in rows:
                    # 필드 뜻: bscTdlvHgt = 실측, tdlvHgt = 예측 (공공데이터포털 명세, 실제 응답으로도 확인)
                    w.writerow([d, hhmm(r.get("obsrvnDt")), num(r.get("bscTdlvHgt")), num(r.get("tdlvHgt")), name])
                f.flush()
                if k % 10 == 0 or k == len(todo):
                    print(f"   {k}/{len(todo)}일 ({d}, {len(rows)}줄)", flush=True)
                try:
                    time.sleep(PAUSE)
                except KeyboardInterrupt:
                    print(f"\n   멈춤. {d}까지 저장됨 → 같은 명령을 다시 실행하면 이어서 받음")
                    return
        if failed:
            print(f"   실패한 날 {len(failed)}일: 다시 돌리면 그 날만 다시 받음")
        print()
    print("끝. tide 폴더의 station_*.csv 를 Claude에게 알려 주세요.")


if __name__ == "__main__":
    main()
