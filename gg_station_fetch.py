"""갯가드 관측소 실측·예측 조위 모으기 (국립해양조사원 조위관측소 실측·예측 조위, 공공데이터포털 15142507)

무엇을 하나
  - 관측소별로 날짜마다 실측 조위와 예측 조위를 받아 CSV 하나에 이어서 저장
  - 중간에 끊겨도 다시 돌리면 이미 받은 날은 건너뜀
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


def fetch_day(key, code, day, minute):
    rows, page = [], 1
    while page <= 10:
        items, total = call(key, code, day, minute, page)
        rows += items
        if not items or len(rows) >= total:
            break
        page += 1
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
            for k, d in enumerate(todo, 1):
                try:
                    rows = fetch_day(key, code, d, a.min)
                except (urllib.error.URLError, RuntimeError, TimeoutError, OSError) as e:
                    failed.append(d)
                    print(f"   {d} 실패: {e}")
                    if "오류 22" in str(e) or "LIMITED" in str(e):
                        print("   하루 호출 한도 초과 → 내일 다시 돌리면 이어서 받음")
                        break
                    time.sleep(1.0)
                    continue
                name = rows[0].get("obsvtrNm", "") if rows else ""
                for r in rows:
                    # 필드 뜻: bscTdlvHgt = 실측, tdlvHgt = 예측 (공공데이터포털 명세, 실제 응답으로도 확인)
                    w.writerow([d, hhmm(r.get("obsrvnDt")), num(r.get("bscTdlvHgt")), num(r.get("tdlvHgt")), name])
                f.flush()
                if k % 10 == 0 or k == len(todo):
                    print(f"   {k}/{len(todo)}일 ({d}, {len(rows)}줄)")
                time.sleep(0.15)
        if failed:
            print(f"   실패한 날 {len(failed)}일: 다시 돌리면 그 날만 다시 받음")
        print()
    print("끝. tide 폴더의 station_*.csv 를 Claude에게 알려 주세요.")


if __name__ == "__main__":
    main()
