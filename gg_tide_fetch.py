"""갯가드 조위 수집 — TideBED 예측 조위(1분) (해양수산부 국립해양조사원, 공공데이터포털)

무엇을 하나
  - 위경도를 넣으면 그 지점의 예측 조위(1분 간격까지)를 받아 CSV로 저장
  - 같이 오는 "가장 가까운 기준항" 이름, 조시차, 조고비, 평균해면을 화면에 요약
  - 지점 조위와 기준항 조위의 고조·저조 시각 차이(분)를 계산해서 보여 줌

인증키
  - 같은 폴더에 datago_key.txt 파일을 만들고 키 한 줄만 붙여 넣기 (또는 환경변수 DATA_GO_KR_KEY)
  - 이 파일은 GitHub에 올리지 말 것 (.gitignore에 들어 있음)

실행 (cmd, 이 폴더에서)
  python gg_tide_fetch.py                              # 기본 지점들, 오늘, 1분 간격
  python gg_tide_fetch.py --date 20261106              # 날짜 지정
  python gg_tide_fetch.py --date 20261106 --min 10     # 10분 간격
  python gg_tide_fetch.py --point 내지점 37.340 126.675  # 지점 직접 지정 (이름 위도 경도)

결과
  tide\\tidebed_<지점>_<날짜>.csv, 화면 요약

API: https://apis.data.go.kr/1192136/tidebed/GetTidebedApiService
  요청: serviceKey, type=json, lat, lot(경도), reqDate(yyyyMMdd), min(1~60), pageNo, numOfRows(최대 300)
  응답 항목: obsvtrNm(기준항), obsrvnDt/obsrvnHgt(기준항 예측), slctdDt/slctdHgt(선택지점 예측),
            jogo(조고비), josi(조시차), msl(평균해면 cm), M2·S2·K1·O1 진폭·위상
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
from datetime import datetime, timedelta

URL = "https://apis.data.go.kr/1192136/tidebed/GetTidebedApiService"

# 기본 지점 — 지도에서 대충 고른 값 [확인 필요: 실제 갯벌 위인지 지도로 확인 후 고칠 것]
DEFAULT_POINTS = [
    ("인천송도관측소부근", 37.338, 126.586),   # 확인용: 기준항이 송도면 조시차 ≈ 0, 조고비 ≈ 1
    ("오이도앞갯벌", 37.340, 126.675),
    ("오이도먼갯벌", 37.325, 126.655),
]


def load_key():
    k = os.environ.get("DATA_GO_KR_KEY", "").strip()
    if k:
        return k
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datago_key.txt")
    if os.path.exists(p):
        with open(p, encoding="utf-8-sig") as f:
            k = f.read().strip()
    if not k:
        sys.exit("인증키가 없습니다. 이 폴더에 datago_key.txt 를 만들고 키 한 줄만 넣어 주세요.")
    return k


def call(key, lat, lon, date, minute, page):
    q = dict(serviceKey=key, type="json", lat=f"{lat:.6f}", lot=f"{lon:.6f}", reqDate=date,
             min=str(minute), pageNo=str(page), numOfRows="300")
    url = URL + "?" + urllib.parse.urlencode(q)
    req = urllib.request.Request(url, headers={"User-Agent": "gaetguard-tide/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        raw = r.read().decode("utf-8", errors="replace")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        raise RuntimeError("JSON이 아닌 응답: " + raw[:300])
    if "response" in data:
        data = data["response"]
    head = data.get("header", {})
    code = str(head.get("resultCode", ""))
    if code not in ("00", "0", "200", ""):
        raise RuntimeError(f"API 오류 {code}: {head.get('resultMsg')}")
    body = data.get("body", {})
    items = (body.get("items") or {}).get("item") or []
    if isinstance(items, dict):
        items = [items]
    return items, int(body.get("totalCount") or 0)


def fetch(key, lat, lon, date, minute):
    rows, page = [], 1
    while True:
        items, total = call(key, lat, lon, date, minute, page)
        rows += items
        if not items or len(rows) >= total:
            break
        page += 1
        time.sleep(0.2)
    return rows


def extrema(ts, hs):
    """고조·저조 찾기 → [(구분, 시각, 높이)]
    - 앞뒤 약 30분(최소 3칸) 안에서 가장 높거나 낮은 점 (자료 간격은 시각에서 계산)
    - 같은 값이 이어지면(높이가 정수 cm라 꼭대기가 평평함) 그 구간의 가운데 시각
    - 한 점이면 앞뒤 점으로 포물선을 그려 꼭짓점 시각 (자료 간격보다 촘촘하게)
    - 하루의 처음·끝 30분 안은 잘린 구간이라 빼고 봄"""
    pts = sorted((t, h) for t, h in zip(ts, hs) if t is not None and h is not None)
    if len(pts) < 7:
        return []
    ts = [p[0] for p in pts]
    hs = [p[1] for p in pts]
    gaps = sorted((b - a).total_seconds() / 60 for a, b in zip(ts, ts[1:]))
    step = max(gaps[len(gaps) // 2], 1e-6)
    win = max(3, int(round(30 / step)))
    n = len(hs)
    out = []
    i = win
    while i < n - win:
        seg = hs[i - win:i + win + 1]
        if hs[i] == max(seg) and hs[i] > min(seg):
            kind = "고조"
        elif hs[i] == min(seg) and hs[i] < max(seg):
            kind = "저조"
        else:
            i += 1
            continue
        j = i
        while j + 1 < n and hs[j + 1] == hs[i]:
            j += 1
        if j > i:
            t = ts[i] + (ts[j] - ts[i]) / 2
        else:
            a, b, c = hs[i - 1], hs[i], hs[i + 1]
            den = a - 2 * b + c
            off = max(-0.5, min(0.5, 0.5 * (a - c) / den)) if den else 0.0
            t = ts[i] + timedelta(minutes=off * step)
        t = (t + timedelta(seconds=30)).replace(second=0, microsecond=0)
        out.append((kind, t, hs[i]))
        i = j + win
    return out


def fnum(v, nd):
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return v


def parse_dt(s):
    s = str(s).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y%m%d%H%M%S", "%Y%m%d%H%M", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    return None


def main():
    ap = argparse.ArgumentParser(description="TideBED 예측 조위 수집")
    ap.add_argument("--date", default=time.strftime("%Y%m%d"), help="yyyyMMdd (기본: 오늘)")
    ap.add_argument("--min", type=int, default=1, help="시간 간격 분 (1~60)")
    ap.add_argument("--point", nargs=3, action="append", metavar=("이름", "위도", "경도"))
    a = ap.parse_args()

    key = load_key()
    points = [(n, float(la), float(lo)) for n, la, lo in a.point] if a.point else DEFAULT_POINTS
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tide")
    os.makedirs(outdir, exist_ok=True)

    print(f"날짜 {a.date}, {a.min}분 간격, 지점 {len(points)}개\n")
    for name, lat, lon in points:
        try:
            rows = fetch(key, lat, lon, a.date, a.min)
        except (urllib.error.URLError, RuntimeError) as e:
            print(f"[{name}] 실패: {e}\n")
            continue
        if not rows:
            print(f"[{name}] 데이터 없음 (바다 위 좌표인지 확인)\n")
            continue
        path = os.path.join(outdir, f"tidebed_{name}_{a.date}.csv")
        cols = list(rows[0].keys())
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        r0 = rows[0]
        print(f"[{name}] 위도 {lat}, 경도 {lon} → 기준항 {r0.get('obsvtrNm')}, "
              f"조시차 {r0.get('josi')}, 조고비 {fnum(r0.get('jogo'), 4)}, 평균해면 {fnum(r0.get('msl'), 1)}cm, {len(rows)}행")
        # 고조·저조 비교
        ts = [parse_dt(r.get("slctdDt")) for r in rows]
        tr = [parse_dt(r.get("obsrvnDt")) for r in rows]
        hs = [float(r.get("slctdHgt") or 0) for r in rows]
        hr = [float(r.get("obsrvnHgt") or 0) for r in rows]
        es, er = extrema(ts, hs), extrema(tr, hr)
        for kind, t, h in es:
            ref = min((x for x in er if x[0] == kind and x[1] and t),
                      key=lambda x: abs((x[1] - t).total_seconds()), default=None)
            if t is None:
                continue
            msg = f"   {kind} {t:%H:%M} {h:.0f}cm"
            if ref and ref[1]:
                d = (t - ref[1]).total_seconds() / 60
                msg += f"  (기준항 {ref[1]:%H:%M} {ref[2]:.0f}cm, 차이 {d:+.0f}분)"
            print(msg)
        print(f"   저장: {path}\n")


if __name__ == "__main__":
    main()
