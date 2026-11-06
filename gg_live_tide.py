"""실시간 조위 → 판정 엔진(gg_redecide.Redecider)이 쓰는 "물" (gg_redecide.Water 와 같은 모양: at(t), latest(t, v))

  예측: 우리 조석 성분 (data/영흥도_조석성분_2022-2024.pkl, gg_tide_harmonic.py --save-coef) → 인터넷 없이 6시간 앞까지
  보정: 매분 관측소 실측(공공데이터포털 조위관측소 API, 영흥도 DT_0043)의 최근 10분 평균 − 같은 때 우리 예측
        (검증한 방식 그대로: 5장 높이 보정, 99%·95% 여유도 이 방식으로 뽑은 값)
  실측이 끊기면: 마지막 보정값을 30분까지 씀, 그 뒤엔 보정 0 + "실측 없음" 표시 [가정]
  실측 1분 자료에 빈 값이 0.0으로 적힌 경우(예측이 1m 넘는데 실측 0)는 빼고 셈 (커넥터와 같은 규칙)

실행
  python gg_live_tide.py --check                 # PC에서: 지금 API로 오늘 실측을 받아 보정값·앞으로 1시간 물높이 (datago_key.txt 필요)
  python gg_live_tide.py --replay "2025-09-11 02:07" --csv tide\\station_DT_0043_1m.csv
                                                 # 지난 실측을 실시간처럼 흘려 시험 (인터넷 없이)
필요: numpy utide (pip install utide), 판정까지 쓰려면 gg_retreat_map.py · 지형 파일
"""
import argparse
import csv
import json
import os
import pickle
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import warnings
from datetime import datetime, timedelta, timezone

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
KST = timezone(timedelta(hours=9))
SURVEY_URL = "https://apis.data.go.kr/1192136/surveyTideLevel/GetSurveyTideLevelApiService"
COEF = os.path.join(HERE, "data", "영흥도_조석성분_2022-2024.pkl")
STATION = "DT_0043"                     # 영흥도 조위관측소
HOLD_OFF_MIN = 30                       # 실측이 끊겨도 마지막 보정값을 쓰는 시간(분) [가정]


def _num(v):
    try:
        return None if v is None or str(v).strip() in ("", "-", "None", "null") else float(v)
    except (TypeError, ValueError):
        return None


def _dt(s):
    s = str(s or "").strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y%m%d%H%M%S", "%Y%m%d%H%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    return None


def _clean(obs, pred_khoa):
    """빈 값이 0.0으로 적힌 것 빼기"""
    return None if (obs == 0 and pred_khoa is not None and pred_khoa > 100) else obs


# ───────── 실측 받는 곳 ─────────
class ApiSource:
    """공공데이터포털 조위관측소 API (1분). 같은 날짜는 60초에 한 번만 다시 받음"""

    def __init__(self, station=STATION, key=None, timeout=20):
        self.station, self.timeout = station, timeout
        self.key = key or os.environ.get("DATA_GO_KR_KEY", "").strip() or self._key_file()
        if not self.key:
            raise SystemExit("인증키가 없습니다: 이 폴더에 datago_key.txt 를 두거나 DATA_GO_KR_KEY 를 설정하세요.")
        if "%" in self.key:
            self.key = urllib.parse.unquote(self.key)
        self.cache = {}

    @staticmethod
    def _key_file():
        for d in (HERE, os.path.join(HERE, "tide_mcp")):
            p = os.path.join(d, "datago_key.txt")
            if os.path.exists(p):
                return open(p, encoding="utf-8-sig").read().strip()
        return ""

    def now(self):
        return np.datetime64(datetime.now(KST).replace(tzinfo=None), "s")

    def _day(self, day):
        hit = self.cache.get(day)
        if hit and time.time() - hit[0] < 60:
            return hit[1]
        rows, page = [], 1
        while page <= 6:
            q = dict(serviceKey=self.key, type="json", obsCode=self.station, reqDate=day, min="1",
                     pageNo=str(page), numOfRows="300")
            req = urllib.request.Request(SURVEY_URL + "?" + urllib.parse.urlencode(q), headers={"User-Agent": "gaetguard-live/1.0"})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    data = json.loads(r.read().decode("utf-8", errors="replace"))
            except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as e:
                raise RuntimeError(f"공공데이터포털 접속 실패 ({type(e).__name__})") from None   # 키는 절대 안 찍음
            body = (data.get("response", data) or {}).get("body") or {}
            items = body.get("items") or {}
            items = items.get("item", []) if isinstance(items, dict) else items
            items = [items] if isinstance(items, dict) else (items or [])
            rows += items
            total = int(body.get("totalCount") or 0)
            if not items or len(rows) >= total:
                break
            page += 1
        out = []
        for r in rows:
            t = _dt(r.get("obsrvnDt"))
            o, p = _num(r.get("bscTdlvHgt")), _num(r.get("tdlvHgt"))
            if t is not None and o is not None:
                o = _clean(o, p)
                if o is not None:
                    out.append((np.datetime64(t, "m"), o))
        self.cache[day] = (time.time(), out)
        return out

    def obs(self, t_from, t_to):
        days = sorted({str(t_from)[:10].replace("-", ""), str(t_to)[:10].replace("-", "")})
        rows = [x for d in days for x in self._day(d)]
        sel = [(t, o) for t, o in rows if t_from <= t <= t_to]
        return np.array([t for t, _ in sel], "datetime64[m]"), np.array([o for _, o in sel], float)


class CsvSource:
    """지난 실측 CSV(date,time,obs_cm,...)를 실시간처럼 흘림 — 시험용. now 이후 자료는 안 줌"""

    def __init__(self, path):
        T, O = [], []
        with open(path, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                o = _num(r.get("obs_cm"))
                p = _num(r.get("pred_cm"))
                if o is None:
                    continue
                o = _clean(o, p)
                if o is None:
                    continue
                T.append(f"{r['date'][:4]}-{r['date'][4:6]}-{r['date'][6:]}T{r['time']}")
                O.append(o)
        self.T = np.array(T, "datetime64[m]")
        o = np.argsort(self.T)
        self.T, self.O = self.T[o], np.array(O, float)[o]

    def obs(self, t_from, t_to):
        i0, i1 = np.searchsorted(self.T, t_from), np.searchsorted(self.T, t_to, side="right")
        return self.T[i0:i1], self.O[i0:i1]


class ListSource(CsvSource):
    """(시각 문자열, 실측 cm) 목록 — 커넥터로 받은 자료 대조용"""

    def __init__(self, rows):
        self.T = np.array([t for t, _ in rows], "datetime64[m]")
        self.O = np.array([o for _, o in rows], float)


# ───────── 판정 엔진용 물 ─────────
class LiveWater:
    """t = 시작 뒤 분(실수). at(t) → (지금부터 분 단위 막히는 시각, 밤?, 보정 cm). gg_redecide.Water 와 바꿔 쓸 수 있음"""

    def __init__(self, G, source, t_start, depth=0.30, coef_path=COEF, horizon=360):
        with open(coef_path, "rb") as f:
            self.coef = pickle.load(f)["coef"]
        self.G, self.src, self.depth, self.horizon = G, source, depth, horizon
        self.t0 = np.datetime64(t_start, "m")
        self.cache, self.ls_cache = {}, {}
        self.last_off, self.last_off_t, self.status = None, None, ""
        self.lat_c = float(G["GY"].mean()) if G is not None else 37.24
        self.lon_c = float(G["GX"].mean()) if G is not None else 126.45

    def pred(self, tt):
        import utide
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return utide.reconstruct(np.asarray(tt).astype("datetime64[s]"), self.coef, verbose=False).h

    def offset_at(self, tm):
        """tm(분)까지 10분 실측 평균 − 같은 때 우리 예측 (cm). 끊기면 마지막 값 30분, 그 뒤 0"""
        try:
            ts, ob = self.src.obs(tm - np.timedelta64(9, "m"), tm)
        except RuntimeError as e:
            ts, ob = np.array([], "datetime64[m]"), np.array([])
            self.status = str(e)
        if len(ts):
            off = float(np.mean(ob - self.pred(ts)))
            self.last_off, self.last_off_t = off, tm
            self.status = f"실측 {len(ts)}분 (마지막 {str(ts[-1])[11:16]})"
            return off
        if self.last_off is not None and (tm - self.last_off_t) <= np.timedelta64(HOLD_OFF_MIN, "m"):
            self.status = f"실측 끊김 → {str(self.last_off_t)[11:16]} 보정값 씀"
            return self.last_off
        self.status = "실측 없음 → 보정 0"
        return 0.0

    def forecast(self, tm, n=None):
        """tm부터 n분 물높이 (m, 보정 포함)"""
        n = n or self.horizon
        off = self.offset_at(tm)
        tt = tm + np.arange(n).astype("timedelta64[m]")
        return tt, (self.pred(tt) + off) / 100.0, off

    def at(self, t):
        import gg_decide as gd
        import gg_retreat_map as rm
        m = int(np.floor(t + 1e-9))
        if m not in self.cache:
            tm = self.t0 + np.timedelta64(m, "m")
            _, Hf, off = self.forecast(tm)
            tc = rm.close_times_fast(self.G, Hf, self.depth)
            night = bool(gd._sun_elev(np.array([tm]), self.lat_c, self.lon_c)[0] < -0.833)
            if len(self.cache) > 4:
                self.cache.pop(min(self.cache))
            self.cache[m] = (tc, night, off)
        tc, night, off = self.cache[m]
        return tc - (t - m), night, off

    def latest(self, t, v):
        import gg_retreat_map as rm
        key = (round(t, 4), round(v, 4))
        if key not in self.ls_cache:
            tcf, _, _ = self.at(t)
            if len(self.ls_cache) > 6:
                self.ls_cache.pop(next(iter(self.ls_cache)))
            self.ls_cache[key] = rm.latest_start(self.G, tcf, v, with_path=True)[:2]
        return self.ls_cache[key]


def show(w, tm):
    tt, H, off = w.forecast(tm, 61)
    print(f"{str(tm)[:16].replace('T', ' ')} 영흥도({STATION}) · {w.status} · 보정(실측 − 우리 예측) {off:+.1f}cm")
    print("   앞으로 1시간 물높이 (우리 예측 + 보정, 약최저저조면 기준 cm): " +
          ", ".join(f"{str(t)[11:16]} {h * 100:.0f}" for t, h in zip(tt[::10], H[::10])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="지금 API로 오늘 실측을 받아 보정값 보여 주기 (PC)")
    ap.add_argument("--replay", help="이 시각을 '지금'으로 보고 지난 실측을 흘림 (예: '2025-09-11 02:07')")
    ap.add_argument("--csv", help="--replay 에 쓸 관측소 CSV (gg_station_fetch.py 결과)")
    ap.add_argument("--coef", default=COEF)
    a = ap.parse_args()
    if a.check:
        src = ApiSource()
        now = np.datetime64(src.now(), "m")
        w = LiveWater(None, src, now, coef_path=a.coef)
        show(w, now)
        ts, ob = src.obs(now - np.timedelta64(60, "m"), now)
        if len(ts):
            d = ob - w.pred(ts)
            print(f"   최근 1시간 실측 {len(ts)}분, 실측 − 우리 예측: 평균 {d.mean():+.1f}cm, 범위 {d.min():+.1f}~{d.max():+.1f}cm")
        return
    if a.replay and a.csv:
        src = CsvSource(a.csv)
        t0 = np.datetime64(a.replay.replace(" ", "T"), "m")
        w = LiveWater(None, src, t0, coef_path=a.coef)
        for k in (0, 10, 30):
            show(w, t0 + np.timedelta64(k, "m"))
        return
    ap.print_help()


if __name__ == "__main__":
    main()
