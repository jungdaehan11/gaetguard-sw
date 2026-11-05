"""갯가드 조위 커넥터 — 국립해양조사원 조위 API를 Claude 커넥터(MCP)로 열어 주는 서버

무엇을 하나
  Claude(claude.ai)가 이 서버의 도구 3개를 직접 부를 수 있게 합니다.
    1) tide_point_summary  위경도 한 점의 하루 고조·저조 시각/높이 + 가장 가까운 기준항과의 차이 (TideBED)
    2) tide_point_series   위경도 한 점의 예측 조위 시계열 (10분 간격 등, 최대 300줄) (TideBED)
    3) station_tide        조위관측소의 실측·예측 조위 비교 (조위관측소 실측·예측 조위 API)

인증키·비밀 주소 (환경변수, 저장소에 올리지 말 것)
  DATA_GO_KR_KEY  공공데이터포털 인증키 (일반 인증키 Decoding/Encoding 어느 쪽이든 됨)
  MCP_SECRET      커넥터 주소에 넣는 비밀 문자열 (16자 이상). 주소 = https://<서버>/<MCP_SECRET>/mcp
                  만들기: python -c "import secrets; print(secrets.token_urlsafe(24))"
  (선택) PORT     기본 8000 (Render 등은 자동으로 넣어 줌)
  내 PC에서 돌릴 때는 환경변수 대신 datago_key.txt, mcp_secret.txt 파일(이 폴더나 위 폴더)도 읽음

실행
  python server.py
"""
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from functools import partial

import anyio
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from starlette.requests import Request
from starlette.responses import PlainTextResponse

KST = timezone(timedelta(hours=9))
TIDEBED_URL = os.environ.get("TIDEBED_URL", "https://apis.data.go.kr/1192136/tidebed/GetTidebedApiService")
SURVEY_URL = os.environ.get("SURVEY_URL",
                            "https://apis.data.go.kr/1192136/surveyTideLevel/GetSurveyTideLevelApiService")
TIMEOUT = float(os.environ.get("API_TIMEOUT", "20"))
MAX_ROWS_OUT = 300          # 도구가 돌려주는 시계열 최대 줄 수 (대화창이 넘치지 않게)
MAX_PAGES = 10              # 한 번에 부르는 최대 페이지 (300줄 × 10)
CACHE_TTL_PRED = 6 * 3600   # 예측값은 잘 안 바뀜 → 6시간 기억
CACHE_TTL_OBS = 120         # 실측값은 계속 들어옴 → 2분

# 자주 쓸 관측소 이름 → 코드. [확인 필요: 국립해양조사원 관측소 코드표로 확인 전]
# 결과에 API가 돌려준 관측소 이름이 같이 나오므로, 틀리면 바로 보입니다.
STATION_ALIASES = {
    "인천": "DT_0001",
    "인천송도": "DT_0052",
}

SOURCE_TIDEBED = "국립해양조사원 TideBED 예측 조위 (공공데이터포털 OpenAPI, 출처표시 조건)"
SOURCE_SURVEY = "국립해양조사원 조위관측소 실측·예측 조위 (공공데이터포털 OpenAPI 15142507, 출처표시 조건)"

ERROR_HINTS = {
    "1": "공공데이터포털 서버 문제 (잠시 뒤 다시)",
    "3": "데이터 없음 (좌표·날짜·관측소 코드 확인)",
    "4": "제공기관 서버 문제 (잠시 뒤 다시)",
    "10": "요청 값이 잘못됨 (날짜 yyyyMMdd, 위경도, 간격 1~60분 확인)",
    "11": "필수 요청 값이 빠짐",
    "12": "API 주소가 없거나 폐기됨",
    "20": "이 API 활용신청이 안 됐거나 일시 중지됨 (공공데이터포털 마이페이지 확인)",
    "22": "하루 호출 한도 초과",
    "23": "초당 호출이 너무 많음 (잠시 뒤 다시)",
    "30": "인증키가 등록되지 않음 (키가 틀렸거나 이 API 승인 전·승인 직후 반영 대기)",
    "31": "활용 기간 만료",
    "99": "원인 불명 오류",
}


class ApiError(Exception):
    pass


# ───────────────────────── 공공데이터포털 호출 ─────────────────────────

HERE = os.path.dirname(os.path.abspath(__file__))


def _from_file(name):
    """PC에서 돌릴 때: 환경변수가 없으면 이 폴더나 위 폴더(gaetguard)의 텍스트 파일에서 읽기"""
    for d in (HERE, os.path.dirname(HERE)):
        p = os.path.join(d, name)
        if os.path.exists(p):
            with open(p, encoding="utf-8-sig") as f:
                v = f.read().strip()
            if v:
                return v
    return ""


def _key():
    k = os.environ.get("DATA_GO_KR_KEY", "").strip() or _from_file("datago_key.txt")
    if not k:
        raise ApiError("서버에 DATA_GO_KR_KEY(공공데이터포털 인증키)가 설정되지 않았습니다.")
    if "%" in k:                       # Encoding 키를 넣었으면 풀어서 한 번만 인코딩되게
        k = urllib.parse.unquote(k)
    return k


def _error_from_text(raw):
    """XML·텍스트 오류 응답에서 코드 뽑기 (키 값은 절대 안 담음)"""
    code = None
    for tag in ("returnReasonCode", "resultCode"):
        m = re.search(rf"<{tag}>\s*([^<\s]+)\s*</{tag}>", raw)
        if m:
            code = m.group(1)
            break
    msg = None
    for tag in ("returnAuthMsg", "resultMsg", "errMsg"):
        m = re.search(rf"<{tag}>\s*([^<]+?)\s*</{tag}>", raw)
        if m:
            msg = m.group(1)
            break
    if code is None and msg is None:
        return "알 수 없는 응답: " + re.sub(r"\s+", " ", raw)[:200]
    hint = ERROR_HINTS.get(str(code).lstrip("0") or "0", "")
    return f"공공데이터포털 오류 {code}: {msg or ''} {('→ ' + hint) if hint else ''}".strip()


def _call(url, params):
    q = dict(params)
    q["serviceKey"] = _key()
    q["type"] = "json"
    full = url + "?" + urllib.parse.urlencode(q)
    req = urllib.request.Request(full, headers={"User-Agent": "gaetguard-tide-mcp/1.0",
                                                "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            raw = r.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        detail = _error_from_text(body) if body.strip() else ""
        raise ApiError(f"공공데이터포털 HTTP {e.code}. {detail}".strip()) from None
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        reason = getattr(e, "reason", e)
        raise ApiError(f"공공데이터포털에 연결하지 못했습니다 ({type(e).__name__}: {reason}). "
                       "서버가 있는 곳에서 apis.data.go.kr 접속이 막혔을 수 있습니다.") from None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        raise ApiError(_error_from_text(raw)) from None
    if isinstance(data, dict) and "response" in data:
        data = data["response"]
    head = (data or {}).get("header") or {}
    code = str(head.get("resultCode", "")).strip()
    if code not in ("", "0", "00", "200"):
        hint = ERROR_HINTS.get(code.lstrip("0") or "0", "")
        raise ApiError(f"API 오류 {code}: {head.get('resultMsg', '')} {('→ ' + hint) if hint else ''}".strip())
    body = data.get("body") or {}
    items = (body.get("items") or {}) if isinstance(body.get("items"), dict) else {"item": body.get("items") or []}
    items = items.get("item") or []
    if isinstance(items, dict):
        items = [items]
    try:
        total = int(body.get("totalCount") or 0)
    except (TypeError, ValueError):
        total = 0
    return items, total


_cache = {}
_cache_lock = threading.Lock()


def _fetch_all(url, params, ttl):
    ck = (url, tuple(sorted(params.items())))
    now = time.time()
    with _cache_lock:
        hit = _cache.get(ck)
        if hit and now - hit[0] < ttl:
            return hit[1]
    rows, page = [], 1
    while page <= MAX_PAGES:
        items, total = _call(url, dict(params, pageNo=str(page), numOfRows="300"))
        rows += items
        if not items or len(rows) >= total:
            break
        page += 1
    with _cache_lock:
        if len(_cache) > 200:
            _cache.clear()
        _cache[ck] = (now, rows)
    return rows


# ───────────────────────── 계산 도우미 ─────────────────────────

def _date(d):
    if d is None or str(d).strip() == "":
        return datetime.now(KST).strftime("%Y%m%d")
    s = re.sub(r"[^0-9]", "", str(d))
    if len(s) != 8:
        raise ApiError(f"날짜는 yyyyMMdd 또는 yyyy-mm-dd 로 주세요 (받은 값: {d}).")
    try:
        datetime.strptime(s, "%Y%m%d")
    except ValueError:
        raise ApiError(f"없는 날짜입니다 (받은 값: {d}).") from None
    return s


def _interval(m):
    m = int(m)
    if not 1 <= m <= 60:
        raise ApiError("간격(interval_min)은 1~60분입니다.")
    return m


def _hm(s, default):
    s = (s or default).strip()
    m = re.fullmatch(r"(\d{1,2}):?(\d{2})", s)
    if not m or int(m.group(1)) > 23 or int(m.group(2)) > 59:
        raise ApiError(f"시각은 HH:MM 으로 주세요 (받은 값: {s}).")
    return int(m.group(1)) * 60 + int(m.group(2))


def _dt(s):
    s = str(s or "").strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y%m%d%H%M%S", "%Y%m%d%H%M", "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%dT%H:%M", "%Y.%m.%d %H:%M", "%Y/%m/%d %H:%M"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    return None


def _num(v):
    try:
        if v is None or str(v).strip() in ("", "-", "None", "null"):
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _r(v, n=1):
    return None if v is None else round(v, n)


def _extrema(ts, hs, step_min):
    """고조·저조: 앞뒤 약 30분(또는 3칸) 안에서 가장 높거나 낮은 점. 같은 값이 이어지면 가운데.
    하루의 처음·끝 30분 안은 잘린 구간이라 빼고 봄."""
    win = max(3, 30 // max(step_min, 1))
    n = len(hs)
    cand = []
    for i in range(win, n - win):
        if hs[i] is None or ts[i] is None:
            continue
        seg = [h for h in hs[i - win:i + win + 1] if h is not None]
        if hs[i] == max(seg) and hs[i] > min(seg):
            cand.append(("고조", i))
        elif hs[i] == min(seg) and hs[i] < max(seg):
            cand.append(("저조", i))
    out, group = [], []
    for c in cand + [(None, None)]:
        if group and (c[0] != group[-1][0] or c[1] - group[-1][1] > win):
            i = group[len(group) // 2][1]
            out.append((group[0][0], i))
            group = []
        if c[0] is not None:
            group.append(c)
    return out


def _need_times(ts, sample):
    if not any(ts):      # 실제 API 시각 형식이 예상과 다를 때 조용히 빈 결과를 내지 않게
        raise ApiError(f"API가 준 시각 형식을 읽지 못했습니다 (예: {sample!r}). server.py의 _dt()에 형식을 추가해야 합니다.")


def _window(ts, t0, t1):
    return [i for i, t in enumerate(ts) if t is not None and t0 <= t.hour * 60 + t.minute <= t1]


# ───────────────────────── 도구 본체 (스레드에서 실행) ─────────────────────────

def _tidebed(lat, lon, date, minute):
    if not (30.0 <= lat <= 40.0 and 122.0 <= lon <= 133.0):
        raise ApiError("위경도가 한국 주변 바다 범위(위도 30~40, 경도 122~133) 밖입니다. 위도·경도 순서를 확인해 주세요.")
    rows = _fetch_all(TIDEBED_URL, dict(lat=f"{lat:.6f}", lot=f"{lon:.6f}", reqDate=date, min=str(minute)),
                      CACHE_TTL_PRED)
    if not rows:
        raise ApiError("데이터가 없습니다. 좌표가 바다·갯벌 위인지(육지면 안 나올 수 있음), 날짜가 제공 범위인지 확인해 주세요.")
    return rows


def point_summary(lat, lon, date):
    d = _date(date)
    rows = _tidebed(lat, lon, d, 1)
    r0 = rows[0]
    ts = [_dt(r.get("slctdDt")) for r in rows]
    _need_times(ts, r0.get("slctdDt"))
    tr = [_dt(r.get("obsrvnDt")) for r in rows]
    hs = [_num(r.get("slctdHgt")) for r in rows]
    hr = [_num(r.get("obsrvnHgt")) for r in rows]
    es = _extrema(ts, hs, 1)
    er = _extrema(tr, hr, 1)
    events = []
    for kind, i in es:
        e = {"구분": kind, "지점_시각": ts[i].strftime("%H:%M"), "지점_높이_cm": _r(hs[i], 0)}
        refs = [j for k, j in er if k == kind]
        if refs:
            j = min(refs, key=lambda j: abs((tr[j] - ts[i]).total_seconds()))
            if abs((tr[j] - ts[i]).total_seconds()) < 4 * 3600:
                e.update({"기준항_시각": tr[j].strftime("%H:%M"), "기준항_높이_cm": _r(hr[j], 0),
                          "시각차_분(지점-기준항)": round((ts[i] - tr[j]).total_seconds() / 60)})
        events.append(e)
    valid = [h for h in hs if h is not None]
    return {
        "지점": {"위도": lat, "경도": lon},
        "날짜": f"{d[:4]}-{d[4:6]}-{d[6:]}",
        "기준항": r0.get("obsvtrNm"),
        "조시차(API 값)": r0.get("josi"),
        "조고비(API 값)": r0.get("jogo"),
        "평균해면_cm(API 값)": r0.get("msl"),
        "고조저조": events,
        "지점_하루최고_cm": _r(max(valid), 0) if valid else None,
        "지점_하루최저_cm": _r(min(valid), 0) if valid else None,
        "자료수": len(rows),
        "참고": ["1분 간격 예측 자료로 계산. 하루 처음·끝 30분 안의 고조·저조는 빠질 수 있음",
                 "예측 조위라서 실제(바람·기압 영향)와 다를 수 있음",
                 "높이 기준면은 API 설명서 확인 필요 [확인 필요]"],
        "출처": SOURCE_TIDEBED,
    }


def point_series(lat, lon, date, interval_min, start, end, include_reference):
    d = _date(date)
    m = _interval(interval_min)
    t0, t1 = _hm(start, "00:00"), _hm(end, "23:59")
    rows = _tidebed(lat, lon, d, m)
    ts = [_dt(r.get("slctdDt")) for r in rows]
    _need_times(ts, rows[0].get("slctdDt"))
    idx = _window(ts, t0, t1)
    if len(idx) > MAX_ROWS_OUT:
        raise ApiError(f"{len(idx)}줄이라 너무 깁니다 (최대 {MAX_ROWS_OUT}). 간격을 늘리거나 시간 범위를 줄여 주세요.")
    if include_reference:
        cols = ["시각", "지점_cm", "기준항_cm"]
        data = [[ts[i].strftime("%H:%M"), _r(_num(rows[i].get("slctdHgt")), 0),
                 _r(_num(rows[i].get("obsrvnHgt")), 0)] for i in idx]
    else:
        cols = ["시각", "지점_cm"]
        data = [[ts[i].strftime("%H:%M"), _r(_num(rows[i].get("slctdHgt")), 0)] for i in idx]
    return {
        "지점": {"위도": lat, "경도": lon},
        "날짜": f"{d[:4]}-{d[4:6]}-{d[6:]}",
        "기준항": rows[0].get("obsvtrNm"),
        "간격_분": m,
        "열": cols,
        "자료": data,
        "참고": "예측 조위. 높이 기준면은 API 설명서 확인 필요 [확인 필요]",
        "출처": SOURCE_TIDEBED,
    }


def station(obs_code, date, interval_min, start, end):
    code = STATION_ALIASES.get(obs_code.strip(), obs_code.strip())
    if not re.fullmatch(r"[A-Za-z]{2}_\d{4}", code):
        raise ApiError(f"관측소 코드는 DT_0001 같은 모양입니다 (받은 값: {obs_code}). "
                       f"이름으로는 {', '.join(STATION_ALIASES)} 만 됩니다.")
    d = _date(date)
    m = _interval(interval_min)
    t0, t1 = _hm(start, "00:00"), _hm(end, "23:59")
    rows = _fetch_all(SURVEY_URL, dict(obsCode=code.upper(), reqDate=d, min=str(m)), CACHE_TTL_OBS)
    if not rows:
        raise ApiError("데이터가 없습니다. 관측소 코드와 날짜를 확인해 주세요.")
    ts = [_dt(r.get("obsrvnDt")) for r in rows]
    _need_times(ts, rows[0].get("obsrvnDt"))
    # 필드 뜻은 공공데이터포털 명세(swagger) 기준: tdlvHgt = 예측조위, bscTdlvHgt = 실측조위
    pred = [_num(r.get("tdlvHgt")) for r in rows]
    obs = [_num(r.get("bscTdlvHgt")) for r in rows]
    idx = _window(ts, t0, t1)
    both = [i for i in idx if pred[i] is not None and obs[i] is not None]
    diff = [obs[i] - pred[i] for i in both]
    last = max((i for i in idx if obs[i] is not None), key=lambda i: ts[i], default=None)
    step = m
    keep = idx
    if len(keep) > MAX_ROWS_OUT:            # 길면 고르게 솎아서 300줄 안으로
        k = -(-len(keep) // MAX_ROWS_OUT)
        keep = keep[::k]
        step = m * k
    r0 = rows[0]
    return {
        "관측소": r0.get("obsvtrNm"),
        "관측소코드": code.upper(),
        "위치": {"위도": _num(r0.get("lat")), "경도": _num(r0.get("lot"))},
        "날짜": f"{d[:4]}-{d[4:6]}-{d[6:]}",
        "마지막_실측": None if last is None else {
            "시각": ts[last].strftime("%H:%M"), "실측_cm": _r(obs[last]), "예측_cm": _r(pred[last]),
            "차이_cm(실측-예측)": _r(obs[last] - pred[last]) if pred[last] is not None else None},
        "실측-예측_차이_cm": None if not diff else {
            "평균": _r(sum(diff) / len(diff)), "최대": _r(max(diff)), "최소": _r(min(diff)), "비교한_점수": len(diff)},
        "간격_분": step,
        "열": ["시각", "실측_cm", "예측_cm"],
        "자료": [[ts[i].strftime("%H:%M"), _r(obs[i]), _r(pred[i])] for i in keep],
        "참고": ["필드 뜻은 공공데이터포털 명세 기준 (tdlvHgt=예측조위, bscTdlvHgt=실측조위)",
                 "관측소 이름이 기대와 다르면 코드가 틀린 것"],
        "출처": SOURCE_SURVEY,
    }


# ───────────────────────── MCP 서버 ─────────────────────────

SECRET_FROM_FILE = not os.environ.get("MCP_SECRET", "").strip()
SECRET = os.environ.get("MCP_SECRET", "").strip() or _from_file("mcp_secret.txt")
PORT = int(os.environ.get("PORT", "8000"))
HOST = os.environ.get("HOST", "0.0.0.0")

mcp = FastMCP(
    "gaetguard-tide",
    instructions=("국립해양조사원 조위 자료(공공데이터포털)를 부르는 도구입니다. "
                  "갯벌 지점은 위경도로 tide_point_summary(고조·저조 요약) 또는 tide_point_series(시계열)를, "
                  "조위관측소 실측은 station_tide를 쓰세요. 날짜는 한국 시각 기준 yyyyMMdd. "
                  "결과의 '출처'와 '참고'를 답에 함께 적어 주세요."),
    host=HOST,
    port=PORT,
    streamable_http_path=f"/{SECRET}/mcp",
    stateless_http=True,
    json_response=True,
    log_level="WARNING",          # 접속 기록(비밀 주소 포함)을 남기지 않음
    # Render·터널 뒤에서는 Host 헤더가 외부 주소라서 localhost 전용 검사를 끔 (대신 비밀 주소로 막음)
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)


async def _run(fn, *args):
    # 막히는 네트워크 호출은 별도 스레드에서. ApiError는 그대로 올려서 isError=True 결과로 감
    return await anyio.to_thread.run_sync(partial(fn, *args))


@mcp.tool()
async def tide_point_summary(lat: float, lon: float, date: str = "") -> dict:
    """위경도 한 점(갯벌·바다)의 하루 고조·저조 시각과 높이, 가장 가까운 기준항 이름·조시차·조고비·평균해면,
    기준항과의 고조·저조 시각 차이(분)를 돌려줍니다. 자료: 국립해양조사원 TideBED 예측 조위(1분 간격).

    lat: 위도 (예: 37.340), lon: 경도 (예: 126.675), date: yyyyMMdd 또는 yyyy-mm-dd (비우면 오늘, 한국 시각)
    """
    return await _run(point_summary, float(lat), float(lon), date)


@mcp.tool()
async def tide_point_series(lat: float, lon: float, date: str = "", interval_min: int = 10,
                            start: str = "00:00", end: str = "23:59", include_reference: bool = False) -> dict:
    """위경도 한 점의 예측 조위 시계열(cm)을 돌려줍니다 (최대 300줄). 자료: 국립해양조사원 TideBED.

    interval_min: 1~60분 (기본 10), start/end: HH:MM (한국 시각), include_reference: 기준항 값도 같이
    """
    return await _run(point_series, float(lat), float(lon), date, interval_min, start, end, include_reference)


@mcp.tool()
async def station_tide(obs_code: str, date: str = "", interval_min: int = 10,
                       start: str = "00:00", end: str = "23:59") -> dict:
    """조위관측소의 실측 조위와 예측 조위를 같이 돌려주고, 차이(실측-예측) 평균·최대·최소와 마지막 실측값을 요약합니다.
    자료: 국립해양조사원 조위관측소 실측·예측 조위 API (따로 활용신청 필요).

    obs_code: 관측소 코드 (예: DT_0001) 또는 이름 '인천', '인천송도' [코드 확인 필요 — 결과의 관측소 이름으로 확인]
    interval_min: 1~60분 (기본 10), start/end: HH:MM
    """
    return await _run(station, obs_code, date, interval_min, start, end)


@mcp.custom_route("/", methods=["GET"])
async def health(_: Request):
    return PlainTextResponse("ok")


def main():
    if len(SECRET) < 16:
        sys.exit("MCP_SECRET 환경변수(16자 이상)를 먼저 설정해 주세요.\n"
                 "만들기: python -c \"import secrets; print(secrets.token_urlsafe(24))\"")
    if not (os.environ.get("DATA_GO_KR_KEY", "").strip() or _from_file("datago_key.txt")):
        print("경고: 인증키(DATA_GO_KR_KEY 또는 datago_key.txt)가 없습니다. 도구를 부르면 오류가 납니다.", file=sys.stderr)
    if SECRET_FROM_FILE:      # 내 PC에서 돌릴 때만 화면에 보여 줌 (클라우드 로그에는 안 남김)
        print(f"커넥터 주소 = <터널 주소>/{SECRET}/mcp", file=sys.stderr)
    print(f"갯가드 조위 커넥터 시작: 포트 {PORT}, 주소 /<MCP_SECRET>/mcp", file=sys.stderr, flush=True)
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
