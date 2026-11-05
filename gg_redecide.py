"""걷는 동안 30초마다 다시 판정 — 사람이 느리거나, 멈추거나, 길을 벗어나면 다른 탈출로 · 피난 · 대기로

처음 판정(gg_decide.py)은 "0.5m/s로, 반응 2분 뒤 출발"을 가정합니다. 실제 사람은 더 느리거나, 멈추거나,
길을 벗어날 수 있으므로 걷는 동안 30초마다 그때 알 수 있는 정보로 다시 계산합니다.
  물  : 그 시각까지 10분 실측 평균으로 높이 보정을 새로 함 (gg_decide.py 와 같은 방식, 미래 실측은 안 씀)
  사람: 지금 위치 + 걸음 속도 = 최근 "걸으라고 했고 실제로 움직인" 5분의 속도 (0.5m/s 넘게는 안 봄)
        · 멈추라고 한 시간(새 길 알려 준 뒤 반응 2분)은 뺌
        · 서 있던 시간도 속도에서는 빼고, 그 대신 시계가 흘러간 만큼 여유가 줄어 듦
          (펄에 잠깐 빠졌다 나온 사람을 "앞으로도 0m/s"로 보지 않게)
        · 위치는 지금 시뮬레이션 값 그대로 [가정: 실제는 위치 장치 오차가 더해짐, 방법 미정]
  검사: 탈출 중 → 남은 길을 그 속도로 걸을 때 (마지막 출발 여유 L) − (99% 여유) ≥ 0 이면 그대로
        피난 중 → 남은 길의 모든 칸에 (여유 뺀) 막히는 시각 전에 닿으면 그대로
        안내 중인 길에서 OFF_M 넘게 벗어나도 다시 계산
  다시 계산(지금 자리, 그 속도, 반응 2분 포함):
        99% 여유로 새 탈출로 → 99% 여유로 피난(gg_decide.py 와 같은 조건)
        → 95% 여유로 탈출 → 95% 여유로 피난
        → 예보만(통계 여유 없이, 칸 반을 건너는 시간만 뺌)으로 탈출
        → 그래도 안 되면 예보로 "가장 오래 물 밖에 있는 쪽": 해안 쪽으로 계속 / 피난처 / 제자리 대기 중 하나
        (99%로 안 될 때 바로 대기로 가면, 걸어서 더 버틸 수 있는 사람을 곧 잠길 자리에 세워 두게 됨 — 시험에서 확인)
        99% 탈출이 아닌 판정이 처음 나온 때 = 구조 요청 시각
        한 번 피난·대기로 바뀌면 탈출로 되돌리지 않음 [가정: 왔다 갔다 하면 사람이 헷갈림]
검증: 실제 실측 물때로 막히는 시각을 다시 계산 → 사람이 서 있거나 지나는 칸이 물 30cm가 되는 때 = 막힘
      (같은 사람을 "재판정 없음: 처음 길 그대로"로도 걸려서 비교)

실행 예:
  python gg_redecide.py --tide tide\\station_DT_0043_1m_우리예측2022-2024.csv --time "2025-09-11 02:07" ^
      --route tide\\판정_youngheung_2025-09-11_탈출로_0207.json
  python gg_redecide.py --tide tide\\station_DT_0043_1m_우리예측2022-2024.csv --batch 20
필요: gg_decide.py, gg_retreat_map.py (같은 폴더), numpy scipy matplotlib
"""
import argparse
import csv
import heapq
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gg_decide as gd       # noqa: E402
import gg_retreat_map as rm  # noqa: E402

TICK = 0.5          # 분: 다시 판정하는 간격 (30초)
SPEED_WIN = 5.0     # 분: 걸음 속도를 재는 구간 ("걸으라고 한" 시간만 셈)
SPEED_MIN_T = 1.0   # 분: 이보다 짧게 쟀으면 앞서 잰 속도(처음엔 기본 속도)를 씀
STILL_M = 3.0       # m: 30초에 이보다 덜 움직였으면 "서 있었음" (속도 계산에서 뺌)
OFF_M = 30.0        # m: 안내 중인 길에서 이만큼 넘게 벗어나면 그 자리에서 다시 계산 [가정]
V_FLOOR = 0.1       # m/s: 멈춘 사람도 이 속도로 봄 (0으로 나누지 않게)
T_END = 330.0       # 분: 판정 뒤 이만큼까지만 따라감 (물 계산 길이 360분 안)
# 95% 시각 오차(분) — gd.MARGIN(99%)과 같은 시험(영흥도 2025-01~2026-11, 우리 예측 + 높이 보정, 물 찰 때)의 95% 값
MARGIN95 = {"밤": [(30, 4.9), (60, 8.3), (120, 14.5)],
            "낮": [(30, 7.0), (60, 11.7), (120, 17.9)]}
LEVELS = [("99%", gd.MARGIN), ("95%", MARGIN95), ("예보만", None)]
NB = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0), (-1, -1, 1.4142), (-1, 1, 1.4142), (1, -1, 1.4142), (1, 1, 1.4142)]


class Water:
    """판정 시각 t0부터 t분 뒤에 '그때까지 알 수 있던 정보'로 만든 막히는 시각 (지금부터 분)"""

    def __init__(self, G, T, O, P, t0, depth):
        self.G, self.T, self.O, self.P, self.t0, self.depth = G, T, O, P, t0, depth
        self.lat_c, self.lon_c = float(G["GY"].mean()), float(G["GX"].mean())
        self.cache = {}
        self.ls_cache = {}

    def latest(self, t, v):
        """같은 시각·같은 속도면 다시 계산하지 않음 (일괄 검증에서 출발 칸 여러 개가 같은 계산을 씀)"""
        key = (round(t, 4), round(v, 4))
        if key not in self.ls_cache:
            tcf, _, _ = self.at(t)
            if len(self.ls_cache) > 6:
                self.ls_cache.pop(next(iter(self.ls_cache)))
            self.ls_cache[key] = rm.latest_start(self.G, tcf, v, with_path=True)[:2]
        return self.ls_cache[key]

    def at(self, t):
        m = int(np.floor(t + 1e-9))
        if m not in self.cache:
            tm = self.t0 + np.timedelta64(m, "m")
            i = int(np.searchsorted(self.T, tm))
            o10, p10 = self.O[max(0, i - 9):i + 1], self.P[max(0, i - 9):i + 1]
            ok = np.isfinite(o10) & np.isfinite(p10)
            off = float(np.mean(o10[ok] - p10[ok])) if ok.any() else 0.0
            Hf = (gd.window(self.T, self.P, tm, gd.HORIZON_H) + off) / 100.0
            tc = rm.close_times_fast(self.G, Hf, self.depth)
            night = bool(gd._sun_elev(np.array([tm]), self.lat_c, self.lon_c)[0] < -0.833)
            if len(self.cache) > 4:
                self.cache.pop(min(self.cache))
            self.cache[m] = (tc, night, off)
        tc, night, off = self.cache[m]
        return tc - (t - m), night, off

    def truth(self):
        Ht = gd.smooth5(gd.window(self.T, self.O, self.t0, gd.HORIZON_H)) / 100.0
        return rm.close_times_fast(self.G, Ht, self.depth)


def margin_tbl(h, night, tbl):
    """gd.margin 과 같은 방식(30분 아래는 30분 값, 120분 넘으면 60~120분 기울기로 늘림), 표만 바꿔 씀"""
    t = tbl["밤" if night else "낮"]
    xs = np.array([x for x, _ in t], float)
    ys = np.array([y for _, y in t], float)
    h = np.asarray(h, float)
    m = np.interp(np.clip(h, xs[0], xs[-1]), xs, ys)
    return np.where(h > xs[-1], ys[-1] + (h - xs[-1]) * (ys[-1] - ys[-2]) / (xs[-1] - xs[-2]), m)


def margin_arr(x, night, tbl=None):
    x = np.asarray(x, float)
    xx = np.where(np.isfinite(x), x, 999)
    m = gd.margin(xx, night) if tbl is None else margin_tbl(xx, night, tbl)
    return np.where(np.isfinite(x), m, 0.0)


def margin_level(level, x, night, v, res):
    """단계별 여유(분). '예보만' = 통계 여유 없이, 칸 가운데에서 대각선 경계까지 걷는 시간만
    (판정은 칸 가운데 도착 시각으로 하지만 사람은 그 칸을 다 건널 때까지 그 칸 안에 있음)"""
    if level == "예보만":
        return np.where(np.isfinite(np.asarray(x, float)), res / 2 * 1.4142 / v / 60.0, 0.0)
    return margin_arr(x, night, dict(LEVELS)[level])


def refuge_path(G, start, tcs, walk, t_start):
    """gd.refuge 와 같은 규칙 + 가는 길: start 칸에서 t_start(분)에 출발, 각 칸 (여유 뺀) 막히는 시각 전 도착으로 닿는 칸 중
    가장 늦게 잠기는 칸 (같으면 가까운 칸). gd.refuge 와 다른 점: 해안선 칸(육지)에도 닿을 수 있음 → 닿으면 그게 가장 좋은 곳
    (해안 바로 앞 칸에서 구조를 기다리라고 하지 않게)"""
    sea, bar = G["sea"], G["barrier"]
    ny, nx = sea.shape
    step = G["res"] / walk / 60.0
    sy, sx = divmod(start, nx)
    best = (tcs[sy, sx], 0.0, start)
    arr = {start: t_start}
    par = {start: -1}
    heap = [(t_start, start)]
    while heap:
        t, k = heapq.heappop(heap)
        if t > arr.get(k, np.inf):
            continue
        y, x = divmod(k, nx)
        if (tcs[y, x], -t) > (best[0], -best[1]):
            best = (tcs[y, x], t, k)
        if bar[y, x]:
            continue                                       # 육지에 닿음: 더 넘어가지 않음
        for dy, dx, dd in NB:
            yy, xx = y + dy, x + dx
            if 0 <= yy < ny and 0 <= xx < nx and (sea[yy, xx] or bar[yy, xx]):
                kk = yy * nx + xx
                ta = t + dd * step
                if ta < tcs[yy, xx] and ta < arr.get(kk, np.inf):
                    arr[kk] = ta
                    par[kk] = k
                    heapq.heappush(heap, (ta, kk))
    path = [best[2]]
    while par[path[-1]] >= 0:
        path.append(par[path[-1]])
    return best[0], best[1], path[::-1]


class Redecider:
    def __init__(self, G, water, origin, walk):
        self.G, self.W, self.walk = G, water, walk
        self.lat0, self.lon0 = origin
        self.kx, self.ky, self.res = G["kx"], G["ky"], G["res"]
        self.B = rm.AREA["B"]
        self.ny, self.nx = G["sea"].shape
        self.cn = ((G["GY"] - self.lat0) * self.ky).ravel()
        self.ce = ((G["GX"] - self.lon0) * self.kx).ravel()
        self.bar = G["barrier"].ravel()
        self.sea = G["sea"].ravel()
        self.plan, self.hist, self.waits, self.log, self.plans = None, [], [], [], []
        self.alert = None
        self.v_last = walk

    # ---- 좌표
    def cell(self, n, e):
        iy = int((self.lat0 + n / self.ky - self.B[2]) * self.ky / self.res)
        ix = int((self.lon0 + e / self.kx - self.B[0]) * self.kx / self.res)
        return min(max(iy, 0), self.ny - 1) * self.nx + min(max(ix, 0), self.nx - 1)

    def center(self, k):
        return np.array([self.cn[k], self.ce[k]])

    def chain(self, parent, k):
        out = [k]
        while not self.bar[out[-1]] and parent[out[-1]] >= 0 and len(out) < 5000:
            out.append(int(parent[out[-1]]))
        return out

    def make_plan(self, mode, t, pos, cells, info):
        # 지금 칸 가운데로 되돌아가지 않고 바로 다음 칸으로 (칸 목록에는 지금 칸도 남겨 검사)
        nxt = cells[1:] if (cells and cells[0] == self.cell(*pos) and len(cells) > 1) else cells
        pts = np.array([pos] + [self.center(c) for c in nxt]) if nxt else np.array([pos])
        s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))] if len(pts) > 1 else np.array([0.0])
        P = dict(id=len(self.plans), mode=mode, t_set=t, pts=pts, cells=np.array([self.cell(*pos)] + list(nxt), dtype=np.int64),
                 s=s, s_last=0.0, info=info, level=info.get("level", "99%"), short=bool(info.get("short", False)),
                 cont=bool(info.get("cont", False)), via_ref=bool(info.get("via_ref", False)))
        self.plans.append(P)
        return P

    def walk_dry(self, t, pos, cells, v, tcf, wait):
        """칸 목록(첫 칸 = 지금 칸)을 따라 걸을 때 예보로 언제 물에 막히나 (지금부터 분, 끝까지 가면 inf)
        칸에 닿기 전에 막히면 그 앞 칸이 막힐 때까지, 칸을 건너는 중에 막히면 그때"""
        tf = tcf.ravel()
        B = self.res / 2 * 1.4142 / v / 60.0
        pts = [np.asarray(pos, float)] + [self.center(k) for k in cells[1:]]
        arr = wait + np.r_[0.0, np.cumsum([np.hypot(*(b - a)) for a, b in zip(pts[:-1], pts[1:])])] / v / 60.0
        for i, k in enumerate(cells):
            if arr[i] + B >= tf[k]:
                if arr[i] < tf[k] or i == 0:
                    return float(tf[k])
                return float(tf[cells[i - 1]])
        return np.inf

    # ---- 다시 계산
    @staticmethod
    def same_way(a, b, k=4):
        """두 칸 목록(첫 칸 = 지금 칸)의 다음 k칸이 같으면 같은 길 → 사람에게 새로 알릴 것이 없으므로 멈추지 않음"""
        return a is not None and b is not None and len(a) > 1 and list(a[1:1 + k]) == list(b[1:1 + k])

    def replan(self, t, pos, v, allow_escape, cur=None):
        """cur: 지금 걷는 탈출로의 남은 칸 (첫 칸 = 지금 칸). 새 길이 같은 길이면 멈추지 않고 계속"""
        tcf, night, off = self.W.at(t)
        c = self.cell(*pos)
        if self.bar[c] or not self.sea[c]:
            return self.make_plan("도착", t, pos, [], dict(why="해안"))
        cy, cx = divmod(c, self.nx)
        lead = float(np.hypot(*(np.asarray(pos) - self.center(c)))) / v / 60.0
        Lp, ch = -np.inf, None
        if allow_escape:
            Lf, parent = self.W.latest(t, v)
            Lp = float(Lf[cy, cx]) - lead
            ch = self.chain(parent, c)
        same = self.same_way(ch, cur)
        for lvl, _ in LEVELS:
            if allow_escape and np.isfinite(Lp):
                sl = Lp - float(margin_level(lvl, Lp, night, v, self.res)) - (0.0 if same else gd.REACT)
                if sl >= 0:
                    return self.make_plan("탈출", t, pos, ch, dict(L=Lp, slack=sl, v=v, off=off, level=lvl, cont=same))
            tcs = np.where(np.isfinite(tcf), tcf - margin_level(lvl, tcf, night, v, self.res), np.inf)
            best_t, t_arr, path = refuge_path(self.G, c, tcs, v, gd.REACT + lead)
            if self.bar[path[-1]]:                          # 피난 규칙(칸마다 여유)으로 해안까지 닿음 → 탈출
                return self.make_plan("탈출", t, pos, path, dict(L=np.nan, slack=np.nan, arrive=t_arr, v=v, off=off, level=lvl,
                                                              via_ref=True, cont=self.same_way(path, cur)))
            if lvl != "예보만":
                if best_t > tcs[cy, cx] + gd.MIN_GAIN:
                    return self.make_plan("피난", t, pos, path, dict(target_tcs=best_t, here_tcs=float(tcs[cy, cx]), arrive=t_arr, v=v,
                                                                  off=off, level=lvl))
                continue
            # 마지막 단계: 예보로 가장 오래 물 밖에 있는 쪽 (지금 길 계속 / 새 길로 해안 쪽 / 피난처 / 제자리)
            B = self.res / 2 * 1.4142 / v / 60.0
            d_wait = float(tcf[cy, cx])
            cand = []
            if cur is not None and len(cur) > 1:
                cand.append((self.walk_dry(t, pos, cur, v, tcf, 0.0), 3, "cur"))
            if ch is not None and np.isfinite(Lp):
                cand.append((self.walk_dry(t, pos, ch, v, tcf, 0.0 if same else gd.REACT), 2, "new"))
            if len(path) > 1:
                cand.append((best_t + B, 1, "ref"))
            if cand:
                d, _, kind = max(cand)
                if d > d_wait + gd.MIN_GAIN:
                    if kind == "cur":
                        return self.make_plan("탈출", t, pos, list(cur), dict(L=np.nan, slack=np.nan, v=v, off=off, level="예보만",
                                                                           short=True, dry=d, cont=True))
                    if kind == "new":
                        return self.make_plan("탈출", t, pos, ch, dict(L=Lp, slack=Lp - B, v=v, off=off, level="예보만",
                                                                     short=True, dry=d, cont=same))
                    return self.make_plan("피난", t, pos, path, dict(target_tcs=best_t, here_tcs=float(tcs[cy, cx]), arrive=t_arr, v=v,
                                                                  off=off, level=lvl, dry=d))
        return self.make_plan("대기", t, pos, [], dict(here_tcf=float(tcf[cy, cx]), v=v, off=off, level="-"))

    # ---- 걸음 속도 (걸으라고 한 시간만)
    def speed(self):
        dist = tm = 0.0
        for (t1, p1), (t2, p2) in zip(self.hist[-2::-1], self.hist[:0:-1]):
            dur = t2 - t1
            wait = sum(max(0.0, min(t2, b) - max(t1, a)) for a, b in self.waits)
            inst = dur - wait
            step = float(np.hypot(*(np.asarray(p2) - np.asarray(p1))))
            if inst <= 1e-9 or step < STILL_M * dur / TICK:
                continue                                   # 멈추라고 했거나, 서 있었음
            frac = min(1.0, (SPEED_WIN - tm) / inst)
            dist += step * frac
            tm += inst * frac
            if tm >= SPEED_WIN - 1e-9:
                break
        if tm >= SPEED_MIN_T:
            self.v_last = float(np.clip(dist / (tm * 60.0), V_FLOOR, self.walk))
        return self.v_last

    # ---- 안내 중인 길 위 어디쯤인가 (뒤로 가지 않게, 앞 150m 안에서)
    def project(self, P, pos):
        pts, s = P["pts"], P["s"]
        if len(pts) < 2:
            return 0.0, float(np.hypot(*(np.asarray(pos) - pts[0])))
        best = (np.inf, P["s_last"])
        for i in range(len(pts) - 1):
            if s[i + 1] < P["s_last"] - 1e-6 or s[i] > P["s_last"] + 150:
                continue
            a, b = pts[i], pts[i + 1]
            ab = b - a
            L2 = float(ab @ ab)
            u = 0.0 if L2 == 0 else float(np.clip((np.asarray(pos) - a) @ ab / L2, 0, 1))
            d = float(np.hypot(*(np.asarray(pos) - (a + u * ab))))
            if d < best[0]:
                best = (d, s[i] + u * np.sqrt(L2))
        P["s_last"] = max(P["s_last"], best[1])
        return P["s_last"], best[0]

    def check(self, t, pos, v):
        P = self.plan
        s_p, d = self.project(P, pos)
        tcf, night, _ = self.W.at(t)
        wait_left = 0.0 if P["cont"] else max(0.0, P["t_set"] + gd.REACT - t)
        ahead = P["s"] >= s_p - 1e-6
        cells, sa = P["cells"][ahead], P["s"][ahead]
        arr = wait_left + (sa - s_p) / v / 60.0
        c_now = self.cell(*pos)
        tf = tcf.ravel()
        if P["mode"] == "탈출" and P["short"]:            # 해안 쪽으로 계속: 예보로 물에 막히는 때가 정했을 때보다 5분 넘게 당겨지지 않으면 그대로
            rest = [c_now] + [int(k) for k in cells if k != c_now]
            dry = self.walk_dry(t, pos, rest, v, tcf, wait_left)
            ok = dry >= P["info"]["dry"] - (t - P["t_set"]) - 5.0 and d <= OFF_M
            return ok, dict(s=s_p, d=d, L=np.nan, slack=dry - (P["info"]["dry"] - (t - P["t_set"])), rest=rest)
        if P["mode"] == "탈출" and not P["via_ref"]:
            L = min(float(tf[c_now]), float(np.min(tf[cells] - arr)) if len(cells) else np.inf)
            sl = L - float(margin_level(P["level"], L, night, v, self.res)) if np.isfinite(L) else np.inf
            ok = sl >= 0 and d <= OFF_M
            return ok, dict(s=s_p, d=d, L=L, slack=sl, rest=[c_now] + [int(k) for k in cells if k != c_now])
        tcs = np.where(np.isfinite(tf), tf - margin_level(P["level"], tf, night, v, self.res), np.inf)
        keep = cells != c_now
        late = arr[keep] - tcs[cells[keep]]
        worst = float(np.max(late)) if len(late) else -np.inf
        ok = worst < 0 and d <= OFF_M
        return ok, dict(s=s_p, d=d, L=np.nan, slack=-worst, rest=[c_now] + [int(k) for k in cells if k != c_now])

    # ---- 30초마다
    def start(self, pos):
        self.hist.append((0.0, tuple(pos)))
        self.plan = self.replan(0.0, pos, self.walk, True)
        self.waits.append((0.0, gd.REACT))
        if self.plan["mode"] != "탈출" or self.plan["level"] != "99%":
            self.alert = 0.0
        self._log(0.0, pos, self.walk, dict(s=0, d=0, L=self.plan["info"].get("L", np.nan),
                                              slack=self.plan["info"].get("slack", np.nan)), "처음 판정")
        return self.plan

    def tick(self, t, pos):
        self.hist.append((t, tuple(pos)))
        P = self.plan
        if P["mode"] in ("도착", "피난처", "대기"):
            return P
        v = self.speed()
        ok, info = self.check(t, pos, v)
        if info["s"] >= P["s"][-1] - 5.0 and info["d"] <= OFF_M:
            P["mode"] = "도착" if P["mode"] == "탈출" else "피난처"
            self._log(t, pos, v, info, "도착" if P["mode"] == "도착" else "피난처 도착")
            return P
        if ok:
            self._log(t, pos, v, info, "")
            return P
        why = f"길에서 {info['d']:.0f}m 벗어남" if info["d"] > OFF_M else f"여유 모자람 ({info['slack']:+.1f}분, {v:.2f}m/s)"
        cur = info.get("rest") if (P["mode"] == "탈출" and info["d"] <= OFF_M) else None
        newP = self.replan(t, pos, v, P["mode"] == "탈출", cur)
        self.plan = newP
        if not newP["cont"]:
            self.waits.append((t, t + gd.REACT))
        if (newP["mode"] != "탈출" or newP["level"] != "99%") and self.alert is None:
            self.alert = t
        lv = {"99%": " (99% 여유)", "95%": " (95% 여유)", "예보만": " (예보만, 통계 여유 없이)"}.get(newP["level"], "")
        if newP["short"]:
            lv = f" (해안 쪽으로 계속, 예보로 {newP['info']['dry']:.0f}분 뒤 물에 막힘)"
        self._log(t, pos, v, info, f"{why} → {newP['mode']}{lv}" + (" · 같은 길 계속" if newP["cont"] else ""))
        return newP

    def _log(self, t, pos, v, info, event):
        self.log.append(dict(t_min=round(t, 2), mode=self.plan["mode"], plan=self.plan["id"], v_est=round(v, 3),
                             n=round(float(pos[0]), 1), e=round(float(pos[1]), 1), along_m=round(float(info.get("s", 0)), 1),
                             off_m=round(float(info.get("d", 0)), 1), slack_min=round(float(info.get("slack", np.nan)), 2),
                             event=event))


# ---------------------------------------------------------------- 가상의 사람
BEHAVIORS = {
    "정상 0.5m/s": dict(v=0.5),
    "느림 0.4m/s": dict(v=0.4),
    "아주 느림 0.3m/s": dict(v=0.3),
    "15분 걷고 10분 멈춤": dict(v=0.5, stop=(17.0, 10.0)),
    "15분 걷고 5분 길 벗어남": dict(v=0.5, off=(17.0, 5.0)),
}
DT = 5 / 60.0       # 분: 사람 움직임 계산 간격


def simulate(G, water, tct, origin, start_ne, beh, redecide=True, walk=0.5):
    """가상의 사람 한 명. redecide=False면 처음 탈출로를 끝까지 (길 벗어나면 드론이 원래 길로 데려옴)"""
    eng = Redecider(G, water, origin, walk)
    P0 = eng.start(np.array(start_ne, float))
    if P0["mode"] != "탈출":
        return None
    v_p = beh["v"]
    stop, off = beh.get("stop"), beh.get("off")
    pos = np.array(start_ne, float)
    path, ps, adopted = P0["pts"].copy(), 0.0, P0["id"]
    free_at = gd.REACT
    next_tick = TICK
    leave = None
    track, modes = [(0.0, pos[0], pos[1])], []
    t, caught, arrived, refuge_t = 0.0, None, None, None
    tf = tct.ravel()
    heading = None
    while t < T_END:
        if redecide and t >= next_tick - 1e-9:
            eng.tick(t, pos.copy())
            next_tick += TICK
            if eng.plan["mode"] == "피난처" and refuge_t is None:
                refuge_t = t
        plan = eng.plan if redecide else P0
        in_stop = stop and stop[0] <= t < stop[0] + stop[1]
        in_off = off and off[0] <= t < off[0] + off[1]
        if in_off:
            if leave is None:
                leave = (pos.copy(), ps)
                seg = np.diff(path, axis=0)
                i = int(np.searchsorted(np.r_[0, np.cumsum(np.hypot(*seg.T))], ps, side="right") - 1)
                d = seg[min(max(i, 0), len(seg) - 1)]
                d = d / max(np.hypot(*d), 1e-9)
                heading = np.array([-d[1], d[0]])          # 진행 방향 오른쪽 직각 (북·동 좌표)
            pos = pos + heading * v_p * DT * 60
        elif in_stop:
            pass
        else:
            if off and leave is not None and t >= off[0] + off[1] and free_at < off[0] + off[1] + gd.REACT:
                free_at = off[0] + off[1] + gd.REACT          # 길 벗어났다가 다시 안내 받음 → 반응 2분
                if not redecide:                               # 드론이 떠난 곳으로 데려온 뒤 원래 길 계속
                    rest = path[np.searchsorted(np.r_[0, np.cumsum(np.hypot(*np.diff(path, axis=0).T))], leave[1], side="right"):]
                    path = np.vstack([pos, leave[0], rest])
                    ps = 0.0
            if redecide and plan["id"] != adopted:
                if not plan["cont"]:
                    free_at = max(free_at, plan["t_set"] + gd.REACT)
                if t >= free_at - 1e-9:
                    path = np.vstack([pos, plan["pts"]]) if np.hypot(*(plan["pts"][0] - pos)) > 0.5 else plan["pts"].copy()
                    ps, adopted = 0.0, plan["id"]
            if t >= free_at - 1e-9 and (not redecide or adopted == plan["id"]) and len(path) > 1:
                ps += v_p * DT * 60
                cum = np.r_[0, np.cumsum(np.hypot(*np.diff(path, axis=0).T))]
                if ps >= cum[-1]:
                    ps = cum[-1]
                i = min(int(np.searchsorted(cum, ps, side="right") - 1), len(path) - 2)
                u = 0 if cum[i + 1] == cum[i] else (ps - cum[i]) / (cum[i + 1] - cum[i])
                pos = path[i] + u * (path[i + 1] - path[i])
                if ps >= cum[-1] - 1e-6 and plan["mode"] in ("탈출", "도착") and (not redecide or adopted == plan["id"]):
                    if eng.bar[eng.cell(*pos)] or not eng.sea[eng.cell(*pos)]:
                        arrived = t + DT
        t += DT
        track.append((t, pos[0], pos[1]))
        modes.append(plan["mode"])
        k = eng.cell(*pos)
        if arrived is not None:
            break
        if eng.sea[k] and t >= tf[k]:
            caught = t
            break
        # 더 움직이지 않는 상태(대기, 피난처 도착)면 그 칸이 막히는 때까지 건너뜀
        settled = redecide and plan["mode"] in ("대기", "피난처") and t >= free_at and not in_off
        if settled or (redecide and plan["mode"] == "피난" and ps >= np.sum(np.hypot(*np.diff(path, axis=0).T)) - 1e-6
                       and adopted == plan["id"] and not in_stop and not in_off):
            if plan["mode"] == "피난":
                plan["mode"] = "피난처"
                refuge_t = t
                eng._log(t, pos, 0, dict(s=plan["s"][-1], d=0), "피난처 도착")
            caught = max(t, float(tf[k])) if eng.sea[k] else None
            if caught is not None and caught > T_END:
                caught = None
            break
    tr = np.array(track)
    minmarg = np.nan
    if arrived is not None:
        ks = [eng.cell(n, e) for _, n, e in tr]
        mm = [tf[k] - tt for (tt, _, _), k in zip(tr, ks) if eng.sea[k] and np.isfinite(tf[k])]
        minmarg = float(min(mm)) if mm else np.inf
    final_mode = (eng.plan if redecide else P0)["mode"]
    return dict(arrived=arrived, caught=caught, alert=eng.alert if redecide else None, final=final_mode, refuge_t=refuge_t,
                walked=float(np.sum(np.hypot(*np.diff(tr[:, 1:], axis=0).T))), track=tr, minmarg=minmarg,
                plans=eng.plans if redecide else [P0], log=eng.log, n_switch=len(eng.plans) - 1 if redecide else 0)


def describe(r, t0):
    hm = lambda x: str(t0 + np.timedelta64(int(round(x * 60)), "s"))[11:16]     # noqa: E731
    if r["arrived"] is not None:
        s = f"{hm(r['arrived'])} 해안 도착 (지나간 칸 중 가장 아슬아슬한 곳도 {r['minmarg']:.0f}분 뒤 잠김)"
    elif r["final"] in ("피난", "피난처") and r["refuge_t"] is not None:
        s = f"{hm(r['refuge_t'])} 피난처 도착, 구조 대기" + (f", 물 30cm {hm(r['caught'])}" if r["caught"] else ", 계산 끝까지 안 잠김")
    elif r["final"] == "피난":
        s = f"{hm(r['caught'])} 피난처 가는 길에 막힘 (물 30cm)" if r["caught"] else "피난처에 못 닿음"
    elif r["final"] == "대기":
        s = "제자리 대기" + (f", 물 30cm {hm(r['caught'])}" if r["caught"] else ", 계산 끝까지 안 잠김")
    else:
        s = f"{hm(r['caught'])} 걷던 길에서 막힘 (물 30cm)" if r["caught"] else "끝까지 못 나옴"
    if r["alert"] is not None:
        s += f" · 구조 요청 {hm(r['alert'])}"
        if r["caught"]:
            s += f" (물 차기 {r['caught'] - r['alert']:.0f}분 전)"
    return s


def run_case(a, G, T, O, P):
    a._TOP = (T, O, P)
    route = json.load(open(a.route, encoding="utf-8"))
    t0 = np.datetime64(a.time.replace(" ", "T"))
    origin = (route["start_lat"], route["start_lon"])
    water = Water(G, T, O, P, t0, a.depth)
    tct = water.truth()
    res = {}
    for name, beh in BEHAVIORS.items():
        r0 = simulate(G, water, tct, origin, (0.0, 0.0), beh, redecide=False, walk=a.walk)
        r1 = simulate(G, water, tct, origin, (0.0, 0.0), beh, redecide=True, walk=a.walk)
        res[name] = (r0, r1)
        print(f"\n[{name}]")
        print(f"   재판정 없음: {describe(r0, t0)}")
        print(f"   재판정 있음: {describe(r1, t0)}")
        for row in r1["log"]:
            if row["event"]:
                print(f"      {str(t0 + np.timedelta64(int(round(row['t_min'] * 60)), 's'))[11:19]}  {row['event']}  "
                      f"(길 따라 {row['along_m']:.0f}m, 속도 {row['v_est']:.2f}m/s)")
    first = res[next(iter(res))][1]
    print(f"\n처음 판정 확인: {first['log'][0]['mode']}, 남은 여유 {first['log'][0]['slack_min']:.1f}분 (판정 파일 {route['slack_min']}분), "
          f"길 {first['plans'][0]['s'][-1]:.0f}m (판정 파일 {route['length_m']}m)")
    base = os.path.join(rm.HERE, "tide", f"재판정_{rm.AREA['key']}_{a.time[:10]}_{a.time[11:].replace(':', '')}")
    with open(base + ".csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["behavior", "redecide", "t_min", "clock", "mode", "plan", "v_est", "n", "e", "along_m", "off_m", "slack_min", "event"])
        for name, (r0, r1) in res.items():
            for row in r1["log"]:
                w.writerow([name, 1, row["t_min"], str(t0 + np.timedelta64(int(round(row["t_min"] * 60)), "s"))[11:19], row["mode"],
                            row["plan"], row["v_est"], row["n"], row["e"], row["along_m"], row["off_m"], row["slack_min"], row["event"]])
    with open(base + "_요약.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["behavior", "redecide", "result"])
        for name, (r0, r1) in res.items():
            w.writerow([name, 0, describe(r0, t0)])
            w.writerow([name, 1, describe(r1, t0)])
    draw_case(base + ".png", G, route, t0, res, tct, a)
    print("저장:", base + ".csv", base + "_요약.csv", base + ".png")


def short(r, t0):
    hm = lambda x: str(t0 + np.timedelta64(int(round(x * 60)), "s"))[11:16]     # noqa: E731
    if r["arrived"] is not None:
        s = f"{hm(r['arrived'])} 해안 도착\n(가장 아슬아슬한 칸도 {r['minmarg']:.0f}분 뒤 잠김)"
    elif r["refuge_t"] is not None:
        s = f"{hm(r['refuge_t'])} 피난처 도착\n→ 그 자리 물 30cm {hm(r['caught'])}" if r["caught"] else f"{hm(r['refuge_t'])} 피난처 도착"
    elif r["final"] == "대기":
        s = f"제자리 대기 → 물 30cm {hm(r['caught'])}"
    else:
        s = f"{hm(r['caught'])} 걷던 길에서 물에 막힘\n(미리 알린 것 없음)" if r["alert"] is None else f"{hm(r['caught'])} 걷던 길에서 물에 막힘"
    if r["alert"] is not None:
        s += f"\n구조 요청 {hm(r['alert'])}" + (f" (물 차기 {r['caught'] - r['alert']:.0f}분 전)" if r["caught"] else "")
    return s


def draw_case(out, G, route, t0, res, tct, a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    for name in ("Malgun Gothic", "Noto Sans CJK KR", "AppleGothic", "NanumGothic"):
        if any(f.name == name for f in fm.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False
    INK, INK2, MUTED, LAND, SEA = "#0b0b0b", "#52514e", "#9a9993", "#e9e7e1", "#c9d3df"
    COLS = ["#1baf7a", "#a8dcc4", "#eda100", "#d6402b"]
    C0, C1, SW = "#8a8984", "#2a78d6", "#eb6834"
    lat0, lon0 = route["start_lat"], route["start_lon"]
    kx, ky = G["kx"], G["ky"]
    to_ll = lambda n, e: (lat0 + np.asarray(n) / ky, lon0 + np.asarray(e) / kx)      # noqa: E731
    _, D = gd.decide_at(G, *a._TOP, t0, a)
    allpts = [np.array(route["path_latlon"])]
    for r0, r1 in res.values():
        for r in (r0, r1):
            la, lo = to_ll(r["track"][:, 1], r["track"][:, 2])
            allpts.append(np.c_[la, lo])
    allp = np.vstack(allpts)
    pad = 0.004
    ylim = (allp[:, 0].min() - pad, allp[:, 0].max() + pad)
    xlim = (allp[:, 1].min() - pad * 1.3, allp[:, 1].max() + pad * 1.3)
    k = len(res)
    fig = plt.figure(figsize=(3.5 * k + 0.6, 9.4), facecolor="white")
    fig.text(0.02, 0.955, "걷는 동안 30초마다 다시 판정 — 같은 사람을 '처음 길 그대로'와 '재판정'으로 걸려 실제 물때로 비교",
             fontsize=15.5, weight="bold", color=INK)
    fig.text(0.02, 0.922, f"영흥도 {str(t0)[:16].replace('T', ' ')} 판정 (탈출, 남은 여유 {route['slack_min']}분, 탈출로 {route['length_m']}m). "
             "재판정: 그때까지의 실측 보정 + 실제로 걸은 속도 + 지금 위치로 다시 계산 → 99% 여유로 안 되면 95% 여유, "
             "그래도 안 되면 예보로 가장 오래 물 밖에 있는 쪽", fontsize=9.6, color=INK2)
    w = 0.96 / k
    base = np.where(G["land"] | G["barrier"], 0, 1).astype(float)
    cls = np.full(D["dec"].shape, np.nan)
    cls[(D["dec"] == 1) & (D["slack"] >= 30)] = 0
    cls[(D["dec"] == 1) & (D["slack"] < 30)] = 1
    cls[D["dec"] == 2] = 2
    cls[D["dec"] == 3] = 3
    rl = np.array(route["path_latlon"])
    for i, (name, (r0, r1)) in enumerate(res.items()):
        ax = fig.add_axes([0.02 + i * w, 0.33, w - 0.012, 0.56])
        ax.pcolormesh(G["GX"], G["GY"], base, cmap=ListedColormap([LAND, SEA]), shading="auto", zorder=0)
        ax.pcolormesh(G["GX"], G["GY"], cls, cmap=ListedColormap(COLS), vmin=-0.5, vmax=3.5, shading="auto", alpha=0.45, zorder=1)
        ax.plot(rl[:, 1], rl[:, 0], color=INK, lw=0.9, ls=(0, (4, 3)), zorder=2)
        for r, col, lw, z in ((r0, C0, 2.6, 3), (r1, C1, 1.8, 4)):
            la, lo = to_ll(r["track"][:, 1], r["track"][:, 2])
            ax.plot(lo, la, color=col, lw=lw, zorder=z, solid_capstyle="round")
            if r["arrived"] is not None:
                ax.scatter([lo[-1]], [la[-1]], s=90, marker="*", color=col, edgecolor="white", linewidth=0.8, zorder=z + 5)
            elif r["refuge_t"] is not None or r["final"] == "대기":
                ax.scatter([lo[-1]], [la[-1]], s=60, marker="^", color=col, edgecolor="white", linewidth=0.8, zorder=z + 5)
            else:
                ax.scatter([lo[-1]], [la[-1]], s=70, marker="X", color=col, edgecolor="white", linewidth=0.8, zorder=z + 5)
        sw = [row for row in r1["log"] if "→" in row["event"]]
        if sw:
            la, lo = to_ll([sw[0]["n"]], [sw[0]["e"]])
            ax.scatter(lo, la, s=46, marker="D", color=SW, edgecolor="white", linewidth=0.8, zorder=12)
        ax.scatter([lon0], [lat0], s=36, color="white", edgecolor=INK, zorder=11)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect(1 / np.cos(np.radians(lat0)))
        ax.set_xticks([])
        ax.set_yticks([])
        for s_ in ax.spines.values():
            s_.set_color("#d8d7d2")
        ax.set_title(name, loc="left", fontsize=12.5, color=INK, weight="bold")
        if i == 0:
            bx, by = xlim[0] + (xlim[1] - xlim[0]) * 0.06, ylim[0] + (ylim[1] - ylim[0]) * 0.03
            ax.plot([bx, bx + 500 / kx], [by, by], color=INK, lw=2.2, zorder=13)
            ax.text(bx + 250 / kx, by + (ylim[1] - ylim[0]) * 0.012, "500 m", ha="center", fontsize=9, color=INK, zorder=13)
        x0 = 0.02 + i * w
        fig.text(x0, 0.305, "재판정 없음 (처음 길 그대로)", fontsize=9.6, color=C0, weight="bold", va="top")
        fig.text(x0, 0.283, short(r0, t0), fontsize=9.4, color=INK, va="top", linespacing=1.4)
        fig.text(x0, 0.215, "재판정 있음", fontsize=9.6, color=C1, weight="bold", va="top")
        evs = [row for row in r1["log"] if "→" in row["event"]]
        txt = short(r1, t0)
        if evs:
            e0 = evs[0]
            why = e0["event"].split(" → ")[0]
            what = e0["event"].split(" → ")[1].replace(" (99% 여유)", " (99%)").replace(" (95% 여유)", " (95%)")
            hm = str(t0 + np.timedelta64(int(round(e0["t_min"] * 60)), "s"))[11:16]
            txt = f"{hm} {why}\n→ {what}" + (f" 외 {len(evs) - 1}번 다시 계산" if len(evs) > 1 else "") + "\n" + txt
        else:
            txt = "끝까지 처음 길 유지\n" + txt
        fig.text(x0, 0.193, txt, fontsize=9.4, color=INK, va="top", linespacing=1.4)
    h = [Line2D([], [], color=INK, lw=0.9, ls=(0, (4, 3)), label="처음 판정한 탈출로"),
         Line2D([], [], color=C0, lw=2.6, label="재판정 없음: 사람이 걸은 길"),
         Line2D([], [], color=C1, lw=1.8, label="재판정 있음: 사람이 걸은 길"),
         Line2D([], [], color=SW, marker="D", ls="none", markersize=6, label="처음으로 판정을 바꾼 곳"),
         Line2D([], [], color=INK2, marker="*", ls="none", markersize=9, label="해안 도착"),
         Line2D([], [], color=INK2, marker="^", ls="none", markersize=7, label="피난처·대기 (구조 기다림)"),
         Line2D([], [], color=INK2, marker="X", ls="none", markersize=7, label="걷다가 물에 막힘"),
         Patch(fc=COLS[1], alpha=0.6, label="바탕: 02:07 판정 (초록 탈출 · 주황 피난 · 빨강 대기)")]
    fig.legend(handles=h, loc="lower left", bbox_to_anchor=(0.02, 0.005), ncol=4, frameon=False, fontsize=9.4)
    fig.text(0.98, 0.012, "물 30cm = 통행 불가로 봄. 사람 위치는 오차 없음, 반응 2분, 갯골 칸 0.5m 낮춤 [가정]. "
             "출처: BADA2024, 개방海 갯골·해안선, 영흥도 조위관측소 실측, 우리 조석 예측", ha="right", fontsize=8, color=MUTED)
    fig.savefig(out, dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tide", required=True)
    ap.add_argument("--time")
    ap.add_argument("--route")
    ap.add_argument("--batch", type=int, default=0, help="밤 큰 간조 N번으로 일괄 검증")
    ap.add_argument("--cells", type=int, default=5, help="일괄 검증: 판정마다 출발 칸 수")
    ap.add_argument("--res", type=float, default=50.0)
    ap.add_argument("--walk", type=float, default=0.5)
    ap.add_argument("--depth", type=float, default=0.30)
    ap.add_argument("--drop", type=float, default=0.5)
    a = ap.parse_args()
    G = rm.build_grid(rm.AREA, a.res, a.drop)
    T, O, P = gd.load(a.tide)
    if a.batch:
        run_batch(a, G, T, O, P)
    else:
        run_case(a, G, T, O, P)


def night_lows(G, T, O, P, k):
    """gd.batch 와 같은 방법으로 밤 큰 간조 k번"""
    lat_c, lon_c = float(G["GY"].mean()), float(G["GX"].mean())
    lows = [i for i in range(60, len(P) - 60) if np.isfinite(P[i]) and P[i] < 100 and P[i] == np.nanmin(P[i - 60:i + 61])]
    ded = []
    for i in lows:
        if not ded or (T[i] - T[ded[-1]]) > np.timedelta64(360, "m"):
            ded.append(i)
    lows = [i for i in ded if gd._sun_elev(T[i:i + 1], lat_c, lon_c)[0] < -0.833]
    lows = [i for i in lows if np.isfinite(O[i:i + 420]).mean() > 0.9 and (T[min(i + 420, len(T) - 1)] - T[i]) == np.timedelta64(420, "m")]
    return [lows[int(j)] for j in np.linspace(0, len(lows) - 1, min(k, len(lows)))] if lows else []


def outcome(r):
    if r["arrived"] is not None:
        return "탈출 성공"
    if r["refuge_t"] is not None:
        return "피난처 도착"
    if r["final"] == "대기":
        return "대기"
    return "걷다가 막힘"


def run_batch(a, G, T, O, P):
    """밤 큰 간조 N번 × 간조 뒤 60·90분 판정 × '99% 탈출'(남은 여유 0~30분, 길 300m 넘음) 칸 N_CELL개 × 사람 5가지
    → 재판정 있음/없음을 실제 물때로 비교"""
    N_CELL = a.cells
    pick = night_lows(G, T, O, P, a.batch)
    print(f"밤 큰 간조 {len(pick)}번: " + ", ".join(str(T[i])[:16].replace("T", " ") for i in pick))
    rng = np.random.default_rng(7)
    ny, nx = G["sea"].shape
    bar = G["barrier"].ravel()
    rows = []
    for n_i, i in enumerate(pick):
        for dm in (60, 90):
            t0 = T[i] + np.timedelta64(dm, "m")
            W = Water(G, T, O, P, t0, a.depth)
            tct = W.truth()
            tcf, night, _ = W.at(0.0)
            Lf, parent = W.latest(0.0, a.walk)
            sl = Lf - margin_arr(Lf, night) - gd.REACT
            ok = G["sea"] & (tcf > 0) & np.isfinite(Lf) & (sl >= 0) & (sl <= 30)
            cand = np.argwhere(ok)
            rng.shuffle(cand)
            chosen = []
            for y, x in cand:
                k, n = y * nx + x, 0
                while not bar[k] and parent[k] >= 0 and n < 5000:
                    k, n = int(parent[k]), n + 1
                if n >= 6:
                    chosen.append((y, x))
                if len(chosen) >= N_CELL:
                    break
            for y, x in chosen:
                origin = (float(G["GY"][y, x]), float(G["GX"][y, x]))
                for name, beh in BEHAVIORS.items():
                    r0 = simulate(G, W, tct, origin, (0.0, 0.0), beh, redecide=False, walk=a.walk)
                    r1 = simulate(G, W, tct, origin, (0.0, 0.0), beh, redecide=True, walk=a.walk)
                    if r0 is None or r1 is None:
                        continue
                    rows.append(dict(low=str(T[i])[:16], after_min=dm, lat=round(origin[0], 6), lon=round(origin[1], 6),
                                     slack0=round(float(sl[y, x]), 1), route_m=round(float(r1["plans"][0]["s"][-1])), behavior=name,
                                     out0=outcome(r0), water0=None if r0["caught"] is None else round(r0["caught"], 1),
                                     out1=outcome(r1), water1=None if r1["caught"] is None else round(r1["caught"], 1),
                                     alert1=None if r1["alert"] is None else round(r1["alert"], 1), switches1=r1["n_switch"],
                                     final1=r1["final"], level1=r1["plans"][-1]["level"]))
        print(f"   {n_i + 1}/{len(pick)} {str(T[i])[:16]} 끝, 지금까지 {len(rows)}건", flush=True)
    out = os.path.join(rm.HERE, "tide", f"재판정검증_{rm.AREA['key']}_밤간조{len(pick)}번.csv")
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summ = summarize(rows)
    with open(out.replace(".csv", "_요약.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0]))
        w.writeheader()
        w.writerows(summ)
    print("저장:", out)


def summarize(rows):
    """사람마다: 재판정 없음/있음 결과 수, 구조 요청~물 30cm(중앙값), 바뀐 결과"""
    out = []
    print("\n" + " | ".join(["사람", "건수", "없음: 탈출·걷다가 막힘", "있음: 탈출·피난처·대기·걷다가 막힘", "요청 없이 막힘",
                              "구조 요청~물 30cm 중앙값", "요청했지만 해안 도착", "없음은 탈출, 있음은 못 함", "물이 1분 넘게 더 일찍 닿음"]))
    for name in BEHAVIORS:
        rs = [r for r in rows if r["behavior"] == name]
        if not rs:
            continue
        c0 = lambda k: sum(r["out0"] == k for r in rs)          # noqa: E731
        c1 = lambda k: sum(r["out1"] == k for r in rs)          # noqa: E731
        lead = [r["water1"] - r["alert1"] for r in rs if r["alert1"] is not None and r["water1"] is not None]
        d = dict(behavior=name, n=len(rs), esc0=c0("탈출 성공"), caught0=c0("걷다가 막힘"),
                 esc1=c1("탈출 성공"), refuge1=c1("피난처 도착"), wait1=c1("대기"), caught1=c1("걷다가 막힘"),
                 caught1_noalert=sum(r["out1"] == "걷다가 막힘" and r["alert1"] is None for r in rs),
                 lead_med=float(np.median(lead)) if lead else np.nan, lead_n=len(lead),
                 alert_but_escaped=sum(r["out1"] == "탈출 성공" and r["alert1"] is not None for r in rs),
                 lost=sum(r["out0"] == "탈출 성공" and r["out1"] != "탈출 성공" for r in rs),
                 worse=sum(r["out0"] != "탈출 성공" and r["out1"] != "탈출 성공" and r["water0"] is not None
                           and r["water1"] is not None and r["water1"] < r["water0"] - 1 for r in rs))
        out.append(d)
        print(f"{name} | {d['n']} | {d['esc0']}·{d['caught0']} | {d['esc1']}·{d['refuge1']}·{d['wait1']}·{d['caught1']} | "
              f"{d['caught1_noalert']} | {d['lead_med']:.0f}분({d['lead_n']}건) | {d['alert_but_escaped']} | {d['lost']} | {d['worse']}")
    return out


if __name__ == "__main__":
    main()
