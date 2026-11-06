"""걷는 동안 30초마다 다시 판정 — 정상 / 주의 / 신고 (2026-11-06 팀 결정: 피난 개념을 뺌)

처음 판정(gg_decide.py)은 "0.5m/s로, 반응 2분 뒤 출발"을 가정합니다. 실제 사람은 더 느리거나, 멈추거나,
길을 벗어날 수 있으므로 걷는 동안 30초마다 그때 알 수 있는 정보로 다시 계산합니다.
  물  : 그 시각까지 10분 실측 평균으로 높이 보정을 새로 함 (gg_decide.py 와 같은 방식, 미래 실측은 안 씀)
  사람: 지금 위치 + 걸음 속도 = 최근 "걸으라고 했고 실제로 움직인" 5분의 속도 (0.5m/s 넘게는 안 봄)
        · 멈추라고 한 시간(새 길 알려 준 뒤 반응 2분)은 뺌
        · 서 있던 시간도 속도에서는 빼고, 그 대신 시계가 흘러간 만큼 여유가 줄어 듦
          (펄에 잠깐 빠졌다 나온 사람을 "앞으로도 0m/s"로 보지 않게)
        · 위치는 지금 시뮬레이션 값 그대로 [가정: 실제는 드론 RTK 위치 + 사람까지 약 8m]
  다시 계산 (지금 자리에서, 잰 속도로, "가장 오래 열린 길" = 해안까지의 탈출로):
        99% 여유로 해안까지 → 정상
        95% 여유로 해안까지 → 주의: 관제 알림, 위치 전송 시작
        그것도 안 되면     → 신고: 같은 길로 해안 쪽으로 "갈 수 있는 데까지" 감, 해경 신고
                             다음 칸이 닿기 전에 잠기면 그 자리에서 멈춤 → 구조품 투하 (그 자리가 잠기기 전)
        새 길이 지금 길과 같으면 사람을 멈춰 세우지 않음(반응 2분 없음)
        위험도는 올라가기만 함 (한 번 신고하면 취소는 관제가 함)
        걸으라고 했는데 3분 넘게 안 움직이면(펄에 빠짐 등) 바로 신고 + 그 자리에 구조품 (2026-11-06 팀 결정)
  구조품 (2026-11-06 팀 결정):
        드론 1대: 안내 드론이 싣고 다니다가 3분 안 움직인 자리 또는 갈 수 있는 끝(멈춤)에서 투하
        드론 2대: 신고 때 두 번째 드론이 구조품을 싣고 출동(신고 + N분에 도착 [가정]), 사람 곁에서 기다리다가
                  실제로 멈춘 자리(3분 안 움직임 또는 멈춤)에서 투하 → 물 30cm 전에 전달됐나를 N = 5·10·15·20분으로 셈
  검사(30초마다): 남은 길을 잰 속도로 걸을 때 (마지막 출발 여유) − (그 단계 여유) ≥ 0 이면 그대로
        갈 수 있는 데까지 가는 중이면: 예보로 물에 막히는 때가 정했을 때보다 5분 넘게 당겨지지 않으면 그대로
        안내 중인 길에서 OFF_M 넘게 벗어나면 그 자리에서 다시 계산
  전송: 주의부터 30초마다 위치(시뮬레이션에선 사람 위치) + 그 자리 물 30cm 예상 시각(예보, 99% 기준)
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
V_FLOOR = 0.1       # m/s: 걸음 속도 아래 한계 (0으로 나누지 않게)
STUCK_MIN = 3.0     # 분: 걸으라고 했는데 이만큼 넘게 안 움직이면 신고 + 그 자리에 구조품 (펄에 빠짐 등, 2026-11-06 팀 결정)
DELIVER_MIN = (5, 10, 15, 20)   # 분: 두 번째 드론(구조품)이 신고 뒤 현장에 오기까지 [가정] — 몇 분 안에 와야 하는지 보려고 여러 값
T_END = 330.0       # 분: 판정 뒤 이만큼까지만 따라감 (물 계산 길이 360분 안)
LEVEL_TABLE = {"99%": gd.MARGIN, "95%": gd.MARGIN95}
TIER_NAME = {0: "정상", 1: "주의", 2: "신고"}


def margin_level(level, x, night):
    x = np.asarray(x, float)
    xx = np.where(np.isfinite(x), x, 999)
    return np.where(np.isfinite(x), gd.margin(xx, night, LEVEL_TABLE[level]), 0.0)


def land_dist(G):
    """칸마다 가장 가까운 해안선 칸까지 거리(m). 한 번만 계산해 G에 둠"""
    if "land_d" not in G:
        from scipy import ndimage
        G["land_d"] = ndimage.distance_transform_edt(~G["barrier"]) * G["res"]
    return G["land_d"]


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
        self.tier = 0
        self.caution = None        # 주의 시각
        self.alert = None          # 신고 시각
        self.drop = None           # (시각, 위치) 구조품 투하
        self.tx = []               # 구조대 전송 기록
        self.stuck_on = False
        self.stuck_events = []     # (시각, 위치) 걸으라 했는데 3분 넘게 안 움직임
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
        """mode: 탈출(정상·주의) / 끝까지(신고, 갈 수 있는 데까지) / 멈춤 / 도착"""
        # 지금 칸 가운데로 되돌아가지 않고 바로 다음 칸으로 (칸 목록에는 지금 칸도 남겨 검사)
        nxt = cells[1:] if (cells and cells[0] == self.cell(*pos) and len(cells) > 1) else cells
        pts = np.array([pos] + [self.center(c) for c in nxt]) if nxt else np.array([pos])
        s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))] if len(pts) > 1 else np.array([0.0])
        P = dict(id=len(self.plans), mode=mode, t_set=t, pts=pts, cells=np.array([self.cell(*pos)] + list(nxt), dtype=np.int64),
                 s=s, s_last=0.0, info=info, level=info.get("level", "-"), cont=bool(info.get("cont", False)))
        self.plans.append(P)
        return P

    def walk_dry(self, pos, cells, v, tcf, wait):
        """칸 목록(첫 칸 = 지금 칸)을 따라 걸을 때 예보로 언제 물에 막히나 (지금부터 분, 끝까지 가면 inf) + 막히는 칸 번호
        칸에 닿기 전에 막히면 그 앞 칸이 막힐 때까지, 칸을 건너는 중에 막히면 그때.
        칸 반(대각선)을 건너는 시간만큼 미리 막힌 것으로 봄 (판정은 칸 가운데 도착 시각, 사람은 칸을 다 건널 때까지 그 칸 안)"""
        tf = tcf.ravel()
        B = self.res / 2 * 1.4142 / v / 60.0
        pts = [np.asarray(pos, float)] + [self.center(k) for k in cells[1:]]
        arr = wait + np.r_[0.0, np.cumsum([np.hypot(*(b - a)) for a, b in zip(pts[:-1], pts[1:])])] / v / 60.0
        for i, k in enumerate(cells):
            if arr[i] + B >= tf[k]:
                if arr[i] < tf[k] or i == 0:
                    return float(tf[k]), i
                return float(tf[cells[i - 1]]), i
        return np.inf, None

    @staticmethod
    def same_way(a, b, k=4):
        """두 칸 목록(첫 칸 = 지금 칸)의 다음 k칸이 같으면 같은 길 → 사람에게 새로 알릴 것이 없으므로 멈추지 않음"""
        return a is not None and b is not None and len(a) > 1 and list(a[1:1 + k]) == list(b[1:1 + k])

    # ---- 다시 계산
    def replan(self, t, pos, v, cur=None):
        """cur: 지금 걷는 길의 남은 칸 (첫 칸 = 지금 칸)"""
        tcf, night, off = self.W.at(t)
        c = self.cell(*pos)
        if self.bar[c] or not self.sea[c]:
            return self.make_plan("도착", t, pos, [], dict(why="해안"))
        cy, cx = divmod(c, self.nx)
        lead = float(np.hypot(*(np.asarray(pos) - self.center(c)))) / v / 60.0
        Lf, parent = self.W.latest(t, v)
        Lp = float(Lf[cy, cx]) - lead
        ch = self.chain(parent, c)
        same = self.same_way(ch, cur)
        react = 0.0 if same else gd.REACT
        if np.isfinite(Lp):
            for lvl in ("99%", "95%"):
                sl = Lp - float(margin_level(lvl, Lp, night)) - react
                if sl >= 0:
                    return self.make_plan("탈출", t, pos, ch, dict(L=Lp, slack=sl, v=v, off=off, level=lvl, cont=same))
        # 신고: 해안 쪽으로 갈 수 있는 데까지 (지금 길 그대로 / 새로 계산한 길 중 예보로 더 오래 물 밖에 있는 쪽, 비슷하면 지금 길)
        d_here = float(tcf[cy, cx])
        cand = []
        if cur is not None and len(cur) > 1:
            cand.append((self.walk_dry(pos, cur, v, tcf, 0.0)[0] + 1.0, "cur", list(cur), True))    # 1분 안 차이면 바꾸지 않음
        if len(ch) > 1:
            cand.append((self.walk_dry(pos, ch, v, tcf, react)[0], "new", ch, same))
        if cand:
            d, which, cells, cont = max(cand, key=lambda c_: c_[0])
            d = d - 1.0 if which == "cur" else d
            if d > d_here:
                return self.make_plan("끝까지", t, pos, cells, dict(L=Lp, slack=np.nan, v=v, off=off, dry=d, cont=cont))
        return self.make_plan("멈춤", t, pos, [], dict(here_tcf=d_here, v=v, off=off, why="더 갈 수 없음"))

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

    def still_minutes(self):
        """걸으라고 한 시간 중, 마지막으로 움직인 뒤 계속 서 있던 시간(분). 멈추라고 한 시간은 세지 않음"""
        still = 0.0
        for (t1, p1), (t2, p2) in zip(self.hist[-2::-1], self.hist[:0:-1]):
            dur = t2 - t1
            inst = dur - sum(max(0.0, min(t2, b) - max(t1, a)) for a, b in self.waits)
            if inst <= 1e-9:
                continue
            if float(np.hypot(*(np.asarray(p2) - np.asarray(p1)))) >= STILL_M * dur / TICK:
                break
            still += inst
        return still

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
        c_now = self.cell(*pos)
        rest = [c_now] + [int(k) for k in cells if k != c_now]
        tf = tcf.ravel()
        if P["mode"] == "탈출":
            arr = wait_left + (sa - s_p) / v / 60.0
            L = min(float(tf[c_now]), float(np.min(tf[cells] - arr)) if len(cells) else np.inf)
            sl = L - float(margin_level(P["level"], L, night)) if np.isfinite(L) else np.inf
            return sl >= 0 and d <= OFF_M, dict(s=s_p, d=d, L=L, slack=sl, rest=rest)
        # 끝까지: 다음 칸이 닿기 전에 잠기면 여기가 끝
        dry, i_fail = self.walk_dry(pos, rest, v, tcf, wait_left)
        if i_fail is not None and i_fail <= 1 and d <= OFF_M:
            return False, dict(s=s_p, d=d, L=np.nan, slack=np.nan, rest=rest, stop_here=True)
        due = P["info"]["dry"] - (t - P["t_set"])
        return dry >= due - 5.0 and d <= OFF_M, dict(s=s_p, d=d, L=np.nan, slack=dry - due, rest=rest)

    # ---- 위험도·투하·전송
    def tier_of(self, P):
        if P["mode"] in ("탈출", "도착"):
            return {"99%": 0, "95%": 1}.get(P["level"], 0)
        return 2

    def after_plan(self, t, pos):
        """새 계획마다: 위험도(올라가기만 함), 주의·신고 시각, 멈춤이면 구조품 투하"""
        self.tier = max(self.tier, self.tier_of(self.plan))
        if self.tier >= 1 and self.caution is None:
            self.caution = t
        if self.tier >= 2 and self.alert is None:
            self.alert = t
        if self.plan["mode"] == "멈춤" and self.drop is None:
            self.drop = (t, np.array(pos, float))

    def send(self, t, pos):
        """주의부터 30초마다 구조대 전송: 드론 RTK 위치(시뮬레이션에선 사람 위치로 대신) + 그 자리 물 30cm 예상 시각"""
        if self.tier < 1:
            return
        tcf, night, _ = self.W.at(t)
        f = float(tcf.ravel()[self.cell(*pos)])
        ok = np.isfinite(f)
        self.tx.append(dict(t_min=round(t, 2), tier=self.tier, n=round(float(pos[0]), 1), e=round(float(pos[1]), 1),
                            mode=self.plan["mode"], water_fc_min=round(t + f, 1) if ok else None,
                            water_99_min=round(t + f - float(gd.margin(f, night)), 1) if ok else None))

    # ---- 30초마다
    def start(self, pos):
        self.hist.append((0.0, tuple(pos)))
        self.plan = self.replan(0.0, pos, self.walk)
        self.waits.append((0.0, gd.REACT))
        self.after_plan(0.0, pos)
        self._log(0.0, pos, self.walk, dict(s=0, d=0, slack=self.plan["info"].get("slack", np.nan)), "처음 판정")
        self.send(0.0, pos)
        return self.plan

    def tick(self, t, pos):
        self.hist.append((t, tuple(pos)))
        P = self.plan
        if P["mode"] in ("도착", "멈춤"):
            self.send(t, pos)
            return P
        v = self.speed()
        still = self.still_minutes()
        if still >= STUCK_MIN and not self.stuck_on:                     # 걸으라 했는데 3분 넘게 안 움직임 → 신고 + 구조품
            self.stuck_on = True
            self.stuck_events.append((t, np.array(pos, float)))
            ev = f"걸으라고 했는데 {STUCK_MIN:.0f}분 넘게 안 움직임 → 그 자리에 구조품"
            if self.tier < 2:
                self.tier = 2
                self.caution = t if self.caution is None else self.caution
                self.alert = t
                ev += " [신고: 해경·위치 전송]"
            if self.drop is None:
                self.drop = (t, np.array(pos, float))
            self._log(t, pos, v, dict(s=P["s_last"], d=0), ev)
        elif still < 1e-9:
            self.stuck_on = False
        if self.stuck_on:                       # 빠져 있는 동안은 길을 다시 짜지 않고 위치·물 예상 시각만 보냄
            self.send(t, pos)
            return P
        ok, info = self.check(t, pos, v)
        if info["s"] >= P["s"][-1] - 5.0 and info["d"] <= OFF_M and self.bar[int(P["cells"][-1])]:
            P["mode"] = "도착"
            self._log(t, pos, v, info, "해안 도착")
            return P
        if ok:
            self._log(t, pos, v, info, "")
            self.send(t, pos)
            return P
        old_tier = self.tier
        if info.get("stop_here"):
            self.plan = self.make_plan("멈춤", t, pos, [], dict(v=v, why="다음 칸이 먼저 잠김"))
            ev = "다음 칸이 닿기 전에 잠김 → 여기서 멈춤 (갈 수 있는 끝)"
        else:
            why = f"길에서 {info['d']:.0f}m 벗어남" if info["d"] > OFF_M else f"여유 모자람 ({info['slack']:+.1f}분, {v:.2f}m/s)"
            cur = info["rest"] if info["d"] <= OFF_M else None
            self.plan = self.replan(t, pos, v, cur)
            lv = {"탈출": f" ({self.plan['level']} 여유로 해안까지)", "끝까지": f" (해안 쪽으로 갈 수 있는 데까지, 예보로 "
                  f"{self.plan['info'].get('dry', np.nan):.0f}분 뒤 물에 막힘)", "멈춤": " (더 갈 수 없음)"}.get(self.plan["mode"], "")
            ev = f"{why} → {self.plan['mode']}{lv}" + (" · 같은 길 계속" if self.plan["cont"] else "")
        if not self.plan["cont"]:
            self.waits.append((t, t + gd.REACT))
        self.after_plan(t, pos)
        if self.tier > old_tier:
            ev += {1: " [주의: 관제 알림·위치 전송]", 2: " [신고: 해경·위치 전송]"}[self.tier]
        if self.plan["mode"] == "멈춤":
            ev += " → 구조품 투하"
        self._log(t, pos, v, info, ev)
        self.send(t, pos)
        return self.plan

    def _log(self, t, pos, v, info, event):
        self.log.append(dict(t_min=round(t, 2), mode=self.plan["mode"], level=self.plan["level"], tier=self.tier,
                             plan=self.plan["id"], v_est=round(v, 3), n=round(float(pos[0]), 1), e=round(float(pos[1]), 1),
                             along_m=round(float(info.get("s", 0)), 1), off_m=round(float(info.get("d", 0)), 1),
                             slack_min=round(float(info.get("slack", np.nan)), 2), event=event))


# ---------------------------------------------------------------- 가상의 사람
BEHAVIORS = {
    "정상 0.5m/s": dict(v=0.5),
    "느림 0.4m/s": dict(v=0.4),
    "아주 느림 0.3m/s": dict(v=0.3),
    "15분 걷고 펄에 빠짐(못 나옴)": dict(v=0.5, stop=(17.0, np.inf)),     # 실제 사례: 빠지면 구조대가 끌어냄 (조위데이터 9장)
    "15분 걷고 5분 길 벗어남": dict(v=0.5, off=(17.0, 5.0)),
}
DT = 5 / 60.0       # 분: 사람 움직임 계산 간격


def simulate(G, water, tct, origin, start_ne, beh, redecide=True, walk=0.5):
    """가상의 사람 한 명. redecide=False면 처음 탈출로를 끝까지 (길 벗어나면 드론이 원래 길로 데려옴)
    stop: (시작 분, 길이) 펄에 빠져 못 움직임 / off: (시작 분, 길이) 안내를 놓치고 진행 방향 오른쪽으로 걸어감"""
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
    track = [(0.0, pos[0], pos[1])]
    t, caught, arrived = 0.0, None, None
    tf = tct.ravel()
    heading = None
    cumlen = lambda pth: np.r_[0, np.cumsum(np.hypot(*np.diff(pth, axis=0).T))]     # noqa: E731
    while t < T_END:
        if redecide and t >= next_tick - 1e-9:
            eng.tick(t, pos.copy())
            next_tick += TICK
        plan = eng.plan if redecide else P0
        in_stop = stop and stop[0] <= t < stop[0] + stop[1]
        in_off = off and off[0] <= t < off[0] + off[1]
        if in_off:
            if leave is None:
                leave = (pos.copy(), ps)
                seg = np.diff(path, axis=0)
                i = int(np.searchsorted(cumlen(path), ps, side="right") - 1)
                d = seg[min(max(i, 0), len(seg) - 1)]
                d = d / max(np.hypot(*d), 1e-9)
                heading = np.array([-d[1], d[0]])          # 진행 방향 오른쪽 직각 (북·동 좌표)
            pos = pos + heading * v_p * DT * 60
        elif not in_stop:
            if off and leave is not None and t >= off[0] + off[1] and free_at < off[0] + off[1] + gd.REACT:
                free_at = off[0] + off[1] + gd.REACT          # 길 벗어났다가 다시 안내 받음 → 반응 2분
                if not redecide:                               # 드론이 떠난 곳으로 데려온 뒤 원래 길 계속
                    rest = path[np.searchsorted(cumlen(path), leave[1], side="right"):]
                    path = np.vstack([pos, leave[0], rest])
                    ps = 0.0
            if redecide and plan["id"] != adopted:
                if not plan["cont"]:
                    free_at = max(free_at, plan["t_set"] + gd.REACT)
                if t >= free_at - 1e-9:
                    path = np.vstack([pos, plan["pts"]]) if np.hypot(*(plan["pts"][0] - pos)) > 0.5 else plan["pts"].copy()
                    ps, adopted = 0.0, plan["id"]
            moving = plan["mode"] in ("탈출", "끝까지", "도착")
            if moving and t >= free_at - 1e-9 and (not redecide or adopted == plan["id"]) and len(path) > 1:
                cum = cumlen(path)
                ps = min(ps + v_p * DT * 60, cum[-1])
                i = min(int(np.searchsorted(cum, ps, side="right") - 1), len(path) - 2)
                u = 0 if cum[i + 1] == cum[i] else (ps - cum[i]) / (cum[i + 1] - cum[i])
                pos = path[i] + u * (path[i + 1] - path[i])
                k_ = eng.cell(*pos)
                if ps >= cum[-1] - 1e-6 and (eng.bar[k_] or not eng.sea[k_]):
                    arrived = t + DT
        t += DT
        track.append((t, pos[0], pos[1]))
        k = eng.cell(*pos)
        if arrived is not None:
            break
        if eng.sea[k] and t >= tf[k]:
            caught = t
            break
        if redecide and plan["mode"] == "멈춤" and t >= free_at and not in_off:
            caught = max(t, float(tf[k])) if eng.sea[k] else None    # 더 움직이지 않음 → 그 칸이 막히는 때까지 건너뜀
            if caught is not None and caught > T_END:
                caught = None
            break
    tr = np.array(track)
    stuck_end = bool(stop and t >= stop[0] and arrived is None)
    minmarg = np.nan
    if arrived is not None:
        ks = [eng.cell(n, e) for _, n, e in tr]
        mm = [tf[k] - tt for (tt, _, _), k in zip(tr, ks) if eng.sea[k] and np.isfinite(tf[k])]
        minmarg = float(min(mm)) if mm else np.inf
    drop = eng.drop if redecide else None
    deliver = deliveries(eng, tr, tf, arrived, caught) if redecide else {}
    drop_water = drop_land = None
    if drop is not None:
        kd = eng.cell(*drop[1])
        drop_water = float(tf[kd]) - drop[0] if eng.sea[kd] and np.isfinite(tf[kd]) else np.inf   # 투하 뒤 그 자리 물 30cm까지 (실제)
        drop_land = float(land_dist(G).ravel()[kd])
    return dict(arrived=arrived, caught=caught, final=(eng.plan if redecide else P0)["mode"], stuck_end=stuck_end,
                stuck_t=stop[0] if stuck_end else None,
                alert=eng.alert if redecide else None, caution=eng.caution if redecide else None,
                drop=drop, drop_water=drop_water, drop_land=drop_land, tx=eng.tx if redecide else [],
                walked=float(np.sum(np.hypot(*np.diff(tr[:, 1:], axis=0).T))), track=tr, minmarg=minmarg,
                plans=eng.plans if redecide else [P0], log=eng.log, n_switch=len(eng.plans) - 1 if redecide else 0,
                deliver=deliver)


def deliveries(eng, tr, tf, arrived, caught):
    """드론 2대: 신고 + N분에 두 번째 드론 도착 → 사람이 멈춰 있는 자리(3분 안 움직임 / 갈 수 있는 끝)에서 투하.
    결과 {N: ("전달", 물까지 분) / ("물 먼저", 늦은 분) / ("필요 없음", None)}. 신고가 없으면 {}"""
    if eng.alert is None:
        return {}
    needs = []                                              # (시작, 끝, 위치, 그 칸 물 30cm 시각)
    for t0, p0 in eng.stuck_events:                        # 다시 걸어 3m 넘게 벗어나면 그 자리는 끝
        later = tr[tr[:, 0] > t0]
        moved = later[np.hypot(later[:, 1] - p0[0], later[:, 2] - p0[1]) >= STILL_M]
        end = float(moved[0, 0]) if len(moved) else np.inf
        k = eng.cell(*p0)
        needs.append((t0, end, p0, float(tf[k]) if eng.sea[k] else np.inf))
    if arrived is None and eng.plan["mode"] == "멈춤" and eng.plans:
        P = eng.plan
        k = eng.cell(*P["pts"][0])
        needs.append((P["t_set"], np.inf, P["pts"][0], float(tf[k]) if eng.sea[k] else np.inf))
    needs.sort(key=lambda x: x[0])
    if not needs and arrived is None and caught is not None:
        return {N: ("걷다가 막힘", None) for N in DELIVER_MIN}
    out = {}
    for N in DELIVER_MIN:
        come = eng.alert + N
        res = ("필요 없음", None)
        for t0, end, p0, water in needs:
            if end <= come:                                 # 드론이 오기 전에 다시 걸어 나감 → 다음 멈춤으로
                continue
            drop_t = max(t0, come)
            res = ("전달", round(water - drop_t, 1)) if drop_t < water else ("물 먼저", round(drop_t - water, 1))
            break
        out[N] = res
    return out


def outcome(r):
    if r["arrived"] is not None:
        return "해안 도착"
    if r.get("stuck_end"):
        return "빠진 채 구조 대기"
    if r["final"] == "멈춤" and r["drop"] is not None:
        return "멈춤·구조 대기"
    return "걷다가 막힘"


def _hm(t0, x, sec=False):
    return str(t0 + np.timedelta64(int(round(x * 60)), "s"))[11:19 if sec else 16]


def describe(r, t0):
    o = outcome(r)
    if o == "해안 도착":
        s = f"{_hm(t0, r['arrived'])} 해안 도착 (지나간 칸 중 가장 아슬아슬한 곳도 {r['minmarg']:.0f}분 뒤 잠김)"
    elif o == "멈춤·구조 대기":
        s = f"{_hm(t0, r['drop'][0])} 갈 수 있는 끝에서 멈춤 (육지까지 {r['drop_land']:.0f}m)" + \
            (f", 그 자리 물 30cm {_hm(t0, r['caught'])}" if r["caught"] else ", 계산 끝까지 안 잠김")
    elif o == "빠진 채 구조 대기":
        s = f"{_hm(t0, r['stuck_t'])} 펄에 빠짐" + (f", 그 자리 물 30cm {_hm(t0, r['caught'])}" if r["caught"] else ", 계산 끝까지 안 잠김")
    else:
        s = f"{_hm(t0, r['caught'])} 걷던 길에서 막힘 (물 30cm)" if r["caught"] else "끝까지 못 나옴"
    if r["caution"] is not None and r["caution"] != r["alert"]:
        s += f" · 주의 {_hm(t0, r['caution'])}"
    if r["alert"] is not None:
        s += f" · 신고 {_hm(t0, r['alert'])}" + (f" (물 차기 {r['caught'] - r['alert']:.0f}분 전)" if r["caught"] else "")
    if r["drop"] is not None:
        s += f" · 구조품(1대) {_hm(t0, r['drop'][0])} (그 자리 물 30cm까지 {r['drop_water']:.0f}분)"
    if r.get("deliver"):
        s += " · 두 번째 드론 " + ", ".join(f"{N}분: {v[0]}" + ("" if v[1] is None else f" {v[1]:.0f}분") for N, v in r["deliver"].items())
    return s


def short(r, t0):
    o = outcome(r)
    if o == "해안 도착":
        s = f"{_hm(t0, r['arrived'])} 해안 도착\n(가장 아슬아슬한 칸도 {r['minmarg']:.0f}분 뒤 잠김)"
    elif o == "멈춤·구조 대기":
        s = f"{_hm(t0, r['drop'][0])} 갈 수 있는 끝에서 멈춤 (육지까지 {r['drop_land']:.0f}m)\n→ 그 자리 물 30cm " + \
            (_hm(t0, r["caught"]) if r["caught"] else "계산 끝까지 없음")
    elif o == "빠진 채 구조 대기":
        s = f"{_hm(t0, r['stuck_t'])} 펄에 빠짐\n→ 그 자리 물 30cm " + (_hm(t0, r["caught"]) if r["caught"] else "계산 끝까지 없음") + \
            ("\n(미리 알린 것 없음)" if r["alert"] is None else "")
    else:
        s = f"{_hm(t0, r['caught'])} 걷던 길에서 물에 막힘" + ("\n(미리 알린 것 없음)" if r["alert"] is None else "")
    if r["caution"] is not None and r["caution"] != r["alert"]:
        s += f"\n주의(관제) {_hm(t0, r['caution'])}"
    if r["alert"] is not None:
        s += f"\n신고(해경) {_hm(t0, r['alert'])}" + (f" · 물 차기 {r['caught'] - r['alert']:.0f}분 전" if r["caught"] else "")
    if r["drop"] is not None:
        s += f"\n구조품 {_hm(t0, r['drop'][0])} · 물까지 {r['drop_water']:.0f}분"
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
            if row["event"] and row["event"] != "처음 판정":
                print(f"      {_hm(t0, row['t_min'], True)}  {row['event']}  (길 따라 {row['along_m']:.0f}m, 속도 {row['v_est']:.2f}m/s)")
    first = res[next(iter(res))][1]
    print(f"\n처음 판정 확인: {first['log'][0]['mode']} {first['log'][0]['level']}, 남은 여유 {first['log'][0]['slack_min']:.1f}분 "
          f"(판정 파일 {route['slack_min']}분), 길 {first['plans'][0]['s'][-1]:.0f}m (판정 파일 {route['length_m']}m)")
    base = os.path.join(rm.HERE, "tide", f"재판정_{rm.AREA['key']}_{a.time[:10]}_{a.time[11:].replace(':', '')}")
    with open(base + ".csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["behavior", "t_min", "clock", "mode", "level", "tier", "plan", "v_est", "n", "e", "along_m", "off_m", "slack_min", "event"])
        for name, (r0, r1) in res.items():
            for row in r1["log"]:
                w.writerow([name, row["t_min"], _hm(t0, row["t_min"], True), row["mode"], row["level"], TIER_NAME[row["tier"]],
                            row["plan"], row["v_est"], row["n"], row["e"], row["along_m"], row["off_m"], row["slack_min"], row["event"]])
    lat0, lon0 = origin
    with open(base + "_구조대전송.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["behavior", "clock", "tier", "mode", "lat", "lon", "water30cm_forecast", "water30cm_99pct"])
        for name, (r0, r1) in res.items():
            for row in r1["tx"]:
                w.writerow([name, _hm(t0, row["t_min"], True), TIER_NAME[row["tier"]], row["mode"],
                            round(lat0 + row["n"] / G["ky"], 6), round(lon0 + row["e"] / G["kx"], 6),
                            "" if row["water_fc_min"] is None else _hm(t0, row["water_fc_min"]),
                            "" if row["water_99_min"] is None else _hm(t0, row["water_99_min"])])
    with open(base + "_요약.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["behavior", "redecide", "result"])
        for name, (r0, r1) in res.items():
            w.writerow([name, 0, describe(r0, t0)])
            w.writerow([name, 1, describe(r1, t0)])
    draw_case(base + ".png", G, route, t0, res, a)
    print("저장:", base + ".csv", base + "_요약.csv", base + "_구조대전송.csv", base + ".png")


def draw_case(out, G, route, t0, res, a):
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
    C0, C1, SW = "#8a8984", "#2a78d6", "#eb6834"
    lat0, lon0 = route["start_lat"], route["start_lon"]
    kx, ky = G["kx"], G["ky"]
    to_ll = lambda n, e: (lat0 + np.asarray(n) / ky, lon0 + np.asarray(e) / kx)      # noqa: E731
    _, D = gd.decide_at(G, *a._TOP, t0, a)
    allp = [np.array(route["path_latlon"])]
    for r0, r1 in res.values():
        for r in (r0, r1):
            la, lo = to_ll(r["track"][:, 1], r["track"][:, 2])
            allp.append(np.c_[la, lo])
    allp = np.vstack(allp)
    pad = 0.004
    ylim = (allp[:, 0].min() - pad, allp[:, 0].max() + pad)
    xlim = (allp[:, 1].min() - pad * 1.3, allp[:, 1].max() + pad * 1.3)
    k = len(res)
    fig = plt.figure(figsize=(3.5 * k + 0.6, 10.4), facecolor="white")
    fig.text(0.02, 0.955, "걷는 동안 30초마다 다시 판정 — 같은 사람을 '처음 길 그대로'와 '재판정'으로 걸려 실제 물때로 비교",
             fontsize=15.5, weight="bold", color=INK)
    fig.text(0.02, 0.922, f"영흥도 {str(t0)[:16].replace('T', ' ')} 처음 판정: 정상 (남은 여유 {route['slack_min']}분, 탈출로 {route['length_m']}m). "
             "재판정: 그때까지의 실측 보정 + 실제로 걸은 속도 + 지금 위치로 해안까지의 길을 다시 계산", fontsize=9.6, color=INK2)
    fig.text(0.02, 0.9, "정상 = 99% 여유로 해안까지 · 주의 = 95% 여유로 해안까지(관제 알림, 위치 전송) · "
             "신고 = 해안까지 못 감 → 같은 길로 갈 수 있는 데까지, 해경 신고, 멈춘 자리에서 잠기기 전에 구조품", fontsize=9.6, color=INK2)
    w = 0.96 / k
    base = np.where(G["land"] | G["barrier"], 0, 1).astype(float)
    cls = gd.dec_classes(D["dec"])
    rl = np.array(route["path_latlon"])
    for i, (name, (r0, r1)) in enumerate(res.items()):
        ax = fig.add_axes([0.02 + i * w, 0.37, w - 0.012, 0.5])
        ax.pcolormesh(G["GX"], G["GY"], base, cmap=ListedColormap([LAND, SEA]), shading="auto", zorder=0)
        ax.pcolormesh(G["GX"], G["GY"], cls, cmap=ListedColormap(gd.DEC_COLS), vmin=-0.5, vmax=2.5, shading="auto", alpha=0.4, zorder=1)
        ax.plot(rl[:, 1], rl[:, 0], color=INK, lw=0.9, ls=(0, (4, 3)), zorder=2)
        for r, col, lw, z in ((r0, C0, 2.6, 3), (r1, C1, 1.8, 4)):
            la, lo = to_ll(r["track"][:, 1], r["track"][:, 2])
            ax.plot(lo, la, color=col, lw=lw, zorder=z, solid_capstyle="round")
            o = outcome(r)
            mk = {"해안 도착": ("*", 90), "멈춤·구조 대기": ("P", 120), "걷다가 막힘": ("X", 70), "빠진 채 구조 대기": ("s", 70)}[o]
            ax.scatter([lo[-1]], [la[-1]], s=mk[1], marker=mk[0], color=col, edgecolor="white", linewidth=0.8, zorder=z + 5)
        sw = [row for row in r1["log"] if row["event"].startswith(("여유 모자람", "길에서", "다음 칸"))]
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
        fig.text(x0, 0.345, "재판정 없음 (처음 길 그대로)", fontsize=9.6, color=C0, weight="bold", va="top")
        fig.text(x0, 0.325, short(r0, t0), fontsize=9.4, color=INK, va="top", linespacing=1.4)
        fig.text(x0, 0.262, "재판정 있음", fontsize=9.6, color=C1, weight="bold", va="top")
        txt = short(r1, t0)
        if sw:
            e0 = sw[0]
            why, what = e0["event"].split(" → ")[0], e0["event"].split(" → ")[1].split(" (")[0].split(" [")[0]
            what = {"탈출": f"탈출 ({e0['level']})", "끝까지": "갈 수 있는 데까지", "멈춤": "멈춤"}.get(what, what)
            txt = f"{_hm(t0, e0['t_min'])} {why}\n→ {what}" + (f" 외 {len(sw) - 1}번 다시 계산" if len(sw) > 1 else "") + "\n" + txt
        else:
            txt = "끝까지 처음 길 유지\n" + txt
        fig.text(x0, 0.242, txt, fontsize=9.4, color=INK, va="top", linespacing=1.4)
    h = [Line2D([], [], color=INK, lw=0.9, ls=(0, (4, 3)), label="처음 판정한 탈출로"),
         Line2D([], [], color=C0, lw=2.6, label="재판정 없음: 사람이 걸은 길"),
         Line2D([], [], color=C1, lw=1.8, label="재판정 있음: 사람이 걸은 길"),
         Line2D([], [], color=SW, marker="D", ls="none", markersize=6, label="처음으로 판정을 바꾼 곳"),
         Line2D([], [], color=INK2, marker="*", ls="none", markersize=9, label="해안 도착"),
         Line2D([], [], color=INK2, marker="P", ls="none", markersize=9, label="멈춤 + 구조품 투하"),
         Line2D([], [], color=INK2, marker="X", ls="none", markersize=7, label="걷다가 물에 막힘"),
         Line2D([], [], color=INK2, marker="s", ls="none", markersize=7, label="펄에 빠진 채 구조 대기")]
    h += [Patch(fc=c, alpha=0.5, label="바탕(처음 판정 시각): " + l.split(":")[0]) for c, l in zip(gd.DEC_COLS, gd.DEC_LABELS)]
    fig.legend(handles=h, loc="lower left", bbox_to_anchor=(0.02, 0.005), ncol=5, frameon=False, fontsize=9.4)
    fig.text(0.98, 0.012, "물 30cm = 통행 불가로 봄. 사람 위치는 오차 없음, 반응 2분, 갯골 칸 0.5m 낮춤 [가정]. "
             "출처: BADA2024, 개방海 갯골·해안선, 영흥도 조위관측소 실측, 우리 조석 예측", ha="right", fontsize=8, color=MUTED)
    fig.savefig(out, dpi=150)


# ---------------------------------------------------------------- 일괄 검증
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


def run_batch(a, G, T, O, P):
    """밤 큰 간조 N번 × 간조 뒤 60·90분 판정 × '정상'(99% 남은 여유 0~30분, 길 300m 넘음) 칸 N_CELL개 × 사람 5가지
    → 재판정 있음/없음을 실제 물때로 비교"""
    pick = night_lows(G, T, O, P, a.batch)
    print(f"밤 큰 간조 {len(pick)}번: " + ", ".join(str(T[i])[:16].replace("T", " ") for i in pick))
    rng = np.random.default_rng(7)
    nx = G["sea"].shape[1]
    bar = G["barrier"].ravel()
    rows = []
    for n_i, i in enumerate(pick):
        for dm in (60, 90):
            t0 = T[i] + np.timedelta64(dm, "m")
            W = Water(G, T, O, P, t0, a.depth)
            tct = W.truth()
            tcf, night, _ = W.at(0.0)
            Lf, parent = W.latest(0.0, a.walk)
            sl = Lf - margin_level("99%", Lf, night) - gd.REACT
            cand = np.argwhere(G["sea"] & (tcf > 0) & np.isfinite(Lf) & (sl >= 0) & (sl <= 30))
            rng.shuffle(cand)
            chosen = []
            for y, x in cand:
                k, n = y * nx + x, 0
                while not bar[k] and parent[k] >= 0 and n < 5000:
                    k, n = int(parent[k]), n + 1
                if n >= 6:
                    chosen.append((y, x))
                if len(chosen) >= a.cells:
                    break
            for y, x in chosen:
                origin = (float(G["GY"][y, x]), float(G["GX"][y, x]))
                for name, beh in BEHAVIORS.items():
                    r0 = simulate(G, W, tct, origin, (0.0, 0.0), beh, redecide=False, walk=a.walk)
                    r1 = simulate(G, W, tct, origin, (0.0, 0.0), beh, redecide=True, walk=a.walk)
                    if r0 is None or r1 is None:
                        continue
                    rnd = lambda x, n=1: None if x is None else round(float(x), n)     # noqa: E731
                    rows.append(dict(low=str(T[i])[:16], after_min=dm, lat=round(origin[0], 6), lon=round(origin[1], 6),
                                     slack0=round(float(sl[y, x]), 1), route_m=round(float(r1["plans"][0]["s"][-1])), behavior=name,
                                     out0=outcome(r0), water0=rnd(r0["caught"]), out1=outcome(r1), water1=rnd(r1["caught"]),
                                     caution1=rnd(r1["caution"]), alert1=rnd(r1["alert"]), switches1=r1["n_switch"],
                                     drop1=None if r1["drop"] is None else rnd(r1["drop"][0]), dropwater1=rnd(r1["drop_water"]),
                                     dropland1=rnd(r1["drop_land"], 0), final1=r1["final"],
                                     **{f"d{N}": "" if N not in r1["deliver"] else
                                        (r1["deliver"][N][0] + ("" if r1["deliver"][N][1] is None else f" {r1['deliver'][N][1]}"))
                                        for N in DELIVER_MIN}))
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
    print("저장:", out, out.replace(".csv", "_요약.csv"))


def summarize(rows):
    """사람마다: 재판정 없음/있음 결과, 위험도별 대응, 구조품 투하"""
    out = []
    print("\n" + " | ".join(["사람", "건수", "없음: 해안·막힘·빠짐", "있음: 해안·멈춤·막힘·빠짐", "신고 없이 막힘", "주의·신고",
                              "신고했지만 해안", "신고~물 30cm 중앙값", "투하·물보다 늦은 투하", "투하 뒤 물까지 중앙값(최소)",
                              "멈춘 곳~육지 중앙값", "없음은 해안, 있음은 못 감", "물이 1분 넘게 더 일찍·늦게"]))
    fin = lambda x: x is not None and np.isfinite(x)        # noqa: E731
    for name in BEHAVIORS:
        rs = [r for r in rows if r["behavior"] == name]
        if not rs:
            continue
        c0 = lambda k: sum(r["out0"] == k for r in rs)          # noqa: E731
        c1 = lambda k: sum(r["out1"] == k for r in rs)          # noqa: E731
        lead = [r["water1"] - r["alert1"] for r in rs if r["alert1"] is not None and r["water1"] is not None]
        dw = [r["dropwater1"] for r in rs if r["drop1"] is not None and fin(r["dropwater1"])]
        dl = [r["dropland1"] for r in rs if r["drop1"] is not None and r["dropland1"] is not None]
        both = [r for r in rs if r["out0"] != "해안 도착" and r["out1"] != "해안 도착" and r["water0"] is not None and r["water1"] is not None]
        d = dict(behavior=name, n=len(rs), shore0=c0("해안 도착"), caught0=c0("걷다가 막힘"), stuck0=c0("빠진 채 구조 대기"),
                 stuck1=c1("빠진 채 구조 대기"),
                 shore1=c1("해안 도착"), stop1=c1("멈춤·구조 대기"), caught1=c1("걷다가 막힘"),
                 caught1_noreport=sum(r["out1"] == "걷다가 막힘" and r["alert1"] is None for r in rs),
                 caution=sum(r["caution1"] is not None for r in rs), report=sum(r["alert1"] is not None for r in rs),
                 report_but_shore=sum(r["out1"] == "해안 도착" and r["alert1"] is not None for r in rs),
                 lead_med=float(np.median(lead)) if lead else np.nan,
                 drops=sum(r["drop1"] is not None for r in rs),
                 drop_late=sum(r["drop1"] is not None and r["dropwater1"] is not None and r["dropwater1"] <= 0 for r in rs),
                 drop_water_med=float(np.median(dw)) if dw else np.nan, drop_water_min=float(np.min(dw)) if dw else np.nan,
                 drop_land_med=float(np.median(dl)) if dl else np.nan,
                 lost=sum(r["out0"] == "해안 도착" and r["out1"] != "해안 도착" for r in rs),
                 earlier=sum(r["water1"] < r["water0"] - 1 for r in both), later=sum(r["water1"] > r["water0"] + 1 for r in both))
        for N in DELIVER_MIN:
            vals = [r[f"d{N}"] for r in rs if r[f"d{N}"]]
            ok = [float(v.split()[1]) for v in vals if v.startswith("전달")]
            d[f"need{N}"] = sum(not v.startswith("필요 없음") for v in vals)
            d[f"ok{N}"] = len(ok)
            d[f"late{N}"] = sum(v.startswith("물 먼저") for v in vals)
            d[f"walk{N}"] = sum(v.startswith("걷다가") for v in vals)
            d[f"okmed{N}"] = float(np.median(ok)) if ok else np.nan
        out.append(d)
        print(f"{name} | {d['n']} | {d['shore0']}·{d['caught0']}·{d['stuck0']} | {d['shore1']}·{d['stop1']}·{d['caught1']}·{d['stuck1']} | "
              f"{d['caught1_noreport']} | "
              f"{d['caution']}·{d['report']} | {d['report_but_shore']} | {d['lead_med']:.0f}분 | {d['drops']}·{d['drop_late']} | "
              f"{d['drop_water_med']:.0f}분({d['drop_water_min']:.0f}) | {d['drop_land_med']:.0f}m | {d['lost']} | {d['earlier']}·{d['later']}")
    print("\n두 번째 드론: 신고 + N분 도착 → 멈춘 자리에서 투하 (필요한 사람 중 물 전에 전달 · 물이 먼저 · 걷다가 막힘, 전달된 경우 물까지 중앙값)")
    for d in out:
        print(f"{d['behavior']:<16} " + " | ".join(
            f"{N}분: {d[f'ok{N}']}/{d[f'need{N}']} 전달, 물 먼저 {d[f'late{N}']}, 걷다가 {d[f'walk{N}']}, 물까지 {d[f'okmed{N}']:.0f}분"
            for N in DELIVER_MIN))
    return out


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


if __name__ == "__main__":
    main()
