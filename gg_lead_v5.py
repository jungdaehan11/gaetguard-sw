"""갯가드 빛길 안내 v5 — 걷는 동안 30초마다 다시 판정한 결과를 드론이 받아서 움직임 (시뮬레이터 시험용)

v5에서 더한 것 (2026-11-06)
  - 드론 SW 안에서 재판정(gg_redecide.Redecider)을 30초마다 돌림. 물은 재생(실측 CSV) 또는 실시간(gg_live_tide)
  - 판정 결과를 드론 동작으로:
       정상·주의(해안까지 감)  → 빛길 안내 그대로. 새 길이 지금 길과 다르면: 사람 발밑에 깜빡임(멈추세요) 2분 → 새 길로 켜짐
       신고(갈 수 있는 데까지) → 같은 길로 계속 안내, 상태에 신고 표시
       멈춤(더 갈 수 없음)     → 사람 발밑에 깜빡임 + 구조품 투하
       걸으라 했는데 3분 안 움직임(펄에 빠짐) → 신고 + 그 자리에 구조품 투하
  - 구조품 투하: 사람 옆 DROP_OFFSET m로 가서 DROP_ALT m까지 내려가 투하 릴레이를 잠깐 켬 → 다시 올라가 사람 발밑에 깜빡임
  - 관제 상태 파일: 30초마다 gg_status.json(최신) + 기록(_status.jsonl) — 위험도, 사람 위치(위경도), 그 자리 물 30cm 예상 시각, 신고·투하
  - 사람 위치: 가상의 사람 실제 위치(--person true) 또는 드론 카메라로 구한 위치(--person cam, gg_person_cam.py)
    카메라일 때는 안내·판정 모두 카메라로 구한 위치만 씀 (가상의 사람 실제 위치는 시험 기록용)
  - 시뮬레이터를 빨리 돌리기(--sim-speed 10): 시간은 기체 시계로 셈, 하트비트도 그만큼 자주 보냄

v4.1 안전장치 그대로: 하트비트 255번 / 모드 바뀌면 명령 멈춤 / Ctrl+C = BRAKE / 소프트웨어 울타리 /
실기체 연결이면 비행 막음(--real) / 시동 꺼질 때까지 하트비트 유지

실행 (시뮬레이터, 계산 폴더에서 — 지형·조위 파일 필요)
  python gg_lead_v5.py tcp:127.0.0.1:5762 --tide tide\\station_DT_0043_1m_우리예측2022-2024.csv --time "2025-09-11 02:07" ^
      --route tide\\판정_youngheung_2025-09-11_탈출로_0207.json --range 2500 --behavior slow --person cam --sim-speed 10
  --behavior normal(0.5m/s) | slow4(0.4m/s) | slow(0.3m/s) | stuck(15분 걷고 펄에 빠져 못 나옴)
  실시간 물때: --tide/--time 대신 --live (공공데이터포털 키 datago_key.txt 필요, 시작 위치는 --route 또는 --origin 위도,경도)
  끝나면: python gg_plot_v5.py gg5_log_....csv

[가정] 투하 높이·거리·릴레이 시간, 사람 반응, 카메라 오차. [확인 필요] 투하 장치·연결 단자(RELAY_PIN2), 카메라
"""
import argparse
import csv
import json
import math
import os
import queue
import sys
import threading
import time
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gg_lead_v4 as v4                     # noqa: E402  (길 계산·기체 연결·빛·이륙은 v4.1 그대로)
from gg_lead_v4 import ON, BLINK            # noqa: E402
from pymavlink import mavutil               # noqa: E402

# ───────── 설정 ─────────
ALT, SPOT_AHEAD, R_SPOT, LED_TILT_DEG = v4.ALT, v4.SPOT_AHEAD, v4.R_SPOT, v4.LED_TILT_DEG
MAX_GAP, MAX_SPEED, OFFPATH_M, OFFPATH_BACK = v4.MAX_GAP, v4.MAX_SPEED, v4.OFFPATH_M, v4.OFFPATH_BACK
ARRIVE_R, STOP_SPEED, STOP_CONFIRM, REACTION = v4.ARRIVE_R, v4.STOP_SPEED, v4.STOP_CONFIRM, v4.REACTION
LIGHT_RELAY = 0        # RELAY_PIN  (실기체 AUX OUT 5) — 빛
DROP_RELAY = 1         # RELAY_PIN2 (실기체 AUX OUT 6 = 55) — 구조품 투하 [확인 필요: 투하 장치·연결 단자]
DROP_OFFSET = 1.5      # m: 사람에게서 이만큼 옆에서 투하 (펄에 빠진 사람이 손 뻗어 닿을 거리) [가정]
DROP_ALT = 5.0         # m: 투하 높이 (평소 7m에서 2m 내려감) [가정]
DROP_PULSE = 1.0       # s: 투하 릴레이를 켜 두는 시간 [가정]
DROP_TIMEOUT = 30.0    # s: 투하 자리에 이 안에 못 가면 그 자리에서 투하
AFTER_DROP_S = 120.0   # s: 투하 뒤 사람 곁에서 깜빡이며 기다리는 시간 (시험은 여기서 끝, 실제는 구조대 올 때까지) [시험용]
SIMPLIFY_M = 25.0      # m: 판정 길(50m 칸 가운데를 잇는 선)을 드론 길로 펼 때 허용 굴곡 → 판정 길에서 최대 약 21m
MIN_LEG = 8.0          # m: 이보다 짧은 구간은 합침
LOOP_DT = 0.25         # s: 계산 주기 (기체 시계)
MAX_RANGE = 60.0
TIER_NAME = {0: "정상", 1: "주의", 2: "신고"}

START, FOLLOW, HOLD, OFFPATH_P, REPLAN, DROP, WAIT, ARRIVED = (
    "출발 대기", "따라가기", "모퉁이 대기", "길 이탈", "새 길 대기", "구조품 투하", "멈춤·구조 대기", "도착")
BEHAVIORS = {"normal": dict(name="정상 0.5m/s", v=0.5), "slow4": dict(name="느림 0.4m/s", v=0.4), "slow": dict(name="아주 느림 0.3m/s", v=0.3),
             "stuck": dict(name="15분 걷고 펄에 빠짐(못 나옴)", v=0.5, stuck_at=17.0 * 60)}   # gg_redecide.BEHAVIORS 와 같음


# ───────── 기체 (v4 + 기체 시계, 빠른 하트비트) ─────────
class Vehicle5(v4.Vehicle):
    def __init__(self, addr, baud, speed=1.0):
        super().__init__(addr, baud)
        self.speed = speed
        self.boot_s = None
        self.boot_wall = None

    def heartbeat(self):
        now = time.time()
        if now - self.last_hb_sent >= 1.0 / self.speed:      # 빠르게 돌리면 기체 시계로 1초마다
            self.m.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
            self.last_hb_sent = now

    def poll(self):
        self.heartbeat()
        while True:
            msg = self.m.recv_match(blocking=False)
            if msg is None:
                break
            if msg.get_srcSystem() != self.m.target_system:
                continue
            typ = msg.get_type()
            tb = getattr(msg, "time_boot_ms", None)
            if tb is not None and typ in ("ATTITUDE", "LOCAL_POSITION_NED", "GLOBAL_POSITION_INT", "SYSTEM_TIME"):
                s = tb / 1000.0
                if self.boot_s is None or s > self.boot_s:
                    self.boot_s, self.boot_wall = s, time.time()
            if typ == "HEARTBEAT" and msg.get_srcComponent() == self.m.target_component:
                self.last_hb_recv = time.time()
                self.mode = mavutil.mode_string_v10(msg)
                self.armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
            elif typ == "LOCAL_POSITION_NED":
                self.ned = (msg.x, msg.y, msg.z)
            elif typ == "GLOBAL_POSITION_INT":
                self.rel_alt = msg.relative_alt / 1000.0
            elif typ == "ATTITUDE":
                self.yaw = math.degrees(msg.yaw)
            elif typ == "COMMAND_ACK" and msg.command == mavutil.mavlink.MAV_CMD_DO_SET_RELAY:
                if msg.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
                    self.relay_ok += 1
                else:
                    self.relay_fail += 1
                    if self.relay_fail == 1:
                        print(f"  ! 릴레이 명령 거절됨 (결과 {msg.result}) — RELAY_PIN / RELAY_PIN2 설정 확인")
            elif typ == "STATUSTEXT":
                txt = msg.text if isinstance(msg.text, str) else msg.text.decode(errors="ignore")
                print("  기체:", txt.strip("\x00"))

    def link_alive(self):
        return time.time() - self.last_hb_recv < 3.0

    def clock(self):
        """시간(초): 실기체·보통 속도는 컴퓨터 시계, 빠르게 돌리는 시뮬레이터는 기체 시계"""
        if self.speed == 1.0 or self.boot_s is None:
            return time.time()
        return self.boot_s + min(time.time() - self.boot_wall, 0.3) * self.speed


# ───────── 판정을 따로 돌리는 곳 (계산이 길어도 비행 명령이 멈추지 않게) ─────────
class EngineThread(threading.Thread):
    def __init__(self, eng):
        super().__init__(daemon=True)
        self.eng = eng
        self.q = queue.Queue()
        self.lock = threading.Lock()
        self.snap = None
        self.err = None
        self.max_lat = 0.0

    def snapshot(self, t):
        e, P = self.eng, self.eng.plan
        last = e.log[-1] if e.log else {}
        tx = e.tx[-1] if e.tx and abs(e.tx[-1]["t_min"] - t) < 1e-6 else None
        return dict(t=t, id=P["id"], mode=P["mode"], level=P["level"], cont=P["cont"], t_set=P["t_set"], pts=P["pts"].copy(),
                    tier=e.tier, caution=e.caution, alert=e.alert, drop=None if e.drop is None else (e.drop[0], e.drop[1].copy()),
                    stuck_on=e.stuck_on, event=last.get("event", "") if last.get("t_min") == round(t, 2) else "",
                    v_est=last.get("v_est"), slack=last.get("slack_min"), tx=tx,
                    tide_off=round(float(e.W.at(t)[2]), 1), tide_status=getattr(e.W, "status", "재생(실측 CSV)"))

    def run(self):
        while True:
            item = self.q.get()
            if item is None:
                return
            t, pos, wall = item
            try:
                self.eng.tick(t, np.asarray(pos, float))
                s = self.snapshot(t)
                s["lat"] = time.time() - wall
                self.max_lat = max(self.max_lat, s["lat"])
            except Exception as ex:                      # 판정이 죽어도 비행은 계속 (마지막 계획 유지) → 기록에 남김
                self.err = f"{t:.1f}분: {ex!r}"
                print("  ! 판정 오류:", self.err)
                continue
            with self.lock:
                self.snap = s

    def latest(self):
        with self.lock:
            return self.snap


# ───────── 판정 길 → 드론 길 ─────────
def drone_path(raw_pts, start):
    """판정 길(칸 가운데를 잇는 선, 북·동 m)을 펴서 드론 길로. 첫 점은 사람의 지금 위치"""
    from shapely.geometry import LineString
    raw = np.asarray(raw_pts, float)
    if len(raw) < 2:
        return None
    s = np.array(LineString(raw[:, ::-1]).simplify(SIMPLIFY_M).coords)[:, ::-1]
    pts = [tuple(start)] + [tuple(p) for p in s[1:]]
    changed = True
    while changed and len(pts) > 2:                      # 짧은 구간 합치기 (첫 점·끝 점은 남김)
        changed = False
        for i in range(1, len(pts) - 1):
            if min(math.dist(pts[i - 1], pts[i]), math.dist(pts[i], pts[i + 1])) < MIN_LEG:
                del pts[i]
                changed = True
                break
    if len(pts) < 2 or sum(math.dist(a, b) for a, b in zip(pts, pts[1:])) < 1.0:
        return None
    return v4.Path(pts)


def path_fits(path, s_from, raw_pts, tol):
    """지금 드론 길의 남은 부분이 새 판정 길에서 tol 안이면 그대로 써도 됨 (같은 길이면 빛이 옆으로 튀지 않게)"""
    from shapely.geometry import LineString, Point
    raw = LineString(np.asarray(raw_pts, float)[:, ::-1])
    end_new = np.asarray(raw_pts, float)[-1]
    if math.dist(path.pts[-1], end_new) > tol:
        return False
    for s in np.linspace(s_from, path.length, 40):
        n, e = path.point(s)[0]
        if raw.distance(Point(e, n)) > tol:
            return False
    return True


# ───────── 가상의 사람 ─────────
class VirtualPerson5:
    """켜짐: 길을 따라 v m/s / 깜빡임: 빛 자국까지 가서 멈춤 / stuck_at(초) 뒤로는 못 움직임 (펄)"""

    def __init__(self, start, v, stuck_at=None):
        self.p = tuple(start)
        self.v, self.stuck_at = v, stuck_at
        self.state = "기다림"
        self.last_signal = None
        self.go_at = 0.0

    def update(self, t, dt, signal, spot, path):
        if self.stuck_at is not None and t >= self.stuck_at:
            self.state = "펄에 빠짐"
            return
        s, d = path.project(self.p)
        if self.last_signal == BLINK and signal == ON:
            self.go_at = t + REACTION
        self.last_signal = signal
        if signal == BLINK:
            dist = math.dist(self.p, spot)
            if dist > 0.35:
                k = min(dist - 0.3, self.v * dt) / dist
                self.p = (self.p[0] + (spot[0] - self.p[0]) * k, self.p[1] + (spot[1] - self.p[1]) * k)
                self.state = "빛 자국으로 감"
            else:
                self.state = "멈춤"
            return
        if signal != ON or t < self.go_at:
            self.state = "멈춤(반응 중)"
            return
        if s >= path.length - 0.3 and abs(d) < OFFPATH_BACK:
            self.state = "도착"
            return
        s2 = min(path.length, s + self.v * dt)
        d2 = d - math.copysign(min(abs(d), 0.1 * dt), d)
        (n, e), u = path.point(s2)
        self.p = (n - u[1] * d2, e + u[0] * d2)
        self.state = "걷는 중"


def unit(a, b, fallback=(1.0, 0.0)):
    dx, dy = b[0] - a[0], b[1] - a[1]
    L = math.hypot(dx, dy)
    return (dx / L, dy / L) if L > 1e-6 else fallback


# ───────── 안내 비행 ─────────
def fly(veh, cfg, log_path):
    import gg_redecide as rd
    eng, t0_clock, to_ll = cfg["eng"], cfg["t0"], cfg["to_ll"]
    beh = BEHAVIORS[cfg["behavior"]]
    P0 = eng.plan
    path = drone_path(P0["pts"], (0.0, 0.0)) if P0["mode"] in ("탈출", "끝까지") else v4.Path([(0.0, 0.0), (0.0, 1.0)])
    if not v4.takeoff(veh):
        return
    veh.m.mav.request_data_stream_send(veh.m.target_system, veh.m.target_component, mavutil.mavlink.MAV_DATA_STREAM_ALL, 10, 1)
    # 출발 배치: 사람은 길 출발점, 드론은 사람에게서 길 따라 R_SPOT 앞 = 이륙한 자리 → 사람 발밑에 깜빡임(출발 대기 2분)
    cmd = path.point_ext(path.ahead(0.0, R_SPOT))[0]
    n0, e0 = veh.ned[0] - cmd[0], veh.ned[1] - cmd[1]
    yaw0 = math.atan2(-cmd[1], -cmd[0])
    t_end = time.time() + 20
    while time.time() < t_end:                           # 사람 쪽으로 돌기
        veh.poll()
        if veh.mode != "GUIDED":
            print(f"조종자가 넘겨받음 ({veh.mode}) → 프로그램 명령 멈춤")
            v4.watch_until_disarmed(veh)
            return
        veh.target(n0 + cmd[0], e0 + cmd[1], ALT, 0, 0, yaw0)
        if veh.yaw is not None and abs((veh.yaw - math.degrees(yaw0) + 540) % 360 - 180) < 5:
            break
        time.sleep(0.05)
    print(f"안내 시작: 판정 {cfg['t0_text']} · 처음 판정 {P0['mode']} {P0['level']} (남은 여유 {P0['info'].get('slack', float('nan')):.1f}분), "
          f"판정 길 {P0['s'][-1]:.0f}m → 드론 길 {path.length:.0f}m 꺾이는 곳 {len(path.corners)}개 · 사람 {beh['name']} · "
          f"사람 위치 {'카메라' if cfg['cam'] else '가상의 사람 실제 위치'} · 물 보정 {eng.W.at(0.0)[2]:+.1f}cm ({getattr(eng.W, 'status', '재생')})")

    eth = EngineThread(eng)
    eth.start()
    eth.snap = eth.snapshot(0.0)
    person = VirtualPerson5((0.0, 0.0), beh["v"], beh.get("stuck_at"))
    loc = None
    if cfg["cam"]:
        import gg_person_cam as pc
        loc = pc.PersonLocator(pos=cfg["cam_pos"], seed=cfg["seed"])
    light = v4.Light(veh, LIGHT_RELAY)
    est = (0.0, 0.0)
    phase, prev_phase = START, FOLLOW
    s_spot = SPOT_AHEAD
    done_corners = set()
    stop_since = None
    hist = []
    plan_id = P0["id"]
    pending = None                                       # 새 길 대기 중인 계획
    phase_t0 = 0.0
    frozen = (0.0, 0.0)                                  # 깜빡임 자리 (사람 발밑, 들어갈 때 고정)
    lead_u = path.point(0.0)[1]
    drop = None                                          # 투하 진행 dict
    drops = []
    drop_u = (1.0, 0.0)
    alt_cmd = ALT
    relay2 = False
    paths = [dict(plan=P0["id"], t=0.0, mode=P0["mode"], pts=[list(map(float, p)) for p in path.pts])]
    events = [(0.0, "처음 판정", f"{P0['mode']} {P0['level']}")]
    rows = []
    end_reason = "시간 초과"
    signal = BLINK
    spot_pt = frozen
    next_tick = rd.TICK
    t_start = veh.clock()
    last = t_start
    wall0 = time.time()
    max_t = cfg["max_time"]
    status_path = os.path.join(os.path.dirname(os.path.abspath(log_path)), "gg_status.json")
    status_log = log_path.replace(".csv", "_status.jsonl")
    open(status_log, "w").close()
    tick_n = 0

    def clock_text(tsec):
        return str(t0_clock + np.timedelta64(int(round(tsec)), "s"))[:19].replace("T", " ")

    def write_status(snap, t):
        nonlocal tick_n
        tick_n += 1
        tx = snap.get("tx") or {}
        lat, lon = to_ll(*est)
        dlat, dlon = to_ll(veh.ned[0] - n0, veh.ned[1] - e0)
        st = dict(version="v5", clock=clock_text(t), t_min=round(t / 60, 2), tier=TIER_NAME[snap["tier"]], mode=snap["mode"],
                  level=snap["level"], event=snap["event"], phase=phase,
                  person=dict(lat=round(lat, 7), lon=round(lon, 7), src="카메라" if loc else "가상의 사람"),
                  drone=dict(lat=round(dlat, 7), lon=round(dlon, 7), alt=round(veh.rel_alt or 0.0, 1)),
                  water30_forecast=None if tx.get("water_fc_min") is None else clock_text(tx["water_fc_min"] * 60),
                  water30_99=None if tx.get("water_99_min") is None else clock_text(tx["water_99_min"] * 60),
                  caution=None if snap["caution"] is None else clock_text(snap["caution"] * 60),
                  alert=None if snap["alert"] is None else clock_text(snap["alert"] * 60),
                  drop=[dict(clock=clock_text(d["t"]), lat=round(to_ll(*d["pos"])[0], 7), lon=round(to_ll(*d["pos"])[1], 7)) for d in drops],
                  stuck=bool(snap["stuck_on"]), v_est=snap["v_est"],
                  tide=dict(offset_cm=snap.get("tide_off"), status=snap.get("tide_status")),
                  slack_min=None if snap["slack"] is None or not np.isfinite(snap["slack"]) else snap["slack"],
                  decide_latency_s=round(snap.get("lat", 0.0) * veh.speed, 2))
        tmp = status_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(st, f, ensure_ascii=False, indent=1)
        os.replace(tmp, status_path)
        with open(status_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(st, ensure_ascii=False) + "\n")

    def enter(ph, t):
        nonlocal phase, phase_t0
        phase, phase_t0 = ph, t

    last_snap_t = -1.0
    try:
        while True:
            now = veh.clock()
            t, dt = now - t_start, max(now - last, 1e-3)
            last = now
            veh.poll()
            if not veh.link_alive():
                end_reason = "기체 신호 끊김"
                print("! 기체 신호가 끊김 → 프로그램 멈춤 (기체는 자체 안전 기능으로 동작)")
                break
            if veh.mode != "GUIDED":
                end_reason = f"조종자가 넘겨받음 ({veh.mode})"
                print(f"! 모드가 {veh.mode}(으)로 바뀜 → 조종자가 넘겨받음, 프로그램 명령 멈춤")
                light.set(ON)
                light.update(now)
                if relay2:
                    veh.relay(DROP_RELAY, False)
                break
            if t > max_t:
                break
            dn, de = (veh.ned[0] - n0, veh.ned[1] - e0)

            # ── 사람 (가상의 사람이 움직이고, 드론은 카메라 또는 실제 위치로 앎) ──
            person.update(t, dt, signal, spot_pt, path)
            if loc is not None:
                e_, lost = loc.step(t, (dn, de), veh.rel_alt or ALT, veh.yaw or 0.0, person.p)
                if e_ is not None:
                    est = e_
            else:
                est, lost = person.p, False
            pn, pe = est
            s_p, d_p = path.project(est)
            hist.append((t, pn, pe))
            while hist and t - hist[0][0] > 2.0:                # 2초 동안 움직인 거리로 속도 (카메라 흔들림에 덜 민감)
                hist.pop(0)
            speed = math.dist(hist[0][1:], hist[-1][1:]) / max(t - hist[0][0], 1e-3) if len(hist) > 2 else 1.0
            gap = math.hypot(dn - person.p[0], de - person.p[1])

            # ── 30초마다 판정에 위치를 보냄, 새 결과를 받음 ──
            if t / 60.0 >= next_tick - 1e-9:
                eth.q.put((next_tick, est, time.time()))
                next_tick += rd.TICK
            snap = eth.latest()
            if snap["t"] != last_snap_t:
                last_snap_t = snap["t"]
                write_status(snap, snap["t"] * 60)
                if snap["event"] and snap["event"] != "처음 판정":
                    events.append((round(t, 1), snap["event"], f"판정 {snap['t']:.1f}분"))
                    print(f"  [{clock_text(t)[11:]}] 판정: {snap['event']}  (위험도 {TIER_NAME[snap['tier']]}, 계산 {snap['lat'] * veh.speed:.1f}초, "
                          f"물 보정 {snap.get('tide_off')}cm · {snap.get('tide_status')})")
            # 새 계획
            if snap["id"] != plan_id:
                plan_id = snap["id"]
                if snap["mode"] in ("탈출", "끝까지") and snap["cont"] and phase in (START, REPLAN):
                    if phase == REPLAN:                  # 이미 멈춰 있음: 기다리는 시간은 그대로, 계획만 바꿈
                        pending = snap
                    else:
                        newp = drone_path(snap["pts"], est)
                        if newp is not None:
                            path = newp
                            lead_u = newp.point(0.0)[1]
                            paths.append(dict(plan=plan_id, t=round(t, 1), mode=snap["mode"], pts=[list(map(float, p)) for p in path.pts]))
                    events.append((round(t, 1), "같은 길(멈춰 있는 중)", f"계획 {plan_id}"))
                elif snap["mode"] in ("탈출", "끝까지"):
                    if snap["cont"] and phase in (FOLLOW, HOLD, OFFPATH_P) and path_fits(path, s_p, snap["pts"], SIMPLIFY_M):
                        events.append((round(t, 1), "같은 길 계속", f"계획 {plan_id}"))
                    elif snap["cont"] and phase in (FOLLOW, HOLD, OFFPATH_P):
                        newp = drone_path(snap["pts"], est)
                        if newp is not None:
                            path, done_corners = newp, set()
                            s_spot = max(SPOT_AHEAD * 0.5, newp.project(spot_pt)[0])
                            phase = FOLLOW
                            paths.append(dict(plan=plan_id, t=round(t, 1), mode=snap["mode"], pts=[list(map(float, p)) for p in path.pts]))
                            events.append((round(t, 1), "같은 길(드론 길만 다시 폄)", f"계획 {plan_id}"))
                    elif phase not in (DROP, WAIT):
                        pending = snap
                        newp = drone_path(snap["pts"], est)
                        lead_u = newp.point(0.0)[1] if newp is not None else lead_u
                        frozen = est
                        enter(REPLAN, t)
                        events.append((round(t, 1), "새 길 대기 시작", f"계획 {plan_id} {snap['mode']}"))
                        print(f"  → 새 길: 사람 발밑에 깜빡임(멈추세요) {rd.gd.REACT:.0f}분 → 새 길로")
                elif snap["mode"] == "멈춤" and phase != DROP and drop is None and not drops:
                    frozen = est
                    drop = dict(stage="접근", t0=t, target=est, u=unit(est, (dn, de)))
                    enter(DROP, t)
                    print("  → 멈춤: 사람 발밑에 깜빡임 + 구조품 투하")
            # 걸으라 했는데 3분 안 움직임 → 그 자리에 구조품
            if snap["drop"] is not None and drop is None and not drops and phase not in (DROP,):
                frozen = est
                drop = dict(stage="접근", t0=t, target=est, u=unit(est, (dn, de)))
                enter(DROP, t)
                print("  → 3분 안 움직임: 신고 + 그 자리에 구조품 투하" if snap["stuck_on"] else "  → 멈춤: 사람 발밑에 깜빡임 + 구조품 투하")

            # ── 단계 ──
            if phase == START and t >= rd.gd.REACT * 60:
                enter(FOLLOW, t)
                events.append((round(t, 1), "출발", ""))
            if phase == REPLAN and t >= phase_t0 + rd.gd.REACT * 60:
                newp = drone_path(pending["pts"], est)
                if newp is not None:
                    path = newp
                    paths.append(dict(plan=pending["id"], t=round(t, 1), mode=pending["mode"], pts=[list(map(float, p)) for p in path.pts]))
                done_corners, s_spot = set(), SPOT_AHEAD * 0.5
                enter(FOLLOW, t)
                events.append((round(t, 1), "새 길로 출발", f"계획 {pending['id']}"))
                pending = None
                s_p, d_p = path.project(est)
            if phase in (FOLLOW, HOLD) and abs(d_p) > OFFPATH_M:
                prev_phase = phase
                enter(OFFPATH_P, t)
                events.append((round(t, 1), "길 이탈", round(abs(d_p), 2)))
            elif phase == OFFPATH_P and abs(d_p) < OFFPATH_BACK:
                enter(prev_phase if prev_phase == HOLD else FOLLOW, t)
                events.append((round(t, 1), "길로 돌아옴", 0))

            pend = [c for c in path.corners if c not in done_corners]
            next_c = pend[0] if pend else None
            if phase == FOLLOW:
                want = s_p + SPOT_AHEAD
                if gap > MAX_GAP:
                    want = s_spot
                s_spot = min(s_spot + MAX_SPEED * dt, max(s_spot, min(want, path.length)))
                if next_c is not None and s_spot >= next_c:
                    s_spot = next_c
                    enter(HOLD, t)
                    stop_since = None
                if s_p >= path.length - 0.3 and abs(d_p) < OFFPATH_BACK and snap["mode"] != "멈춤":
                    enter(ARRIVED, t)
                    events.append((round(t, 1), "도착", ""))
            elif phase == HOLD:
                c_pt, _ = path.point(next_c)
                if math.dist(est, c_pt) <= ARRIVE_R and speed < STOP_SPEED:
                    stop_since = stop_since if stop_since is not None else t
                    if t - stop_since >= STOP_CONFIRM:
                        done_corners.add(next_c)
                        enter(FOLLOW, t)
                else:
                    stop_since = None

            # ── 빛 자국 목표점 · 드론 목표 ──
            alt_cmd = ALT
            yaw_to = None
            if phase in (FOLLOW, HOLD, OFFPATH_P, ARRIVED):
                spot_pt, _ = path.point(s_spot)
                goal = path.drone_spot(s_spot, R_SPOT)
                if phase == ARRIVED:
                    goal, _ = path.point_ext(path.length + R_SPOT + SPOT_AHEAD)
            elif phase in (START, REPLAN):
                spot_pt = frozen
                goal = (frozen[0] + lead_u[0] * R_SPOT, frozen[1] + lead_u[1] * R_SPOT)
            elif phase == WAIT:
                spot_pt = frozen
                goal = (frozen[0] + drop_u[0] * R_SPOT, frozen[1] + drop_u[1] * R_SPOT)
            else:                                        # DROP
                spot_pt = frozen
                tg, u = drop["target"], drop["u"]
                if drop["stage"] == "접근":
                    goal = (tg[0] + u[0] * DROP_OFFSET, tg[1] + u[1] * DROP_OFFSET)
                    alt_cmd = DROP_ALT
                    there = math.hypot(dn - goal[0], de - goal[1]) < 0.5 and abs((veh.rel_alt or ALT) - DROP_ALT) < 0.4
                    if there or t - drop["t0"] > DROP_TIMEOUT:
                        veh.relay(DROP_RELAY, True)
                        relay2 = True
                        drop.update(stage="투하", t_rel=t)
                        drops.append(dict(t=t, pos=(dn, de), person=est, alt=veh.rel_alt, offset=math.hypot(dn - person.p[0], de - person.p[1])))
                        events.append((round(t, 1), "구조품 투하", f"사람에게서 {drops[-1]['offset']:.1f}m, 높이 {veh.rel_alt:.1f}m"))
                        print(f"  → 구조품 투하 ({clock_text(t)[11:]}): 사람에게서 {drops[-1]['offset']:.1f}m, 높이 {veh.rel_alt:.1f}m")
                elif drop["stage"] == "투하":
                    goal = (tg[0] + u[0] * DROP_OFFSET, tg[1] + u[1] * DROP_OFFSET)
                    alt_cmd = DROP_ALT
                    if t - drop["t_rel"] >= DROP_PULSE:
                        veh.relay(DROP_RELAY, False)
                        relay2 = False
                        drop["stage"] = "올라감"
                if drop["stage"] == "올라감":
                    goal = (tg[0] + u[0] * R_SPOT, tg[1] + u[1] * R_SPOT)
                    if math.hypot(dn - goal[0], de - goal[1]) < 0.7 and abs((veh.rel_alt or 0) - ALT) < 0.5:
                        drop_u = u
                        enter(WAIT, t)
                yaw_to = tg
            # 걷기 다시 시작 (빠졌다가 나온 경우): 멈춤 계획이 아니고 3분 규칙도 풀렸으면 다시 따라가기
            if phase == WAIT and snap["mode"] in ("탈출", "끝까지") and not snap["stuck_on"]:
                enter(FOLLOW, t)
                s_spot = path.project(est)[0] + SPOT_AHEAD * 0.5
            vel = (0.0, 0.0)
            dx, dy = goal[0] - cmd[0], goal[1] - cmd[1]
            dist = math.hypot(dx, dy)
            step = min(dist, MAX_SPEED * dt)
            if dist > 1e-6:
                cmd = (cmd[0] + dx / dist * step, cmd[1] + dy / dist * step)
                vel = (dx / dist * step / dt, dy / dist * step / dt)
            alt_now = veh.rel_alt if veh.rel_alt is not None else ALT
            r_now = max(alt_now, 0.5) * math.tan(math.radians(LED_TILT_DEG))
            aim = yaw_to or spot_pt
            if phase in (FOLLOW, HOLD, OFFPATH_P) and abs(math.dist((dn, de), spot_pt) - r_now) > 0.5:
                s_aim = path.aim_s((dn, de), r_now, s_spot)
                if s_aim is not None:
                    aim = path.point_ext(s_aim)[0]
            if math.hypot(*cmd) > MAX_RANGE:
                print("! 울타리(MAX_RANGE) 밖 목표 → 기다림")
                cmd, vel = (dn, de), (0.0, 0.0)
            yaw = math.atan2(aim[1] - de, aim[0] - dn)
            veh.target(n0 + cmd[0], e0 + cmd[1], alt_cmd, vel[0], vel[1], yaw)

            signal = ON if phase in (FOLLOW, ARRIVED) else BLINK
            light.set(signal)
            light.update(now)

            yr = math.radians(veh.yaw) if veh.yaw is not None else yaw
            r = max(alt_now, 0.0) * math.tan(math.radians(LED_TILT_DEG))
            spot_real = (dn + r * math.cos(yr), de + r * math.sin(yr))
            sp_s, sp_off = path.project(spot_real)
            ps_true, pd_true = path.project(person.p)
            rows.append(dict(t=round(t, 2), clock=clock_text(t)[11:], phase=phase, signal=signal, tier=TIER_NAME[snap["tier"]],
                             plan=plan_id, eng_mode=snap["mode"], person_state=person.state,
                             person_n=round(person.p[0], 2), person_e=round(person.p[1], 2), est_n=round(pn, 2), est_e=round(pe, 2),
                             est_err=round(math.dist(est, person.p), 2), lost=int(bool(lost)),
                             person_s=round(ps_true, 2), offset=round(pd_true, 2),
                             drone_n=round(dn, 2), drone_e=round(de, 2), alt=round(alt_now, 2), gap=round(gap, 2),
                             spot_s=round(s_spot, 2), spot_tn=round(spot_pt[0], 2), spot_te=round(spot_pt[1], 2),
                             spot_n=round(spot_real[0], 2), spot_e=round(spot_real[1], 2),
                             spot_err=round(math.dist(spot_real, spot_pt), 2), spot_off=round(sp_off, 2),
                             spot_ahead=round(sp_s - ps_true, 2), relay=int(bool(light.relay_on)), relay2=int(relay2),
                             yaw_cmd=round(math.degrees(yaw), 1), yaw=None if veh.yaw is None else round(veh.yaw, 1), mode=veh.mode))
            if len(rows) % int(60 / LOOP_DT) == 0:
                print(f"{clock_text(t)[11:]} | {phase} | {snap['mode']} {TIER_NAME[snap['tier']]} | 사람 {person.state} "
                      f"길 {ps_true:6.0f}/{path.length:.0f}m 옆 {pd_true:+.1f}m | 드론 거리 {gap:4.1f}m | 빛 {signal}"
                      + (f" | 카메라 오차 {math.dist(est, person.p):.2f}m" if loc else ""))
            if phase == ARRIVED and math.dist(cmd, goal) < 0.3 and math.hypot(dn - goal[0], de - goal[1]) < 1.0:
                end_reason = "사람 도착"
                break
            if phase == WAIT and drops and t - drops[-1]["t"] >= AFTER_DROP_S:
                end_reason = f"구조품 투하 뒤 {AFTER_DROP_S:.0f}초 곁에서 대기 (시험 끝)"
                break
            time.sleep(max(0.0, LOOP_DT / veh.speed - 0.002))
    except KeyboardInterrupt:
        end_reason = "Ctrl+C (그 자리 정지)"
        veh.set_mode("BRAKE")
        print("\nCtrl+C → BRAKE(그 자리 정지). 조종자가 넘겨받으세요.")
    eth.q.put(None)
    wall = time.time() - wall0

    if relay2:
        veh.relay(DROP_RELAY, False)
    if end_reason.startswith(("사람 도착", "시간 초과", "구조품")):
        light.set(ON)
        light.update(veh.clock())
        veh.set_mode("LAND")
        print(f"{end_reason} → 착륙 명령 보냄")
    if end_reason != "기체 신호 끊김":
        v4.watch_until_disarmed(veh)

    if not rows:
        return
    with open(log_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    meta = dict(version="v5", t0=cfg["t0_text"], origin=cfg["origin"], behavior=cfg["behavior"], behavior_name=beh["name"],
                person_src="카메라(" + cfg["cam_pos"] + ")" if cfg["cam"] else "가상의 사람", water=cfg["water_src"],
                tide=cfg["tide"], route=cfg["route"],
                alt=ALT, r_spot=round(R_SPOT, 2), spot_ahead=SPOT_AHEAD, offpath_m=OFFPATH_M, led_tilt=LED_TILT_DEG,
                drop_offset=DROP_OFFSET, drop_alt=DROP_ALT, simplify_m=SIMPLIFY_M, sim_speed=veh.speed, wall_s=round(wall, 1),
                paths=paths, plans=[dict(id=P["id"], t_set=P["t_set"], mode=P["mode"], level=P["level"], cont=P["cont"],
                                         pts=np.round(P["pts"], 1).tolist()) for P in eng.plans],
                engine_log=eng.log, tx=eng.tx, drops=[dict(d, pos=list(d["pos"]), person=list(d["person"])) for d in drops],
                caution=eng.caution, alert=eng.alert, stuck=[(t_, list(map(float, p))) for t_, p in eng.stuck_events],
                events=events, end_reason=end_reason, engine_error=eth.err, engine_max_latency_s=round(eth.max_lat * veh.speed, 2),
                relay_ok=veh.relay_ok, relay_fail=veh.relay_fail, light_sent=light.sent)
    with open(log_path.replace(".csv", "_path.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, default=float)
    summary(rows, meta, cfg, log_path)


def summary(rows, meta, cfg, log_path):
    """실제 물때(실측)로 검증: 사람이 지난 칸 여유 / 신고~물 / 투하~물"""
    eng, W = cfg["eng"], cfg["eng"].W
    print("\n── 요약 ──")
    print(f"끝난 이유: {meta['end_reason']} · 안내 {rows[-1]['t'] / 60:.1f}분 (실제 시간 {meta['wall_s'] / 60:.1f}분, 시뮬레이터 {meta['sim_speed']:g}배)")
    print(f"판정 계획 {len(meta['plans'])}개 · 드론 길 {len(meta['paths'])}번 · 판정 계산 최대 {meta['engine_max_latency_s']:.1f}초"
          + (f" · 판정 오류 {meta['engine_error']}" if meta["engine_error"] else ""))
    for t, what, info in meta["events"]:
        if what not in ("출발",):
            print(f"   {str(cfg['t0'] + np.timedelta64(int(round(t)), 's'))[11:19]}  {what}  {info}")
    if not hasattr(W, "truth"):
        print("실시간 물때 → 실제 물때 검증은 나중에 실측으로")
        return
    tct = W.truth().ravel()
    pass_m, ks = [], []
    for r in rows:
        k = eng.cell(r["person_n"], r["person_e"])
        if eng.sea[k] and np.isfinite(tct[k]):
            pass_m.append(tct[k] - r["t"] / 60.0)
            ks.append(k)
    last = rows[-1]
    k_end = eng.cell(last["person_n"], last["person_e"])
    if meta["end_reason"] == "사람 도착":
        print(f"실제 물때로 검증: 사람이 지난 칸 중 가장 아슬아슬한 곳도 지난 뒤 {min(pass_m):.1f}분에 잠김"
              + (" → 늦지 않음" if min(pass_m) > 0 else " → 늦음!"))
    else:
        water = float(tct[k_end]) if eng.sea[k_end] else np.inf
        print(f"실제 물때로 검증: 사람이 멈춘 칸 물 30cm {str(cfg['t0'] + np.timedelta64(int(round(water * 60)), 's'))[11:16] if np.isfinite(water) else '없음'}"
              + (f" · 신고 뒤 {water - meta['alert']:.0f}분" if meta["alert"] is not None and np.isfinite(water) else ""))
        for d in meta["drops"]:
            print(f"   구조품 투하 {str(cfg['t0'] + np.timedelta64(int(round(d['t'])), 's'))[11:19]} → 그 자리 물 30cm까지 {water - d['t'] / 60:.0f}분, "
                  f"사람에게서 {d['offset']:.1f}m")
    walk = [r for r in rows if r["phase"] == FOLLOW and r["person_state"] == "걷는 중"]
    if walk:
        off = sorted(abs(r["spot_off"]) for r in walk)
        print(f"빛 자국 (따라가는 동안): 길에서 벗어난 거리 중앙값 {off[len(off) // 2]:.2f}m / 95% {off[int(len(off) * .95)]:.2f}m")
    if cfg["cam"]:
        er = sorted(r["est_err"] for r in rows)
        print(f"카메라로 구한 사람 위치 오차: 중앙값 {er[len(er) // 2]:.2f}m / 95% {er[int(len(er) * .95)]:.2f}m / 최대 {er[-1]:.2f}m, "
              f"놓친 시간 {sum(r['lost'] for r in rows) * LOOP_DT:.0f}초")
    print(f"릴레이 명령 수락 {meta['relay_ok']} / 거절 {meta['relay_fail']}")
    print(f"기록: {log_path} · 상태 기록: {log_path.replace('.csv', '_status.jsonl')} → 그림: python gg_plot_v5.py {os.path.basename(log_path)}")


def main():
    global MAX_RANGE
    ap = argparse.ArgumentParser(description="갯가드 빛길 안내 v5 (재판정 받기)")
    ap.add_argument("addr", nargs="?", default="tcp:127.0.0.1:5762")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--real", action="store_true", help="실기체 비행 허용 (설정 수정·허락·안전 조종자 준비 뒤에만)")
    ap.add_argument("--tide", help="조위 CSV (재생: 그 시각까지의 실측만 씀)")
    ap.add_argument("--time", help="판정 시각 (재생)")
    ap.add_argument("--live", action="store_true", help="실시간 물때 (공공데이터포털)")
    ap.add_argument("--route", help="gg_decide.py 탈출로 파일 → 출발 위치")
    ap.add_argument("--origin", help="출발 위치 위도,경도 (--route 대신)")
    ap.add_argument("--behavior", default="normal", choices=list(BEHAVIORS))
    ap.add_argument("--person", default="true", choices=["true", "cam"], help="사람 위치: 가상의 사람 실제 위치 / 카메라로 구함")
    ap.add_argument("--cam-pos", default="rtk", choices=["rtk", "gps"], help="카메라 계산에 쓰는 드론 위치")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--range", type=float, help="울타리 거리(m) — 시뮬레이터에서만")
    ap.add_argument("--sim-speed", type=float, default=1.0, help="시뮬레이터 배속 (SITL --speedup 과 같게, 시뮬레이터에서만)")
    ap.add_argument("--walk", type=float, default=0.5)
    ap.add_argument("--depth", type=float, default=0.30)
    a = ap.parse_args()

    local = v4.is_local(a.addr)
    if (a.range or a.sim_speed != 1.0) and (not local or a.real):
        print("울타리 거리·배속은 시뮬레이터에서만 바꿀 수 있습니다.")
        return
    if not local and not a.real:
        print("실기체 연결로 보여 비행을 막았습니다.")
        return
    if a.real:
        print("실기체 비행 전 확인: 배터리·지상국 안전 설정 수정 / 관리자 허락 / 안전 조종자 조종기 대기(스로틀 가운데) / "
              "비행 구역 사람 통제 / 투하 장치 점검 / 비행승인 여부 [법규 확인: 항공안전법 제127조]")
        if input("모두 확인했으면 '비행' 입력: ").strip() != "비행":
            print("취소했습니다.")
            return
    if a.range:
        MAX_RANGE = a.range

    import gg_decide as gd
    import gg_redecide as rd
    import gg_retreat_map as rm
    if a.route:
        route = json.load(open(a.route, encoding="utf-8"))
        origin = (route["start_lat"], route["start_lon"])
    elif a.origin:
        origin = tuple(float(x) for x in a.origin.split(","))
    else:
        print("--route 또는 --origin 이 필요합니다.")
        return
    print("지형 격자 만드는 중...")
    G = rm.build_grid(rm.AREA, 50.0, 0.5)
    if a.live:
        import gg_live_tide as lt
        t0 = np.datetime64(datetime.now(lt.KST).strftime("%Y-%m-%dT%H:%M"))        # 한국 시각 (컴퓨터 시간대와 무관)
        water = lt.LiveWater(G, lt.ApiSource(), t0, a.depth)
        water_src = "실시간 (공공데이터포털 영흥도 실측 + 저장한 조석 성분)"
    else:
        if not (a.tide and a.time):
            print("--tide 와 --time (재생) 또는 --live 가 필요합니다.")
            return
        T, O, P = gd.load(a.tide)
        t0 = np.datetime64(a.time.replace(" ", "T"))
        water = rd.Water(G, T, O, P, t0, a.depth)
        water_src = f"재생: {os.path.basename(a.tide)} ({a.time}부터, 그때까지의 실측만)"
    eng = rd.Redecider(G, water, origin, a.walk)
    P0 = eng.start(np.zeros(2))
    print(f"처음 판정: {P0['mode']} {P0['level']}, 판정 길 {P0['s'][-1]:.0f}m")
    beh = BEHAVIORS[a.behavior]
    L = float(P0["s"][-1])
    far = float(np.max(np.hypot(P0["pts"][:, 0], P0["pts"][:, 1])))
    if far > MAX_RANGE - 5:
        print(f"판정 길 끝이 출발점에서 {far:.0f}m → 울타리 {MAX_RANGE:.0f}m 밖. 시뮬레이터면 --range 를 늘리세요.")
        return
    kx, ky = G["kx"], G["ky"]
    cfg = dict(eng=eng, t0=t0, t0_text=str(t0)[:16].replace("T", " "), origin=origin, behavior=a.behavior,
               cam=a.person == "cam", cam_pos=a.cam_pos, seed=a.seed, water_src=water_src,
               tide=os.path.abspath(a.tide) if a.tide else None, route=os.path.abspath(a.route) if a.route else None,
               to_ll=lambda n, e: (origin[0] + n / ky, origin[1] + e / kx),
               max_time=min(rd.T_END * 60, L / beh["v"] * 1.4 + gd.REACT * 60 + 900))
    veh = Vehicle5(a.addr, a.baud, a.sim_speed)
    if veh.wait_vehicle() is None:
        print("연결 실패: 시뮬레이터가 켜져 있는지, 주소·포트가 맞는지 확인하세요")
        return
    print(f"연결됨: 기체 번호 {veh.m.target_system}, 하트비트 255번")
    fly(veh, cfg, time.strftime("gg5_log_%Y%m%d_%H%M%S.csv"))


if __name__ == "__main__":
    main()
