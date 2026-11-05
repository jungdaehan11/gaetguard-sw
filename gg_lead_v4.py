"""갯가드 빛길 안내 v4.1 — 빛 자국을 늘 "사람 3m 앞 길 위"에 찍기, 드론도 길을 따라 ㄱ자로 (시뮬레이터 시험용)

v4.1에서 바뀐 점 (2026-11-04)
  - 모퉁이에서 원호로 돌지 않음. 드론도 길을 따라 ㄱ자로 감
  - 규칙: LED가 고정이라 빛 자국은 늘 드론에서 R_SPOT(약 4.9m) 떨어진 곳에 떨어짐
         → 드론 자리 = 빛 자국에서 직선거리 4.9m인 곳
         → 곧은 길: 빛 자국에서 길 따라 4.9m 앞
         → 모퉁이 4.9m 안: 빛 자국 둘레 4.9m 원 위에서 새 길 쪽으로 고르게 돌림
           (꺾인 새 길 위에 딱 맞추면 기수가 순간적으로 초당 30° 넘게 돌아야 해서 실기체 제한 20°에 걸림
            → 고르게 돌리면 초당 약 9°, 대신 드론이 길에서 최대 약 1m 바깥으로 살짝 둥글게 돎)
  - 모퉁이: 빛 자국이 모퉁이에 고정 + 깜빡임 → 사람이 와서 멈춤(1초) → 바로 켜짐,
           드론이 새 길로 나가며 빛 자국이 꺾인 길 위로 앞서감 → 사람이 따라옴
  - 모퉁이 바깥 빈 공간 필요 없음

v3에서 바뀐 점 (팀 결정 2026-11-04)
  - "좌우 쓸기"를 뺌 → 빛 신호는 2가지
       켜짐   = 따라오세요 (빛 자국이 사람 3m 앞 길 위에 있음)
       깜빡임 = 빛 자국까지 와서 멈추세요
  - 곧은 길: 빛 자국이 사람 3m 앞 길 위를 같이 감
  - 꺾이는 곳: ① 빛 자국이 모퉁이에 닿으면 거기 고정하고 깜빡임
              ② 사람이 모퉁이에 와서 멈추면
              ③ 다시 켜짐, 빛 자국이 꺾인 길 위로 앞서감 → 사람이 방향을 틀어 따라옴
  - 길 이탈(2m 넘게 벗어남): 빛 자국을 그 자리에 두고 깜빡임 → 사람이 빛 자국(길 위)으로 돌아옴
  - LED는 앞으로 35° 기울여 고정했다고 가정 → 고도 7m에서 빛 자국은 드론 앞 약 4.9m
    드론은 빛 자국에서 직선거리 4.9m인 길 위 점에 떠서 빛 자국을 바라봄
  - v3 안전장치 그대로: 하트비트 255번 / 모드 바뀌면 명령 멈춤 / Ctrl+C = BRAKE / 소프트웨어 울타리 /
    실기체 연결이면 비행 막음(--real) / 시동 꺼질 때까지 하트비트 유지 / --light-test

실행
  python gg_lead_v4.py                         # 시뮬레이터 (기본 tcp:127.0.0.1:5762)
  python gg_lead_v4.py tcp:127.0.0.1:5763      # 5762가 안 되면
  python gg_lead_v4.py COM5 --light-test       # 땅에서 빛만 시험: 시동 안 걺 (프로펠러 빼고, USB만)
  python gg_lead_v4.py --path-json tide\\판정_youngheung_2025-09-11_탈출로_0207.json --fit 55   # 계산된 탈출로 모양을 55m로 줄여서
  python gg_lead_v4.py --path-json ... --range 2500     # 시뮬레이터에서 실제 크기 그대로 (울타리 늘림, 시뮬레이터만)
  끝나면: python gg_plot_v4.py gg_log_....csv  # 결과 그림

주의
  - 실기체 비행은 막아 둠. 설정 수정·관리자 허락·안전 조종자 준비 뒤에만 --real 로 풀 것
  - 사람 위치는 지금 "가상의 사람"(VirtualPerson). 실제 시연에서 사람 위치를 어떻게 알지는 아직 정하지 않음
"""
import argparse
import collections
import csv
import json
import math
import os
import time

os.environ.setdefault("MAVLINK20", "1")       # MAVLink2로 주고받기 (MAVLink1도 읽음)
from pymavlink import mavutil                 # noqa: E402

# ───────── 설정 (현장에 맞게 바꾸는 곳) ─────────
ALT = 7.0            # 비행 고도 (m) — 계획 5~10m
SPOT_AHEAD = 3.0     # 빛 자국을 사람보다 길 위로 몇 m 앞에 둘지 (m)
LED_TILT_DEG = 35.0  # LED를 수직에서 앞으로 기울인 각도 [가정] → 빛 자국 = 드론 앞 ALT×tan(35°) ≈ 4.9m
MAX_GAP = 12.0       # 사람과 이보다 멀어지면 드론이 기다림 (m)
WALK = 0.5           # 가상 사람 걸음 (m/s) — 밤 갯벌 보행 가정 [확인 필요]
MAX_SPEED = 1.5      # 드론 목표점이 움직이는 최대 속도 (m/s)
OFFPATH_M = 2.0      # 길 중심에서 이보다 벗어나면 "경로 이탈" (m) — 안전 길 폭 4m 가정 [확인 필요]
OFFPATH_BACK = 1.0   # 이 안으로 돌아오면 이탈 해제 (m)
ARRIVE_R = 1.0       # 모퉁이에서 이 안에 들어와 멈추면 "모퉁이 도착" (m)
STOP_SPEED = 0.15    # 이보다 느리면 "멈춤"으로 봄 (m/s)
STOP_CONFIRM = 1.0   # 멈춤이 이만큼 이어지면 모퉁이를 돎 (s)
BLINK_HZ = 1.0       # 깜빡임 횟수 (초당)
LIGHT_RELAY = 0      # 0번 = RELAY_PIN(54, AUX OUT 5) [확인 필요: AUX 5에 지금 연결된 장치]
MAX_RANGE = 60.0     # 출발점에서 이보다 멀리 안 감 (m) — 소프트웨어 울타리
LOOP_DT = 0.25       # 계산 주기 (s)
MAX_TIME = 300       # 안내 최대 시간 (s)

R_SPOT = ALT * math.tan(math.radians(LED_TILT_DEG))   # 드론 바로 아래 ↔ 빛 자국 (m)

# 길: 출발점 기준 (북쪽 m, 동쪽 m). 북 25m → 동 15m → 북 15m (총 55m)
PATH = [(0.0, 0.0), (25.0, 0.0), (25.0, 15.0), (40.0, 15.0)]
PATH_INFO = dict(source="기본 시험 길")        # --path-json 으로 탈출로를 넣으면 바뀜

# 가상 사람 시나리오 (시험용, 길 위 거리 기준)
REST_S, REST_LEN = 10.0, 8.0      # 10m 지점에서 8초 쉼 → 빛 자국·드론도 기다리는지
DRIFT_S = 44.0                    # 44m 지점(마지막 곧은 길)에서 옆으로 벗어나기 시작 → 깜빡임이 켜지는지
DRIFT_SPEED = 0.3                 # 벗어나는 속도 (m/s, 옆으로)
REACTION = 1.0                    # 깜빡임 → 켜짐을 보고 다시 걷기 시작할 때까지 (s) [가정]

# SET_POSITION_TARGET_LOCAL_NED: 위치 + 속도 + 방향(yaw) 사용, 가속도·회전속도 무시
POS_VEL_YAW = 0b0000100111000000

ON, BLINK, OFF = "켜짐", "깜빡임", "꺼짐"
MEANING = {ON: "따라오세요", BLINK: "빛 자국까지 와서 멈추세요", OFF: "-"}
FOLLOW, HOLD, OFFPATH_P, ARRIVED = "따라가기", "모퉁이 대기", "길 이탈", "도착"


def _dp(pts, tol):
    """더글러스-포커: tol(m)보다 작은 굴곡은 펴기"""
    if len(pts) < 3:
        return list(pts)
    (n1, e1), (n2, e2) = pts[0], pts[-1]
    L = math.hypot(n2 - n1, e2 - e1) or 1e-9
    dmax, imax = 0.0, 0
    for i, (n, e) in enumerate(pts[1:-1], 1):
        d = abs((n2 - n1) * (e1 - e) - (n1 - n) * (e2 - e1)) / L
        if d > dmax:
            dmax, imax = d, i
    if dmax <= tol:
        return [pts[0], pts[-1]]
    return _dp(pts[:imax + 1], tol)[:-1] + _dp(pts[imax:], tol)


def load_path_json(fn, fit=None, min_leg=8.0):
    """gg_decide.py 가 만든 탈출로(path_ne_m: 출발점 기준 북·동 m)를 읽음.
    fit(m)을 주면 전체 길이를 그만큼으로 줄이고(모양 유지), 줄인 뒤 min_leg보다 짧은 구간은 합침 (시연장 크기용)"""
    with open(fn, encoding="utf-8") as f:
        info = json.load(f)
    pts = [tuple(map(float, p)) for p in info["path_ne_m"]]
    L0 = sum(math.dist(a, b) for a, b in zip(pts, pts[1:]))
    k = 1.0
    if fit:
        k = fit / L0
        pts = [(n * k, e * k) for n, e in pts]
        pts = _dp(pts, 1.0)
        changed = True
        while changed and len(pts) > 2:          # 짧은 구간 없애기 (가운데 점부터)
            changed = False
            for i in range(1, len(pts) - 1):
                if min(math.dist(pts[i - 1], pts[i]), math.dist(pts[i], pts[i + 1])) < min_leg:
                    del pts[i]
                    changed = True
                    break
    return pts, dict(source=os.path.basename(fn), scale=round(k, 4), original_length_m=round(L0, 1),
                     start_lat=info.get("start_lat"), start_lon=info.get("start_lon"), decided_at=info.get("time"))


# ───────── 길 계산 ─────────
class Path:
    def __init__(self, pts):
        self.pts = [tuple(map(float, p)) for p in pts]
        self.cum = [0.0]
        for a, b in zip(self.pts, self.pts[1:]):
            self.cum.append(self.cum[-1] + math.dist(a, b))
        self.length = self.cum[-1]
        self.corners = self.cum[1:-1]

    def seg_dir(self, i):
        (n1, e1), (n2, e2) = self.pts[i], self.pts[i + 1]
        L = self.cum[i + 1] - self.cum[i]
        return (n2 - n1) / L, (e2 - e1) / L

    def point(self, s):
        """길 위 거리 s의 점과 진행 방향. 모퉁이 정확히 위면 들어오는 방향"""
        s = max(0.0, min(s, self.length))
        for i in range(len(self.pts) - 1):
            if s <= self.cum[i + 1] or i == len(self.pts) - 2:
                u = self.seg_dir(i)
                k = s - self.cum[i]
                n1, e1 = self.pts[i]
                return (n1 + u[0] * k, e1 + u[1] * k), u
        raise ValueError

    def point_ext(self, s):
        """길 밖으로도 곧게 연장 (출발 전은 첫 방향, 도착 뒤는 끝 방향)"""
        if s > self.length:
            (n, e), u = self.point(self.length)
            k = s - self.length
            return (n + u[0] * k, e + u[1] * k), u
        if s < 0:
            (n, e), u = self.point(0.0)
            return (n + u[0] * s, e + u[1] * s), u
        return self.point(s)

    def ahead(self, s0, r):
        """s0 점에서 길을 따라 앞으로 가다가 s0 점과 직선거리가 처음 r이 되는 곳의 길 위 거리"""
        p0, _ = self.point_ext(s0)
        s = s0
        while math.dist(self.point_ext(s + 0.1)[0], p0) < r:
            s += 0.1
            if s - s0 > 4 * r + 50:
                return s0 + r
        lo, hi = s, s + 0.1
        for _ in range(30):
            mid = (lo + hi) / 2
            if math.dist(self.point_ext(mid)[0], p0) >= r:
                hi = mid
            else:
                lo = mid
        return hi

    def drone_spot(self, s_spot, r):
        """빛 자국이 s_spot일 때 드론이 떠 있을 곳 (빛 자국에서 직선거리 r).
        곧은 길: 빛 자국에서 길 따라 r 앞.
        모퉁이 r 안쪽: 빛 자국을 중심으로 반지름 r 원 위에서 들어오는 방향 → 나가는 방향으로
        빛 자국이 모퉁이에 다가가는 만큼 고르게 돌림 → 기수 회전이 초당 약 9°로 고르게 (실기체 제한 20°)
        드론은 길에서 최대 약 1m 바깥으로 살짝 둥글게 돎"""
        P, _ = self.point(s_spot)
        for i, c in enumerate(self.corners):
            y = c - s_spot
            if 0.0 <= y < r:
                u_in, u_out = self.seg_dir(i), self.seg_dir(i + 1)
                phi = math.atan2(u_in[0] * u_out[1] - u_in[1] * u_out[0], u_in[0] * u_out[0] + u_in[1] * u_out[1])
                th = phi * (1.0 - y / r)
                c_, s_ = math.cos(th), math.sin(th)
                d = (u_in[0] * c_ - u_in[1] * s_, u_in[0] * s_ + u_in[1] * c_)
                return (P[0] + r * d[0], P[1] + r * d[1])
        return self.point_ext(self.ahead(s_spot, r))[0]

    def aim_s(self, d, r, s_ref, span=6.0):
        """점 d에서 직선거리 r인 길 위 점들 중 s_ref에 가장 가까운 것의 길 위 거리 (없으면 None)"""
        f = lambda x: math.dist(self.point_ext(x)[0], d) - r
        best = None
        x, prev = s_ref - span, f(s_ref - span)
        while x < s_ref + span:
            x2 = x + 0.1
            cur = f(x2)
            if prev * cur <= 0:
                lo, hi = x, x2
                for _ in range(25):
                    mid = (lo + hi) / 2
                    if f(lo) * f(mid) <= 0:
                        hi = mid
                    else:
                        lo = mid
                root = (lo + hi) / 2
                if best is None or abs(root - s_ref) < abs(best - s_ref):
                    best = root
            x, prev = x2, cur
        return best

    def project(self, p):
        """점 p를 길에 투영 → (길 위 거리 s, 옆으로 벗어난 거리 d, +는 진행 방향 오른쪽)"""
        best = None
        for i in range(len(self.pts) - 1):
            n1, e1 = self.pts[i]
            u = self.seg_dir(i)
            L = self.cum[i + 1] - self.cum[i]
            k = max(0.0, min(L, (p[0] - n1) * u[0] + (p[1] - e1) * u[1]))
            q = n1 + u[0] * k, e1 + u[1] * k
            dist = math.dist(p, q)
            side = (p[1] - q[1]) * u[0] - (p[0] - q[0]) * u[1]
            if best is None or dist < best[1] - 1e-9:
                best = (self.cum[i] + k, dist, 1.0 if side >= 0 else -1.0)
        return best[0], best[1] * best[2]


# ───────── 가상의 사람 (시뮬레이터 시험용) ─────────
class VirtualPerson:
    """평소엔 길을 따라 걷고, 깜빡임을 보면 빛 자국까지 걸어가서 멈춤"""
    def __init__(self, path):
        self.path = path
        self.p = path.pts[0]
        self.rest_until = None
        self.rest_done = False
        self.drift_done = False
        self.state = "걷는 중"

    def update(self, t, dt, signal, spot):
        s, d = self.path.project(self.p)
        if s >= self.path.length - 0.3 and abs(d) < OFFPATH_BACK:
            self.state = "도착"
            return
        if getattr(self, "last_signal", None) == BLINK and signal == ON:
            self.go_at = t + REACTION
        self.last_signal = signal
        if signal == ON and t < getattr(self, "go_at", 0.0):
            self.state = "멈춤(반응 중)"
            return
        if signal == BLINK:
            if abs(d) > OFFPATH_M * 0.5:
                self.drift_done = True
            dist = math.dist(self.p, spot)
            if dist > 0.35:
                k = min(dist - 0.3, WALK * dt) / dist
                self.p = (self.p[0] + (spot[0] - self.p[0]) * k, self.p[1] + (spot[1] - self.p[1]) * k)
                self.state = "빛 자국으로 감"
            else:
                self.state = "멈춤"
            return
        if not self.rest_done and s >= REST_S:
            if self.rest_until is None:
                self.rest_until = t + REST_LEN
            if t < self.rest_until:
                self.state = "쉬는 중"
                return
            self.rest_done = True
        drifting = s >= DRIFT_S and not self.drift_done
        s2 = min(self.path.length, s + WALK * dt)
        d2 = d + DRIFT_SPEED * dt if drifting else d - math.copysign(min(abs(d), 0.1 * dt), d)
        (n, e), u = self.path.point(s2)
        self.p = (n - u[1] * d2, e + u[0] * d2)
        self.state = "걷는 중(길 벗어나는 중)" if drifting else "걷는 중"


# ───────── 기체 연결·상태 ─────────
class Vehicle:
    def __init__(self, addr, baud):
        print("연결 중:", addr)
        self.m = mavutil.mavlink_connection(addr, baud=baud, source_system=255, source_component=191)
        self.last_hb_sent = 0.0
        self.last_hb_recv = 0.0
        self.mode = None
        self.armed = False
        self.ned = None
        self.rel_alt = None
        self.yaw = None
        self.relay_ok = self.relay_fail = 0

    def wait_vehicle(self, timeout=30):
        t_end = time.time() + timeout
        while time.time() < t_end:
            self.heartbeat()
            hb = self.m.recv_match(type="HEARTBEAT", blocking=True, timeout=1)
            if hb and hb.type != mavutil.mavlink.MAV_TYPE_GCS and \
                    hb.autopilot != mavutil.mavlink.MAV_AUTOPILOT_INVALID:
                self.m.target_system, self.m.target_component = hb.get_srcSystem(), hb.get_srcComponent()
                self.last_hb_recv = time.time()
                return hb
        return None

    def heartbeat(self):
        now = time.time()
        if now - self.last_hb_sent >= 1.0:
            self.m.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
            self.last_hb_sent = now

    def poll(self):
        self.heartbeat()
        while True:
            msg = self.m.recv_match(blocking=False)
            if msg is None:
                break
            typ = msg.get_type()
            if msg.get_srcSystem() != self.m.target_system:
                continue
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
                        print(f"  ! 빛(릴레이) 명령 거절됨 (결과 {msg.result}) — RELAY_PIN 설정 확인")
            elif typ == "STATUSTEXT":
                txt = msg.text if isinstance(msg.text, str) else msg.text.decode(errors="ignore")
                print("  기체:", txt.strip("\x00"))

    def link_alive(self):
        return time.time() - self.last_hb_recv < 3.0

    def set_mode(self, name):
        self.m.set_mode(self.m.mode_mapping()[name])

    def wait_mode(self, name, timeout=5):
        t_end = time.time() + timeout
        while time.time() < t_end:
            self.poll()
            if self.mode == name:
                return True
            time.sleep(0.1)
        return False

    def relay(self, num, on):
        self.m.mav.command_long_send(self.m.target_system, self.m.target_component,
                                     mavutil.mavlink.MAV_CMD_DO_SET_RELAY, 0, num, 1 if on else 0, 0, 0, 0, 0, 0)

    def target(self, n, e, alt, vn, ve, yaw):
        self.m.mav.set_position_target_local_ned_send(
            0, self.m.target_system, self.m.target_component, mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            POS_VEL_YAW, n, e, -alt, vn, ve, 0, 0, 0, 0, yaw, 0)


class Light:
    """빛 신호 → 릴레이 켜기/끄기. 바뀔 때만 명령을 보냄"""
    def __init__(self, veh, relay_num):
        self.veh, self.num = veh, relay_num
        self.signal = OFF
        self.relay_on = None
        self.sent = 0

    def set(self, signal):
        self.signal = signal

    def update(self, now):
        want = (int(now * BLINK_HZ * 2) % 2 == 0) if self.signal == BLINK else (self.signal == ON)
        if want != self.relay_on:
            self.veh.relay(self.num, want)
            self.relay_on = want
            self.sent += 1


def is_local(addr):
    return addr.startswith(("tcp:127.", "tcp:localhost", "udp:127.", "udpin:127.", "udpin:0.0.0.0"))


# ───────── 땅에서 빛만 시험 ─────────
def light_test(veh):
    veh.poll()
    if veh.armed:
        print("시동이 걸려 있어 빛 시험을 하지 않습니다.")
        return
    print(f"빛 시험: 릴레이 {LIGHT_RELAY}번 (실기체 AUX OUT {5 + LIGHT_RELAY}). 시동은 걸지 않습니다.")
    if input("AUX 단자에 지금 무엇이 연결돼 있는지 확인했고, 프로펠러를 뺐으면 y: ").strip().lower() != "y":
        print("취소했습니다.")
        return
    light = Light(veh, LIGHT_RELAY)
    for sig, sec in ((ON, 5), (BLINK, 6), (OFF, 2)):
        print(f"  {sig} ({MEANING[sig]}) {sec}초")
        light.set(sig)
        t_end = time.time() + sec
        while time.time() < t_end:
            veh.poll()
            light.update(time.time())
            time.sleep(0.05)
    time.sleep(1.0)
    veh.poll()
    print(f"릴레이 명령 {light.sent}번 보냄 / 수락 {veh.relay_ok} / 거절 {veh.relay_fail}")


def watch_until_disarmed(veh, max_s=600):
    """명령은 안 보내고, 시동이 꺼질 때까지 하트비트만 보내며 지켜봄"""
    print("기체 시동이 꺼질 때까지 연결 확인 신호를 계속 보냅니다 (프로그램을 끄지 마세요)")
    last_mode = veh.mode
    t_end = time.time() + max_s
    try:
        while time.time() < t_end:
            veh.poll()
            if veh.mode != last_mode:
                print(f"  모드: {veh.mode}")
                last_mode = veh.mode
            if not veh.armed:
                print("시동 꺼짐 → 프로그램 끝")
                return
            if not veh.link_alive():
                print("! 기체 신호 끊김 → 프로그램 끝")
                return
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\n지켜보기 중단 (하트비트 멈춤 → 기체가 '지상국 끊김'으로 귀환할 수 있음)")


# ───────── 안내 비행 ─────────
def takeoff(veh):
    veh.m.mav.request_data_stream_send(veh.m.target_system, veh.m.target_component,
                                       mavutil.mavlink.MAV_DATA_STREAM_ALL, 4, 1)
    veh.set_mode("GUIDED")
    if not veh.wait_mode("GUIDED"):
        print("GUIDED 모드로 못 바꿈 — Mission Planner 메시지 확인")
        return False
    for _ in range(40):
        veh.m.arducopter_arm()
        t_end = time.time() + 3
        while time.time() < t_end and not veh.armed:
            veh.poll()
            time.sleep(0.1)
        if veh.armed:
            break
        print("시동 대기 중 (GPS·센서 준비 중)...")
    if not veh.armed:
        print("시동 실패: 위의 '기체: PreArm' 문구를 확인하세요")
        return False
    print("시동 걸림 → 이륙")
    veh.m.mav.command_long_send(veh.m.target_system, veh.m.target_component,
                                mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0, ALT)
    t_end = time.time() + 60
    while time.time() < t_end:
        veh.poll()
        if veh.mode != "GUIDED":
            print(f"조종자가 넘겨받음 ({veh.mode}) → 프로그램 명령 멈춤")
            watch_until_disarmed(veh)
            return False
        if veh.rel_alt is not None and veh.rel_alt > ALT * 0.9 and veh.ned is not None:
            return True
        time.sleep(0.1)
    print("60초 안에 목표 고도에 못 올라감 → 착륙")
    veh.set_mode("LAND")
    watch_until_disarmed(veh)
    return False


def fly(veh, log_path):
    path = Path(PATH)
    if not takeoff(veh):
        return
    # 출발 배치: 사람은 길 출발점, 빛 자국은 3m 앞, 드론은 빛 자국에서 직선 R_SPOT 앞 = 이륙한 자리
    (sn, se), _ = path.point(SPOT_AHEAD)
    cmd = path.drone_spot(SPOT_AHEAD, R_SPOT)
    n0, e0 = veh.ned[0] - cmd[0], veh.ned[1] - cmd[1]
    print(f"이륙 완료 ({veh.rel_alt:.1f}m) → 안내 시작. 길 {path.length:.0f}m, 꺾이는 곳 {len(path.corners)}개. "
          f"사람은 이륙 자리 {math.dist(cmd, path.pts[0]):.1f}m 뒤에서 출발, 빛 자국은 드론 앞 {R_SPOT:.1f}m")

    # 준비: 드론이 빛 자국(사람 쪽)을 바라볼 때까지 제자리에서 돎 → 그 뒤에 사람 출발
    yaw0 = math.atan2(se - cmd[1], sn - cmd[0])
    t_end = time.time() + 20
    print("준비: 드론이 사람 쪽으로 도는 중...")
    while time.time() < t_end:
        veh.poll()
        if veh.mode != "GUIDED":
            print(f"조종자가 넘겨받음 ({veh.mode}) → 프로그램 명령 멈춤")
            watch_until_disarmed(veh)
            return
        veh.target(n0 + cmd[0], e0 + cmd[1], ALT, 0, 0, yaw0)
        if veh.yaw is not None and abs((veh.yaw - math.degrees(yaw0) + 540) % 360 - 180) < 5:
            break
        time.sleep(LOOP_DT)

    person = VirtualPerson(path)
    light = Light(veh, LIGHT_RELAY)
    s_spot = SPOT_AHEAD
    done_corners = set()
    phase = FOLLOW
    stop_since = None
    hist = collections.deque()
    events = []
    t0 = last = time.time()
    rows = []
    end_reason = "시간 초과"
    signal = ON
    spot_pt = (sn, se)
    try:
        while True:
            now = time.time()
            t, dt = now - t0, max(now - last, 1e-3)
            last = now
            veh.poll()
            if not veh.link_alive():
                end_reason = "기체 신호 끊김"
                print("! 기체 신호가 3초 넘게 안 옴 → 프로그램 멈춤 (기체는 자체 안전 기능으로 동작)")
                break
            if veh.mode != "GUIDED":
                end_reason = f"조종자가 넘겨받음 ({veh.mode})"
                print(f"! 모드가 {veh.mode}(으)로 바뀜 → 조종자가 넘겨받음, 프로그램 명령 멈춤")
                light.set(ON)
                light.update(now)
                break
            if t > MAX_TIME:
                break

            # ── 사람 (지금은 가상의 사람) ──
            person.update(t, dt, signal, spot_pt)
            pn, pe = person.p
            s_p, d_p = path.project((pn, pe))
            hist.append((now, pn, pe))
            while hist and now - hist[0][0] > 1.0:
                hist.popleft()
            speed = math.dist(hist[0][1:], hist[-1][1:]) / max(now - hist[0][0], 1e-3) if len(hist) > 2 else 1.0
            dn, de = (veh.ned[0] - n0, veh.ned[1] - e0)
            gap = math.hypot(dn - pn, de - pe)

            # ── 단계 정하기 ──
            if phase != OFFPATH_P and abs(d_p) > OFFPATH_M:
                prev_phase = phase
                phase = OFFPATH_P
                events.append((t, "길 이탈", round(abs(d_p), 2)))
                print(f"  → 경로 이탈 {abs(d_p):.1f}m: 빛 자국 그 자리에서 깜빡임, 드론 기다림")
            elif phase == OFFPATH_P and abs(d_p) < OFFPATH_BACK:
                phase = prev_phase if prev_phase == HOLD else FOLLOW
                events.append((t, "길로 돌아옴", 0))
                print("  → 길로 돌아옴: 다시 켜짐")

            pending = [c for c in path.corners if c not in done_corners]
            next_c = pending[0] if pending else None
            if phase == FOLLOW:
                want = s_p + SPOT_AHEAD
                if gap > MAX_GAP:
                    want = s_spot                     # 너무 멀면 기다림
                # 빛 자국은 앞으로만, 초당 MAX_SPEED보다 빨리 안 감 (모퉁이 출발 때 갑자기 튀지 않게)
                s_spot = min(s_spot + MAX_SPEED * dt, max(s_spot, min(want, path.length)))
                if next_c is not None and s_spot >= next_c:
                    s_spot = next_c
                    phase = HOLD
                    stop_since = None
                    events.append((t, "모퉁이 대기 시작", next_c))
                    print(f"  → 빛 자국이 모퉁이({next_c:.0f}m)에 닿음: 깜빡임, 모퉁이까지 와서 멈추세요")
                if s_p >= path.length - 0.3 and abs(d_p) < OFFPATH_BACK:
                    phase = ARRIVED
            elif phase == HOLD:
                c_pt, _ = path.point(next_c)
                at_corner = math.dist((pn, pe), c_pt) <= ARRIVE_R
                if at_corner and speed < STOP_SPEED:
                    stop_since = stop_since or now
                    if now - stop_since >= STOP_CONFIRM:
                        done_corners.add(next_c)
                        phase = FOLLOW
                        events.append((t, "모퉁이 출발", next_c))
                        print("  → 사람이 모퉁이에서 멈춤: 다시 켜짐, 빛 자국이 꺾인 길로 앞서감")
                else:
                    stop_since = None

            # ── 빛 자국 목표점 · 드론 목표 ──
            # 드론 자리 = 빛 자국에서 직선거리 R_SPOT (곧은 길: 길 따라 앞, 모퉁이: 새 길 쪽으로 고르게 돌림)
            spot_pt, _ = path.point(s_spot)
            goal = path.drone_spot(s_spot, R_SPOT)
            if phase == ARRIVED:                 # 도착: 사람과 거리를 더 두고 내림
                goal, _ = path.point_ext(path.length + R_SPOT + SPOT_AHEAD)
            vel = (0.0, 0.0)
            dx, dy = goal[0] - cmd[0], goal[1] - cmd[1]
            dist = math.hypot(dx, dy)
            step = min(dist, MAX_SPEED * dt)
            if dist > 1e-6:
                cmd = (cmd[0] + dx / dist * step, cmd[1] + dy / dist * step)
                vel = (dx / dist * step / dt, dy / dist * step / dt)
            # 바라볼 곳: 보통은 빛 자국 목표점.
            # 드론이 목표보다 많이 늦어 빛 자국 거리가 0.5m 넘게 어긋나면(모퉁이에서 다시 출발할 때 등)
            # 지금 자리·고도에서 빛 자국이 실제로 떨어질 길 위 점을 바라봄 → 빛 자국이 길 밖으로 안 나감
            alt_now = veh.rel_alt if veh.rel_alt is not None else ALT
            r_now = max(alt_now, 0.5) * math.tan(math.radians(LED_TILT_DEG))
            aim = spot_pt
            if abs(math.dist((dn, de), spot_pt) - r_now) > 0.5:
                s_aim = path.aim_s((dn, de), r_now, s_spot)
                if s_aim is not None:
                    aim = path.point_ext(s_aim)[0]
            if math.hypot(*cmd) > MAX_RANGE:     # 소프트웨어 울타리
                print("! 울타리(MAX_RANGE) 밖 목표 → 기다림")
                cmd = (dn, de)
                vel = (0.0, 0.0)
            yaw = math.atan2(aim[1] - de, aim[0] - dn)
            veh.target(n0 + cmd[0], e0 + cmd[1], ALT, vel[0], vel[1], yaw)

            signal = ON if phase in (FOLLOW, ARRIVED) else BLINK
            light.set(signal)
            light.update(now)

            # ── 빛 자국 실제 위치 (드론 위치 + 기수 방향 + 고도로 계산) ──
            alt = veh.rel_alt if veh.rel_alt is not None else ALT
            yr = math.radians(veh.yaw) if veh.yaw is not None else yaw
            r = max(alt, 0.0) * math.tan(math.radians(LED_TILT_DEG))
            spot_real = (dn + r * math.cos(yr), de + r * math.sin(yr))
            _, spot_off = path.project(spot_real)
            spot_ahead = path.project(spot_real)[0] - s_p

            rows.append(dict(t=round(t, 2), phase=phase, signal=signal, person_state=person.state,
                             person_n=round(pn, 2), person_e=round(pe, 2), person_s=round(s_p, 2),
                             offset=round(d_p, 2), drone_n=round(dn, 2), drone_e=round(de, 2),
                             alt=round(alt, 2), gap=round(gap, 2),
                             spot_s=round(s_spot, 2), spot_tn=round(spot_pt[0], 2), spot_te=round(spot_pt[1], 2),
                             spot_n=round(spot_real[0], 2), spot_e=round(spot_real[1], 2),
                             spot_err=round(math.dist(spot_real, spot_pt), 2), spot_off=round(spot_off, 2),
                             spot_ahead=round(spot_ahead, 2), relay=int(bool(light.relay_on)),
                             yaw_cmd=round(math.degrees(yaw), 1),
                             yaw=None if veh.yaw is None else round(veh.yaw, 1), mode=veh.mode))
            if len(rows) % 4 == 0:
                print(f"{t:6.1f}s | {phase} | 사람 {s_p:5.1f}m ({person.state}) 옆 {d_p:+.1f}m | 드론 거리 {gap:4.1f}m | "
                      f"빛 자국 사람 앞 {spot_ahead:+.1f}m, 길에서 {abs(spot_off):.1f}m | 빛 {signal}")

            if phase == ARRIVED and math.dist(cmd, goal) < 0.3 and math.hypot(dn - goal[0], de - goal[1]) < 1.0:
                end_reason = "사람 도착"
                break
            time.sleep(LOOP_DT)
    except KeyboardInterrupt:
        end_reason = "Ctrl+C (그 자리 정지)"
        veh.set_mode("BRAKE")
        print("\nCtrl+C → BRAKE(그 자리 정지). 조종자가 넘겨받으세요.")

    if end_reason in ("사람 도착", "시간 초과"):
        light.set(ON)
        light.update(time.time())
        veh.set_mode("LAND")
        print(f"{end_reason} → 착륙 명령 보냄")
    if end_reason != "기체 신호 끊김":
        watch_until_disarmed(veh)

    if rows:
        with open(log_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        with open(log_path.replace(".csv", "_path.json"), "w", encoding="utf-8") as f:
            json.dump(dict(version="v4.1", path=PATH, path_info=PATH_INFO, offpath_m=OFFPATH_M, spot_ahead=SPOT_AHEAD, alt=ALT,
                           r_spot=round(R_SPOT, 2), led_tilt=LED_TILT_DEG, corners=path.corners,
                           events=events, end_reason=end_reason), f, ensure_ascii=False)
        summary(rows, events, light, veh, end_reason, log_path)


def summary(rows, events, light, veh, end_reason, log_path):
    print("\n── 요약 ──")
    print(f"끝난 이유: {end_reason}, 안내 {rows[-1]['t']:.0f}초, 사람 {rows[-1]['person_s']:.1f}m 걸음")
    gaps = [r["gap"] for r in rows]
    walk = [r for r in rows if r["phase"] == FOLLOW and r["person_state"] == "걷는 중"]
    walk_spot = [r for r in walk if r["spot_s"] < Path(PATH).length - 0.01]   # 도착점에 닿은 뒤는 뺌
    print(f"사람-드론 거리: 최소 {min(gaps):.1f}m" +
          (f", 곧은 길 걷는 동안 {min(r['gap'] for r in walk):.1f}~{max(r['gap'] for r in walk):.1f}m" if walk else ""))
    if walk_spot:
        off = sorted(abs(r["spot_off"]) for r in walk_spot)
        ah = [r["spot_ahead"] for r in walk_spot]
        print(f"빛 자국 (따라가는 동안): 사람 앞 {min(ah):.1f}~{max(ah):.1f}m (목표 {SPOT_AHEAD:.0f}m), "
              f"길에서 벗어난 거리 중앙값 {off[len(off) // 2]:.2f}m / 95% {off[int(len(off) * .95)]:.2f}m / 최대 {off[-1]:.2f}m")
    hold = [r for r in rows if r["phase"] == HOLD]
    if hold:
        print(f"빛 자국 (모퉁이 대기 중): 모퉁이에서 최대 {max(r['spot_err'] for r in hold):.2f}m")
    starts = [e for e in events if e[1] == "모퉁이 대기 시작"]
    for k, e in enumerate(starts):
        t_go = next((x[0] for x in events if x[1] == "모퉁이 출발" and x[2] == e[2]), None)
        stood = [r["t"] for r in rows if r["phase"] == HOLD and r["person_state"] == "멈춤" and e[0] <= r["t"] <= (t_go or 1e9)]
        if t_go:
            print(f"모퉁이 {k + 1}: 깜빡임 {e[0]:.1f}초 → 다시 켜짐 {t_go:.1f}초"
                  + (f" (사람이 모퉁이에 서 있던 시간 약 {t_go - stood[0]:.1f}초)" if stood else ""))
    hold_gap = [r["gap"] for r in rows if r["phase"] == HOLD]
    if hold_gap:
        print(f"모퉁이 대기 중 사람-드론 거리 최소 {min(hold_gap):.1f}m")
    off_ev = next((e for e in events if e[1] == "길 이탈"), None)
    drift = next((r for r in rows if r["person_state"].startswith("걷는 중(길 벗어나")), None)
    if off_ev and drift:
        back = next((e for e in events if e[1] == "길로 돌아옴"), None)
        print(f"길 이탈: {drift['t']:.1f}초에 벗어나기 시작 → {off_ev[0]:.1f}초에 깜빡임 "
              f"(최대 {max(abs(r['offset']) for r in rows):.1f}m)" + (f" → {back[0]:.1f}초에 돌아옴" if back else ""))
    print(f"빛(릴레이) 명령: {light.sent}번 보냄 / 수락 {veh.relay_ok} / 거절 {veh.relay_fail}")
    print(f"기록 저장: {log_path}  → 그림: python gg_plot_v4.py {os.path.basename(log_path)}")


def main():
    ap = argparse.ArgumentParser(description="갯가드 빛길 안내 v4.1")
    ap.add_argument("addr", nargs="?", default="tcp:127.0.0.1:5762")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--light-test", action="store_true", help="시동 없이 빛(릴레이)만 시험")
    ap.add_argument("--real", action="store_true", help="실기체 비행 허용 (설정 수정·허락·안전 조종자 준비 뒤에만)")
    ap.add_argument("--path-json", help="gg_decide.py 탈출로 파일 (path_ne_m)을 길로 씀")
    ap.add_argument("--fit", type=float, help="탈출로 전체 길이를 이 m로 줄임 (모양 유지, 시연장 크기)")
    ap.add_argument("--range", type=float, help="울타리 거리(m) 바꾸기 — 시뮬레이터 연결에서만")
    a = ap.parse_args()

    global PATH, MAX_RANGE, MAX_TIME, REST_S, DRIFT_S, PATH_INFO
    if a.path_json:
        PATH, PATH_INFO = load_path_json(a.path_json, a.fit)
        L = sum(math.dist(p, q) for p, q in zip(PATH, PATH[1:]))
        legs = [math.dist(p, q) for p, q in zip(PATH, PATH[1:])]
        REST_S = 0.18 * L                                   # 기본 길(55m)에서 10m 지점 = 18%
        DRIFT_S = max(L - 11.0, L - 0.6 * legs[-1])         # 마지막 곧은 길에서 옆으로 벗어나기
        MAX_TIME = int(L / WALK * 1.35 + 30 * len(PATH) + 120)
        print(f"길: {a.path_json} → {len(PATH)}점, {L:.0f}m (원래 {PATH_INFO['original_length_m']:.0f}m, 배율 {PATH_INFO['scale']}), "
              f"최대 시간 {MAX_TIME}초")
    if a.range:
        if not is_local(a.addr) or a.real:
            print("울타리 거리는 시뮬레이터에서만 바꿀 수 있습니다.")
            return
        MAX_RANGE = a.range
        print(f"울타리 {MAX_RANGE:.0f}m (시뮬레이터)")
    far = max(math.hypot(n, e) for n, e in PATH)
    if far > MAX_RANGE - 5:
        print(f"길 끝이 출발점에서 {far:.0f}m → 울타리 {MAX_RANGE:.0f}m 안에 안 들어옵니다. --fit 으로 줄이거나 (시뮬레이터면) --range 를 늘리세요.")
        return

    if not a.light_test and not is_local(a.addr) and not a.real:
        print("실기체 연결로 보여 비행을 막았습니다. 땅에서 빛만 시험하려면 --light-test 를 붙이세요.")
        return
    if a.real and not a.light_test:
        print("실기체 비행 전 확인: 배터리·지상국 안전 설정 수정 / 관리자 허락 / 안전 조종자 조종기 대기(스로틀 가운데) / "
              "비행 구역 사람 통제 / 비행승인 여부 [법규 확인: 항공안전법 제127조]")
        if input("모두 확인했으면 '비행' 입력: ").strip() != "비행":
            print("취소했습니다.")
            return

    veh = Vehicle(a.addr, a.baud)
    if veh.wait_vehicle() is None:
        print("연결 실패: 시뮬레이터가 켜져 있는지, 주소·포트가 맞는지 확인하세요")
        return
    print(f"연결됨: 기체 번호 {veh.m.target_system}, 하트비트 255번으로 보내는 중")
    if a.light_test:
        light_test(veh)
        return
    fly(veh, time.strftime("gg_log_%Y%m%d_%H%M%S.csv"))


if __name__ == "__main__":
    main()
