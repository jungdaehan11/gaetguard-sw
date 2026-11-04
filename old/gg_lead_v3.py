"""갯가드 빛길 안내 v3 — 꺾인 길 + 빛 신호 3가지 + 안전장치 (시뮬레이터 시험용)

v2에서 바뀐 점
  1) 꺾인 길(PATH)을 따라 사람보다 "길 위 거리"로 7m 앞을 유지
  2) 빛 신호 3가지 (팀 계획표 그대로)
       켜짐      = 따라오세요
       깜빡임    = 멈추세요   (사람이 길에서 벗어났을 때)
       좌우 쓸기 = 방향 트세요 (사람이 꺾이는 곳 5m 앞에 왔을 때)
     빛은 릴레이 0번(실기체 RELAY_PIN 54 = AUX OUT 5)으로 켜고 끔
     ※ v2는 "너무 멀어짐"에도 깜빡였는데, 계획표 뜻에 맞게 "길 벗어남"에만 깜빡이도록 바꿈
  3) 드론은 사람 쪽을 바라봄 → 앞으로 기울여 단 LED가 사람 바로 앞 바닥을 비춤
     좌우 쓸기는 기체를 좌우로 천천히 돌려서 빛 자국을 움직임
  4) 연결 확인 신호(하트비트)를 지상국 번호 255로 1초마다 보냄
     → 이 프로그램이 죽으면 기체가 "지상국 끊김"으로 보고 귀환 (실기체 FS_GCS_ENABLE=1)
  5) 조종자가 모드 스위치를 돌리면(GUIDED가 아니게 되면) 바로 명령을 멈춤
  6) Ctrl+C 누르면 그 자리에 멈춤(BRAKE) → 조종자가 넘겨받음
  7) 비행 기록을 CSV로 저장 → gg_plot_v3.py 로 그림
  8) 사람은 이륙 자리 7m 뒤에서 출발 (드론이 사람 바로 위에서 뜨지 않게)
  9) 안내가 끝나도 기체 시동이 꺼질 때까지 하트비트를 계속 보냄
     (먼저 끄면 착륙·조종 중에 "지상국 끊김"이 걸려 귀환해 버릴 수 있음 — 시뮬레이터 4.5에서 확인)

실행
  python gg_lead_v3.py                         # 시뮬레이터 (기본 tcp:127.0.0.1:5762)
  python gg_lead_v3.py tcp:127.0.0.1:5763      # 5762가 안 되면
  python gg_lead_v3.py COM5 --light-test       # 땅에서 빛만 시험: 시동 안 걺 (프로펠러 빼고, USB만)

주의
  - 실기체 비행은 막아 둠. 설정 수정·관리자 허락·안전 조종자 준비 뒤에만 --real 로 풀 것
  - 사람 위치는 지금 "가상의 사람"(VirtualPerson). 실제 시연에서 사람 위치를 어떻게 알지는 아직 정하지 않음
"""
import argparse
import csv
import json
import math
import os
import sys
import time

os.environ.setdefault("MAVLINK20", "1")       # MAVLink2로 주고받기 (MAVLink1도 읽음)
from pymavlink import mavutil                 # noqa: E402

# ───────── 설정 (현장에 맞게 바꾸는 곳) ─────────
ALT = 7.0            # 비행 고도 (m) — 계획 5~10m
LEAD = 7.0           # 사람보다 길 위로 앞설 거리 (m) — 계획 5~10m
MAX_GAP = 12.0       # 사람과 이보다 멀어지면 드론이 기다림 (m)
MIN_GAP = 5.0        # 꺾이는 곳 등에서 이보다 가까워지면 목표점을 더 앞으로 (m) — 계획 5~10m의 아래 끝
WALK = 0.5           # 가상 사람 걸음 (m/s) — 밤 갯벌 보행 가정 [확인 필요]
MAX_SPEED = 1.5      # 목표점이 움직이는 최대 속도 (m/s)
OFFPATH_M = 2.0      # 길 중심에서 이보다 벗어나면 "경로 이탈" (m) — 안전 길 폭 4m 가정 [확인 필요]
OFFPATH_BACK = 1.0   # 이 안으로 돌아오면 이탈 해제 (m)
SWEEP_BEFORE = 5.0   # 꺾이는 곳 몇 m 앞부터 좌우 쓸기 (m)
SWEEP_DEG = 20.0     # 좌우 쓸기 폭 (±도)
SWEEP_RATE = 18.0    # 좌우 쓸기 속도 (도/초) — 실기체 ATC_SLEW_YAW 2000(=20도/초)보다 느리게
BLINK_HZ = 1.0       # 깜빡임 횟수 (초당)
LIGHT_RELAY = 0      # 0번 = RELAY_PIN(54, AUX OUT 5) [확인 필요: AUX 5에 지금 연결된 장치]
MAX_RANGE = 60.0     # 출발점에서 이보다 멀리 안 감 (m) — 소프트웨어 울타리
LOOP_DT = 0.25       # 계산 주기 (s)
MAX_TIME = 240       # 안내 최대 시간 (s)

# 길: 출발점 기준 (북쪽 m, 동쪽 m). 북 25m → 동 15m → 북 15m (총 55m)
PATH = [(0.0, 0.0), (25.0, 0.0), (25.0, 15.0), (40.0, 15.0)]

# 가상 사람 시나리오 (시험용)
REST_AT, REST_LEN = 20.0, 8.0     # 20초에 8초 쉼 → 드론도 그 자리에서 기다리는지
DRIFT_AT = 95.0                   # 95초(마지막 곧은 길)에 옆으로 벗어나기 시작 → 깜빡임이 켜지는지
DRIFT_SPEED = 0.3                 # 벗어나는 속도 (m/s, 옆으로)
RETURN_SPEED = 0.3                # 깜빡임을 보고 길로 돌아오는 속도 (m/s)

# SET_POSITION_TARGET_LOCAL_NED: 위치 + 속도 + 방향(yaw) 사용, 가속도·회전속도 무시
POS_VEL_YAW = 0b0000100111000000

ON, BLINK, SWEEP, OFF = "켜짐", "깜빡임", "좌우 쓸기", "꺼짐"
MEANING = {ON: "따라오세요", BLINK: "멈추세요", SWEEP: "방향 트세요", OFF: "-"}


# ───────── 길 계산 ─────────
class Path:
    def __init__(self, pts, extend):
        self.pts = [tuple(map(float, p)) for p in pts]
        # 드론은 도착점 너머로 extend m 더 갈 수 있게 (사람이 도착해도 드론 밑으로 안 들어가게)
        (n1, e1), (n2, e2) = self.pts[-2], self.pts[-1]
        L = math.hypot(n2 - n1, e2 - e1)
        self.drone_pts = self.pts + [(n2 + (n2 - n1) / L * extend, e2 + (e2 - e1) / L * extend)]
        self.cum = [0.0]
        for a, b in zip(self.drone_pts, self.drone_pts[1:]):
            self.cum.append(self.cum[-1] + math.dist(a, b))
        self.length = self.cum[len(self.pts) - 1]          # 사람 도착점까지 길이
        self.corners = self.cum[1:len(self.pts) - 1]       # 꺾이는 곳 (길 위 거리)

    def point(self, s):
        s = max(0.0, min(s, self.cum[-1]))
        for i in range(len(self.drone_pts) - 1):
            if s <= self.cum[i + 1] or i == len(self.drone_pts) - 2:
                (n1, e1), (n2, e2) = self.drone_pts[i], self.drone_pts[i + 1]
                L = self.cum[i + 1] - self.cum[i]
                u = (n2 - n1) / L, (e2 - e1) / L
                k = s - self.cum[i]
                return (n1 + u[0] * k, e1 + u[1] * k), u
        raise ValueError

    def project(self, p):
        """점 p를 사람 길(도착점까지)에 투영 → (길 위 거리 s, 옆으로 벗어난 거리 d, +는 진행 방향 오른쪽)"""
        best = None
        for i in range(len(self.pts) - 1):
            (n1, e1), (n2, e2) = self.pts[i], self.pts[i + 1]
            L = self.cum[i + 1] - self.cum[i]
            u = (n2 - n1) / L, (e2 - e1) / L
            k = max(0.0, min(L, (p[0] - n1) * u[0] + (p[1] - e1) * u[1]))
            q = n1 + u[0] * k, e1 + u[1] * k
            dist = math.dist(p, q)
            side = (p[1] - q[1]) * u[0] - (p[0] - q[0]) * u[1]     # 오른쪽(+)
            if best is None or dist < best[1] - 1e-9:
                best = (self.cum[i] + k, dist, 1.0 if side >= 0 else -1.0)
        return best[0], best[1] * best[2]


# ───────── 가상의 사람 (시뮬레이터 시험용) ─────────
class VirtualPerson:
    def __init__(self, path):
        self.path = path
        self.s, self.d = 0.0, 0.0
        self.drift_done = False
        self.state = "걷는 중"

    def pos(self):
        (n, e), u = self.path.point(min(self.s, self.path.length))
        return n - u[1] * self.d, e + u[0] * self.d       # 오른쪽 수직 = (-u_e, u_n)

    def update(self, t, dt, light):
        resting = REST_AT <= t < REST_AT + REST_LEN
        drifting = t >= DRIFT_AT and not self.drift_done
        if light == BLINK:                     # 멈추세요 → 앞으로 안 가고 길 쪽으로 돌아옴
            self.drift_done = True
            self.d -= math.copysign(min(abs(self.d), RETURN_SPEED * dt), self.d)
            self.state = "멈춤·길로 돌아옴"
        elif resting:
            self.state = "쉬는 중"
        elif self.s >= self.path.length:
            self.state = "도착"
        else:
            self.s = min(self.path.length, self.s + WALK * dt)
            if drifting:
                self.d += DRIFT_SPEED * dt
                self.state = "걷는 중(길 벗어나는 중)"
            else:
                self.d -= math.copysign(min(abs(self.d), 0.1 * dt), self.d)   # 빛 따라 길 가운데로
                self.state = "걷는 중"


# ───────── 기체 연결·상태 ─────────
class Vehicle:
    def __init__(self, addr, baud):
        print("연결 중:", addr)
        self.m = mavutil.mavlink_connection(addr, baud=baud, source_system=255, source_component=191)
        self.last_hb_sent = 0.0
        self.last_hb_recv = 0.0
        self.mode = None
        self.armed = False
        self.ned = None               # (북, 동, 아래)
        self.rel_alt = None
        self.yaw = None               # 실제 기수 방향 (도)
        self.relay_ok = self.relay_fail = 0
        self.statustexts = []

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
        """지상국(255번) 연결 확인 신호, 1초마다"""
        now = time.time()
        if now - self.last_hb_sent >= 1.0:
            self.m.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
            self.last_hb_sent = now

    def poll(self):
        """쌓인 메시지를 모두 읽어 상태를 갱신"""
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
                txt = txt.strip("\x00")
                self.statustexts.append(txt)
                print("  기체:", txt)

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
        if self.signal == BLINK:
            want = int(now * BLINK_HZ * 2) % 2 == 0
        else:
            want = self.signal in (ON, SWEEP)
        if want != self.relay_on:
            self.veh.relay(self.num, want)
            self.relay_on = want
            self.sent += 1


def tri(t, amp, rate):
    """삼각파: ±amp를 rate(도/초)로 오감"""
    period = 4 * amp / rate
    x = (t % period) / period
    return amp * (4 * x - 1 if x < 0.5 else 3 - 4 * x)


def is_local(addr):
    return addr.startswith(("tcp:127.", "tcp:localhost", "udp:127.", "udpin:127.", "udpin:0.0.0.0"))


# ───────── 땅에서 빛만 시험 ─────────
def light_test(veh):
    veh.poll()
    if veh.armed:
        print("시동이 걸려 있어 빛 시험을 하지 않습니다.")
        return
    print(f"빛 시험: 릴레이 {LIGHT_RELAY}번 (실기체 AUX OUT {5 + LIGHT_RELAY}). 시동은 걸지 않습니다.")
    ok = input("AUX 단자에 지금 무엇이 연결돼 있는지 확인했고, 프로펠러를 뺐으면 y: ").strip().lower()
    if ok != "y":
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


# ───────── 안내 비행 ─────────
def fly(veh, log_path):
    path = Path(PATH, LEAD)
    veh.m.mav.request_data_stream_send(veh.m.target_system, veh.m.target_component,
                                       mavutil.mavlink.MAV_DATA_STREAM_ALL, 4, 1)
    veh.set_mode("GUIDED")
    if not veh.wait_mode("GUIDED"):
        print("GUIDED 모드로 못 바꿈 — Mission Planner 메시지 확인")
        return
    # 시동: GPS·센서가 준비될 때까지 재시도
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
        return
    print("시동 걸림 → 이륙")
    veh.m.mav.command_long_send(veh.m.target_system, veh.m.target_component,
                                mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0, ALT)
    t_end = time.time() + 60
    while time.time() < t_end:
        veh.poll()
        if veh.mode != "GUIDED":
            print(f"조종자가 넘겨받음 ({veh.mode}) → 프로그램 명령 멈춤")
            return
        if veh.rel_alt is not None and veh.rel_alt > ALT * 0.9:
            break
        time.sleep(0.1)
    else:
        print("60초 안에 목표 고도에 못 올라감 → 착륙")
        veh.set_mode("LAND")
        watch_until_disarmed(veh)
        return
    while veh.ned is None:
        veh.poll()
        time.sleep(0.1)
    # 길 출발점 = 이륙한 자리에서 길을 따라 LEAD m 뒤 → 사람은 드론 밑이 아니라 뒤에서 출발
    (ln, le), _ = path.point(LEAD)
    n0, e0 = veh.ned[0] - ln, veh.ned[1] - le
    print(f"이륙 완료 ({veh.rel_alt:.1f}m) → 안내 시작. 길 {path.length:.0f}m, 꺾이는 곳 {len(path.corners)}개, "
          f"사람은 이륙 자리 {LEAD:.0f}m 뒤에서 출발")

    person = VirtualPerson(path)
    light = Light(veh, LIGHT_RELAY)
    light.set(ON)
    s_cmd = LEAD
    offpath = False
    t0 = last = time.time()
    rows = []
    end_reason = "시간 초과"
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

            # 사람 위치 (지금은 가상의 사람)
            person.update(t, dt, light.signal)
            pn, pe = person.pos()
            s_p, d_p = path.project((pn, pe))
            dn, de = (veh.ned[0] - n0, veh.ned[1] - e0) if veh.ned else (None, None)
            gap = math.hypot(dn - pn, de - pe) if dn is not None else None

            # 경로 이탈 판정 (2m 넘으면 이탈, 1m 안으로 오면 해제)
            if not offpath and abs(d_p) > OFFPATH_M:
                offpath = True
                print(f"  → 경로 이탈 {abs(d_p):.1f}m: 깜빡임(멈추세요), 드론 기다림")
            elif offpath and abs(d_p) < OFFPATH_BACK:
                offpath = False
                print("  → 길로 돌아옴: 다시 켜짐(따라오세요)")
            near_corner = any(-1.0 <= c - s_p <= SWEEP_BEFORE for c in path.corners)
            signal = BLINK if offpath else (SWEEP if near_corner else ON)
            light.set(signal)

            # 드론 목표: 사람보다 길 위로 LEAD m 앞. 이탈했거나 너무 멀면 기다림. 사람 쪽으로는 안 돌아감
            waiting = offpath or (gap is not None and gap > MAX_GAP)
            s_want = s_cmd if waiting else s_p + LEAD
            if not waiting and gap is not None and gap < MIN_GAP:   # 꺾이는 곳에서 너무 가까워지면 더 앞으로
                s_want += MIN_GAP - gap
            s_want = min(s_want, path.cum[-1])
            step = max(0.0, min(MAX_SPEED * dt, s_want - s_cmd))
            s_cmd += step
            (tn, te), u = path.point(s_cmd)
            if math.hypot(tn, te) > MAX_RANGE:                     # 소프트웨어 울타리
                s_cmd -= step
                (tn, te), u = path.point(s_cmd)
                step = 0.0
            vn, ve = u[0] * step / dt, u[1] * step / dt
            # 방향: 드론이 사람을 바라봄 (+ 좌우 쓸기)
            ref_n, ref_e = (dn, de) if dn is not None else (tn, te)
            yaw = math.atan2(pe - ref_e, pn - ref_n)
            if signal == SWEEP:
                yaw += math.radians(tri(t, SWEEP_DEG, SWEEP_RATE))
            veh.target(n0 + tn, e0 + te, ALT, vn, ve, yaw)
            light.update(now)

            if person.state == "도착" and (gap is None or gap > 3.0):
                end_reason = "사람 도착"
                break

            rows.append(dict(t=round(t, 2), person_n=round(pn, 2), person_e=round(pe, 2),
                             person_s=round(s_p, 2), offset=round(d_p, 2), person_state=person.state,
                             drone_n=None if dn is None else round(dn, 2),
                             drone_e=None if de is None else round(de, 2),
                             alt=None if veh.rel_alt is None else round(veh.rel_alt, 2),
                             gap=None if gap is None else round(gap, 2), signal=signal,
                             relay=int(bool(light.relay_on)), yaw_cmd=round(math.degrees(yaw), 1),
                             yaw=None if veh.yaw is None else round(veh.yaw, 1), mode=veh.mode))
            if len(rows) % 4 == 0:
                g = "-" if gap is None else f"{gap:4.1f}m"
                print(f"{t:6.1f}s | 사람 {s_p:5.1f}m ({person.state}) 옆 {d_p:+.1f}m | 거리 {g} | "
                      f"빛 {signal}({MEANING[signal]})")
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

    # 기록 저장
    if rows:
        with open(log_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        with open(log_path.replace(".csv", "_path.json"), "w", encoding="utf-8") as f:
            json.dump(dict(path=PATH, offpath_m=OFFPATH_M, lead=LEAD, alt=ALT, corners=path.corners,
                           end_reason=end_reason), f, ensure_ascii=False)
        summary(rows, light, veh, end_reason, log_path)


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


def summary(rows, light, veh, end_reason, log_path):
    gaps = [r["gap"] for r in rows if r["gap"] is not None]
    walk = [r["gap"] for r in rows[len(rows) // 6:]
            if r["gap"] is not None and r["signal"] == ON and r["person_state"] == "걷는 중"]
    print("\n── 요약 ──")
    print(f"끝난 이유: {end_reason}, 안내 {rows[-1]['t']:.0f}초, 사람 {rows[-1]['person_s']:.1f}m 걸음")
    if gaps:
        print(f"사람-드론 거리: 전체 최소 {min(gaps):.1f}m (가장 가까웠을 때)")
    if walk:
        print(f"  곧은 길 걷는 동안 {min(walk):.1f}~{max(walk):.1f}m (목표 {LEAD:.0f}m, 계획 5~10m)")
    drift = next((r for r in rows if r["person_state"].startswith("걷는 중(길 벗어나")), None)
    blink = next((r for r in rows if r["signal"] == BLINK), None)
    if drift and blink:
        print(f"경로 이탈: {drift['t']:.1f}초에 벗어나기 시작 → {blink['t']:.1f}초에 깜빡임 "
              f"(최대 {max(abs(r['offset']) for r in rows):.1f}m 벗어남)")
    corners = Path(PATH, LEAD).corners
    swept = sum(1 for c in corners
                if any(r["signal"] == SWEEP and -1.0 <= c - r["person_s"] <= SWEEP_BEFORE for r in rows))
    print(f"좌우 쓸기: 꺾이는 곳 {len(corners)}개 중 {swept}곳에서 켜짐")
    print(f"빛(릴레이) 명령: {light.sent}번 보냄 / 수락 {veh.relay_ok} / 거절 {veh.relay_fail}")
    print(f"기록 저장: {log_path}  → 그림: python gg_plot_v3.py {os.path.basename(log_path)}")


def main():
    ap = argparse.ArgumentParser(description="갯가드 빛길 안내 v3")
    ap.add_argument("addr", nargs="?", default="tcp:127.0.0.1:5762")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--light-test", action="store_true", help="시동 없이 빛(릴레이)만 시험")
    ap.add_argument("--real", action="store_true", help="실기체 비행 허용 (설정 수정·허락·안전 조종자 준비 뒤에만)")
    a = ap.parse_args()

    if not a.light_test and not is_local(a.addr) and not a.real:
        print("실기체 연결로 보여 비행을 막았습니다. 땅에서 빛만 시험하려면 --light-test 를 붙이세요.")
        return
    if a.real and not a.light_test:
        print("실기체 비행 전 확인: 배터리·지상국 안전 설정 수정 / 관리자 허락 / 안전 조종자 조종기 대기 / "
              "비행 구역 사람 통제 / 비행승인 여부 [법규 확인: 항공안전법 제127조]")
        if input("모두 확인했으면 '비행' 입력: ").strip() != "비행":
            print("취소했습니다.")
            return

    veh = Vehicle(a.addr, a.baud)
    hb = veh.wait_vehicle()
    if hb is None:
        print("연결 실패: 시뮬레이터가 켜져 있는지, 주소·포트가 맞는지 확인하세요")
        return
    print(f"연결됨: 기체 번호 {veh.m.target_system}, 하트비트 255번으로 보내는 중")
    if a.light_test:
        light_test(veh)
        return
    log_path = time.strftime("gg_log_%Y%m%d_%H%M%S.csv")
    fly(veh, log_path)


if __name__ == "__main__":
    main()
