"""갯가드 빛길 안내 - 시뮬레이터(SITL) 첫 연결 시험: 사람 앞 7m 앞장서기 (v2: 부드럽게)

하는 일
  1) 가상 드론에 연결  2) GUIDED 모드  3) 시동  4) 7m 이륙
  5) 가상의 사람이 북쪽으로 초속 0.5m로 걷는 동안, 드론이 사람 7m 앞을 유지하며 앞장섬
  6) 사람과의 거리에 따라 빛 신호(화면 출력)를 바꿈  7) 착륙

실행 (Mission Planner 시뮬레이터가 켜진 상태에서)
  python gg_sitl_lead.py                      # 기본: tcp:127.0.0.1:5762
  python gg_sitl_lead.py tcp:127.0.0.1:5763   # 5762가 안 되면

주의: 시뮬레이터 전용입니다. 실기체에 그대로 쓰지 마세요.
"""
import math
import sys
import time

from pymavlink import mavutil

ADDR = sys.argv[1] if len(sys.argv) > 1 else "tcp:127.0.0.1:5762"
ALT = 7.0        # 비행 고도 (m)
LEAD = 7.0       # 사람 앞 거리 (m) - 계획값 5~10m의 가운데
WALK = 0.5       # 사람 걸음 (m/s) - 밤 갯벌 보행 가정 [확인 필요]
DURATION = 60    # 안내 시간 (s)
STOP_AT = 30     # 이 시각에 사람이 멈췄다고 가정 (s) - 멈춤 신호 시험용
STOP_LEN = 8     # 멈춰 있는 시간 (s)

# SET_POSITION_TARGET_LOCAL_NED: 위치 + 속도를 함께 보냄 (가속도·방향은 무시)
#  - 위치만 보내면 드론이 목표를 "쫓아가서" 뒤처지고 툭툭 움직임
#  - 걷는 속도를 같이 알려주면 같은 속도로 부드럽게 따라감
POS_VEL = 0b0000110111000000
MAX_SPEED = 1.5   # 처음 자리 잡을 때 등 목표점이 움직이는 최대 속도 (m/s)


def recv(m, typ, timeout=3.0):
    return m.recv_match(type=typ, blocking=True, timeout=timeout)


def light_signal(gap):
    """드론-사람 거리로 빛 신호를 고름 (실기체에서는 LED 릴레이 명령으로 바뀔 자리)"""
    if gap > 10.0:
        return "깜빡임 (멈추세요, 너무 멀어짐)"
    return "켜짐 (따라오세요)"


def main():
    print("연결 중:", ADDR)
    m = mavutil.mavlink_connection(ADDR, source_system=250)
    hb = m.wait_heartbeat(timeout=30)
    if hb is None:
        print("연결 실패: 시뮬레이터가 켜져 있는지, 주소가 맞는지 확인하세요")
        return
    kind = "ArduPilot" if hb.autopilot == mavutil.mavlink.MAV_AUTOPILOT_ARDUPILOTMEGA else str(hb.autopilot)
    print(f"연결됨: 기체 번호 {m.target_system}, 펌웨어 {kind}")

    # 위치 정보를 초당 4번 받도록 요청
    m.mav.request_data_stream_send(m.target_system, m.target_component,
                                   mavutil.mavlink.MAV_DATA_STREAM_ALL, 4, 1)

    # GUIDED 모드
    m.set_mode(m.mode_mapping()["GUIDED"])
    time.sleep(1)

    # 시동: GPS·센서가 준비될 때까지 재시도
    armed = False
    for _ in range(40):
        m.arducopter_arm()
        t_end = time.time() + 3
        while time.time() < t_end:
            hb = recv(m, "HEARTBEAT", 1)
            if hb and hb.get_srcSystem() == m.target_system and \
                    hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED:
                armed = True
                break
        if armed:
            break
        print("시동 대기 중 (GPS·센서 준비 중)...")
    if not armed:
        print("시동 실패: Mission Planner 메시지 창의 'PreArm' 문구를 확인하세요")
        return
    print("시동 걸림")

    # 이륙
    m.mav.command_long_send(m.target_system, m.target_component,
                            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0, ALT)
    t_end = time.time() + 60
    while time.time() < t_end:
        p = recv(m, "GLOBAL_POSITION_INT", 2)
        if p and p.relative_alt / 1000.0 > ALT * 0.9:
            break
    print("이륙 완료, 안내 시작")

    # 사람 출발점 = 지금 드론 바로 아래
    lp = recv(m, "LOCAL_POSITION_NED", 5)
    n0, e0 = (lp.x, lp.y) if lp else (0.0, 0.0)

    t0 = time.time()
    walked = 0.0
    last = t0
    cmd_n = n0                      # 실제로 보내는 목표점 (천천히 움직이게 제한)
    gaps = []
    while time.time() - t0 < DURATION:
        now = time.time()
        t = now - t0
        dt = max(now - last, 1e-3)
        stopped = STOP_AT <= t < STOP_AT + STOP_LEN
        if not stopped:
            walked += WALK * dt
        last = now
        person_n = n0 + walked

        # 드론 목표: 사람보다 LEAD m 앞 (10m 넘게 벌어지면 더 나가지 않음)
        lp = recv(m, "LOCAL_POSITION_NED", 1)
        drone_n = lp.x if lp else None
        want_n = person_n + LEAD
        if drone_n is not None and drone_n - person_n > 10.0:
            want_n = drone_n            # 기다림
        # 목표점은 MAX_SPEED보다 빨리 움직이지 않게 (처음에 확 튀어나가지 않도록)
        step = max(-MAX_SPEED * dt, min(MAX_SPEED * dt, want_n - cmd_n))
        cmd_n += step
        vel_n = step / dt               # 목표점이 움직이는 속도 = 드론에 알려줄 속도
        m.mav.set_position_target_local_ned_send(
            0, m.target_system, m.target_component, mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            POS_VEL, cmd_n, e0, -ALT, vel_n, 0, 0, 0, 0, 0, 0, 0)

        if drone_n is not None:
            gap = drone_n - person_n
            gaps.append(gap)
            state = "멈춤" if stopped else "걷는 중"
            print(f"{t:5.1f}s | 사람 {person_n - n0:5.1f}m ({state}) | 드론 {drone_n - n0:5.1f}m | "
                  f"앞 거리 {gap:4.1f}m | 빛: {light_signal(gap)}")
        time.sleep(0.25)

    # 착륙
    m.set_mode(m.mode_mapping()["LAND"])
    print("착륙 명령 보냄")
    if gaps:
        steady = gaps[len(gaps) // 4:]          # 처음 출발 구간 빼고
        print(f"요약: 앞 거리 최소 {min(steady):.1f}m, 최대 {max(steady):.1f}m (목표 {LEAD:.0f}m, 허용 5~10m)")


if __name__ == "__main__":
    main()
