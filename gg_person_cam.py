"""드론 카메라로 사람 위치 구하기 — 화면 속 사람 발 위치 → 땅 위 좌표 (북·동 m)

  ① 사람 찾기: 화면에서 사람 상자(bounding box)를 찾고, 상자 아래 가운데 = 발 위치(픽셀)
       - 시뮬레이터: SimDetector — 가상의 사람 실제 위치를 카메라 모델로 화면에 비추고 오차·놓침을 넣음
       - 실기체: YoloDetector — 카메라 영상(USB·RTSP)에 YOLO 사람 찾기 [시험 안 함: 영상 없음]
  ② 땅 위 좌표: 발 픽셀 → 카메라 방향 → 드론 위치(RTK)·기수·짐벌 기울기·고도로 땅(평평하다고 봄)과 만나는 점
       갯벌은 평평한 편이라 "땅 = 이륙한 곳 높이"로 봄 [가정]
  ③ 칼만 필터(등속도): 흔들림을 줄이고, 튀는 값(3σ 밖)은 버림, 3초 넘게 못 찾으면 "놓침"

카메라는 짐벌이 기울기·좌우 기울어짐을 잡아 준다고 봄(기울기 = 짐벌 명령 각, 좌우 = 0, 방향 = 기수) [확인 필요: 기체 짐벌]
오차 가정 (SimDetector 기본값) [가정, 실기체에서 다시 재야 함]
  발 픽셀 흔들림 3px, 한 장에서 못 찾을 확률 10%, 짐벌 기울기 치우침 ±1°(고정)+0.3°, 기수 치우침 ±2°(고정)+0.5°,
  고도 치우침 ±0.3m(고정)+0.1m, 드론 위치: RTK 0.03m / 일반 GPS 1.5m(천천히 변함)

실행 (자체 시험): python gg_person_cam.py            → RTK·일반 GPS일 때 사람 위치 오차
"""
import argparse
import math

import numpy as np


# ───────── 카메라 ─────────
class Camera:
    """핀홀 카메라. hfov: 가로 화각(도). 짐벌 기울기 tilt(도, 아래가 +), 방향 = 드론 기수"""

    def __init__(self, w=1280, h=720, hfov=80.0):
        self.w, self.h = w, h
        self.fx = self.fy = (w / 2) / math.tan(math.radians(hfov) / 2)
        self.cx, self.cy = w / 2, h / 2

    @staticmethod
    def axes(yaw_deg, tilt_deg, roll_deg=0.0):
        """카메라 축(북·동·아래): 앞 f, 오른쪽 r, 화면 아래 d"""
        y, b, g = math.radians(yaw_deg), math.radians(tilt_deg), math.radians(roll_deg)
        f = np.array([math.cos(b) * math.cos(y), math.cos(b) * math.sin(y), math.sin(b)])
        r = np.array([-math.sin(y), math.cos(y), 0.0])
        d = np.array([-math.sin(b) * math.cos(y), -math.sin(b) * math.sin(y), math.cos(b)])
        if g:                                     # 광축을 중심으로 좌우 기울어짐
            r, d = r * math.cos(g) + d * math.sin(g), -r * math.sin(g) + d * math.cos(g)
        return f, r, d

    def project(self, drone_ne, alt, yaw, tilt, ground_ne, roll=0.0):
        """땅 위 점 → 픽셀 (화면 밖이면 None)"""
        f, r, d = self.axes(yaw, tilt, roll)
        q = np.array([ground_ne[0] - drone_ne[0], ground_ne[1] - drone_ne[1], alt])
        z = q @ f
        if z <= 0.1:
            return None
        u, v = self.cx + self.fx * (q @ r) / z, self.cy + self.fy * (q @ d) / z
        return (u, v) if 0 <= u < self.w and 0 <= v < self.h else None

    def ground(self, drone_ne, alt, yaw, tilt, uv, roll=0.0):
        """픽셀 → 땅 위 점 (북·동). 광선이 땅으로 안 가면 None"""
        f, r, d = self.axes(yaw, tilt, roll)
        ray = f + r * (uv[0] - self.cx) / self.fx + d * (uv[1] - self.cy) / self.fy
        if ray[2] <= 1e-3:
            return None
        lam = alt / ray[2]
        return (drone_ne[0] + lam * ray[0], drone_ne[1] + lam * ray[1])

    def tilt_to(self, drone_ne, alt, target_ne):
        """target이 화면 가운데 높이에 오게 하는 짐벌 기울기(도)"""
        return math.degrees(math.atan2(alt, max(math.dist(drone_ne, target_ne), 0.1)))


# ───────── 칼만 필터 (등속도) ─────────
class Tracker:
    def __init__(self, acc=0.6, meas=0.4, gate=3.0, lost_s=3.0):
        self.acc, self.meas, self.gate, self.lost_s = acc, meas, gate, lost_s
        self.x = None
        self.P = None
        self.t = None
        self.t_meas = None
        self.rejects = 0
        self.n_ok = self.n_rej = 0

    def predict(self, t):
        if self.x is None:
            return
        dt = max(t - self.t, 0.0)
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        q = self.acc ** 2
        Q = q * np.array([[dt ** 4 / 4, 0, dt ** 3 / 2, 0], [0, dt ** 4 / 4, 0, dt ** 3 / 2],
                          [dt ** 3 / 2, 0, dt ** 2, 0], [0, dt ** 3 / 2, 0, dt ** 2]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.t = t

    def update(self, t, z, meas=None):
        R = np.eye(2) * (meas or self.meas) ** 2
        if self.x is None:
            self.x = np.array([z[0], z[1], 0.0, 0.0])
            self.P = np.diag([R[0, 0], R[1, 1], 1.0, 1.0])
            self.t = self.t_meas = t
            return True
        self.predict(t)
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)
        y = np.asarray(z) - H @ self.x
        S = H @ self.P @ H.T + R
        if float(y @ np.linalg.solve(S, y)) > self.gate ** 2:       # 튀는 값
            self.rejects += 1
            self.n_rej += 1
            if self.rejects >= 5:                                   # 5번 이어서 튀면 새로 시작 (사람을 다시 찾은 것)
                self.x, self.rejects = None, 0
                return self.update(t, z, meas)
            return False
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P
        self.t_meas, self.rejects = t, 0
        self.n_ok += 1
        return True

    def lost(self, t):
        return self.t_meas is None or t - self.t_meas > self.lost_s

    def pos(self, t):
        if self.x is None:
            return None
        dt = max(t - self.t, 0.0)
        return (self.x[0] + self.x[2] * dt, self.x[1] + self.x[3] * dt)


# ───────── 사람 찾기: 시뮬레이터 ─────────
class SimDetector:
    """가상의 사람 실제 위치 → 오차가 있는 드론 자세·위치로 화면에 비춘 발 픽셀 (놓침 포함).
    드론 SW 쪽은 이 픽셀과 '자기가 아는'(오차 있는) 자세로 땅 좌표를 다시 구함 → 실제와 같은 길"""

    def __init__(self, cam, seed=1, px=3.0, p_miss=0.10, tilt_bias=1.0, tilt_noise=0.3, yaw_bias=2.0, yaw_noise=0.5,
                 roll_noise=0.5, alt_bias=0.3, alt_noise=0.1, pos="rtk"):
        self.cam = cam
        self.rng = np.random.default_rng(seed)
        self.px, self.p_miss = px, p_miss
        self.tilt_b = self.rng.normal(0, tilt_bias)
        self.yaw_b = self.rng.normal(0, yaw_bias)
        self.alt_b = self.rng.normal(0, alt_bias)
        self.tilt_n, self.yaw_n, self.roll_n, self.alt_n = tilt_noise, yaw_noise, roll_noise, alt_noise
        self.pos_sigma = 0.03 if pos == "rtk" else 1.5
        self.pos_err = self.rng.normal(0, self.pos_sigma, 2)
        self.t_last = None

    def known_pose_error(self, t):
        """드론이 아는 위치의 오차 (일반 GPS는 1분 정도에 걸쳐 천천히 변함)"""
        if self.t_last is not None:
            dt = t - self.t_last
            a = math.exp(-dt / 60.0)
            self.pos_err = a * self.pos_err + math.sqrt(1 - a * a) * self.rng.normal(0, self.pos_sigma, 2)
        self.t_last = t
        return self.pos_err

    def detect(self, t, drone_true_ne, alt_true, yaw_cmd, tilt_cmd, person_true_ne):
        """→ (픽셀 또는 None, 드론이 아는 자세 dict). 실제 카메라 방향 = 명령 + 치우침 + 흔들림"""
        perr = self.known_pose_error(t)
        known = dict(ne=(drone_true_ne[0] + perr[0], drone_true_ne[1] + perr[1]), alt=alt_true + self.alt_b + self.rng.normal(0, self.alt_n),
                     yaw=yaw_cmd, tilt=tilt_cmd)
        yaw_real = yaw_cmd + self.yaw_b + self.rng.normal(0, self.yaw_n)
        tilt_real = tilt_cmd + self.tilt_b + self.rng.normal(0, self.tilt_n)
        roll_real = self.rng.normal(0, self.roll_n)
        if self.rng.random() < self.p_miss:
            return None, known
        uv = self.cam.project(drone_true_ne, alt_true, yaw_real, tilt_real, person_true_ne, roll_real)
        if uv is None:
            return None, known
        return (uv[0] + self.rng.normal(0, self.px), uv[1] + self.rng.normal(0, self.px)), known


# ───────── 사람 찾기: 실기체 (YOLO) ─────────
class YoloDetector:
    """카메라 영상에서 사람(COCO 0번) 상자를 찾아 발 픽셀(상자 아래 가운데)을 돌려줌.
    [시험 안 함: 이 작업 환경엔 카메라 영상이 없음] 필요: pip install ultralytics opencv-python
    밤에는 일반 카메라로 사람이 잘 안 보일 수 있음 → 열화상이면 열화상용으로 학습한 모델 필요 [확인 필요]"""

    def __init__(self, source=0, model="yolov8n.pt", conf=0.35):
        import cv2
        from ultralytics import YOLO
        self.cap = cv2.VideoCapture(source)
        self.model = YOLO(model)
        self.conf = conf

    def detect(self):
        ok, frame = self.cap.read()
        if not ok:
            return None
        res = self.model.predict(frame, classes=[0], conf=self.conf, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return None
        b = res.boxes
        i = int(np.argmax(b.conf.cpu().numpy()))
        x1, y1, x2, y2 = b.xyxy[i].cpu().numpy()
        return ((x1 + x2) / 2, y2)                     # 발 = 상자 아래 가운데


# ───────── 한 덩어리: 드론 SW가 부르는 곳 ─────────
class PersonLocator:
    """detect → ground → tracker. 드론 SW에서 매 계산 주기마다 step() 호출"""

    def __init__(self, cam=None, detector=None, rate_hz=5.0, **kw):
        self.cam = cam or Camera()
        self.det = detector or SimDetector(self.cam, **kw)
        self.trk = Tracker()
        self.period = 1.0 / rate_hz
        self.t_next = 0.0
        self.last_raw = None

    def step(self, t, drone_true_ne, alt_true, yaw_deg, person_true_ne=None):
        """→ (추정 위치 또는 None, 놓침?). 짐벌은 추정 위치(없으면 진짜 위치 근처)를 보게 기울임"""
        if t >= self.t_next:
            self.t_next = t + self.period
            target = self.trk.pos(t) or person_true_ne or drone_true_ne
            tilt = min(max(self.cam.tilt_to(drone_true_ne, alt_true, target), 20.0), 90.0)
            if isinstance(self.det, SimDetector):
                uv, known = self.det.detect(t, drone_true_ne, alt_true, yaw_deg, tilt, person_true_ne)
            else:
                uv, known = self.det.detect(), dict(ne=drone_true_ne, alt=alt_true, yaw=yaw_deg, tilt=tilt)
            if uv is not None:
                g = self.cam.ground(known["ne"], known["alt"], known["yaw"], known["tilt"], uv)
                if g is not None:
                    self.last_raw = g
                    self.trk.update(t, g)
        self.trk.predict(t)
        return self.trk.pos(t), self.trk.lost(t)


def self_test(pos_mode, seed=1, secs=240.0):
    """사람이 0.5m/s로 꺾인 길을 걷고, 드론은 v4.1처럼 사람 7.9m 앞·고도 7m에서 사람 쪽을 봄 → 위치 오차"""
    rng = np.random.default_rng(seed)
    loc = PersonLocator(pos=pos_mode, seed=seed)
    pts = [(0, 0), (40, 0), (40, 30), (80, 30)]
    seg = [(np.array(a, float), np.array(b, float)) for a, b in zip(pts, pts[1:])]
    lens = [np.linalg.norm(b - a) for a, b in seg]
    err, raw_err, lost = [], [], 0
    dt = 0.25
    for k in range(int(secs / dt)):
        t = k * dt
        s = min(0.5 * t, sum(lens) - 0.01)
        i, s0 = 0, s
        while s0 > lens[i]:
            s0 -= lens[i]
            i += 1
        a, b = seg[i]
        u = (b - a) / lens[i]
        p = a + u * s0 + np.array([-u[1], u[0]]) * 0.3 * math.sin(t / 7)      # 길 위에서 살짝 흔들리며 걸음
        drone = p + u * 7.9 + rng.normal(0, 0.2, 2)
        yaw = math.degrees(math.atan2(-u[1], -u[0]))                          # 사람 쪽을 봄
        est, is_lost = loc.step(t, tuple(drone), 7.0 + rng.normal(0, 0.1), yaw, tuple(p))
        if est is not None and t > 3:
            err.append(math.dist(est, p))
            if loc.last_raw is not None:
                raw_err.append(math.dist(loc.last_raw, p))
        lost += bool(is_lost) and t > 3
    e, r = np.array(err), np.array(raw_err)
    return dict(med=float(np.median(e)), p95=float(np.percentile(e, 95)), mx=float(e.max()),
                raw_med=float(np.median(r)), raw_p95=float(np.percentile(r, 95)), lost=lost * dt,
                ok=loc.trk.n_ok, rej=loc.trk.n_rej)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=20, help="자체 시험 반복 (치우침을 매번 새로 뽑음)")
    a = ap.parse_args()
    for mode, name in (("rtk", "드론 RTK"), ("gps", "드론 일반 GPS")):
        rs = [self_test(mode, seed=s) for s in range(a.runs)]
        med = np.median([r["med"] for r in rs])
        p95 = np.median([r["p95"] for r in rs])
        worst = max(r["p95"] for r in rs)
        raw = np.median([r["raw_p95"] for r in rs])
        lost = sum(r["lost"] for r in rs)
        print(f"{name}: 사람 위치 오차 중앙값 {med:.2f}m, 95% {p95:.2f}m (시험 {a.runs}번 중 가장 나쁜 95% {worst:.2f}m), "
              f"필터 전 한 장 95% {raw:.2f}m, 놓침 합계 {lost:.0f}초")


if __name__ == "__main__":
    main()
