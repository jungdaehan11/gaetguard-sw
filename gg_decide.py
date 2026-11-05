"""탈출 / 피난 / 대기 판정 (인수인계 3장 ④·⑤) + 실제 물때로 검증 + 탈출로를 드론 경로로

판정 시각 t에 "그때 알 수 있던 정보"만 씀
  ① 앞으로 물높이 = 우리 조석 예측 + (t까지 10분 실측 평균 − 같은 때 예측)        ← 높이 보정 (5장)
  ② 바다와 이어진 물, 깊이 30cm부터 통행 불가 → 칸마다 막히는 시각               ← gg_retreat_map.py
  ③ 가장 오래 열린 길 → 칸마다 마지막 출발 시각 L                                 ← gg_retreat_map.py
  ④ 안전 여유 = 몇 분 앞을 보는지 · 밤/낮별 99% 시각 오차 (영흥도 2025~26 시험)
  ⑤ 남은 여유 = L − t − 반응 2분 − 안전 여유
     ≥ 0 → 탈출 (그 길로 육지까지)
     < 0 → 피난: 지금 걸어서 (각 칸 막히기 전·여유 빼고) 닿는 칸 중 가장 늦게 잠기는 곳이 지금 자리보다
           MIN_GAIN분 넘게 늦게 잠기면 그곳으로
     그것도 없으면 → 대기 (제자리, 구조·구명장비)
검증: 실제 실측 물때로 막히는 시각을 다시 계산해서
  탈출 → 판정한 그 길을 반응 2분 뒤 출발해 걸었을 때 모든 칸을 막히기 전에 지났나
  피난 → 피난처가 약속한 시각(안전 여유 뺀 시각)보다 일찍 잠겼나

실행 예:
  python gg_decide.py --tide tide\\station_DT_0043_1m_우리예측2022-2024.csv --times "2025-09-11 01:30" "2025-09-11 02:07" "2025-09-11 02:30" "2025-09-11 03:00" --path-at "2025-09-11 02:07"
  python gg_decide.py --tide tide\\station_DT_0043_1m_우리예측2022-2024.csv --batch 20     (밤 큰 간조 20번으로 세 방식 비교)
  (--tide: date,time,obs_cm,pred_cm 모양. pred_cm = 우리 예측(gg_tide_harmonic.py --api 결과), obs_cm = 실측)
"""
import argparse
import csv
import heapq
import json
import os

import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gg_retreat_map as rm  # noqa: E402

REACT = 2.0          # 분: 판정을 듣고 움직이기 시작할 때까지 (인수인계 ⑤)
MIN_GAIN = 15.0      # 분: 피난처가 지금 자리보다 이만큼 넘게 늦게 잠겨야 피난으로 봄 [가정]
HORIZON_H = 6 * 60   # 분: 판정 뒤 계산할 길이
# 99% 시각 오차(분) — 영흥도 2025-01~2026-11, 우리 예측(2022~2024 학습) + 높이 보정, 물 찰 때 (gg_tide_correction_test.py)
MARGIN = {"밤": [(30, 8.6), (60, 13.0), (120, 20.9)],
          "낮": [(30, 16.9), (60, 24.7), (120, 36.0)]}


def margin(h, night):
    """몇 분 앞(h)을 보는지에 따른 99% 안전 여유(분). 30분보다 짧으면 30분 값, 120분 넘으면 60~120분 기울기로 늘림"""
    tbl = MARGIN["밤" if night else "낮"]
    xs = np.array([x for x, _ in tbl], float)
    ys = np.array([y for _, y in tbl], float)
    h = np.asarray(h, float)
    m = np.interp(np.clip(h, xs[0], xs[-1]), xs, ys)
    slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
    return np.where(h > xs[-1], ys[-1] + (h - xs[-1]) * slope, m)


def load(path):
    T, O, P = [], [], []
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            T.append(f"{r['date'][:4]}-{r['date'][4:6]}-{r['date'][6:]}T{r['time']}")
            O.append(float(r["obs_cm"]) if r["obs_cm"] else np.nan)
            P.append(float(r["pred_cm"]) if r["pred_cm"] else np.nan)
    T = np.array(T, dtype="datetime64[m]")
    o = np.argsort(T)
    T, O, P = T[o], np.array(O)[o], np.array(P)[o]
    O[(O == 0) & (P > 100)] = np.nan
    return T, O, P


def window(T, X, t0, n):
    """t0부터 n분 (빠진 분은 보간)"""
    i0 = int(np.searchsorted(T, t0))
    x = X[i0:i0 + n].astype(float)
    ok = np.isfinite(x)
    if ok.sum() < 2:
        raise SystemExit(f"{t0} 뒤 자료가 없습니다.")
    return np.interp(np.arange(len(x)), np.nonzero(ok)[0], x[ok])


def smooth5(x):
    k = np.ones(5) / 5
    return np.convolve(np.r_[[x[0]] * 2, x, [x[-1]] * 2], k, "valid")


def refuge(G, start, tcs, walk):
    """start 칸에서 반응 시간 뒤 출발, 각 칸 막히기 전(tcs) 도착 조건으로 닿는 칸 중 tcs 가장 늦은 칸 (동률이면 가까운 칸)"""
    sea = G["sea"]
    ny, nx = sea.shape
    step = G["res"] / walk / 60.0
    nb = [(-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1), (-1, -1, 1.4142), (-1, 1, 1.4142), (1, -1, 1.4142), (1, 1, 1.4142)]
    sy, sx = start
    best = (tcs[sy, sx], 0.0, sy, sx)
    arr = {(sy, sx): REACT}
    heap = [(REACT, sy, sx)]
    while heap:
        t, y, x = heapq.heappop(heap)
        if t > arr.get((y, x), np.inf):
            continue
        if (tcs[y, x], -t) > (best[0], -best[1]):
            best = (tcs[y, x], t, y, x)
        for dy, dx, dd in nb:
            yy, xx = y + dy, x + dx
            if 0 <= yy < ny and 0 <= xx < nx and sea[yy, xx]:
                ta = t + dd * step
                if ta < tcs[yy, xx] and ta < arr.get((yy, xx), np.inf):
                    arr[(yy, xx)] = ta
                    heapq.heappush(heap, (ta, yy, xx))
    return best


def decide_at(G, T, O, P, t_now, a):
    n = HORIZON_H
    i_now = int(np.searchsorted(T, t_now))
    o10 = O[max(0, i_now - 9):i_now + 1]
    p10 = P[max(0, i_now - 9):i_now + 1]
    ok = np.isfinite(o10) & np.isfinite(p10)
    offset = float(np.mean(o10[ok] - p10[ok])) if ok.any() else 0.0
    Hf = (window(T, P, t_now, n) + offset) / 100.0                    # ① 예보 + 높이 보정 (m)
    Ht = smooth5(window(T, O, t_now, n)) / 100.0                      # 실제 (검증용)
    Tw = t_now + np.arange(len(Hf)).astype("timedelta64[m]")
    lon_c, lat_c = (G["GX"][0, 0] + G["GX"][0, -1]) / 2, (G["GY"][0, 0] + G["GY"][-1, 0]) / 2
    night = bool(_sun_elev(np.array([t_now]), lat_c, lon_c)[0] < -0.833)
    tcf = rm.close_times(G, Tw, Hf, a.depth)                          # ② 예보로 막히는 시각 (판정 시각부터 분)
    Lf, parent, order = rm.latest_start(G, tcf, a.walk, with_path=True)   # ③
    walk_now = G["sea"] & (tcf > 0)                                   # 지금 서 있을 수 있는 칸
    m_L = margin(np.where(np.isfinite(Lf), Lf, 999), night)          # ④
    slack = Lf - m_L - REACT                                          # ⑤ 남은 여유
    escape = walk_now & (slack >= 0)
    tcs = tcf - margin(np.where(np.isfinite(tcf), tcf, 999), night)   # 여유 뺀 막히는 시각
    tcs = np.where(np.isfinite(tcf), tcs, np.inf)
    dec = np.zeros(G["sea"].shape, np.int8)                           # 0 바다/육지, 1 탈출, 2 피난, 3 대기
    dec[escape] = 1
    ref_target = {}
    for y, x in zip(*np.nonzero(walk_now & ~escape)):
        best_t, _, ry, rx = refuge(G, (y, x), tcs, a.walk)
        if best_t > tcs[y, x] + MIN_GAIN:
            dec[y, x] = 2
            ref_target[(y, x)] = (ry, rx, best_t)
        else:
            dec[y, x] = 3
    # 검증: 실제 물때
    tct = rm.close_times(G, Tw, Ht, a.depth)
    Lt = rm.latest_start(G, tct, a.walk)
    ny, nx = G["sea"].shape
    S = np.full(ny * nx, np.inf)                 # 판정한 길을 따라갈 때 실제로 남는 시간 (최소값)
    bar = G["barrier"].ravel()
    tct_f = tct.ravel()
    step = G["res"] / a.walk / 60.0
    for k in order:                              # 육지에서 바깥으로 확정된 순서 → 부모가 먼저
        if bar[k]:
            continue
        p = parent[k]
        if p < 0:
            continue
        dy, dx = abs(k // nx - p // nx), abs(k % nx - p % nx)
        dd = 1.4142 if dy and dx else 1.0
        S[k] = min(tct_f[k], (np.inf if bar[p] else S[p]) - dd * step)
    S = S.reshape(ny, nx)
    esc_fail = escape & (S < REACT)              # 반응 2분 뒤 출발했는데 길 어딘가가 먼저 막힘
    ref_fail = 0
    for (y, x), (ry, rx, bt) in ref_target.items():
        if tct[ry, rx] < bt:                     # 피난처가 약속(여유 뺀 시각)보다 일찍 잠김
            ref_fail += 1
    res = dict(time=str(t_now)[:16].replace("T", " "), night=night, offset_cm=round(offset, 1),
               walkable=int(walk_now.sum()), escape=int(escape.sum()), refuge=int((dec == 2).sum()),
               wait=int((dec == 3).sum()), escape_fail=int(esc_fail.sum()), refuge_fail=ref_fail,
               true_escape_possible=int((walk_now & (Lt >= REACT)).sum()),
               missed_escape=int((walk_now & ~escape & (Lt >= REACT + 30)).sum()))
    return res, dict(dec=dec, slack=slack, Lf=Lf, parent=parent, tcf=tcf, tct=tct, Hf=Hf, Ht=Ht, Tw=Tw, esc_fail=esc_fail)


def _sun_elev(T, lat, lon):
    import gg_tide_correction_test as ct
    return ct.sun_elev(T, lat, lon)


def export_path(G, D, t_now, out, a):
    """예시: 탈출 판정 칸 중 남은 여유 5~20분, 육지까지 길이 긴 칸 하나 → 탈출로를 꺾인 점으로"""
    from shapely.geometry import LineString
    dec, slack, parent = D["dec"], D["slack"], D["parent"]
    ny, nx = dec.shape
    bar = G["barrier"].ravel()
    fx0, fx1, fy0, fy1 = rm.AREA.get("focus", (-999, 999, -999, 999))
    infocus = (G["GX"] >= fx0) & (G["GX"] <= fx1) & (G["GY"] >= fy0) & (G["GY"] <= fy1)
    cand = np.argwhere((dec == 1) & (slack >= 5) & (slack <= 20) & infocus)
    if not len(cand):
        print("   예시로 쓸 탈출 칸 없음")
        return

    def chain(k):
        out = [k]
        while not bar[out[-1]] and parent[out[-1]] >= 0 and len(out) < 5000:
            out.append(parent[out[-1]])
        return out
    lens = []
    for y, x in cand[:: max(1, len(cand) // 400)]:
        c = chain(y * nx + x)
        if infocus.ravel()[c[-1]]:                   # 도착(해안)도 범위 안
            lens.append((len(c), y, x))
    if not lens:
        print("   범위 안에서 끝나는 예시 탈출로 없음")
        return
    _, y0, x0 = max(lens)
    ch = chain(y0 * nx + x0)
    lon = np.array([G["GX"].ravel()[k] for k in ch])
    lat = np.array([G["GY"].ravel()[k] for k in ch])
    kx, ky = G["kx"], G["ky"]
    north, east = (lat - lat[0]) * ky, (lon - lon[0]) * kx
    simp = LineString(np.c_[east, north]).simplify(30.0)            # 30m 안쪽 굴곡은 펴서 꺾인 점만
    e_s, n_s = np.array(simp.coords).T
    pts_ne = [(round(float(n), 1), round(float(e), 1)) for n, e in zip(n_s, e_s)]
    dist = float(np.sum(np.hypot(np.diff(north), np.diff(east))))
    info = dict(time=str(t_now)[:16].replace("T", " "), start_lat=round(float(lat[0]), 6), start_lon=round(float(lon[0]), 6),
                end_lat=round(float(lat[-1]), 6), end_lon=round(float(lon[-1]), 6), length_m=round(dist),
                walk_min=round(dist / a.walk / 60, 1), last_start_min=round(float(D["Lf"][y0, x0]), 1),
                slack_min=round(float(slack[y0, x0]), 1),
                path_ne_m=pts_ne,
                path_latlon=[(round(float(lat[0] + n / ky), 6), round(float(lon[0] + e / kx), 6)) for n, e in pts_ne],
                note="path_ne_m = 출발점 기준 (북쪽 m, 동쪽 m). gg_lead_v4.py 의 PATH 와 같은 모양. 격자 50m(실제 해상도 150m)")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=1)
    print(f"   예시 탈출로: 출발 ({info['start_lat']}, {info['start_lon']}) → 해안 {info['length_m']}m, 걸어서 {info['walk_min']}분, "
          f"남은 여유 {info['slack_min']}분, 꺾인 점 {len(pts_ne)}개 → {out}")
    return info


def draw(out, A, G, results, Ds, path_info):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    for name in ("Malgun Gothic", "Noto Sans CJK KR", "AppleGothic", "NanumGothic"):
        if any(f.name == name for f in fm.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False
    INK, INK2, MUTED, LAND, SEA = "#0b0b0b", "#52514e", "#9a9993", "#e9e7e1", "#c9d3df"
    COLS = ["#1baf7a", "#a8dcc4", "#eda100", "#d6402b"]       # 탈출 여유 30분↑ / 탈출 0~30분 / 피난 / 대기
    B = A["B"]
    k = len(results)
    fig = plt.figure(figsize=(4.6 * k + 1.0, 6.6), facecolor="white")
    fig.text(0.02, 0.955, f"{A['name']} 탈출 · 피난 · 대기 판정 — 그 시각까지의 정보(우리 예측 + 실측 보정 + 99% 안전 여유)만으로",
             fontsize=15, weight="bold", color=INK)
    fig.text(0.02, 0.915, "칸마다 판정. 검증은 실제 물때(관측소 실측)로 다시 계산. 보행 0.5m/s, 반응 2분, 물 깊이 30cm부터 통행 불가, "
             "갯골 칸 0.5m 낮춤 [가정]", fontsize=10, color=INK2)
    w = 0.94 / k
    for i, (r, D) in enumerate(zip(results, Ds)):
        ax = fig.add_axes([0.02 + i * w, 0.17, w - 0.01, 0.66])
        base = np.where(G["land"] | G["barrier"], 0, 1).astype(float)
        ax.pcolormesh(G["GX"], G["GY"], base, cmap=ListedColormap([LAND, SEA]), shading="auto", zorder=0)
        cls = np.full(D["dec"].shape, np.nan)
        cls[(D["dec"] == 1) & (D["slack"] >= 30)] = 0
        cls[(D["dec"] == 1) & (D["slack"] < 30)] = 1
        cls[D["dec"] == 2] = 2
        cls[D["dec"] == 3] = 3
        ax.pcolormesh(G["GX"], G["GY"], cls, cmap=ListedColormap(COLS), vmin=-0.5, vmax=3.5, shading="auto", zorder=1)
        if D["esc_fail"].any():
            ax.scatter(G["GX"][D["esc_fail"]], G["GY"][D["esc_fail"]], s=3, color=INK, zorder=3)
        if path_info and path_info["time"] == r["time"]:
            pl = np.array(path_info["path_latlon"])
            ax.plot(pl[:, 1], pl[:, 0], color=INK, lw=1.8, zorder=4)
            ax.scatter([pl[0, 1]], [pl[0, 0]], s=30, color="white", edgecolor=INK, zorder=5)
        ax.set_xlim(B[0], B[1])
        ax.set_ylim(B[2], B[3])
        ax.set_aspect(1 / np.cos(np.radians((B[2] + B[3]) / 2)))
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#d8d7d2")
        tot = max(r["walkable"], 1)
        ax.set_title(f"{r['time'][11:]}  ({'밤' if r['night'] else '낮'}, 보정 {r['offset_cm']:+.0f}cm)", loc="left",
                     fontsize=12.5, color=INK)
        ax.text(0.0, -0.03, f"탈출 {r['escape'] / tot * 100:.0f}% · 피난 {r['refuge'] / tot * 100:.0f}% · 대기 {r['wait'] / tot * 100:.0f}%\n"
                f"틀린 탈출 지시 {r['escape_fail']}칸 / 탈출 {r['escape']}칸", transform=ax.transAxes, va="top", fontsize=10,
                color=INK2, linespacing=1.5)
    h = [Patch(fc=COLS[0], label="탈출 (남은 여유 30분 넘게)"), Patch(fc=COLS[1], label="탈출 (남은 여유 0~30분)"),
         Patch(fc=COLS[2], label="피난 (더 늦게 잠기는 곳으로)"), Patch(fc=COLS[3], label="대기 (구조·구명장비)"),
         Patch(fc=SEA, label="이미 물속"), Patch(fc=LAND, label="육지")]
    if path_info:
        from matplotlib.lines import Line2D
        h.append(Line2D([], [], color=INK, lw=1.8, label=f"예시 탈출로 ({path_info['time'][11:]}, {path_info['length_m']}m)"))
    fig.legend(handles=h, loc="lower left", bbox_to_anchor=(0.02, 0.0), ncol=4, frameon=False, fontsize=10)
    fig.text(0.98, 0.02, "출처: BADA2024, 개방海 갯골·해안선, 영흥도 조위관측소 실측, 우리 조석 예측(2022~2024 실측으로 학습)",
             ha="right", fontsize=8.5, color=MUTED)
    fig.savefig(out, dpi=140)


def batch(G, T, O, P, a, k):
    """밤 큰 간조 k번(예측 100cm 아래, 해 진 뒤)마다 간조 뒤 60·90·120분에 판정 → 세 방식 비교
       ① 예측만 + 여유 없음  ② 보정 + 여유 없음  ③ 보정 + 99% 여유 (지금 계획)
       틀린 탈출 지시 = 탈출이라고 했는데 그 길을 반응 2분 뒤 걸으면 실제 물때로 어딘가 먼저 막힘"""
    lat_c, lon_c = float(G["GY"].mean()), float(G["GX"].mean())
    lows = []
    for i in range(60, len(P) - 60):
        if np.isfinite(P[i]) and P[i] < 100 and P[i] == np.nanmin(P[i - 60:i + 61]):
            lows.append(i)
    ded = []                                      # 평평한 바닥에서 같은 간조가 여러 번 잡히지 않게 6시간 안은 하나만
    for i in lows:
        if not ded or (T[i] - T[ded[-1]]) > np.timedelta64(360, "m"):
            ded.append(i)
    lows = [i for i in ded if _sun_elev(T[i:i + 1], lat_c, lon_c)[0] < -0.833]
    lows = [i for i in lows if np.isfinite(O[i:i + 420]).mean() > 0.9 and (T[min(i + 420, len(T) - 1)] - T[i]) == np.timedelta64(420, "m")]
    pick = [lows[int(j)] for j in np.linspace(0, len(lows) - 1, min(k, len(lows)))] if lows else []
    print(f"밤 큰 간조 후보 {len(lows)}번 중 {len(pick)}번 사용: " + ", ".join(str(T[i])[:16].replace("T", " ") for i in pick))
    variants = {"① 예측만, 여유 없음": (False, False), "② 보정, 여유 없음": (True, False), "③ 보정 + 99% 여유": (True, True)}
    tot = {v: dict(esc=0, fail=0, stand=0) for v in variants}
    rows = []
    for i in pick:
        for dm in (60, 90, 120):
            t_now = T[i] + np.timedelta64(dm, "m")
            j = int(np.searchsorted(T, t_now))
            Ht = smooth5(window(T, O, t_now, HORIZON_H)) / 100.0
            tct = rm.close_times(G, None, Ht, a.depth)
            ny, nx = G["sea"].shape
            for vname, (corr, marg) in variants.items():
                o10, p10 = O[max(0, j - 9):j + 1], P[max(0, j - 9):j + 1]
                ok = np.isfinite(o10) & np.isfinite(p10)
                off = float(np.mean(o10[ok] - p10[ok])) if (corr and ok.any()) else 0.0
                Hf = (window(T, P, t_now, HORIZON_H) + off) / 100.0
                tcf = rm.close_times(G, None, Hf, a.depth)
                Lf, parent, order = rm.latest_start(G, tcf, a.walk, with_path=True)
                stand = G["sea"] & (tcf > 0)
                m = margin(np.where(np.isfinite(Lf), Lf, 999), True) if marg else 0.0
                esc = stand & (Lf - m - REACT >= 0)
                S = np.full(ny * nx, np.inf)
                bar, tf = G["barrier"].ravel(), tct.ravel()
                step = G["res"] / a.walk / 60.0
                for kk in order:
                    if bar[kk] or parent[kk] < 0:
                        continue
                    p = parent[kk]
                    dd = 1.4142 if (abs(kk // nx - p // nx) and abs(kk % nx - p % nx)) else 1.0
                    S[kk] = min(tf[kk], (np.inf if bar[p] else S[p]) - dd * step)
                fail = esc & (S.reshape(ny, nx) < REACT)
                tot[vname]["esc"] += int(esc.sum())
                tot[vname]["fail"] += int(fail.sum())
                tot[vname]["stand"] += int(stand.sum())
                rows.append(dict(low=str(T[i])[:16], after_min=dm, variant=vname, offset_cm=round(off, 1),
                                 stand=int(stand.sum()), escape=int(esc.sum()), escape_fail=int(fail.sum())))
    print("\n방식                  탈출 지시(칸)   틀린 탈출 지시   비율     탈출 지시 비율")
    for vname, t in tot.items():
        print(f"{vname:<20} {t['esc']:>12,}   {t['fail']:>12,}   {t['fail'] / max(t['esc'], 1) * 100:6.2f}%   "
              f"{t['esc'] / max(t['stand'], 1) * 100:5.1f}%")
    out = os.path.join(rm.HERE, "tide", f"판정검증_{rm.AREA['key']}_밤간조{len(pick)}번.csv")
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("저장:", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tide", required=True)
    ap.add_argument("--times", nargs="+", default=[])
    ap.add_argument("--batch", type=int, default=0, help="밤 큰 간조 N번 골라 세 방식 비교 검증")
    ap.add_argument("--path-at")
    ap.add_argument("--res", type=float, default=50.0)
    ap.add_argument("--walk", type=float, default=0.5)
    ap.add_argument("--depth", type=float, default=0.30)
    ap.add_argument("--drop", type=float, default=0.5)
    a = ap.parse_args()
    A = rm.AREA
    G = rm.build_grid(A, a.res, a.drop)
    T, O, P = load(a.tide)
    if a.batch:
        batch(G, T, O, P, a, a.batch)
        return
    results, Ds, path_info = [], [], None
    day = a.times[0][:10]
    base = os.path.join(rm.HERE, "tide", f"판정_{A['key']}_{day}")
    for ts in a.times:
        t_now = np.datetime64(ts.replace(" ", "T"))
        r, D = decide_at(G, T, O, P, t_now, a)
        tot = max(r["walkable"], 1)
        print(f"{r['time']} ({'밤' if r['night'] else '낮'}, 보정 {r['offset_cm']:+.1f}cm): 서 있을 수 있는 칸 {r['walkable']:,} → "
              f"탈출 {r['escape']:,} ({r['escape'] / tot * 100:.0f}%), 피난 {r['refuge']:,} ({r['refuge'] / tot * 100:.0f}%), "
              f"대기 {r['wait']:,} ({r['wait'] / tot * 100:.0f}%)")
        print(f"   검증(실제 물때): 틀린 탈출 지시 {r['escape_fail']}칸, 피난처가 약속보다 일찍 잠김 {r['refuge_fail']}칸, "
              f"실제로는 30분 넘게 여유 있었는데 탈출 못 하게 한 칸 {r['missed_escape']}")
        if a.path_at and ts == a.path_at:
            path_info = export_path(G, D, t_now, base + f"_탈출로_{ts[11:].replace(':', '')}.json", a)
        results.append(r)
        Ds.append(D)
    with open(base + ".csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)
    draw(base + ".png", A, G, results, Ds, path_info)
    print("저장:", base + ".csv", base + ".png")


if __name__ == "__main__":
    main()
