"""퇴로 마감 지도 — 갯벌 칸마다 "몇 시까지 출발하면 육지까지 걸어 나올 수 있나"

계산 (인수인계 3장 핵심 기능 1·③)
  1) 지형: BADA2024 수심 → 약최저저조면 위 높이 (높이 = Z0 − 수심). 해안선 위 점은 약최고고조면 높이로 둠 [가정]
     → 둘 사이를 직선 보간해 격자(기본 50m)로. BADA 간격이 약 150m라 실제 해상도는 150m
  2) 갯골(상세) 안의 칸은 DROP m 낮춤 [가정: 갯골 깊이 자료 없음]
  3) 매 분 물높이(관측소 실측)로: 바다와 물로 이어진 칸만 물에 잠김 (고립된 웅덩이는 안 셈)
     → 물 깊이 DEPTH m 이상이면 그 칸은 지나갈 수 없음 → 그 칸이 막히는 시각 T_close
  4) "가장 오래 열린 길": 육지(해안선)에서 거꾸로 퍼지며, 칸 c에서 출발해 길 위 모든 칸을 막히기 전에 지나
     육지에 닿을 수 있는 가장 늦은 출발 시각 L(c) = max(경로) min(경로 위 칸의 T_close − 거기까지 걸리는 시간)
     걸음 속도 WALK m/s [확인 필요: 밤 갯벌 보행]
출력: 지도 PNG + 칸별 CSV (경도, 위도, 높이, 막히는 시각, 마지막 출발 시각)

실행 예 (영흥도, 2025-09-11 새벽 밀물, 관측소 실측):
  python gg_retreat_map.py --tide tide\\station_DT_0043_1m.csv --start "2025-09-11 00:50" --end "2025-09-11 06:30" --mark "2025-09-11 02:07"
필요: numpy scipy matplotlib (지형은 terrain\\youngheung\\ 의 gg_terrain_area.py 결과)
"""
import argparse
import csv
import heapq
import json
import os
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TRAP = 10          # 분: 자기 칸이 잠기기 이만큼 넘게 전에 퇴로가 끊기면 "함정 칸"
AREA = dict(name="영흥도", B=(126.38, 126.56, 37.19, 37.31), dir=os.path.join(HERE, "terrain", "youngheung"),
            key="youngheung", Z0=4.334,                      # TideBED 영흥도 평균해면 433.4cm (BADA 평균해면 기준 → 약최저저조면 기준)
            AHHW=8.72,                                       # 약최고고조면 ≈ 평균해면 + (M2+S2+K1+O1): 4.384 + 4.337 (우리 성분 2022~2024) [계산값]
            station=("영흥도 조위관측소", 37.23861, 126.42861))


def load_layers(A):
    d = A["dir"]
    pts = np.loadtxt(os.path.join(d, f"bada2024_{A['key']}.csv"), delimiter=",", skiprows=1, usecols=(0, 1, 2),
                     encoding="utf-8-sig")
    lon, lat, depth = pts[:, 0], pts[:, 1], pts[:, 2]

    def gj(name):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return json.load(open(p, encoding="utf-8"))["features"]
        with zipfile.ZipFile(p + ".zip") as z:
            return json.loads(z.read(name).decode("utf-8"))["features"]
    coast = gj(f"coast_artificial_{A['key']}.geojson") + gj(f"coast_natural_{A['key']}.geojson")
    gaetgol = gj(f"gaetgol_detail_2017_{A['key']}.geojson")
    return lon, lat, A["Z0"] - depth, coast, gaetgol


def parts(geom):
    g = geom["coordinates"]
    if geom["type"] in ("LineString",):
        return [np.array(g)[:, :2]]
    if geom["type"] in ("MultiLineString", "Polygon"):
        return [np.array(p)[:, :2] for p in g]
    if geom["type"] == "MultiPolygon":
        return [np.array(r)[:, :2] for poly in g for r in poly]
    return []


def densify(line, step_deg):
    out = [line[:1]]
    for a, b in zip(line[:-1], line[1:]):
        n = max(1, int(np.ceil(np.hypot(*(b - a)) / step_deg)))
        out.append(a + (b - a) * (np.arange(1, n + 1)[:, None] / n))
    return np.vstack(out)


def build_grid(A, res, drop):
    from scipy.interpolate import griddata
    from scipy import ndimage
    from matplotlib.path import Path
    B = A["B"]
    lat0 = (B[2] + B[3]) / 2
    kx, ky = 111320 * np.cos(np.radians(lat0)), 110950.0
    xs = np.arange(B[0] + res / kx / 2, B[1], res / kx)
    ys = np.arange(B[2] + res / ky / 2, B[3], res / ky)
    GX, GY = np.meshgrid(xs, ys)
    ny, nx = GX.shape
    lon, lat, elev, coast, gaetgol = load_layers(A)

    def cell(px, py):
        return ((py - B[2]) * ky / res).astype(int), ((px - B[0]) * kx / res).astype(int)

    # 해안선 → 막힘 칸(육지 경계) + 보간용 점(약최고고조면 높이)
    barrier = np.zeros((ny, nx), bool)
    cpts = []
    for f in coast:
        for ln in parts(f["geometry"]):
            dl = densify(ln, 10 / kx)
            iy, ix = cell(dl[:, 0], dl[:, 1])
            ok = (iy >= 0) & (iy < ny) & (ix >= 0) & (ix < nx)
            barrier[iy[ok], ix[ok]] = True
            cpts.append(densify(ln, 40 / kx))
    cpts = np.vstack(cpts)
    # 해안선 조각 사이 틈으로 바다가 섬 안까지 새지 않게 한 칸 두껍게
    barrier = ndimage.binary_dilation(barrier, iterations=1)
    # 바다 쪽: 깊은 바다 칸에서 시작해 해안선을 넘지 않고 이어진 칸 (4방향)
    iy, ix = cell(lon, lat)
    ok = (iy >= 0) & (iy < ny) & (ix >= 0) & (ix < nx)
    seeds = np.zeros((ny, nx), bool)
    deep = ok & (elev < -2.0)
    seeds[iy[deep], ix[deep]] = True
    seeds &= ~barrier
    lab, _ = ndimage.label(~barrier, structure=[[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    sea_side = np.isin(lab, np.unique(lab[seeds])) & ~barrier
    # 높이: BADA 점 + 해안선 점(약최고고조면)으로 직선 보간
    P = np.r_[np.c_[lon * kx, lat * ky], np.c_[cpts[:, 0] * kx, cpts[:, 1] * ky]]
    V = np.r_[elev, np.full(len(cpts), A["AHHW"])]
    Z = griddata(P, V, (GX * kx, GY * ky), method="linear")
    Zn = griddata(P, V, (GX * kx, GY * ky), method="nearest")
    Z = np.where(np.isfinite(Z), Z, Zn)
    # 갯골 칸 낮추기
    ggl = np.zeros((ny, nx), bool)
    if drop > 0:
        cc = np.c_[GX.ravel(), GY.ravel()]
        for f in gaetgol:
            for ring in parts(f["geometry"]):
                iy2, ix2 = cell(ring[:, 0], ring[:, 1])
                ok2 = (iy2 >= 0) & (iy2 < ny) & (ix2 >= 0) & (ix2 < nx)
                ggl[iy2[ok2], ix2[ok2]] = True                       # 꼭짓점이 걸친 칸
                x0, x1, y0, y1 = ring[:, 0].min(), ring[:, 0].max(), ring[:, 1].min(), ring[:, 1].max()
                sel = np.nonzero((cc[:, 0] >= x0) & (cc[:, 0] <= x1) & (cc[:, 1] >= y0) & (cc[:, 1] <= y1))[0]
                if len(sel):
                    inside = Path(ring).contains_points(cc[sel])
                    ggl.ravel()[sel[inside]] = True                  # 가운데가 안에 든 칸
        ggl &= sea_side
        Z = np.where(ggl, Z - drop, Z)
    land = ~sea_side & ~barrier
    return dict(GX=GX, GY=GY, Z=Z, sea=sea_side, barrier=barrier, land=land, seeds=seeds, gaetgol=ggl,
                res=res, kx=kx, ky=ky)


def load_tide(path, start, end):
    t0, t1 = np.datetime64(start.replace(" ", "T")), np.datetime64(end.replace(" ", "T"))
    T, H = [], []
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if "time" in r and "date" not in r:          # 바다누리 정리 CSV
                t = np.datetime64(r["time"][:16].replace(" ", "T"))
                v = r["tide_cm"]
            else:                                         # 관측소 API CSV
                t = np.datetime64(f"{r['date'][:4]}-{r['date'][4:6]}-{r['date'][6:]}T{r['time']}")
                v = r["obs_cm"]
            if t0 <= t <= t1:
                T.append(t)
                H.append(float(v) if v not in ("", None) else np.nan)
    T, H = np.array(T, dtype="datetime64[m]"), np.array(H)
    o = np.argsort(T)
    T, H = T[o], H[o]
    H[H == 0] = np.nan                                    # 0으로 적힌 빈 값
    ok = np.isfinite(H)
    H = np.interp(np.arange(len(H)), np.nonzero(ok)[0], H[ok])
    k = np.ones(5) / 5                                    # 5분 평균 (1분 출렁임 줄임)
    Hs = np.convolve(np.r_[[H[0]] * 2, H, [H[-1]] * 2], k, "valid")
    return T, Hs / 100.0                                  # m


def close_times(G, T, H, depth_lim):
    """칸이 막히는 첫 시각(분 번호): 바다와 이어진 물 깊이 ≥ depth_lim"""
    from scipy import ndimage
    Z, sea, seeds = G["Z"], G["sea"], G["seeds"]
    tc = np.full(Z.shape, np.inf)
    st8 = np.ones((3, 3))
    for i, h in enumerate(H):
        wet = sea & (Z < h)
        lab, _ = ndimage.label(wet, structure=st8)
        conn = np.isin(lab, np.unique(lab[seeds & wet]))
        conn &= lab > 0
        blocked = conn & (h - Z >= depth_lim) & ~np.isfinite(tc)
        tc[blocked] = i
    return tc


def latest_start(G, tc, walk):
    """가장 오래 열린 길: 육지에서 거꾸로. 값 = 마지막 출발 분 번호"""
    sea, barrier = G["sea"], G["barrier"]
    ny, nx = sea.shape
    res = G["res"]
    L = np.full((ny, nx), -np.inf)
    heap = []
    for iy, ix in zip(*np.nonzero(barrier)):
        L[iy, ix] = np.inf
        heapq.heappush(heap, (-np.inf, iy, ix))
    nb = [(-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1), (-1, -1, 1.4142), (-1, 1, 1.4142), (1, -1, 1.4142), (1, 1, 1.4142)]
    step_min = res / walk / 60.0
    done = np.zeros((ny, nx), bool)
    while heap:
        negv, iy, ix = heapq.heappop(heap)
        if done[iy, ix]:
            continue
        done[iy, ix] = True
        v = -negv
        for dy, dx, dd in nb:
            y, x = iy + dy, ix + dx
            if 0 <= y < ny and 0 <= x < nx and sea[y, x] and not done[y, x]:
                cand = min(tc[y, x], v - dd * step_min)
                if cand > L[y, x]:
                    L[y, x] = cand
                    heapq.heappush(heap, (-cand, y, x))
    return L


def hhmm(T, i):
    if not np.isfinite(i):
        return "끝까지 열림"
    i = int(np.floor(i))
    return str(T[min(max(i, 0), len(T) - 1)])[11:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tide", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--mark", help="이 시각에 '지금 출발하면 나올 수 있는 곳'을 따로 표시")
    ap.add_argument("--res", type=float, default=50.0)
    ap.add_argument("--walk", type=float, default=0.5, help="m/s [확인 필요]")
    ap.add_argument("--depth", type=float, default=0.30, help="통행 불가 물 깊이 m")
    ap.add_argument("--drop", type=float, default=0.5, help="갯골 칸 낮춤 m [가정]")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args()
    A = AREA
    G = build_grid(A, a.res, a.drop)
    T, H = load_tide(a.tide, a.start, a.end)
    print(f"{A['name']} 격자 {G['Z'].shape[1]}×{G['Z'].shape[0]} ({a.res:.0f}m), 바다 쪽 칸 {G['sea'].sum():,}, "
          f"갯골 칸 {G['gaetgol'].sum():,}, 해안선 칸 {G['barrier'].sum():,}")
    print(f"물때 {T[0]} ~ {T[-1]} ({len(T)}분), 물높이 {H.min():.2f} → {H.max():.2f} m, 가장 낮은 때 {T[np.argmin(H)]}")
    tc = close_times(G, T, H, a.depth)
    L = latest_start(G, tc, a.walk)
    i0 = int(np.argmin(H))
    walkable = G["sea"] & (tc > i0)                       # 가장 낮은 물때에 걸을 수 있던 갯벌 칸
    cell_km2 = (a.res / 1000) ** 2
    fin = walkable & np.isfinite(tc) & np.isfinite(L)
    lead = tc[fin] - L[fin]
    print(f"가장 낮은 물때에 걸을 수 있던 갯벌 {walkable.sum():,}칸 (약 {walkable.sum() * cell_km2:.1f}km²)")
    print(f"자기 칸이 막히기 전에 퇴로가 먼저 끊기는 칸 ('밀물은 뒤에서 먼저 온다'): 10분 넘게 먼저 {np.mean(lead > 10) * 100:.0f}%, "
          f"30분 넘게 {np.mean(lead > 30) * 100:.0f}%, 60분 넘게 {np.mean(lead > 60) * 100:.0f}% (최대 {lead.max():.0f}분), "
          f"10분 넘게 먼저인 칸 {int(np.sum(lead > TRAP)):,}칸")
    Lw = L[walkable]
    for q in (10, 50, 90):
        print(f"   마지막 출발 시각 {q}%: {hhmm(T, np.percentile(Lw[np.isfinite(Lw)], q))}")
    mark_i = None
    if a.mark:
        mark_i = int((np.datetime64(a.mark.replace(" ", "T")) - T[0]) / np.timedelta64(1, "m"))
        late = walkable & (L < mark_i)
        print(f"{a.mark} 기준: 이미 퇴로가 끊긴 갯벌 {late.sum():,}칸 ({late.sum() / walkable.sum() * 100:.0f}%), "
              f"아직 나올 수 있는 갯벌 {(walkable & (L >= mark_i)).sum():,}칸")
    out = a.out or os.path.join(HERE, "tide", f"퇴로마감_{A['key']}_{str(T[0])[:10]}_w{a.walk}_d{a.drop}")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out + ".csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["lon", "lat", "height_DL_m", "gaetgol", "close_time", "last_start"])
        for iy, ix in zip(*np.nonzero(walkable)):
            w.writerow([f"{G['GX'][iy, ix]:.5f}", f"{G['GY'][iy, ix]:.5f}", f"{G['Z'][iy, ix]:.2f}", int(G["gaetgol"][iy, ix]),
                        hhmm(T, tc[iy, ix]), hhmm(T, L[iy, ix])])
    if not a.no_plot:
        draw(out + ".png", A, G, T, H, tc, L, walkable, mark_i, a)
    print("저장:", out + ".csv", "" if a.no_plot else out + ".png")


def draw(out, A, G, T, H, tc, L, walkable, mark_i, a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch
    for name in ("Malgun Gothic", "Noto Sans CJK KR", "AppleGothic", "NanumGothic"):
        if any(f.name == name for f in fm.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False
    INK, INK2, MUTED = "#0b0b0b", "#52514e", "#9a9993"
    LAND, SEA = "#e9e7e1", "#c9d3df"
    B = A["B"]
    i0 = int(np.argmin(H))
    # 마지막 출발 시각을 30분 구간으로 (가장 낮은 물때 기준 몇 분 뒤)
    edges = [i0 + m for m in (0, 30, 60, 90, 120, 150, 180)]
    labels = []
    for e0, e1 in zip(edges[:-1], edges[1:]):
        labels.append(f"{hhmm(T, e0)}~{hhmm(T, e1)}")
    labels.append(f"{hhmm(T, edges[-1])} 뒤")
    cols = ["#7a1f12", "#b4401f", "#eb6834", "#eda100", "#f3cf6b", "#bfe3cf", "#1baf7a"]   # 이를수록 진한 주황·빨강
    Lm = np.where(walkable, np.minimum(L, edges[-1] + 1), np.nan)
    fig = plt.figure(figsize=(13.5, 8.6), facecolor="white")
    ax = fig.add_axes([0.04, 0.07, 0.64, 0.80])
    base = np.where(G["land"] | G["barrier"], 0, 1).astype(float)
    ax.pcolormesh(G["GX"], G["GY"], base, cmap=ListedColormap([LAND, SEA]), shading="auto", zorder=0)
    ax.pcolormesh(G["GX"], G["GY"], Lm, cmap=ListedColormap(cols), norm=BoundaryNorm(edges + [edges[-1] + 2], len(cols)),
                  shading="auto", zorder=1)
    if mark_i is not None:
        late = walkable & (L < mark_i)
        ax.contour(G["GX"], G["GY"], late.astype(float), levels=[0.5], colors=[INK], linewidths=1.2, zorder=3)
    ax.contour(G["GX"], G["GY"], (G["gaetgol"] & walkable).astype(float), levels=[0.5], colors=["#2a78d6"],
               linewidths=0.6, zorder=2)
    # 함정 칸: 내 자리는 아직 안 잠겼는데 퇴로가 TRAP분 넘게 먼저 끊김 ('밀물은 뒤에서 먼저 온다')
    with np.errstate(invalid="ignore"):
        trap = walkable & np.isfinite(L) & ((tc - L) > TRAP)
    ax.scatter(G["GX"][trap], G["GY"][trap], s=2.2, color=INK, linewidths=0, zorder=4)
    name, sy, sx = A["station"]
    ax.scatter([sx], [sy], s=50, color=INK, edgecolor="white", zorder=5)
    ax.annotate(name, (sx, sy), xytext=(8, 6), textcoords="offset points", fontsize=10, color=INK,
                bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.2), zorder=5)
    ax.set_xlim(B[0], B[1])
    ax.set_ylim(B[2], B[3])
    ax.set_aspect(1 / np.cos(np.radians((B[2] + B[3]) / 2)))
    ax.tick_params(colors=INK2, labelsize=9)
    for s in ax.spines.values():
        s.set_color("#d8d7d2")
    kx = G["kx"]
    x0, y0 = B[0] + (B[1] - B[0]) * 0.04, B[2] + (B[3] - B[2]) * 0.05
    ax.plot([x0, x0 + 1000 / kx], [y0, y0], color=INK, lw=2.5, zorder=6)
    ax.text(x0 + 500 / kx, y0 + 0.0015, "1 km", ha="center", fontsize=9.5, color=INK, zorder=6)
    day = str(T[0])[:10]
    fig.text(0.04, 0.95, f"{A['name']} 퇴로 마감 시각 — 이 시각까지 출발하면 육지까지 걸어 나올 수 있음", fontsize=16.5,
             weight="bold", color=INK)
    fig.text(0.04, 0.915, f"{day} 밤 밀물 · 물높이는 {name} 실측(5분 평균) · 보행 {a.walk}m/s · 물 깊이 {a.depth * 100:.0f}cm부터 통행 불가 · "
             f"갯골 칸 {a.drop}m 낮춤 [가정]", fontsize=10.5, color=INK2)
    h1 = [Patch(fc=c, ec="none", label=l) for c, l in zip(cols, labels)]
    fig.legend(handles=h1, loc="upper left", bbox_to_anchor=(0.70, 0.87), frameon=False, fontsize=10,
                     title=f"마지막 출발 시각 (가장 낮은 물때 {hhmm(T, i0)} 기준)", title_fontsize=10.5, alignment="left")
    from matplotlib.lines import Line2D
    h2 = [Line2D([], [], color=INK, marker="o", ls="none", markersize=3,
                 label=f"검은 점: 내 자리는 아직 안 잠겼는데\n퇴로가 {TRAP}분 넘게 먼저 끊기는 곳"),
          Line2D([], [], color="#2a78d6", lw=1, label="파란 선: 갯골(상세 2017) 칸 테두리"),
          Patch(fc=SEA, ec="none", label="가장 낮은 물때에도 물속"), Patch(fc=LAND, ec="none", label="육지")]
    if mark_i is not None:
        h2.insert(0, Line2D([], [], color=INK, lw=1.2, label=f"검은 선과 바다 사이: {hhmm(T, mark_i)}에\n이미 퇴로가 끊긴 곳"))
    fig.legend(handles=h2, loc="upper left", bbox_to_anchor=(0.70, 0.575), frameon=False, fontsize=10, alignment="left")
    # 물높이 작은 그림
    bx = fig.add_axes([0.715, 0.165, 0.26, 0.12])
    hrs = (T - T[0]) / np.timedelta64(1, "h")
    bx.plot(hrs, H, color=INK, lw=1.6)
    if mark_i is not None:
        bx.axvline(mark_i / 60, color=INK2, lw=1, ls=(0, (3, 2)))
    ticks = np.arange(0, hrs[-1] + 0.01, 1.0)
    bx.set_xticks(ticks, [str(T[0] + np.timedelta64(int(t * 60), "m"))[11:16] for t in ticks], fontsize=8.5, color=INK2)
    bx.tick_params(axis="y", labelsize=8.5, colors=INK2)
    bx.set_title(f"{name} 실측 물높이 (m, 약최저저조면 기준)", fontsize=9.5, color=INK, loc="left")
    for s in ("top", "right"):
        bx.spines[s].set_visible(False)
    fig.text(0.70, 0.02, "출처: 국립해양조사원 BADA2024 격자형수심(150m),\n개방海 갯골·해안선, 조위관측소 실측(공공데이터포털)\n"
             "해안선 높이 = 약최고고조면 [가정], BADA 빈칸은 직선 보간\n→ 해안 쪽 갯벌 높이는 추정값",
             fontsize=8.5, color=MUTED, va="bottom", linespacing=1.45)
    fig.savefig(out, dpi=150)


if __name__ == "__main__":
    main()
