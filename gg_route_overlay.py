"""처음부터 끝까지 한 장: 조위·지형 계산 → 판정 → 탈출로 → 드론 빛길 안내(시뮬레이터)

  - 바탕: 판정 시각의 탈출/피난/대기 지도 (gg_decide.py 와 같은 계산)
  - 검은 선: 그 판정이 고른 탈출로 (가장 오래 열린 길)
  - 파란 점·회색 선·검은 굵은 선: 시뮬레이터에서 빛 자국 · 드론 · 가상의 사람 (gg_lead_v4.py --path-json, 실제 크기)
  - 검증: 가상의 사람이 실제로 걸은 시각표(판정 + 반응 2분 + 시뮬레이터 시간)를 실제 물때와 맞대어
          지나간 모든 칸이 그 사람이 지나가기 전까지 막히지 않았는지 → 가장 아슬아슬했던 곳의 여유(분)

실행 예:
  python gg_route_overlay.py --tide tide\\station_DT_0043_1m_우리예측2022-2024.csv --time "2025-09-11 02:07" ^
      --route tide\\판정_youngheung_2025-09-11_탈출로_0207.json --log gg_log_....csv
"""
import argparse
import csv
import json
import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gg_decide as gd       # noqa: E402
import gg_retreat_map as rm  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tide", required=True)
    ap.add_argument("--time", required=True)
    ap.add_argument("--route", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    opt = types.SimpleNamespace(res=50.0, walk=0.5, depth=0.30, drop=0.5)
    A = rm.AREA
    G = rm.build_grid(A, opt.res, opt.drop)
    T, O, P = gd.load(a.tide)
    t_now = np.datetime64(a.time.replace(" ", "T"))
    res, D = gd.decide_at(G, T, O, P, t_now, opt)
    route = json.load(open(a.route, encoding="utf-8"))
    lat0, lon0 = route["start_lat"], route["start_lon"]
    kx, ky = G["kx"], G["ky"]
    rows = list(csv.DictReader(open(a.log, encoding="utf-8-sig")))
    f = lambda r, k: float(r[k]) if r.get(k) not in ("", None, "None") else np.nan   # noqa: E731
    tt = np.array([f(r, "t") for r in rows])
    pn, pe = np.array([f(r, "person_n") for r in rows]), np.array([f(r, "person_e") for r in rows])
    dn, de = np.array([f(r, "drone_n") for r in rows]), np.array([f(r, "drone_e") for r in rows])
    sn, se = np.array([f(r, "spot_n") for r in rows]), np.array([f(r, "spot_e") for r in rows])
    sig = [r.get("signal", "") for r in rows]
    to_ll = lambda n, e: (lat0 + n / ky, lon0 + e / kx)       # noqa: E731
    plat, plon = to_ll(pn, pe)
    # 검증: 사람이 지난 칸마다 (실제로 막힌 시각 − 사람이 거기 있던 시각)
    B = A["B"]
    iy = ((plat - B[2]) * ky / opt.res).astype(int)
    ix = ((plon - B[0]) * kx / opt.res).astype(int)
    ok = (iy >= 0) & (iy < G["Z"].shape[0]) & (ix >= 0) & (ix < G["Z"].shape[1])
    tct = D["tct"]
    clock_min = gd.REACT + tt / 60.0                              # 판정 시각부터 분
    sea_ok = ok & G["sea"][iy.clip(0, G["Z"].shape[0] - 1), ix.clip(0, G["Z"].shape[1] - 1)]
    left = tct[iy[sea_ok], ix[sea_ok]] - clock_min[sea_ok]
    worst = float(np.min(left)) if len(left) else np.nan
    wi = int(np.argmin(left)) if len(left) else 0
    t_end = clock_min[-1]
    print(f"판정 {a.time}: 탈출로 {route['length_m']}m, 남은 여유(판정) {route['slack_min']}분")
    print(f"시뮬레이터: 사람 {tt[-1] / 60:.1f}분 걸어 도착 → 실제 시각 {str(t_now + np.timedelta64(int(round(t_end)), 'm'))[11:16]}")
    print(f"실제 물때로 검증: 지나간 칸 {sea_ok.sum()}번 기록, 가장 아슬아슬했던 곳의 여유 {worst:.1f}분 "
          f"(사람이 지나간 뒤 {worst:.1f}분 만에 그 칸이 막힘)" + (" → 안전" if worst > 0 else " → 위험!"))
    # 그림
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    for name in ("Malgun Gothic", "Noto Sans CJK KR", "AppleGothic", "NanumGothic"):
        if any(ff.name == name for ff in fm.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False
    INK, INK2, MUTED, LAND, SEA = "#0b0b0b", "#52514e", "#9a9993", "#e9e7e1", "#c9d3df"
    COLS = gd.DEC_COLS
    fig = plt.figure(figsize=(12.5, 8.6), facecolor="white")
    fig.text(0.04, 0.955, "계산 → 판정 → 탈출로 → 드론 빛길 안내, 한 장으로 (시뮬레이터, 실제 크기)", fontsize=16, weight="bold", color=INK)
    fig.text(0.04, 0.92, f"영흥도 {a.time} · 그때까지의 정보(우리 예측 + 실측 보정 + 99% 여유)로 판정 → 고른 탈출로를 "
             f"ArduCopter 3.6.12 시뮬레이터 드론이 빛 자국으로 안내 · 가상의 사람 0.5m/s", fontsize=10.5, color=INK2)
    ax = fig.add_axes([0.04, 0.08, 0.62, 0.8])
    base = np.where(G["land"] | G["barrier"], 0, 1).astype(float)
    ax.pcolormesh(G["GX"], G["GY"], base, cmap=ListedColormap([LAND, SEA]), shading="auto", zorder=0)
    cls = gd.dec_classes(D["dec"])
    ax.pcolormesh(G["GX"], G["GY"], cls, cmap=ListedColormap(COLS), vmin=-0.5, vmax=2.5, shading="auto", alpha=0.6, zorder=1)
    rl = np.array(route["path_latlon"])
    ax.plot(rl[:, 1], rl[:, 0], color=INK, lw=1, ls=(0, (4, 3)), zorder=3)
    dlat, dlon = to_ll(dn, de)
    ax.plot(dlon, dlat, color="#62615d", lw=1.0, zorder=4)
    slat, slon = to_ll(sn, se)
    on = np.array([s == "켜짐" for s in sig])
    bl = np.array([s == "깜빡임" for s in sig])
    step = max(1, len(rows) // 600)
    ax.scatter(slon[on][::step], slat[on][::step], s=6, color="#2a78d6", zorder=5)
    ax.scatter(slon[bl], slat[bl], s=10, color="#eb6834", marker="s", zorder=6)
    ax.plot(plon, plat, color=INK, lw=2.0, zorder=7)
    ax.scatter([plon[0]], [plat[0]], s=60, color="white", edgecolor=INK, zorder=8)
    ax.scatter([plon[-1]], [plat[-1]], s=110, color=INK, marker="*", zorder=8)
    if len(left):
        wl = np.nonzero(sea_ok)[0][wi]
        ax.annotate(f"가장 아슬아슬한 곳: 지나간 뒤 {worst:.0f}분에 잠김", (plon[wl], plat[wl]), xytext=(18, -24),
                    textcoords="offset points", fontsize=10, color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8),
                    bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5), zorder=9)
    pad = 0.006
    ax.set_xlim(min(rl[:, 1].min(), plon.min()) - pad * 1.6, max(rl[:, 1].max(), plon.max()) + pad * 1.6)
    ax.set_ylim(min(rl[:, 0].min(), plat.min()) - pad, max(rl[:, 0].max(), plat.max()) + pad)
    ax.set_aspect(1 / np.cos(np.radians(lat0)))
    ax.set_xticks([])
    ax.set_yticks([])
    for s_ in ax.spines.values():
        s_.set_color("#d8d7d2")
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    bx, by = x0 + (x1 - x0) * 0.05, y0 + (y1 - y0) * 0.04
    ax.plot([bx, bx + 500 / kx], [by, by], color=INK, lw=2.5, zorder=9)
    ax.text(bx + 250 / kx, by + (y1 - y0) * 0.012, "500 m", ha="center", fontsize=9.5, color=INK, zorder=9)
    h = [Patch(fc=c, alpha=0.6, label=l.split(":")[0] + " (" + l.split(": ")[1].split(" (")[0] + ")") for c, l in zip(COLS, gd.DEC_LABELS)] + [
         Line2D([], [], color=INK, lw=1, ls=(0, (4, 3)), label="판정이 고른 탈출로"),
         Line2D([], [], color=INK, lw=2, label="가상의 사람이 걸은 길"),
         Line2D([], [], color="#2a78d6", marker="o", ls="none", markersize=4, label="빛 자국 · 켜짐 (따라오세요)"),
         Line2D([], [], color="#eb6834", marker="s", ls="none", markersize=5, label="빛 자국 · 깜빡임 (와서 멈추세요)"),
         Line2D([], [], color="#62615d", lw=1, label="드론이 날아간 길")]
    fig.legend(handles=h, loc="upper left", bbox_to_anchor=(0.68, 0.86), frameon=False, fontsize=10)
    arr = str(t_now + np.timedelta64(int(round(t_end)), "m"))[11:16]
    fig.text(0.685, 0.53, f"판정 {a.time[11:]}: 정상\n· 마지막 출발까지 {route['last_start_min']:.0f}분, 99% 여유·반응 2분 빼고 {route['slack_min']:.0f}분 남음\n"
             f"· 탈출로 {route['length_m']}m, 꺾이는 곳 {len(route['path_ne_m']) - 2}개\n\n시뮬레이터\n· 사람 {tt[-1] / 60:.0f}분 걸어 {arr} 해안 도착\n"
             f"· 모퉁이마다 깜빡임 → 멈춤 → 새 길로\n· 빛 자국은 사람 3m 앞이라 이 축척에선 겹쳐 보임\n\n실제 물때로 검증\n· 지나간 칸 중 가장 아슬아슬한 곳도\n  지나간 뒤 {worst:.0f}분에 잠김 → "
             + ("늦지 않음" if worst > 0 else "늦음"), fontsize=10.5, color=INK, va="top", linespacing=1.55)
    fig.text(0.685, 0.05, "가정: 갯골 칸 0.5m 낮춤, 보행 0.5m/s, 물 깊이 30cm 통행 불가,\n해안선 = 약최고고조면. 실기체·현장 검증 전\n"
             "출처: BADA2024, 개방海 갯골·해안선, 영흥도 조위관측소 실측", fontsize=8.5, color=MUTED, va="bottom", linespacing=1.45)
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.log)), "처음부터끝까지_" + os.path.basename(a.route).replace(".json", ".png"))
    fig.savefig(out, dpi=150)
    print("그림:", out)


if __name__ == "__main__":
    main()
