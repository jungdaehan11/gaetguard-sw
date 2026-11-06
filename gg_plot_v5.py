"""gg_lead_v5.py 비행 기록 → 한 장 그림 (발표용)
  ① 지도: 판정 바탕 + 드론 길(처음·새 길) + 가상의 사람이 걸은 길 + 주의·신고·구조품 자리
  ② 시간: 위험도(정상·주의·신고)와 빛 신호(켜짐·깜빡임)
  ③ 같은 사람을 드론 없이 판정만 돌린 결과(gg_redecide.simulate)와 걸은 거리 비교
  ④ 실제 물때(실측)로 검증
실행 (계산 폴더에서): python gg_plot_v5.py gg5_log_....csv
"""
import csv
import json
import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load(path):
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    num = ("t", "person_n", "person_e", "est_n", "est_e", "est_err", "drone_n", "drone_e", "alt", "gap")
    for r in rows:
        for k in num:
            r[k] = float(r[k]) if r.get(k) not in ("", None, "None") else np.nan
    meta = json.load(open(path.replace(".csv", "_path.json"), encoding="utf-8"))
    if meta.get("version") != "v5":
        sys.exit("v5 기록이 아닙니다")
    return rows, meta


def spans(t, vals):
    out, i0 = [], 0
    for i in range(1, len(vals) + 1):
        if i == len(vals) or vals[i] != vals[i0]:
            out.append((t[i0], t[min(i, len(t) - 1)], vals[i0]))
            i0 = i
    return out


def main(path):
    import gg_decide as gd
    import gg_redecide as rd
    import gg_retreat_map as rm
    rows, meta = load(path)
    G = rm.build_grid(rm.AREA, 50.0, 0.5)
    T, O, P = gd.load(meta["tide"])
    t0 = np.datetime64(meta["t0"].replace(" ", "T"))
    opt = types.SimpleNamespace(res=50.0, walk=0.5, depth=0.30, drop=0.5)
    _, D = gd.decide_at(G, T, O, P, t0, opt)
    W = rd.Water(G, T, O, P, t0, 0.30)
    tct = W.truth()
    lat0, lon0 = meta["origin"]
    kx, ky = G["kx"], G["ky"]
    to_ll = lambda n, e: (lat0 + np.asarray(n) / ky, lon0 + np.asarray(e) / kx)      # noqa: E731
    hm = lambda sec: str(t0 + np.timedelta64(int(round(sec)), "s"))[11:16]            # noqa: E731
    tt = np.array([r["t"] for r in rows])
    pn, pe = np.array([r["person_n"] for r in rows]), np.array([r["person_e"] for r in rows])
    dn, de = np.array([r["drone_n"] for r in rows]), np.array([r["drone_e"] for r in rows])

    # 드론 없이 판정만 (같은 사람, 같은 물)
    ref = rd.simulate(G, W, tct, (lat0, lon0), (0.0, 0.0), rd.BEHAVIORS[meta["behavior_name"]], redecide=True)
    eng = rd.Redecider(G, W, (lat0, lon0), 0.5)
    tf = tct.ravel()

    # 검증
    ks = [eng.cell(n, e) for n, e in zip(pn, pe)]
    arrived = meta["end_reason"] == "사람 도착"
    lines = []
    if arrived:
        mm = min(tf[k] - t / 60 for k, t in zip(ks, tt) if eng.sea[k] and np.isfinite(tf[k]))
        lines.append(f"· {hm(tt[-1])} 해안 도착")
        lines.append(f"· 지나간 칸 중 가장 아슬아슬한 곳도 지난 뒤 {mm:.0f}분에 잠김")
    else:
        k = ks[-1]
        water = float(tf[k]) if eng.sea[k] else np.inf
        lines.append(f"· 사람이 멈춘 칸 물 30cm: {hm(water * 60) if np.isfinite(water) else '계산 끝까지 없음'}")
        if meta["alert"] is not None and np.isfinite(water):
            lines.append(f"· 신고 {hm(meta['alert'] * 60)} → 물 차기 {water - meta['alert']:.0f}분 전")
        for d in meta["drops"]:
            lines.append(f"· 구조품 {hm(d['t'])} → 물 차기 {water - d['t'] / 60:.0f}분 전\n   (사람에게서 {d['offset']:.1f}m)")
    rshort = rd.short(ref, t0).split("\n")
    if "est_err" in rows[0] and meta["person_src"].startswith("카메라"):
        er = np.array([r["est_err"] for r in rows])
        lines.append(f"· 카메라로 구한 사람 위치 오차\n   중앙값 {np.median(er):.2f}m, 95% {np.percentile(er, 95):.2f}m")

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
    INK, INK2, MUTED, LAND, SEA, GRID = "#0b0b0b", "#52514e", "#9a9993", "#e9e7e1", "#c9d3df", "#e6e5e1"
    ON_C, BLINK_C, NEW_C = "#2a78d6", "#eb6834", "#eb6834"
    TIERC = dict(zip(("정상", "주의", "신고"), gd.DEC_COLS))

    fig = plt.figure(figsize=(15.5, 9.4), facecolor="white")
    fig.text(0.03, 0.955, "드론이 30초마다 다시 판정받아 움직임 — 시뮬레이터 (ArduCopter 3.6.12, 실제 크기)", fontsize=17, weight="bold", color=INK)
    fig.text(0.03, 0.925, f"영흥도 {meta['t0']} · 가상의 사람: {meta['behavior_name']} · 사람 위치: {meta['person_src']} · "
             f"물: {meta['water'].split(' (')[0]} · 끝난 이유: {meta['end_reason']}", fontsize=10.5, color=INK2)

    # ① 지도
    ax = fig.add_axes([0.02, 0.17, 0.40, 0.71])
    base = np.where(G["land"] | G["barrier"], 0, 1).astype(float)
    ax.pcolormesh(G["GX"], G["GY"], base, cmap=ListedColormap([LAND, SEA]), shading="auto", zorder=0)
    ax.pcolormesh(G["GX"], G["GY"], gd.dec_classes(D["dec"]), cmap=ListedColormap(gd.DEC_COLS), vmin=-0.5, vmax=2.5,
                  shading="auto", alpha=0.45, zorder=1)
    for i, p in enumerate(meta["paths"]):
        q = np.array(p["pts"])
        la, lo = to_ll(q[:, 0], q[:, 1])
        ax.plot(lo, la, color=INK if i == 0 else NEW_C, lw=1.1 if i == 0 else 1.8, ls=(0, (4, 3)), zorder=3 if i == 0 else 6)
    dla, dlo = to_ll(dn, de)
    ax.plot(dlo, dla, color="#62615d", lw=0.9, zorder=4)
    pla, plo = to_ll(pn, pe)
    ax.plot(plo, pla, color=INK, lw=2.2, zorder=5)
    ax.scatter([lon0], [lat0], s=60, color="white", edgecolor=INK, zorder=8)
    ax.annotate(f"출발 {meta['t0'][11:]}", (lon0, lat0), xytext=(8, -14), textcoords="offset points", fontsize=10, color=INK)
    end_mk = ("*", 160) if arrived else ("s", 70)
    ax.scatter([plo[-1]], [pla[-1]], s=end_mk[1], marker=end_mk[0], color=INK, edgecolor="white", zorder=9)
    marks = []
    for key, lab in (("caution", "주의"), ("alert", "신고")):
        if meta[key] is not None and not (key == "caution" and meta["caution"] == meta["alert"]):
            i = int(np.argmin(np.abs(tt - meta[key] * 60)))
            marks.append((plo[i], pla[i], f"{lab} {hm(meta[key] * 60)}", TIERC[lab]))
    for d in meta["drops"]:
        la, lo = to_ll(d["pos"][0], d["pos"][1])
        marks.append((float(lo), float(la), f"구조품 {hm(d['t'])}", BLINK_C))
    for k, (x, y, lab, c) in enumerate(marks):
        ax.scatter([x], [y], s=70, marker="D", color=c, edgecolor="white", linewidth=1, zorder=10)
        ax.annotate(lab, (x, y), xytext=(12, 10 - 16 * k), textcoords="offset points", fontsize=10.5, color=INK,
                    bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.2), zorder=11)
    allla = np.r_[pla, *[to_ll(np.array(p["pts"])[:, 0], 0)[0] for p in meta["paths"]]]
    alllo = np.r_[plo, *[to_ll(0, np.array(p["pts"])[:, 1])[1] for p in meta["paths"]]]
    pad = 0.004
    ax.set_xlim(alllo.min() - pad * 1.6, alllo.max() + pad * 1.6)
    ax.set_ylim(allla.min() - pad, allla.max() + pad)
    ax.set_aspect(1 / np.cos(np.radians(lat0)))
    ax.set_xticks([])
    ax.set_yticks([])
    for s_ in ax.spines.values():
        s_.set_color("#d8d7d2")
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    bx, by = x0 + (x1 - x0) * 0.05, y0 + (y1 - y0) * 0.03
    ax.plot([bx, bx + 500 / kx], [by, by], color=INK, lw=2.5, zorder=12)
    ax.text(bx + 250 / kx, by + (y1 - y0) * 0.012, "500 m", ha="center", fontsize=9.5, color=INK, zorder=12)
    ax.set_title("① 위에서 본 모습", loc="left", fontsize=13, color=INK, pad=8)
    h = [Line2D([], [], color=INK, lw=1.1, ls=(0, (4, 3)), label="처음 판정 길 (드론 길)"),
         Line2D([], [], color=NEW_C, lw=1.1, ls=(0, (4, 3)), label="다시 판정한 길")] if len(meta["paths"]) > 1 else \
        [Line2D([], [], color=INK, lw=1.1, ls=(0, (4, 3)), label="판정 길 (드론 길)")]
    h += [Line2D([], [], color=INK, lw=2.2, label="가상의 사람이 걸은 길"), Line2D([], [], color="#62615d", lw=0.9, label="드론")]
    h += [Patch(fc=c, alpha=0.45, label="바탕(처음 판정 때): " + lab.split(":")[0]) for c, lab in zip(gd.DEC_COLS, gd.DEC_LABELS)]
    fig.legend(handles=h, loc="lower left", bbox_to_anchor=(0.02, 0.02), ncol=2, fontsize=9.5, frameon=False)

    # ② 시간: 위험도·빛
    tmin = tt / 60.0
    xmax = max(tmin[-1], ref["track"][-1, 0]) + 1
    a = fig.add_axes([0.47, 0.66, 0.50, 0.2])
    for lab, y in (("위험도", 1.0), ("빛", 0.0)):
        a.text(-0.01, y, lab, transform=a.get_yaxis_transform(), ha="right", va="center", fontsize=10.5, color=INK)
    for s0, s1, v in spans(tmin, [r["tier"] for r in rows]):
        a.barh(1.0, s1 - s0, left=s0, height=0.55, color=TIERC[v], edgecolor="white", linewidth=1)
        if s1 - s0 > xmax * 0.06:
            a.text((s0 + s1) / 2, 1.0, v, ha="center", va="center", fontsize=10, color="white", weight="bold")
    for s0, s1, v in spans(tmin, [r["signal"] for r in rows]):
        a.barh(0.0, max(s1 - s0, xmax * 0.002), left=s0, height=0.55, color=ON_C if v == "켜짐" else BLINK_C, lw=0)
    last_x, lvl = -1e9, 0
    for s0, s1, v in spans(tmin, [r["phase"] for r in rows]):
        if v in ("출발 대기", "새 길 대기", "구조품 투하", "멈춤·구조 대기") and s1 - s0 > 0.3:
            x = (s0 + s1) / 2
            lvl = 1 - lvl if x - last_x < xmax * 0.09 else 0          # 가까우면 아래로 엇갈려 (겹치지 않게)
            a.annotate(v, (x, -0.32 - 0.3 * lvl), ha="center", va="top", fontsize=9.5, color=INK)
            last_x = x
    a.set_ylim(-1.0, 1.45)
    a.set_xlim(0, xmax)
    a.set_yticks([])
    a.set_title("② 위험도와 빛 신호 (파랑 = 켜짐: 따라오세요, 주황 = 깜빡임: 와서 멈추세요)", loc="left", fontsize=12, color=INK, pad=8)
    for s in ("top", "right", "left"):
        a.spines[s].set_visible(False)
    a.tick_params(colors=INK2)
    xt = a.get_xticks()
    a.set_xticks(xt[(xt >= 0) & (xt <= xmax)])
    a.set_xticklabels([hm(x * 60) for x in a.get_xticks()])

    # ③ 걸은 거리: 드론 시뮬레이터 vs 판정만
    b = fig.add_axes([0.47, 0.36, 0.50, 0.2])
    walked = np.r_[0, np.cumsum(np.hypot(np.diff(pn), np.diff(pe)))]
    rt = ref["track"]
    rw = np.r_[0, np.cumsum(np.hypot(np.diff(rt[:, 1]), np.diff(rt[:, 2])))]
    b.plot(rt[:, 0], rw, color=MUTED, lw=2.4, ls=(0, (5, 3)), label="드론 없이 판정만 (gg_redecide)")
    b.plot(tmin, walked, color=INK, lw=2.0, label="드론 시뮬레이터 (이 기록)")
    b.set_xlim(0, xmax)
    b.set_ylim(0, None)
    b.grid(axis="y", color=GRID, lw=0.8)
    b.set_axisbelow(True)
    for s in ("top", "right", "left"):
        b.spines[s].set_visible(False)
    b.tick_params(colors=INK2)
    xt = b.get_xticks()
    b.set_xticks(xt[(xt >= 0) & (xt <= xmax)])
    b.set_xticklabels([hm(x * 60) for x in b.get_xticks()])
    b.legend(loc="upper left", fontsize=9.5, frameon=False)
    b.set_title("③ 가상의 사람이 걸은 거리 (m) — 같은 사람·같은 물로 비교", loc="left", fontsize=12, color=INK, pad=8)

    # ④ 결과
    ev = [e for e in meta["events"] if e[1] not in ("출발", "같은 길 계속", "도착")]
    def short_ev(x):
        x = x.replace("걸으라고 했는데 3분 넘게 안 움직임", "3분 안 움직임").split(" [")[0].split(" (")[0]
        return x if len(x) <= 30 else x[:29] + "…"
    evl = [f"{str(t0 + np.timedelta64(int(round(e[0])), 's'))[11:16]} {short_ev(e[1])}" for e in ev][:8]
    fig.text(0.47, 0.27, "④ 결과 (실제 물때 = 그 밤 실측으로 검증)", fontsize=12, color=INK)
    fig.text(0.47, 0.245, "\n".join(lines), fontsize=10.5, color=INK, va="top", linespacing=1.45)
    fig.text(0.47, 0.062, "드론 없이 판정만: " + " · ".join(x.strip("()").lstrip("→ ") for x in rshort[:3]), fontsize=10, color=INK2, va="top")
    fig.text(0.76, 0.245, "드론이 받은 판정·동작\n" + "\n".join(evl), fontsize=9.5, color=INK, va="top", linespacing=1.5)
    fig.text(0.97, 0.02, "가정: 투하 높이·거리, 사람 반응, 카메라 오차, 갯골 칸 0.5m 낮춤, 물 30cm 통행 불가. 실기체·현장 검증 전. "
             "출처: BADA2024, 개방海 갯골·해안선, 영흥도 조위관측소 실측", ha="right", fontsize=8.3, color=MUTED)
    out = path.replace(".csv", ".png")
    fig.savefig(out, dpi=140)
    print("그림 저장:", out)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python gg_plot_v5.py gg5_log_....csv")
    else:
        main(sys.argv[1])
