"""gg_lead_v3.py 비행 기록(CSV)을 그림으로 → 발표용
실행: python gg_plot_v3.py gg_log_YYYYmmdd_HHMMSS.csv   (matplotlib 필요: python -m pip install matplotlib)
"""
import csv
import json
import math
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
from matplotlib import font_manager as fm                         # noqa: E402
from matplotlib.lines import Line2D                               # noqa: E402
from matplotlib.patches import Polygon, Circle                    # noqa: E402

for name in ("Malgun Gothic", "AppleGothic", "Pretendard", "NanumGothic", "Noto Sans CJK KR"):
    if any(f.name == name for f in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = name
        break
plt.rcParams["axes.unicode_minus"] = False

INK, INK2, GRID, CORRIDOR = "#1f1f1e", "#62615d", "#e6e5e1", "#ecebe7"
SIG = {"켜짐": ("#2a78d6", "o", "켜짐 = 따라오세요"),
       "깜빡임": ("#eb6834", "s", "깜빡임 = 멈추세요"),
       "좌우 쓸기": ("#1baf7a", "^", "좌우 쓸기 = 방향 트세요")}


def num(v):
    return None if v in ("", "None", None) else float(v)


def load(path):
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("t", "person_n", "person_e", "person_s", "offset", "drone_n", "drone_e", "alt", "gap",
                  "yaw_cmd", "yaw"):
            r[k] = num(r.get(k))
    meta = json.load(open(path.replace(".csv", "_path.json"), encoding="utf-8"))
    return rows, meta


def corridor(ax, pts, w):
    for (n1, e1), (n2, e2) in zip(pts, pts[1:]):
        L = math.hypot(n2 - n1, e2 - e1)
        un, ue = (n2 - n1) / L, (e2 - e1) / L
        rn, re_ = -ue * w, un * w
        poly = [(e1 + re_, n1 + rn), (e2 + re_, n2 + rn), (e2 - re_, n2 - rn), (e1 - re_, n1 - rn)]
        ax.add_patch(Polygon(poly, closed=True, fc=CORRIDOR, ec="none", zorder=0))
    for n, e in pts:
        ax.add_patch(Circle((e, n), w, fc=CORRIDOR, ec="none", zorder=0))
    ax.plot([p[1] for p in pts], [p[0] for p in pts], color="#b9b8b2", lw=1, ls=(0, (4, 3)), zorder=1)


def spans(rows, sig):
    out, start = [], None
    for a, b in zip(rows, rows[1:] + [None]):
        if a["signal"] == sig and start is None:
            start = a["t"]
        if start is not None and (b is None or b["signal"] != sig):
            out.append((start, a["t"]))
            start = None
    return out


def main(path):
    rows, meta = load(path)
    pts = meta["path"]
    fig = plt.figure(figsize=(15, 8.2), facecolor="white")
    fig.text(0.04, 0.95, "갯가드 빛길 안내 v3 · 시뮬레이터 비행 기록", fontsize=18, weight="bold", color=INK)
    fig.text(0.04, 0.93, f"가상의 사람이 꺾인 길 {sum(math.dist(a, b) for a, b in zip(pts, pts[1:])):.0f}m를 걷고, "
             f"드론이 길 위로 {meta['lead']:.0f}m 앞에서 고도 {meta['alt']:.0f}m로 앞장섬\n끝난 이유: {meta['end_reason']}",
             fontsize=11.5, color=INK2, va="top", linespacing=1.5)

    # ① 위에서 본 길
    ax = fig.add_axes([0.04, 0.07, 0.40, 0.77])
    corridor(ax, pts, meta["offpath_m"])
    ax.plot([r["person_e"] for r in rows], [r["person_n"] for r in rows], color=INK, lw=1.6, zorder=6)
    for sig, (c, mk, _) in SIG.items():
        sel = [r for i, r in enumerate(rows) if r["signal"] == sig and r["drone_n"] is not None and i % 2 == 0]
        ax.scatter([r["drone_e"] for r in sel], [r["drone_n"] for r in sel], s=34, color=c, marker=mk,
                   edgecolor="white", linewidth=1.2, zorder=4)
    ax.scatter([pts[0][1]], [pts[0][0]], s=90, color=INK, marker="o", zorder=5)
    ax.annotate("출발", (pts[0][1], pts[0][0]), xytext=(8, -14), textcoords="offset points", fontsize=11, color=INK)
    ax.scatter([pts[-1][1]], [pts[-1][0]], s=120, color=INK, marker="*", zorder=5)
    ax.annotate("도착(육지·피난 지점 가정)", (pts[-1][1], pts[-1][0]), xytext=(8, 4), textcoords="offset points",
                fontsize=11, color=INK)
    drift = next((r for r in rows if r["signal"] == "깜빡임"), None)
    if drift:
        far = max((r for r in rows if r["signal"] == "깜빡임"), key=lambda r: abs(r["offset"]))
        ax.annotate(f"길에서 {abs(far['offset']):.1f}m 벗어남\n→ 깜빡임, 드론 기다림", (far["person_e"], far["person_n"]),
                    xytext=(24, -30), textcoords="offset points", fontsize=11, color=INK,
                    arrowprops=dict(arrowstyle="-", color=INK2, lw=1))
    swept = {round(c, 1) for c in meta["corners"]
             if any(r["signal"] == "좌우 쓸기" and -1.0 <= c - r["person_s"] <= 5.0 for r in rows)}
    for k, (n, e) in enumerate(pts[1:-1]):
        if round(meta["corners"][k], 1) in swept:
            ax.annotate(f"꺾이는 곳 {k + 1}: 좌우 쓸기", (e, n), xytext=(8, 10) if k % 2 == 0 else (10, -20),
                        textcoords="offset points", fontsize=11, color=INK)
    ax.set_aspect("equal")
    ax.set_xlabel("동쪽 (m)", color=INK2)
    ax.set_ylabel("북쪽 (m)", color=INK2)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=INK2)
    ax.set_title("① 위에서 본 모습", loc="left", fontsize=13, color=INK, pad=10)
    hs = [Line2D([], [], color=INK, lw=2, label="사람이 걸은 길")]
    hs += [Line2D([], [], color=c, marker=mk, ls="none", markersize=8, markeredgecolor="white", label="드론 · " + lab)
           for c, mk, lab in SIG.values()]
    hs += [Polygon([(0, 0)], fc=CORRIDOR, label=f"안전한 길 폭 (±{meta['offpath_m']:.0f}m)")]
    ax.legend(handles=hs, loc="lower left", bbox_to_anchor=(1.02, 0.0), fontsize=10.5, frameon=False)

    # ②③ 시간에 따른 값 (같은 시간축, 각자 한 축)
    t = [r["t"] for r in rows]
    for i, (key, title, band, ylab) in enumerate((
            ("gap", "② 사람-드론 거리", (5, 10), "m"),
            ("offset", "③ 사람이 길 가운데서 벗어난 거리", None, "m"))):
        a = fig.add_axes([0.52, 0.56 - i * 0.46, 0.45, 0.31])
        for sig in ("깜빡임", "좌우 쓸기"):
            for t0, t1 in spans(rows, sig):
                a.axvspan(t0, t1, color=SIG[sig][0], alpha=0.14, lw=0)
        if band:
            a.axhspan(*band, color=CORRIDOR, lw=0)
            a.text(t[-1], band[1] - 0.3, "계획 5~10m", ha="right", va="top", fontsize=10, color=INK2)
        vals = [abs(r[key]) if key == "offset" else r[key] for r in rows]
        a.plot(t, vals, color=INK, lw=2)
        if key == "offset":
            a.axhline(meta["offpath_m"], color=SIG["깜빡임"][0], lw=1.2, ls=(0, (4, 3)))
            a.text(t[-1], meta["offpath_m"] + 0.08, f"이탈 기준 {meta['offpath_m']:.0f}m", ha="right", va="bottom",
                   fontsize=10, color=INK2)
        for sig in ("깜빡임", "좌우 쓸기"):
            for t0, t1 in spans(rows, sig):
                a.text((t0 + t1) / 2, 1.0, sig, transform=a.get_xaxis_transform(), ha="center", va="bottom",
                       fontsize=10, color=INK)
        a.set_title(title, loc="left", fontsize=13, color=INK, pad=16)
        a.set_ylabel(ylab, color=INK2)
        a.set_ylim(0, 11 if key == "gap" else 3)
        a.grid(axis="y", color=GRID, lw=0.8)
        a.set_axisbelow(True)
        for s in ("top", "right", "left"):
            a.spines[s].set_visible(False)
        a.spines["bottom"].set_color("#b9b8b2")
        a.tick_params(colors=INK2)
        if i == 1:
            a.set_xlabel("안내 시작 후 시간 (초)", color=INK2)
    out = path.replace(".csv", ".png")
    fig.savefig(out, dpi=130)
    print("그림 저장:", out)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python gg_plot_v3.py gg_log_....csv")
    else:
        main(sys.argv[1])
