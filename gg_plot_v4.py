"""gg_lead_v4.py 비행 기록(CSV)을 그림으로 → 발표용
실행: python gg_plot_v4.py gg_log_YYYYmmdd_HHMMSS.csv   (matplotlib 필요: python -m pip install matplotlib)
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

INK, INK2, GRID, CORRIDOR, DRONE = "#1f1f1e", "#62615d", "#e6e5e1", "#ecebe7", "#9a9993"
ON_C, BLINK_C = "#2a78d6", "#eb6834"
NUMS = ["time", "t", "person_n", "person_e", "person_s", "offset", "drone_n", "drone_e", "alt", "gap",
        "spot_s", "spot_tn", "spot_te", "spot_n", "spot_e", "spot_err", "spot_off", "spot_ahead", "yaw_cmd", "yaw"]


def load(path):
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in NUMS:
            if k in r:
                r[k] = None if r[k] in ("", "None") else float(r[k])
    meta = json.load(open(path.replace(".csv", "_path.json"), encoding="utf-8"))
    if not str(meta.get("version", "")).startswith("v4"):
        sys.exit(f"{path} 는 v4 기록이 아닙니다 (v3 이하). v3 기록은 python gg_plot_v3.py {path} 로 그리세요.")
    return rows, meta


def corridor(ax, pts, w):
    for (n1, e1), (n2, e2) in zip(pts, pts[1:]):
        L = math.hypot(n2 - n1, e2 - e1)
        un, ue = (n2 - n1) / L, (e2 - e1) / L
        rn, re_ = -ue * w, un * w
        ax.add_patch(Polygon([(e1 + re_, n1 + rn), (e2 + re_, n2 + rn), (e2 - re_, n2 - rn), (e1 - re_, n1 - rn)],
                             closed=True, fc=CORRIDOR, ec="none", zorder=0))
    for n, e in pts:
        ax.add_patch(Circle((e, n), w, fc=CORRIDOR, ec="none", zorder=0))
    ax.plot([p[1] for p in pts], [p[0] for p in pts], color="#b9b8b2", lw=1, ls=(0, (4, 3)), zorder=1)


def spans(rows, test):
    out, start = [], None
    for a, b in zip(rows, rows[1:] + [None]):
        if test(a) and start is None:
            start = a["t"]
        if start is not None and (b is None or not test(b)):
            out.append((start, a["t"]))
            start = None
    return out


def style(a):
    a.grid(axis="y", color=GRID, lw=0.8)
    a.set_axisbelow(True)
    for s in ("top", "right", "left"):
        a.spines[s].set_visible(False)
    a.spines["bottom"].set_color("#b9b8b2")
    a.tick_params(colors=INK2)


def main(path):
    rows, meta = load(path)
    pts = meta["path"]
    total = sum(math.dist(a, b) for a, b in zip(pts, pts[1:]))
    fig = plt.figure(figsize=(15, 9.2), facecolor="white")
    fig.text(0.04, 0.955, f"갯가드 빛길 안내 {meta.get('version', 'v4')} · 빛 자국을 사람 3m 앞 길 위에", fontsize=18, weight="bold", color=INK)
    pi = meta.get("path_info") or {}
    src = ""
    if pi.get("decided_at"):
        src = (f"\n길: {pi['decided_at']} 판정으로 계산한 영흥도 탈출로({pi['original_length_m']:.0f}m)"
               + (f"를 모양 그대로 {total:.0f}m로 줄임 (배율 {pi['scale']})" if pi.get("scale", 1) < 0.999 else " 실제 크기"))
    fig.text(0.04, 0.935, f"가상의 사람이 꺾인 길 {total:.0f}m를 걸음 · 고도 {meta['alt']:.0f}m · LED {meta['led_tilt']:.0f}° 기울임 "
             f"→ 빛 자국은 드론 앞 {meta['r_spot']:.1f}m{src}\n끝난 이유: {meta['end_reason']}",
             fontsize=11.5, color=INK2, va="top", linespacing=1.5)

    # ① 위에서 본 모습
    ax = fig.add_axes([0.03, 0.06, 0.40, 0.74])
    corridor(ax, pts, meta["offpath_m"])
    ax.plot([r["drone_e"] for r in rows], [r["drone_n"] for r in rows], color=DRONE, lw=1.2, zorder=2)
    for sig, c, mk in (("켜짐", ON_C, "o"), ("깜빡임", BLINK_C, "s")):
        sel = [r for i, r in enumerate(rows) if r["signal"] == sig and i % 3 == 0]
        ax.scatter([r["spot_e"] for r in sel], [r["spot_n"] for r in sel], s=26, color=c, marker=mk,
                   edgecolor="white", linewidth=1.0, zorder=4)
    ax.plot([r["person_e"] for r in rows], [r["person_n"] for r in rows], color=INK, lw=1.8, zorder=5)
    ax.scatter([pts[0][1]], [pts[0][0]], s=80, color=INK, zorder=6)
    ax.annotate("출발", (pts[0][1], pts[0][0]), xytext=(8, -14), textcoords="offset points", fontsize=11, color=INK)
    ax.scatter([pts[-1][1]], [pts[-1][0]], s=120, color=INK, marker="*", zorder=6)
    ax.annotate("도착(해안 가정)", (pts[-1][1], pts[-1][0]), xytext=(-10, 8), textcoords="offset points",
                fontsize=11, color=INK, ha="right")
    many = len(pts) - 2 > 4                  # 모퉁이가 많으면(실제 크기 탈출로) 이름만, 왼쪽에
    for k, (n, e) in enumerate(pts[1:-1]):
        if many:
            ax.annotate(f"모퉁이 {k + 1}", (e, n), xytext=(-10, -4), textcoords="offset points", fontsize=10, color=INK, ha="right")
        else:
            ax.annotate(f"모퉁이 {k + 1}\n깜빡임 → 멈춤 → 새 길로", (e, n), xytext=(-12, 14) if k % 2 == 0 else (12, -34),
                        textcoords="offset points", fontsize=10.5, color=INK, ha="right" if k % 2 == 0 else "left")
    off = [r for r in rows if r["phase"] == "길 이탈"]
    if off:
        far = max(rows, key=lambda r: abs(r["offset"]))
        ax.annotate(f"길에서 {abs(far['offset']):.1f}m 벗어남\n→ 빛 자국 깜빡임, 그쪽으로 돌아옴",
                    (far["person_e"], far["person_n"]), xytext=(22, -6), textcoords="offset points",
                    fontsize=10.5, color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=1))
    ax.set_aspect("equal")
    ax.set_xlabel("동쪽 (m)", color=INK2)
    ax.set_ylabel("북쪽 (m)", color=INK2)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=INK2)
    ax.set_title("① 위에서 본 모습", loc="left", fontsize=13, color=INK, pad=10)
    hs = [Line2D([], [], color=INK, lw=2, label="사람이 걸은 길"),
          Line2D([], [], color=ON_C, marker="o", ls="none", markersize=7, markeredgecolor="white", label="빛 자국 · 켜짐 = 따라오세요"),
          Line2D([], [], color=BLINK_C, marker="s", ls="none", markersize=7, markeredgecolor="white",
                 label="빛 자국 · 깜빡임 = 와서 멈추세요"),
          Line2D([], [], color=DRONE, lw=1.5, label="드론이 날아간 길"),
          Polygon([(0, 0)], fc=CORRIDOR, label=f"안전한 길 폭 (±{meta['offpath_m']:.0f}m)")]
    ax.legend(handles=hs, loc="lower left", bbox_to_anchor=(1.0, 0.0), fontsize=10.5, frameon=False)

    # ②③④ 시간에 따른 값 (같은 시간축, 각자 한 축)
    t = [r["t"] for r in rows]
    blink_spans = spans(rows, lambda r: r["signal"] == "깜빡임")

    def label_spans(a):
        k = 0
        for t0, t1 in blink_spans:
            mid = min(rows, key=lambda r: abs(r["t"] - (t0 + t1) / 2))
            if mid["phase"] == "길 이탈":
                lab = "길 이탈" if len(blink_spans) <= 5 else "이탈"
            else:
                k += 1
                lab = f"모퉁이 {k}" if len(blink_spans) <= 5 else str(k)       # 많으면 번호만 (겹치지 않게)
            a.axvspan(t0, t1, color=BLINK_C, alpha=0.13, lw=0)
            a.text((t0 + t1) / 2, 1.0, lab, transform=a.get_xaxis_transform(), ha="center", va="bottom",
                   fontsize=10, color=INK)

    panels = [("② 사람-드론 거리 (m)", 0.62), ("③ 빛 자국이 사람보다 앞선 거리 (m)", 0.36),
              ("④ 길 가운데서 벗어난 거리 (m)", 0.10)]
    for i, (title, y0) in enumerate(panels):
        a = fig.add_axes([0.53, y0, 0.44, 0.17])
        a.set_xlim(t[0] - 3, t[-1] + 3)
        label_spans(a)
        if i == 0:
            a.axhspan(5, 10, color=CORRIDOR, lw=0)
            a.text(t[-1], 9.7, "계획 5~10m", ha="right", va="top", fontsize=10, color=INK2)
            a.plot(t, [r["gap"] for r in rows], color=INK, lw=2)
            a.set_ylim(0, 11)
        elif i == 1:
            a.axhline(meta["spot_ahead"], color=ON_C, lw=1.2, ls=(0, (4, 3)))
            a.text(t[-1], meta["spot_ahead"] + 0.15, f"목표 {meta['spot_ahead']:.0f}m", ha="right", va="bottom",
                   fontsize=10, color=INK2)
            a.plot(t, [None if r["phase"] == "도착" else r["spot_ahead"] for r in rows], color=INK, lw=2)
            a.set_ylim(-1, 5)
        else:
            a.axhline(meta["offpath_m"], color=BLINK_C, lw=1.2, ls=(0, (4, 3)))
            a.text(t[-1], meta["offpath_m"] + 0.08, f"이탈 기준 {meta['offpath_m']:.0f}m", ha="right", va="bottom",
                   fontsize=10, color=INK2)
            a.plot(t, [abs(r["offset"]) for r in rows], color=INK, lw=2)
            a.plot(t, [None if r["phase"] == "도착" else abs(r["spot_off"]) for r in rows], color=ON_C, lw=1.6)
            a.text(t[0], 2.75, "검정 = 사람,  파랑 = 빛 자국", fontsize=10, color=INK2, va="top")
            a.set_ylim(0, 3)
            a.set_xlabel("안내 시작 후 시간 (초)", color=INK2)
        a.set_title(title, loc="left", fontsize=12.5, color=INK, pad=16)
        style(a)
    out = path.replace(".csv", ".png")
    fig.savefig(out, dpi=130)
    print("그림 저장:", out)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python gg_plot_v4.py gg_log_....csv")
    else:
        main(sys.argv[1])
