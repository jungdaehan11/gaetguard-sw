"""실측 보정 방법 비교 시험 — "높이로 고칠까, 시각으로 고칠까"

입력: gg_station_fetch.py 가 만든 tide\\station_<코드>_<간격>m.csv (date, time, obs_cm, pred_cm)
하는 일
  1) 자료 확인: 빠진 분, 실측-예측 차이 분포
  2) 고조·저조 시각 차이: 실측 극값 시각 - 예측 극값 시각 (분)
  3) 물 차오를 때 같은 높이를 지나는 시각 차이 (분) — 퇴로 마감에 바로 쓰이는 값
  4) 지금(t)까지의 실측으로 H분 뒤 물높이를 맞히기 — 방법 4가지 비교
     A 예측만        : 예측(t+H)
     B 높이 보정     : 예측(t+H) + [지금 실측 - 지금 예측]   (지금 계획의 ① 실시간 보정, 실측은 최근 10분 평균)
     C 시각 보정     : 최근 W분 실측에 가장 잘 맞게 예측을 τ분 밀고 → 예측(t+H-τ)
     D 시각+높이 보정: τ와 높이 c를 같이 맞춤 → 예측(t+H-τ) + c
     → 오차를 물이 찰 때 / 빠질 때 / 정조(물 멈춤 근처)로 나눠 보고, 물이 찰 때는 "분" 단위로도 봄

실행: python gg_tide_correction_test.py tide\\station_DT_0052_1m.csv   (numpy, matplotlib 필요)
결과: 화면 요약 + 같은 이름_보정시험.png / _보정시험.csv
"""
import csv
import os
import sys
from datetime import datetime

import numpy as np

HORIZONS = (30, 60, 120)        # 몇 분 뒤를 맞히나
LAG_MAX = 40                    # 시각 보정에서 찾는 범위 ±분
WINDOWS = (60, 180)             # 시각 보정에 쓰는 최근 실측 길이(분)
LEVELS = (200, 300, 400, 500, 600)   # 물 차오를 때 지나는 높이 (cm, 약최저저조면 기준) — 갯벌 높이 1.5~5m 부근
ISSUE_EVERY = 10                # 몇 분마다 예보를 낸다고 볼지


def load(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    t, o, p = [], [], []
    for r in rows:
        try:
            dt = datetime.strptime(r["date"] + r["time"], "%Y%m%d%H:%M")
        except ValueError:
            continue
        t.append(dt)
        o.append(float(r["obs_cm"]) if r["obs_cm"] not in ("", None) else np.nan)
        p.append(float(r["pred_cm"]) if r["pred_cm"] not in ("", None) else np.nan)
    order = np.argsort(np.array(t, dtype="datetime64[m]"))
    t = np.array(t, dtype="datetime64[m]")[order]
    o, p = np.array(o)[order], np.array(p)[order]
    keep = np.r_[True, t[1:] != t[:-1]]                      # 같은 시각 두 번이면 하나만
    return t[keep], o[keep], p[keep]


def to_minute_grid(t, o, p):
    """1분 간격 빈칸 없는 배열로 (빠진 분은 NaN)"""
    m = ((t - t[0]) / np.timedelta64(1, "m")).astype(int)
    n = m[-1] + 1
    O, P = np.full(n, np.nan), np.full(n, np.nan)
    O[m], P[m] = o, p
    T = t[0] + np.arange(n).astype("timedelta64[m]")
    step = int(np.median(np.diff(m))) if len(m) > 1 else 1
    if step > 1:          # 10분 자료 등 → 예측은 보간, 실측은 그 칸만
        ok = ~np.isnan(P)
        P = np.interp(np.arange(n), np.arange(n)[ok], P[ok])
    return T, O, P, step


def despike(O, P, lim=30.0):
    """튀는 값 지우기: 실측이 앞뒤 5분 중앙값과 lim cm 넘게 다르면 빈칸으로.
    (2026년 3월 인천송도 1분 자료에 앞뒤는 600cm인데 한 분만 0.0으로 적힌 곳이 31군데 있었음)"""
    n = len(O)
    pad = np.r_[np.full(5, np.nan), O, np.full(5, np.nan)]
    nb = np.stack([pad[5 + k:5 + k + n] for k in (-5, -4, -3, -2, -1, 1, 2, 3, 4, 5)])
    with np.errstate(invalid="ignore"):
        med = np.nanmedian(nb, axis=0)
        bad = (np.abs(O - med) > lim) | ((O == 0) & (P > 100))
    O = O.copy()
    O[bad] = np.nan
    return O, int(bad.sum())


def smooth(x, w):
    """NaN을 빼고 가운데 이동평균"""
    ok = (~np.isnan(x)).astype(float)
    xs = np.where(np.isnan(x), 0, x)
    k = np.ones(w)
    num, den = np.convolve(xs, k, "same"), np.convolve(ok, k, "same")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den >= w * 0.6, num / den, np.nan)


def extrema(h, win=45):
    """고조·저조 (앞뒤 win분 안 최대·최소, 평평하면 가운데)"""
    out, n, i = [], len(h), win
    while i < n - win:
        seg = h[i - win:i + win + 1]
        if np.isnan(h[i]) or np.isnan(seg).any():
            i += 1
            continue
        kind = "고조" if h[i] == seg.max() and h[i] > seg.min() else "저조" if h[i] == seg.min() and h[i] < seg.max() else None
        if kind is None:
            i += 1
            continue
        j = i
        while j + 1 < n and h[j + 1] == h[i]:
            j += 1
        out.append((kind, (i + j) / 2, h[i]))
        i = j + win
    return out


def crossings(h, level, rising=True):
    """h가 level을 아래→위(물 참)로 지나는 시각(분, 소수)"""
    a, b = h[:-1], h[1:]
    m = (a < level) & (b >= level) if rising else (a > level) & (b <= level)
    m &= ~np.isnan(a) & ~np.isnan(b)
    i = np.nonzero(m)[0]
    return i + (level - a[i]) / (b[i] - a[i])


def window_sums(x, w):
    """끝이 t인 길이 w 구간의 합 (NaN 제외)과 개수"""
    ok = ~np.isnan(x)
    c = np.r_[0, np.cumsum(np.where(ok, x, 0))]
    k = np.r_[0, np.cumsum(ok)]
    idx = np.arange(len(x))
    lo = np.maximum(idx - w + 1, 0)
    return c[idx + 1] - c[lo], k[idx + 1] - k[lo]


def fit_lag(O, P, W):
    """각 t에서 최근 W분 실측에 맞는 τ (C: 높이 그대로, D: 높이 c도 같이)"""
    n = len(O)
    lags = np.arange(-LAG_MAX, LAG_MAX + 1)
    best_c = np.full(n, np.inf)
    best_d = np.full(n, np.inf)
    tau_c, tau_d, off_d = np.zeros(n), np.zeros(n), np.zeros(n)
    for L in lags:
        Ps = np.full(n, np.nan)          # 예측(s - L)
        if L >= 0:
            Ps[L:] = P[:n - L]
        else:
            Ps[:n + L] = P[-L:]
        d = O - Ps
        s1, cnt = window_sums(d, W)
        s2, _ = window_sums(d * d, W)
        good = cnt >= W * 0.6
        with np.errstate(invalid="ignore", divide="ignore"):
            sse_c = np.where(good, s2 / cnt, np.inf)
            mean = s1 / cnt
            sse_d = np.where(good, s2 / cnt - mean * mean, np.inf)
        u = sse_c < best_c
        best_c[u], tau_c[u] = sse_c[u], L
        u = sse_d < best_d
        best_d[u], tau_d[u], off_d[u] = sse_d[u], L, mean[u]
    return tau_c, tau_d, off_d


def P_at(P, idx):
    """소수 인덱스에서 예측값 (선형 보간, 범위 밖은 NaN)"""
    idx = np.asarray(idx, float)
    ok = (idx >= 0) & (idx <= len(P) - 1)
    out = np.full(idx.shape, np.nan)
    out[ok] = np.interp(idx[ok], np.arange(len(P)), P)
    return out


def q(x, p):
    x = x[~np.isnan(x)]
    return float(np.percentile(x, p)) if len(x) else np.nan


def main(path):
    t, o, p = load(path)
    T, O, P, step = to_minute_grid(t, o, p)
    n = len(T)
    days = n / 1440
    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    say(f"파일: {os.path.basename(path)}  기간 {T[0]} ~ {T[-1]} ({days:.1f}일), 자료 간격 {step}분")
    if step == 1:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            O, nbad = despike(O, P)
        say(f"튀는 실측값(앞뒤와 30cm 넘게 다르거나 0.0) {nbad}개를 빈칸으로 처리")
    miss = np.isnan(O[::step]).mean() * 100
    say(f"실측 빠진 비율 {miss:.1f}%  (예측 빠짐 {np.isnan(P).mean() * 100:.1f}%)")
    R = O - P
    say(f"실측-예측 차이(cm): 평균 {np.nanmean(R):+.1f}, 표준편차 {np.nanstd(R):.1f}, "
        f"5% {q(R, 5):+.1f}, 50% {q(R, 50):+.1f}, 95% {q(R, 95):+.1f}, 최소 {np.nanmin(R):+.1f}, 최대 {np.nanmax(R):+.1f}")
    Os = smooth(O, 11 if step == 1 else 1)          # 1분 자료의 출렁임(파랑 등)을 11분 평균으로 누름

    # 2) 고조·저조 시각 차이
    if step == 1:
        eo, ep = extrema(np.round(Os, 1)), extrema(P)
        rows = []
        for kind, io, ho in eo:
            cand = [(abs(io - ip), ip, hp) for k, ip, hp in ep if k == kind and abs(io - ip) < 90]
            if cand:
                _, ip, hp = min(cand)
                rows.append((kind, io - ip, ho - hp))
        for kind in ("고조", "저조"):
            dt = np.array([r[1] for r in rows if r[0] == kind])
            dh = np.array([r[2] for r in rows if r[0] == kind])
            if len(dt):
                say(f"{kind} {len(dt)}번: 시각 차이(실측-예측) 평균 {dt.mean():+.1f}분, 5~95% {q(dt, 5):+.0f}~{q(dt, 95):+.0f}분 / "
                    f"높이 차이 평균 {dh.mean():+.1f}cm, 5~95% {q(dh, 5):+.0f}~{q(dh, 95):+.0f}cm")

    # 3) 물 찰 때 같은 높이를 지나는 시각 차이
    say("물 찰 때 같은 높이 지나는 시각 차이 (실측-예측, 분, + = 실측이 늦음):")
    for L in LEVELS:
        co, cp = crossings(Os, L), crossings(P, L)
        d = np.array([c - cp[np.argmin(abs(cp - c))] for c in co if len(cp) and abs(cp - c).min() < 120])
        if len(d):
            say(f"   {L}cm: {len(d)}번, 평균 {d.mean():+.1f}, 50% {q(d, 50):+.1f}, 5~95% {q(d, 5):+.1f}~{q(d, 95):+.1f}, "
                f"|차이| 99% {q(abs(d), 99):.1f}")

    # 4) H분 뒤 맞히기
    O10, c10 = window_sums(O, 10)
    with np.errstate(invalid="ignore", divide="ignore"):
        Onow = np.where(c10 >= 6, O10 / c10, np.nan)          # 최근 10분 평균 실측
    Pnow = P_at(P, np.arange(n) - 4.5)                         # 같은 구간 가운데의 예측
    fits = {W: fit_lag(O, P, W) for W in WINDOWS}
    issue = np.arange(max(WINDOWS), n, ISSUE_EVERY)
    issue = issue[~np.isnan(Onow[issue])]
    slope = np.gradient(P)                                     # cm/분
    results = []
    say("")
    say(f"H분 뒤 물높이 맞히기 ({len(issue)}번 예보, 정답 = 실측 11분 평균)")
    say("   방법                       H   전체 RMS  |오차|95%  |오차|99%   물찰때 95%  빠질때 95%  정조 95%   물찰때 시각오차 95%·99%(분)")
    for H in HORIZONS:
        tgt = issue + H
        ok = tgt < n
        i0, i1 = issue[ok], tgt[ok]
        truth = Os[i1]
        ph = np.where(slope[i1] > 0.5, "찰때", np.where(slope[i1] < -0.5, "빠질때", "정조"))
        meth = {"A 예측만": P[i1], "B 높이 보정": P[i1] + (Onow[i0] - Pnow[i0])}
        for W in WINDOWS:
            tc, td, cd = fits[W]
            meth[f"C 시각 보정({W}분)"] = P_at(P, i1 - tc[i0])
            meth[f"D 시각+높이({W}분)"] = P_at(P, i1 - td[i0]) + cd[i0]
        for name, f in meth.items():
            e = truth - f
            rise = (ph == "찰때") & (slope[i1] > 1.0)
            tmin = np.abs(e[rise] / slope[i1][rise])          # 높이 오차를 그 순간 물 차는 속도로 나눔 → 분
            r = dict(method=name, H=H, n=int((~np.isnan(e)).sum()),
                     rms=float(np.sqrt(np.nanmean(e ** 2))), p95=q(np.abs(e), 95), p99=q(np.abs(e), 99),
                     rise95=q(np.abs(e[ph == "찰때"]), 95), fall95=q(np.abs(e[ph == "빠질때"]), 95),
                     slack95=q(np.abs(e[ph == "정조"]), 95), rise_min95=q(tmin, 95), rise_min99=q(tmin, 99))
            results.append(r)
            say(f"   {name:<22} {H:>3}  {r['rms']:8.1f}  {r['p95']:9.1f}  {r['p99']:9.1f}   {r['rise95']:9.1f}  "
                f"{r['fall95']:9.1f}  {r['slack95']:8.1f}   {r['rise_min95']:6.1f} · {r['rise_min99']:.1f}")
    tau = fits[180][1]
    say("")
    say(f"D(180분)가 찾은 시각 차이 τ: 50% {q(tau[issue], 50):+.0f}분, 5~95% {q(tau[issue], 5):+.0f}~{q(tau[issue], 95):+.0f}분, "
        f"±{LAG_MAX}분 끝에 걸린 비율 {np.mean(np.abs(tau[issue]) >= LAG_MAX) * 100:.0f}%")

    base = os.path.splitext(path)[0] + "_보정시험"
    with open(base + ".csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)
    with open(base + ".txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    draw(base + ".png", T, R, results, path)
    print(f"\n저장: {base}.txt / .csv / .png")


def draw(out, T, R, results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    for name in ("Malgun Gothic", "Noto Sans CJK KR", "AppleGothic", "NanumGothic"):
        if any(f.name == name for f in fm.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False
    INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
    COL = {"A": "#9a9993", "B": "#eb6834", "C": "#1baf7a", "D": "#2a78d6"}
    fig = plt.figure(figsize=(13, 8), facecolor="white")
    fig.text(0.05, 0.95, "실측 보정 방법 비교 — 높이로 고칠까, 시각으로 고칠까", fontsize=16, weight="bold", color=INK)
    fig.text(0.05, 0.915, f"{os.path.basename(path)} · 국립해양조사원 조위관측소 실측·예측 조위 (공공데이터포털)",
             fontsize=10.5, color=INK2)
    a = fig.add_axes([0.06, 0.58, 0.9, 0.27])
    a.plot(T.astype("datetime64[s]").astype(datetime), R, color=INK, lw=0.6)
    a.axhline(0, color="#b9b8b2", lw=1)
    a.set_title("① 실측 − 예측 (cm)", loc="left", fontsize=12, color=INK)
    b = fig.add_axes([0.06, 0.13, 0.9, 0.34])
    names = list(dict.fromkeys(r["method"] for r in results))
    H = sorted({r["H"] for r in results})
    wbar = 0.8 / len(names)
    for k, nm in enumerate(names):
        vals = [next(r["rise_min95"] for r in results if r["method"] == nm and r["H"] == h) for h in H]
        x = np.arange(len(H)) + (k - (len(names) - 1) / 2) * wbar
        b.bar(x, vals, width=wbar * 0.9, color=COL[nm[0]], alpha=1.0 if "180" in nm or nm[0] in "AB" else 0.55,
              label=nm)
    b.set_xticks(np.arange(len(H)), [f"{h}분 뒤" for h in H])
    b.set_title("② 물 찰 때 시각 오차 95% (분) — 낮을수록 좋음", loc="left", fontsize=12, color=INK)
    b.legend(frameon=False, fontsize=9.5, ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.07))
    for ax in (a, b):
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=INK2)
    fig.savefig(out, dpi=130)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python gg_tide_correction_test.py tide\\station_DT_0052_1m.csv")
    else:
        main(sys.argv[1])
