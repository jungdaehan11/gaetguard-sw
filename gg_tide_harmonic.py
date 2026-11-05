"""조석 성분 뽑기 → 우리 예측 만들기 → 다른 해 실측과 비교

실제 시스템이 출동 전에 할 일("조석 조화 성분으로 인터넷 없이 예보")을 그대로 시험합니다.
  1) 학습 기간 실측(1시간 평균)으로 조석 성분을 뽑음 (UTide, 노드 보정 포함)
     시각은 바다누리 그대로(한국 시각)를 씀 → 위상은 그리니치 기준이 아니라 한국 시각 기준 값 (예측에는 영향 없음)
  2) 검증 기간을 1분 간격으로 예측 → 실측과 비교 (높이 오차, 고조·저조 시각 차이)
  3) 검증 기간을 gg_tide_correction_test.py 가 읽는 모양(date,time,obs_cm,pred_cm)으로 저장 → 보정 시험

입력: tide\\badanuri_<관측소>_1m.csv (gg_badanuri_parse.py 결과)
실행: python gg_tide_harmonic.py tide\\badanuri_영흥도_1m.csv --train 2021 2023 --test 2024
필요: pip install utide numpy
"""
import argparse
import csv
import os
import warnings

import numpy as np


def load(path):
    t, h = [], []
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            t.append(r["time"][:16])
            h.append(float(r["tide_cm"]) if r["tide_cm"] else np.nan)
    return np.array(t, dtype="datetime64[m]"), np.array(h)


def hourly(t, h):
    """정시 기준 앞뒤 30분 평균, 45분 이상 있을 때만"""
    hr = t.astype("datetime64[h]") + np.where((t - t.astype("datetime64[h]")) >= np.timedelta64(30, "m"), 1, 0).astype("timedelta64[h]")
    keys, inv = np.unique(hr, return_inverse=True)
    ok = np.isfinite(h)
    s = np.bincount(inv, np.where(ok, h, 0))
    c = np.bincount(inv, ok.astype(float))
    m = c >= 45
    return keys[m], s[m] / c[m]


def extrema(h, win):
    out, n, i = [], len(h), win
    while i < n - win:
        seg = h[i - win:i + win + 1]
        if not np.isfinite(h[i]) or not np.isfinite(seg).all():
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


def q(x, p):
    x = np.asarray(x)[np.isfinite(x)]
    return float(np.percentile(x, p)) if len(x) else np.nan


def main():
    import utide
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--train", nargs=2, type=int, default=[2021, 2023])
    ap.add_argument("--test", nargs="+", type=int, default=[2024])
    ap.add_argument("--lat", type=float, default=37.2386)
    a = ap.parse_args()
    t, h = load(a.csv)
    yr = t.astype("datetime64[Y]").astype(int) + 1970
    tr = (yr >= a.train[0]) & (yr <= a.train[1])
    th, hh = hourly(t[tr], h[tr])
    print(f"학습 {a.train[0]}~{a.train[1]}: 1시간 평균 {len(th):,}개 (1분 자료 {np.isfinite(h[tr]).sum():,}분에서)")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        coef = utide.solve(th.astype("datetime64[s]"), hh, lat=a.lat, method="ols", conf_int="none", trend=False,
                           nodal=True, constit="auto", verbose=False)
    order = np.argsort(-coef["A"])
    print(f"평균해면(학습 기간) {coef['mean']:.1f}cm, 성분 {len(coef['name'])}개. 큰 순서:")
    print("   " + ", ".join(f"{coef['name'][i]} {coef['A'][i]:.1f}cm/{coef['g'][i]:.0f}°" for i in order[:12]))
    base = os.path.splitext(a.csv)[0]
    with open(base + f"_성분_{a.train[0]}-{a.train[1]}.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["name", "amp_cm", "phase_deg", "freq_cph"])
        for i in order:
            w.writerow([coef["name"][i], f"{coef['A'][i]:.3f}", f"{coef['g'][i]:.2f}", f"{coef['aux']['frq'][i]:.8f}"])
    for y in a.test:
        m = yr == y
        tt, ho = t[m], h[m]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hp = utide.reconstruct(tt.astype("datetime64[s]"), coef, verbose=False).h
        r = ho - hp
        print(f"\n검증 {y}: 실측 있는 분 {np.isfinite(ho).sum():,}")
        print(f"   실측-우리예측(cm): 평균 {np.nanmean(r):+.1f}, RMS {np.sqrt(np.nanmean(r ** 2)):.1f}, "
              f"5~95% {q(r, 5):+.1f}~{q(r, 95):+.1f}, |차이| 99% {q(np.abs(r), 99):.1f}")
        # 고조·저조 (실측은 11분 평균)
        k = np.ones(11)
        ok = np.isfinite(ho).astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            hs = np.convolve(np.where(np.isfinite(ho), ho, 0), k, "same") / np.convolve(ok, k, "same")
        hs[np.convolve(ok, k, "same") < 7] = np.nan
        eo, ep = extrema(np.round(hs, 1), 45), extrema(hp, 45)
        for kind in ("고조", "저조"):
            pi = np.array([e[1] for e in ep if e[0] == kind])
            dt, dh = [], []
            for kk, io, ho_ in eo:
                if kk != kind or not len(pi):
                    continue
                j = np.argmin(abs(pi - io))
                if abs(pi[j] - io) < 90:
                    dt.append(io - pi[j])
                    dh.append(ho_ - [e[2] for e in ep if e[0] == kind][j])
            dt, dh = np.array(dt), np.array(dh)
            print(f"   {kind} {len(dt)}번: 시각 차이(실측-예측) 평균 {dt.mean():+.1f}분, 5~95% {q(dt, 5):+.0f}~{q(dt, 95):+.0f}분, "
                  f"|차이| 99% {q(abs(dt), 99):.0f}분 / 높이 차이 평균 {dh.mean():+.1f}cm, 5~95% {q(dh, 5):+.0f}~{q(dh, 95):+.0f}cm")
        out = base + f"_{y}_우리예측.csv"
        with open(out, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["date", "time", "obs_cm", "pred_cm", "station"])
            for i in range(len(tt)):
                s = str(tt[i])
                w.writerow([s[:10].replace("-", ""), s[11:16], "" if not np.isfinite(ho[i]) else f"{ho[i]:.0f}",
                            f"{hp[i]:.1f}", "우리예측"])
        print(f"   저장: {out}")


if __name__ == "__main__":
    main()
