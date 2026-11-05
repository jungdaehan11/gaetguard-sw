"""조석 성분 뽑기 → 우리 예측 만들기 → 다른 해 실측과 비교

실제 시스템이 출동 전에 할 일("조석 조화 성분으로 인터넷 없이 예보")을 그대로 시험합니다.
  1) 학습 기간 실측(1시간 평균)으로 조석 성분을 뽑음 (UTide, 노드 보정 포함)
     시각은 바다누리 그대로(한국 시각)를 씀 → 위상은 그리니치 기준이 아니라 한국 시각 기준 값 (예측에는 영향 없음)
  2) 검증 기간을 1분 간격으로 예측 → 실측과 비교 (높이 오차, 고조·저조 시각 차이)
  3) 검증 기간을 gg_tide_correction_test.py 가 읽는 모양(date,time,obs_cm,pred_cm)으로 저장 → 보정 시험

입력: tide\\badanuri_<관측소>_1m.csv (gg_badanuri_parse.py 결과)
실행: python gg_tide_harmonic.py tide\\badanuri_영흥도_1m.csv --train 2021 2023 --test 2024
     python gg_tide_harmonic.py tide\\badanuri_영흥도_1m.csv --train 2022 2024 --test --api tide\\station_DT_0043_1m.csv
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


def predict(coef, t, chunk=100_000):
    """나눠서 계산 (한 번에 100만 분 넘게 넣으면 메모리 부족)"""
    import utide
    out = np.empty(len(t))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(0, len(t), chunk):
            out[i:i + chunk] = utide.reconstruct(t[i:i + chunk].astype("datetime64[s]"), coef, verbose=False).h
    return out


def compare(label, ho, hp):
    """실측 ho와 예측 hp 비교: 높이 차이, 고조·저조 시각·높이 차이"""
    r = ho - hp
    print(f"   실측-{label}(cm): 평균 {np.nanmean(r):+.1f}, RMS {np.sqrt(np.nanmean(r ** 2)):.1f}, "
          f"5~95% {q(r, 5):+.1f}~{q(r, 95):+.1f}, |차이| 99% {q(np.abs(r), 99):.1f}")
    k = np.ones(11)
    ok = np.isfinite(ho).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        hs = np.convolve(np.where(np.isfinite(ho), ho, 0), k, "same") / np.convolve(ok, k, "same")
    hs[np.convolve(ok, k, "same") < 7] = np.nan                  # 실측은 11분 평균
    eo, ep = extrema(np.round(hs, 1), 45), extrema(hp, 45)
    for kind in ("고조", "저조"):
        pi = np.array([e[1] for e in ep if e[0] == kind])
        ph = np.array([e[2] for e in ep if e[0] == kind])
        dt, dh = [], []
        for kk, io, ho_ in eo:
            if kk != kind or not len(pi):
                continue
            j = np.argmin(abs(pi - io))
            if abs(pi[j] - io) < 90:
                dt.append(io - pi[j])
                dh.append(ho_ - ph[j])
        dt, dh = np.array(dt), np.array(dh)
        if len(dt):
            print(f"   {kind} {len(dt)}번: 시각 차이 평균 {dt.mean():+.1f}분, 5~95% {q(dt, 5):+.0f}~{q(dt, 95):+.0f}분, "
                  f"|차이| 99% {q(abs(dt), 99):.0f}분 / 높이 차이 평균 {dh.mean():+.1f}cm, 5~95% {q(dh, 5):+.0f}~{q(dh, 95):+.0f}cm")


def write_station(path, tt, ho, hp):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["date", "time", "obs_cm", "pred_cm", "station"])
        for i in range(len(tt)):
            s = str(tt[i])
            w.writerow([s[:10].replace("-", ""), s[11:16], "" if not np.isfinite(ho[i]) else f"{ho[i]:.0f}",
                        f"{hp[i]:.1f}", "우리예측"])
    print(f"   저장: {path}")


def main():
    import utide
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--train", nargs=2, type=int, default=[2021, 2023])
    ap.add_argument("--test", nargs="*", type=int, default=[2024])
    ap.add_argument("--lat", type=float, default=37.2386)
    ap.add_argument("--api", help="gg_station_fetch.py 결과 CSV (실측+국립해양조사원 예측) → 같은 시각에 우리 예측과 비교")
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
        hp = predict(coef, tt)
        print(f"\n검증 {y}: 실측 있는 분 {np.isfinite(ho).sum():,}")
        compare("우리예측", ho, hp)
        write_station(base + f"_{y}_우리예측.csv", tt, ho, hp)
    if a.api:                     # API 자료(실측 + 국립해양조사원 예측)와 같은 시각에 우리 예측 → 셋 비교
        ta, oa, pa = [], [], []
        with open(a.api, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                ta.append(f"{r['date'][:4]}-{r['date'][4:6]}-{r['date'][6:]}T{r['time']}")
                oa.append(float(r["obs_cm"]) if r["obs_cm"] else np.nan)
                pa.append(float(r["pred_cm"]) if r["pred_cm"] else np.nan)
        ta, oa, pa = np.array(ta, dtype="datetime64[m]"), np.array(oa), np.array(pa)
        o = np.argsort(ta, kind="stable")
        ta, oa, pa = ta[o], oa[o], pa[o]
        oa[(oa == 0) & (pa > 100)] = np.nan                       # 0으로 적힌 빈 값
        hp = predict(coef, ta)
        print(f"\nAPI 자료 {os.path.basename(a.api)}: {ta[0]} ~ {ta[-1]}, 실측 있는 분 {np.isfinite(oa).sum():,}")
        # 빠진 날이 있으면 1분 간격이 끊기므로 비교는 그대로(같은 시각끼리), 고조·저조는 하루 단위로 끊긴 곳에서 조금 빠질 수 있음
        compare("국립해양조사원 예측", oa, pa)
        compare("우리예측", oa, hp)
        d = hp - pa
        print(f"   우리예측-국립해양조사원 예측(cm): 평균 {np.nanmean(d):+.1f}, RMS {np.sqrt(np.nanmean(d ** 2)):.1f}, "
              f"|차이| 99% {q(np.abs(d), 99):.1f}")
        write_station(os.path.splitext(a.api)[0] + f"_우리예측{a.train[0]}-{a.train[1]}.csv", ta, oa, hp)


if __name__ == "__main__":
    main()
