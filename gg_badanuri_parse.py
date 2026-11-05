"""바다누리 조위관측소 1분 자료(txt 또는 zip) → 정리 CSV 하나 + 품질 확인

입력: tide\\badanuri\\ 안의 zip 또는 "2021년 영흥도 조위관측소.txt" 같은 파일 (UTF-8, 탭 구분, 빈 값은 "-")
     머리 3줄: 관측소명, 위도, 경도 / 열: 관측시간, 조위(cm), 수온, 염분, 파고..., 풍속(m/s), 풍향(16points), 풍향(deg), 기온, 기압, 시정
출력: tide\\badanuri_<관측소명>_1m.csv   열: time, tide_cm, wind_ms, wind_deg, air_c, pres_hpa, flag
     flag: "" 정상 / "spike" 앞뒤 5분 중앙값과 30cm 넘게 다름 / "zero" 0으로 적힌 빈 값 / "stuck" 30분 넘게 같은 값 → 모두 tide_cm 비움

실행: python gg_badanuri_parse.py            (tide\\badanuri 의 파일 전부)
"""
import csv
import glob
import io
import os
import sys
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "tide", "badanuri")


def texts():
    """(파일이름, 내용) — zip 안의 txt도 꺼냄"""
    for p in sorted(glob.glob(os.path.join(SRC, "*"))):
        if p.lower().endswith(".zip"):
            with zipfile.ZipFile(p) as z:
                for info in z.infolist():
                    name = info.filename
                    if not info.flag_bits & 0x800:
                        try:
                            name = name.encode("cp437").decode("cp949")
                        except UnicodeError:
                            pass
                    if name.lower().endswith(".txt"):
                        yield name, z.read(info)
        elif p.lower().endswith(".txt"):
            yield os.path.basename(p), open(p, "rb").read()


def decode(b):
    for enc in ("utf-8-sig", "cp949"):
        try:
            return b.decode(enc)
        except UnicodeDecodeError:
            pass
    return b.decode("utf-8", errors="replace")


def num(s):
    s = s.strip()
    if s in ("", "-"):
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def parse(text):
    meta, rows, header = {}, [], None
    for line in io.StringIO(text):
        line = line.rstrip("\r\n")
        if not line.strip():
            continue
        if header is None:
            if line.startswith("관측시간"):
                header = [h.strip() for h in line.split("\t")]
                continue
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip().lstrip("﻿")] = v.strip()
            continue
        rows.append(line.split("\t"))
    col = {h: i for i, h in enumerate(header or [])}

    def get(r, name):
        i = col.get(name)
        return num(r[i]) if i is not None and i < len(r) else np.nan
    # 연도마다 시각 모양이 다름 (2021 "2021-01-01 00:00:00", 2024 "2024/01/01 00:00")
    t = np.array([r[0].strip()[:16].replace("/", "-").replace(".", "-") for r in rows], dtype="datetime64[m]")
    data = dict(tide_cm=[get(r, "조위(cm)") for r in rows], wind_ms=[get(r, "풍속(m/s)") for r in rows],
                wind_deg=[get(r, "풍향(deg)") for r in rows], air_c=[get(r, "기온(℃)") for r in rows],
                pres_hpa=[get(r, "기압(hPa)") for r in rows])
    return meta, t, {k: np.array(v, float) for k, v in data.items()}


def despike(x, lim=30.0):
    n = len(x)
    pad = np.r_[np.full(5, np.nan), x, np.full(5, np.nan)]
    nb = np.stack([pad[5 + k:5 + k + n] for k in (-5, -4, -3, -2, -1, 1, 2, 3, 4, 5)])
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(nb, axis=0)
    zero = (x == 0) & (med > 100)
    spike = (np.abs(x - med) > lim) & ~zero
    return zero, spike


def stuck(x, minutes=30):
    """같은 값이 minutes분 넘게 그대로 → 센서 멈춤으로 봄 (2024-06-04 10:50~12:03 210cm 고정 등)"""
    out = np.zeros(len(x), bool)
    same = np.r_[False, (x[1:] == x[:-1]) & np.isfinite(x[1:])]
    start = None
    for i, v in enumerate(np.r_[same, False]):
        if v and start is None:
            start = i - 1
        elif not v and start is not None:
            if i - start >= minutes:
                out[start:i] = True
            start = None
    return out


def main():
    parts, meta = [], {}
    for name, b in texts():
        m, t, d = parse(decode(b))
        meta = meta or m
        parts.append((name, t, d))
        print(f"{name}: {len(t):,}줄, {t[0]} ~ {t[-1]}, 조위 있는 분 {np.isfinite(d['tide_cm']).mean() * 100:.1f}%")
    if not parts:
        sys.exit(f"{SRC} 에 txt 나 zip 이 없습니다.")
    t = np.concatenate([p[1] for p in parts])
    order = np.argsort(t, kind="stable")
    t = t[order]
    d = {k: np.concatenate([p[2][k] for p in parts])[order] for k in parts[0][2]}
    keep = np.r_[True, t[1:] != t[:-1]]
    t, d = t[keep], {k: v[keep] for k, v in d.items()}
    full = np.arange(t[0], t[-1] + np.timedelta64(1, "m"), np.timedelta64(1, "m"))
    print(f"\n관측소 {meta.get('관측소명')} ({meta.get('위도')}, {meta.get('경도')}), {t[0]} ~ {t[-1]}")
    print(f"시각 줄 {len(t):,} / 있어야 할 분 {len(full):,} (빠진 줄 {len(full) - len(t):,})")
    x = d["tide_cm"]
    zero, spike = despike(x)
    stk = stuck(x) & ~zero & ~spike
    flag = np.where(zero, "zero", np.where(spike, "spike", np.where(stk, "stuck", "")))
    x = np.where(zero | spike | stk, np.nan, x)
    print(f"조위 빈 값('-') {np.isnan(d['tide_cm']).sum():,}분, 0으로 적힌 빈 값 {zero.sum():,}분, 튀는 값 {spike.sum():,}분, "
          f"30분 넘게 같은 값(멈춤) {stk.sum():,}분")
    print(f"정리 뒤 조위 있는 분 {np.isfinite(x).sum():,} ({np.isfinite(x).mean() * 100:.2f}%), "
          f"범위 {np.nanmin(x):.0f} ~ {np.nanmax(x):.0f}cm, 평균 {np.nanmean(x):.1f}cm")
    yrs = t.astype("datetime64[Y]").astype(int) + 1970
    for y in np.unique(yrs):
        m = yrs == y
        print(f"   {y}: 조위 있는 분 {np.isfinite(x[m]).mean() * 100:.2f}%, 평균 {np.nanmean(x[m]):.1f}cm, "
              f"풍속 있는 분 {np.isfinite(d['wind_ms'][m]).mean() * 100:.0f}%, 기압 {np.isfinite(d['pres_hpa'][m]).mean() * 100:.0f}%")
    # 긴 빈 구간
    ok = np.isfinite(x)
    edges = np.flatnonzero(np.diff(np.r_[1, ok.astype(int), 1]))
    gaps = [(t[a] if a < len(t) else t[-1], b - a) for a, b in zip(edges[::2], edges[1::2]) if b - a >= 60]
    print(f"1시간 넘게 빈 구간 {len(gaps)}개" + (": " + ", ".join(f"{g[0]} {g[1] // 60}시간" for g in gaps[:8]) if gaps else ""))
    out = os.path.join(HERE, "tide", f"badanuri_{meta.get('관측소명', 'station')}_1m.csv")
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["time", "tide_cm", "wind_ms", "wind_deg", "air_c", "pres_hpa", "flag"])
        fmt = lambda v, nd: "" if not np.isfinite(v) else f"{v:.{nd}f}"   # noqa: E731
        for i in range(len(t)):
            w.writerow([str(t[i]).replace("T", " "), fmt(x[i], 0), fmt(d["wind_ms"][i], 1), fmt(d["wind_deg"][i], 0),
                        fmt(d["air_c"][i], 1), fmt(d["pres_hpa"][i], 1), flag[i]])
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
