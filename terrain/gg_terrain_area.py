"""갯벌 지형 자르기 + 지도 (BADA2024 수심, 갯골 상세·약식, 자연·인공 해안선) — 지역 선택

입력 (gaetguard\\terrain 폴더, 국립해양조사원 바다누리·개방海에서 받은 그대로)
  BADA2024_격자형수심_v1.1.nc, 갯골(상세).zip, 갯골(약식).zip, 자연해안선.zip, 인공해안선.zip
출력 (gaetguard\\terrain\\<지역>)
  bada2024_<지역>.csv             lon, lat, 평균해면 아래 수심(m), 약최저저조면 위 높이(m)
  *_<지역>.geojson                갯골·해안선 (위경도, 범위 안만)
  <이름>_지형지도.png

실행: python gg_terrain_area.py youngheung   (또는 siheung)
필요: pip install netCDF4 pyshp pyproj shapely scipy matplotlib
"""
import json
import os
import sys
import tempfile
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# 지역마다: 자를 범위 B(경도0, 경도1, 위도0, 위도1), 따로 표시할 계산 범위 S, 평균해면 높이 Z0(m),
# 참고 저조·고조(m)와 그 날짜, 표시할 점. Z0·저조·고조는 TideBED(커넥터) 값 [기준면 추정]
AREAS = {
    "youngheung": dict(
        name="영흥도", B=(126.38, 126.56, 37.19, 37.31), S=None,
        Z0=4.334, LOW=1.42, HIGH=7.19, day="2026-11-06", tide_src="TideBED 영흥도 (37.2386, 126.4286)",
        points=[("영흥도 조위관측소 (DT_0043)", 37.23861, 126.42861, "o", (8, 6))],
        scale=(0.04, 0.05, "white")),
    "siheung": dict(
        name="시흥·오이도", B=(126.55, 126.80, 37.28, 37.42), S=(126.62, 126.78, 37.31, 37.40),
        Z0=4.58, LOW=1.41, HIGH=7.74, day="2026-11-06", tide_src="TideBED 오이도 앞 (37.340, 126.675)",
        points=[("인천송도 조위관측소", 37.33805, 126.58611, "o", (8, 6)),
                ("오이도 앞 지점 [위치 대충]", 37.340, 126.675, "^", (8, 6)),
                ("오이도 먼 지점 [대충 고른 점 → 방조제 위]", 37.325, 126.655, "^", (8, -14))],
        scale=(0.86, 0.05, "#0b0b0b")),
}
A = AREAS[sys.argv[1] if len(sys.argv) > 1 else "youngheung"]
KEY = sys.argv[1] if len(sys.argv) > 1 else "youngheung"
OUT = os.path.join(HERE, KEY)
B, S, Z0, LOW, HIGH = A["B"], A["S"], A["Z0"], A["LOW"], A["HIGH"]
SHP = {"gaetgol_detail_2017": "갯골(상세).zip", "gaetgol_simple_2019": "갯골(약식).zip",
       "coast_artificial": "인공해안선.zip", "coast_natural": "자연해안선.zip"}


def crop_bada():
    import netCDF4
    nc = netCDF4.Dataset(os.path.join(HERE, "BADA2024_격자형수심_v1.1.nc"))
    lon, lat, z = (nc[k][:].filled(np.nan) for k in ("LON", "LAT", "MSL"))   # MSL = 평균해면 아래 수심(m, 아래가 +)
    m = (lon >= B[0]) & (lon <= B[1]) & (lat >= B[2]) & (lat <= B[3])
    lon, lat, z = lon[m], lat[m], z[m]
    with open(os.path.join(OUT, f"bada2024_{KEY}.csv"), "w", encoding="utf-8-sig") as f:
        f.write("lon,lat,depth_below_msl_m,height_above_DL_m\n")
        for x, y, d in zip(lon, lat, z):
            f.write(f"{x:.5f},{y:.5f},{d:.2f},{Z0 - d:.2f}\n")
    print(f"BADA2024: {len(lon)}점")
    return lon, lat, Z0 - z


def unzip(zpath, dst):
    with zipfile.ZipFile(zpath) as z:
        for info in z.infolist():
            name = info.filename
            if not info.flag_bits & 0x800:
                try:
                    name = name.encode("cp437").decode("cp949")
                except UnicodeError:
                    pass
            norm = os.path.normpath(name)
            if norm.startswith("..") or os.path.isabs(norm) or info.is_dir() or not norm.lower().endswith(
                    (".shp", ".shx", ".dbf", ".prj", ".cpg")):
                continue
            out = os.path.join(dst, os.path.basename(norm))
            with z.open(info) as fi, open(out, "wb") as fo:
                while chunk := fi.read(1 << 20):
                    fo.write(chunk)
    return [os.path.join(dst, n) for n in os.listdir(dst) if n.lower().endswith(".shp")][0]


def crop_shp(key, zname, tmp):
    import pyproj
    import shapefile
    import shapely
    from shapely.geometry import box, mapping, shape
    d = os.path.join(tmp, key)
    os.makedirs(d)
    shp = unzip(os.path.join(HERE, zname), d)
    crs = pyproj.CRS.from_wkt(open(shp[:-4] + ".prj").read())
    to_ll = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    to_xy = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    xs, ys = to_xy.transform([B[0], B[1], B[0], B[1]], [B[2], B[2], B[3], B[3]])
    cpg = shp[:-4] + ".cpg"
    r = shapefile.Reader(shp, encoding=open(cpg).read().strip() if os.path.exists(cpg) else "utf-8")
    bb, feats = box(B[0], B[2], B[1], B[3]), []
    keep = ("CLFCDE", "GRIDLV", "AREA", "MNYMD", "SORDAT", "CSLN_KND", "EXMN_YR", "RGN_NM", "SGG_CD", "CSLN_SE")
    for sr in r.iterShapeRecords(bbox=[min(xs), min(ys), max(xs), max(ys)]):
        g = shape(sr.shape.__geo_interface__)
        if g.is_empty:
            continue
        g = shapely.transform(g, lambda c: np.c_[to_ll.transform(c[:, 0], c[:, 1])])
        g = (g if g.is_valid else g.buffer(0)).intersection(bb)
        if not g.is_empty:
            feats.append({"type": "Feature", "geometry": mapping(g),
                          "properties": {k: v for k, v in sr.record.as_dict().items() if k in keep}})
    with open(os.path.join(OUT, f"{key}_{KEY}.geojson"), "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": feats}, f, ensure_ascii=False)
    print(f"{key}: {len(feats)}개")
    return feats


def draw(lon, lat, elev, layers):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle
    from scipy.spatial import cKDTree
    for name in ("Malgun Gothic", "Noto Sans CJK KR", "AppleGothic", "NanumGothic"):
        if any(f.name == name for f in fm.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False
    INK, INK2, MUTED, ORANGE, AQUA = "#0b0b0b", "#52514e", "#9a9993", "#eb6834", "#1baf7a"
    kx, ky = 111320 * np.cos(np.radians(37.35)), 110950.0
    tree = cKDTree(np.c_[lon * kx, lat * ky])
    GX, GY = np.meshgrid(np.arange(B[0], B[1], 0.0004), np.arange(B[2], B[3], 0.0004))
    dist, idx = tree.query(np.c_[GX.ravel() * kx, GY.ravel() * ky])
    G = elev[idx].reshape(GX.shape)
    G[dist.reshape(GX.shape) > 110] = np.nan     # 자료 점에서 110m 넘게 떨어진 곳 = 자료 없음 (BADA 간격 약 150m)
    bounds = [-20, -5, -2, 0, LOW, 3, Z0, 9]
    cols = ["#104281", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb", "#eef4fc"]

    fig = plt.figure(figsize=(13, 8.4), facecolor="white")
    ax = fig.add_axes([0.05, 0.07, 0.66, 0.80])
    ax.set_facecolor("#f1f0ec")
    ax.pcolormesh(GX, GY, G, cmap=ListedColormap(cols), norm=BoundaryNorm(bounds, len(cols)), shading="auto", zorder=1)

    def plot(feats, **kw):
        for f in feats:
            g = f["geometry"]
            for p in (g["coordinates"] if g["type"].startswith("Multi") else [g["coordinates"]]):
                if g["type"].endswith("Polygon"):
                    for ring in p:
                        a = np.array(ring)[:, :2]
                        ax.fill(a[:, 0], a[:, 1], **kw)
                else:
                    a = np.array(p)[:, :2]
                    ax.plot(a[:, 0], a[:, 1], **kw)
    plot(layers["gaetgol_detail_2017"], color=ORANGE, alpha=0.75, lw=0, zorder=3)
    plot(layers["gaetgol_simple_2019"], color=AQUA, lw=1.4, zorder=4)
    plot(layers["coast_artificial"], color=INK, lw=1.6, zorder=5)
    plot(layers["coast_natural"], color=INK, lw=1.3, ls=(0, (3, 2)), zorder=5)
    reg = {}
    for f in layers["coast_artificial"] + layers["coast_natural"]:
        n = f["properties"].get("RGN_NM")
        if n and n != "기타":
            g = f["geometry"]
            parts = g["coordinates"] if g["type"].startswith("Multi") else [g["coordinates"]]
            reg.setdefault(n, []).append(np.vstack([np.array(p)[:, :2] for p in parts]).mean(0))
    for n, ps in reg.items():
        x, y = np.median(np.array(ps), 0)
        ax.text(x, y, n.split()[-1] if "광역시" in n else n.replace("경기도 ", ""), fontsize=10.5, color=INK2,
                ha="center", va="center", zorder=7, bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.5))
    if S:
        ax.add_patch(Rectangle((S[0], S[2]), S[1] - S[0], S[3] - S[2], fill=False, ec=INK2, lw=1, ls=(0, (5, 3)), zorder=6))
        ax.text(S[0] + 0.002, S[3] - 0.002, "시흥 계산 범위 (인수인계 문서)", fontsize=9.5, color=INK2, va="top", zorder=7)
    for name, y, x, mk, off in A["points"]:
        ax.scatter([x], [y], s=55, marker=mk, color=INK, edgecolor="white", linewidth=1.2, zorder=8)
        ax.annotate(name, (x, y), xytext=off, textcoords="offset points", fontsize=10, color=INK, zorder=8,
                    bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.2))
    ax.set_xlim(B[0], B[1])
    ax.set_ylim(B[2], B[3])
    ax.set_aspect(1 / np.cos(np.radians(37.35)))
    ax.tick_params(colors=INK2, labelsize=9)
    for s in ax.spines.values():
        s.set_color("#d8d7d2")
    ax.set_xlabel("경도", color=INK2)
    ax.set_ylabel("위도", color=INK2)
    fx, fy, sc = A["scale"]          # 축척 막대 자리(범위 비율)와 색 (어두운 바다 위면 흰색)
    x0, y0 = B[0] + (B[1] - B[0]) * fx, B[2] + (B[3] - B[2]) * fy
    ax.plot([x0, x0 + 1000 / kx], [y0, y0], color=sc, lw=2.5, zorder=8)
    ax.text(x0 + 500 / kx, y0 + 0.0015, "1 km", ha="center", fontsize=9.5, color=sc, zorder=8)
    fig.text(0.05, 0.950, f"{A['name']} 갯벌 지형 (BADA2024 수심 + 갯골 + 해안선)", fontsize=17, weight="bold", color=INK)
    fig.text(0.05, 0.915, f"높이 = 약최저저조면 기준 (평균해면 {Z0:.2f}m − BADA 수심, 평균해면 높이는 {A['tide_src']} 값 [기준면 추정])",
             fontsize=10.5, color=INK2)
    lab = ["−5m 아래", "−5 ~ −2m", "−2 ~ 0m", f"0 ~ {LOW:.2f}m", f"{LOW:.2f} ~ 3m", f"3 ~ {Z0:.2f}m", f"{Z0:.2f}m 위"]
    h1 = [Patch(fc=c, ec="none", label=l) for c, l in zip(cols, lab)]
    h1.append(Patch(fc="#f1f0ec", ec="#d8d7d2", label="자료 없음 (육지 또는 BADA 빈칸)"))
    fig.legend(handles=h1[::-1], loc="upper left", bbox_to_anchor=(0.73, 0.87), frameon=False, fontsize=10,
               title="갯벌·바닥 높이 (약최저저조면 기준)", title_fontsize=10.5, alignment="left")
    h2 = [Patch(fc=ORANGE, alpha=0.75, ec="none", label=f"갯골 상세 2017 (면, {len(layers['gaetgol_detail_2017'])}개)"),
          Line2D([], [], color=AQUA, lw=1.6, label=f"갯골 약식 2019 (선, {len(layers['gaetgol_simple_2019'])}개)"),
          Line2D([], [], color=INK, lw=1.6, label="인공해안선"),
          Line2D([], [], color=INK, lw=1.3, ls=(0, (3, 2)), label="자연해안선")]
    fig.legend(handles=h2, loc="upper left", bbox_to_anchor=(0.73, 0.53), frameon=False, fontsize=10,
               title="물길·육지 경계", title_fontsize=10.5, alignment="left")
    fig.text(0.735, 0.31, f"읽는 법\n· {A['day']} 저조 {LOW:.2f}m → {LOW:.2f}m 위\n  (옅은 3칸)가 그 저조 때 드러남\n"
             f"· 같은 날 고조 {HIGH:.2f}m\n· BADA는 평균해면 위쪽 얕은 곳 자료가 거의 없음\n"
             "  (갯벌 윗부분·해안 가까이가 빈칸일 수 있음)\n· 갯골 상세는 2017년, 약식은 2019년 자료",
             fontsize=9.8, color=INK2, va="top", linespacing=1.55)
    fig.text(0.735, 0.035, "출처: 국립해양조사원 BADA2024 격자형수심 v1.1(150m),\n개방海 갯골(상세)·갯골(약식)·자연/인공해안선,\n"
             "TideBED 예측 조위 (공공데이터포털)", fontsize=8.8, color=MUTED, va="bottom", linespacing=1.5)
    out = os.path.join(OUT, f"{A['name'].replace('·', '')}_지형지도.png")
    fig.savefig(out, dpi=150)
    print("지도:", out)


def main():
    os.makedirs(OUT, exist_ok=True)
    lon, lat, elev = crop_bada()
    with tempfile.TemporaryDirectory() as tmp:
        layers = {k: crop_shp(k, z, tmp) for k, z in SHP.items()}
    draw(lon, lat, elev, layers)


if __name__ == "__main__":
    main()
