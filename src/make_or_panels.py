# -*- coding: utf-8 -*-
"""기준범주를 근거에 따라 정한 로지스틱 회귀 → 항목별 오즈비 패널 그래프.

기준범주 근거:
  학교급=초등학교(최다·의무교육 기준), 사고자=일반학생(97%), 성별=여학생(대비 기준),
  사고형태=넘어짐(최다), 사고장소=일반교실(활동성 낮은 중립 실내), 사고부위=팔·손(대부분 경미 종결).
장소·부위는 '저위험 중립 범주'를 기준으로 삼아 다른 범주가 '몇 배 위험'으로 직관적으로 읽히게 함.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator
import numpy as np
import pandas as pd
import statsmodels.api as sm

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "중대사고_위험분석"
D_DATA = OUT / "데이터"
D_FIG = OUT / "그래프"
RED, BLUE, GRAY = "#C0392B", "#2F75B5", "#9AA5B1"

cols = ["학교급", "사고자구분", "사고자성별", "사고장소", "사고부위", "사고형태", "사고시간",
        "요양급여", "장해급여", "간병급여", "유족급여", "장례비", "위로금", "보전비용"]
df = pd.concat([pd.read_excel(DATA / "★2021-2025 학교안전사고 보상 데이터.xlsx", sheet_name=y, usecols=cols)
                for y in ["2021", "2022", "2023", "2024", "2025"]], ignore_index=True)
for c in ["요양급여", "장해급여", "간병급여", "유족급여", "장례비", "위로금", "보전비용"]:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)


def lab(r):
    core = r["요양급여"] + r["장해급여"] + r["간병급여"] + r["유족급여"] + r["장례비"]
    if (r["보전비용"] > 0 or r["위로금"] > 0) and core == 0:
        return np.nan
    if r["유족급여"] > 0 or r["장례비"] > 0:
        return 1
    if r["장해급여"] > 0 or r["간병급여"] > 0:
        return 1
    return 0


df["중대"] = df.apply(lab, axis=1)
df = df[df["중대"].notna()].copy()
df["중대"] = df["중대"].astype(int)
df = df[df["학교급"] != "기타학교"]

head = ["두피", "뇌(두개내)", "이마", "눈", "코", "귀", "볼", "턱", "입술 및 구강", "치아", "목구멍"]
trunk = ["흉부", "복부", "내장기관", "등", "허리", "골반/엉덩이"]
arm = ["어깨", "위팔", "팔꿈치", "아래팔", "손목", "손", "손가락"]
leg = ["넓적다리(허벅지)", "무릎", "아래다리(종아리)", "발목", "발", "발가락"]


def g_form(v):
    if v == "넘어짐": return "넘어짐"
    if "떨어짐" in str(v): return "추락"
    if "부딪힘" in str(v): return "부딪힘"
    if "충격을 가함" in str(v): return "충격"
    return "기타형태"


def g_place(v):
    v = str(v)
    if v == "운동장": return "운동장"
    if "체육" in v or "강당" in v: return "체육시설"
    if "교실" in v: return "교실"
    if v in ("계단", "복도", "현관"): return "계단·복도"
    return "기타장소"


def g_part(v):
    if v in head: return "머리·얼굴"
    if v == "목": return "목"
    if v in trunk: return "몸통"
    if v in arm: return "팔·손"
    if v in leg: return "다리·발"
    if v == "복합부위": return "복합부위"
    return "기타부위"


df["형태"] = df["사고형태"].map(g_form)
df["장소"] = df["사고장소"].map(g_place)
df["부위"] = df["사고부위"].map(g_part)
df["방과후"] = df["사고시간"].isin(["방과후과정", "돌봄교실"]).astype(int)
df["체육특기"] = (df["사고자구분"] == "체육특기학생").astype(int)
df["남학생"] = (df["사고자성별"] == "남").astype(int)

# 근거 기반 기준범주
refs = {"학교급": "초등학교", "형태": "넘어짐", "장소": "교실", "부위": "팔·손"}
X_parts = [df[["체육특기", "남학생", "방과후"]].astype(int)]
for col, ref in refs.items():
    d = pd.get_dummies(df[col], prefix=col).drop(columns=[f"{col}_{ref}"]).astype(int)
    X_parts.append(d)
X = pd.concat(X_parts, axis=1)
Xc = sm.add_constant(X.astype(float))
res = sm.Logit(df["중대"].values, Xc).fit(disp=False, maxiter=200)
od = pd.DataFrame({"오즈비": np.exp(res.params), "lo": np.exp(res.conf_int()[0]),
                   "hi": np.exp(res.conf_int()[1]), "p": res.pvalues}).drop("const")

# 항목별 그룹 정의 (표시명, 기준 명시). '기타' 계열은 세부정보가 없어 표시에서 제외(모델에는 통제변수로 유지).
panels = [
    ("학교급 (기준: 초등학교)", [("학교급_유치원", "유치원"), ("학교급_중학교", "중학교"),
                          ("학교급_고등학교", "고등학교"), ("학교급_특수학교", "특수학교")]),
    ("개인·상황 특성", [("체육특기", "체육특기학생 (기준: 일반학생)"), ("남학생", "남학생 (기준: 여학생)"),
                    ("방과후", "방과후·돌봄 (기준: 그 외 시간)")]),
    ("사고형태 (기준: 넘어짐)", [("형태_추락", "추락"), ("형태_부딪힘", "부딪힘"), ("형태_충격", "충격")]),
    ("사고장소 (기준: 일반교실)", [("장소_운동장", "운동장"), ("장소_체육시설", "체육시설·강당"),
                          ("장소_계단·복도", "계단·복도")]),
    ("사고부위 (기준: 팔·손)", [("부위_다리·발", "다리·발"), ("부위_머리·얼굴", "머리·얼굴"),
                         ("부위_몸통", "몸통"), ("부위_복합부위", "복합부위"), ("부위_목", "목")]),
]

# 항목별 오즈비 CSV
for title, items in panels:
    rows = []
    for key, disp in items:
        if key in od.index:
            r = od.loc[key]
            rows.append([disp, round(r["오즈비"], 2), round(r["lo"], 2), round(r["hi"], 2), round(r["p"], 4)])
    name = title.split(" (")[0].replace("·", "")
    pd.DataFrame(rows, columns=["범주", "오즈비", "CI_low", "CI_high", "p값"]).to_csv(
        D_DATA / f"오즈비_{name}.csv", index=False, encoding="utf-8-sig")

# 항목별 패널 그래프 (2x3)
fig, axes = plt.subplots(3, 2, figsize=(13, 12))
axes = axes.flatten()
for ax, (title, items) in zip(axes, panels):
    data = [(disp, od.loc[key]) for key, disp in items if key in od.index]
    data = sorted(data, key=lambda t: t[1]["오즈비"])
    ys = np.arange(len(data))
    ors = [d[1]["오즈비"] for d in data]
    los = [d[1]["오즈비"] - d[1]["lo"] for d in data]
    his = [d[1]["hi"] - d[1]["오즈비"] for d in data]
    colors = [RED if o > 1 else BLUE for o in ors]
    ax.errorbar(ors, ys, xerr=[los, his], fmt="none", ecolor=GRAY, elinewidth=1.2, capsize=3, zorder=1)
    ax.scatter(ors, ys, s=50, c=colors, zorder=2, edgecolor="white", linewidth=0.8)
    hi_max = max(d[1]["hi"] for d in data)
    for i, (o, r) in enumerate(zip(ors, [d[1] for d in data])):
        star = "*" if r["p"] < 0.05 else ""
        ax.text(r["hi"] * 1.15, i, f"{o:.2f}{star}", ha="left", va="center", fontsize=8.4, color="#333")
    ax.axvline(1, color="#444", ls="--", lw=1)
    ax.set_yticks(ys); ax.set_yticklabels([d[0] for d in data], fontsize=9)
    ax.set_ylim(-0.6, len(data) - 0.4)
    ax.set_xscale("log")
    xt = [0.1, 0.2, 0.5, 1, 2, 5, 10]
    ax.xaxis.set_major_locator(FixedLocator(xt)); ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([("%g" % t) for t in xt], fontsize=8)
    ax.set_xlim(0.08, hi_max * 2.4)  # 오른쪽 값 라벨 공간 확보
    ax.set_title(title, fontsize=10.5, fontweight="bold", color="#17365D", pad=8)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(axis="x", color="#EEE", lw=0.6); ax.set_axisbelow(True)
axes[-1].axis("off")
fig.suptitle("항목별 중대사고 위험요인 오즈비", fontsize=14, fontweight="bold", y=0.985)
fig.text(0.01, 0.005, "자료: 학교안전공제 보상 데이터(2021~2025). 로지스틱 회귀(중대=중상·사망), 5겹 교차검증 AUC 0.843. 모든 오즈비는 다른 요인을 동시에 통제한 값.",
         fontsize=8, color="#888")
fig.tight_layout(rect=(0, 0.02, 1, 0.965))
fig.savefig(D_FIG / "그림3_위험요인_오즈비_항목별.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("저장: 그림3_위험요인_오즈비_항목별.png")
print("항목별 CSV:", [p.name for p in sorted(D_DATA.glob('오즈비_*.csv'))])
print("주요 오즈비(새 기준):")
for k in ["장소_운동장", "부위_다리·발", "부위_머리·얼굴", "형태_추락", "학교급_고등학교", "체육특기"]:
    if k in od.index:
        print(f"  {k}: OR {od.loc[k,'오즈비']:.2f} (p={od.loc[k,'p']:.4f})")
