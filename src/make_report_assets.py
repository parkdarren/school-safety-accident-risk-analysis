# -*- coding: utf-8 -*-
"""B 보고서용 데이터 CSV + 그래프 생성 → 중대사고_위험분석/ 폴더."""
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
NEW = ROOT / "중대사고_위험분석"
D_DATA = NEW / "데이터"
D_FIG = NEW / "그래프"
for d in (D_DATA, D_FIG, NEW / "보고서"):
    d.mkdir(parents=True, exist_ok=True)

NAVY = "#17365D"
RED = "#C0392B"
ORANGE = "#E8973A"
BLUE = "#2F75B5"
GRAY = "#9AA5B1"

# ---- 원자료 로드 + 중대 라벨 ----
cols = ["학교급", "사고자구분", "사고자성별", "사고장소", "사고부위", "사고형태", "사고당시활동", "사고시간",
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
overall = df["중대"].mean() * 10000


def rate_table(col, minn=3000, keep=None):
    g = df.groupby(col)["중대"]
    t = pd.DataFrame({"전체건수": g.size(), "중대건수": g.sum()})
    t["중대화율_만분율"] = (t["중대건수"] / t["전체건수"] * 10000).round(1)
    if keep is not None:
        t = t.loc[[k for k in keep if k in t.index]]
    else:
        t = t[t["전체건수"] >= minn]
    return t.sort_values("중대화율_만분율", ascending=False)


# ---- 중대화율 CSV 저장 ----
t_school = rate_table("학교급", keep=["유치원", "초등학교", "중학교", "고등학교", "특수학교"])
t_type = rate_table("사고자구분", keep=["일반학생", "체육특기학생", "특수학교(학급)학생"])
t_form = rate_table("사고형태", minn=3000)
t_act = rate_table("사고당시활동", minn=3000)
for name, t in [("학교급", t_school), ("사고자구분", t_type), ("사고형태", t_form), ("사고당시활동", t_act)]:
    t.to_csv(D_DATA / f"중대화율_{name}.csv", encoding="utf-8-sig")

# 오즈비·지역중증도·중증도파일 복사
shutil.copy(DATA / "중대화_위험요인_오즈비.csv", D_DATA / "중대화_위험요인_오즈비.csv")
shutil.copy(DATA / "사고빈도_지역_중증도.csv", D_DATA / "사고빈도_지역_중증도.csv")
for lv in ["경상", "중상", "사망"]:
    shutil.copy(DATA / f"보상_{lv}_지역.csv", D_DATA / f"보상_{lv}_지역.csv")


# ---- 그림 1: 지역별 중증도 빈도 3패널 (사고빈도_지역_중증도.csv에서 직접 생성) ----
sevreg = pd.read_csv(D_DATA / "사고빈도_지역_중증도.csv")
sevreg = sevreg[sevreg["지역"] != "전국"].set_index("지역")
SEV_COLOR = {"경상": BLUE, "중상": ORANGE, "사망": RED}
SEV_RATE = {"경상": "경상_학생1만명당", "중상": "중상_학생1만명당", "사망": "사망_학생1만명당"}
SEV_CNT = {"경상": "경상_건수", "중상": "중상_건수", "사망": "사망_건수"}
order_reg = sevreg["경상_학생1만명당"].sort_values(ascending=False).index
fig, axes = plt.subplots(1, 3, figsize=(15, 7.5), sharey=False, gridspec_kw={"wspace": 0.32})
nat_tot = {lv: int(sevreg[SEV_CNT[lv]].sum()) for lv in ["경상", "중상", "사망"]}
nat_rt = {lv: sevreg[SEV_CNT[lv]].sum() / sevreg["학생수_5년인년"].sum() * 10000 for lv in ["경상", "중상", "사망"]}
for ax, lv in zip(axes, ["경상", "중상", "사망"]):
    vals = sevreg.loc[order_reg, SEV_RATE[lv]]
    ax.barh(vals.index, vals, color=SEV_COLOR[lv], height=0.72, edgecolor="white", linewidth=0.6)
    vmax = vals.max() if vals.max() > 0 else 1
    for reg, v in vals.items():
        ax.text(v + vmax * 0.02, reg, f"{v:,.3f}" if lv != "경상" else f"{v:,.1f}",
                va="center", ha="left", fontsize=8, color="#333")
    ax.invert_yaxis()
    ax.set_title(f"{lv}  (전국 {nat_rt[lv]:.3f} / 1만명, {nat_tot[lv]:,}건)", fontsize=10.5,
                 fontweight="bold", color=SEV_COLOR[lv], pad=8)
    ax.set_xlim(0, vmax * 1.2)
    ax.set_xlabel("학생 1만 명당 건수", fontsize=9.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=9); ax.grid(axis="x", color="#EEE", lw=0.7); ax.set_axisbelow(True)
fig.suptitle("지역별 중증도 구성별 빈도 (학생 1만 명당 건수)", fontsize=13, fontweight="bold", y=0.98)
fig.text(0.01, 0.01, "자료: 학교안전공제 보상 데이터(2021~2025), 교육통계 학생수", fontsize=8, color="#888")
fig.tight_layout(rect=(0, 0.02, 1, 0.96))
fig.savefig(D_FIG / "그림1_지역별_중증도_빈도.png", dpi=300, bbox_inches="tight")
plt.close(fig)


# ---- 그림 2: 누가·어디서·어떤 행위가 더 크게 다치나 (4패널 중대화율) ----
def hbar(ax, t, title, label_col="중대화율_만분율"):
    t2 = t.sort_values(label_col)
    colors = [RED if v >= overall * 1.5 else (ORANGE if v >= overall else BLUE) for v in t2[label_col]]
    ax.barh(t2.index, t2[label_col], color=colors, height=0.7, edgecolor="white", linewidth=0.6)
    for i, (idx, v) in enumerate(t2[label_col].items()):
        ax.text(v + t2[label_col].max() * 0.02, i, f"{v:.1f}", va="center", fontsize=8.5, color="#333")
    ax.axvline(overall, color=GRAY, ls="--", lw=1)
    ax.set_title(title, fontsize=11, fontweight="bold", color=NAVY, pad=6)
    ax.set_xlim(0, t2[label_col].max() * 1.16)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=9)
    ax.grid(axis="x", color="#EEE", lw=0.7)
    ax.set_axisbelow(True)


fig, axes = plt.subplots(2, 2, figsize=(13, 9))
hbar(axes[0, 0], t_school, "학교급별")
hbar(axes[0, 1], t_type, "사고자 구분")
hbar(axes[1, 0], t_form.head(8), "사고형태 (상위 8)")
hbar(axes[1, 1], t_act.head(8), "사고당시활동 (상위 8)")
fig.suptitle(f"누가·어디서·어떤 행위가 더 크게 다치나 — 중대화율(사고 1만 건당 중대 건수)   ·   점선=전체 평균 {overall:.1f}",
             fontsize=13, fontweight="bold", y=0.98)
fig.text(0.01, 0.01, "자료: 학교안전공제 보상 데이터(2021~2025). 중대=중상·사망(급여 유형 기준). 표본 3,000건 이상 범주만 표시.",
         fontsize=8, color="#888")
fig.tight_layout(rect=(0, 0.02, 1, 0.96))
fig.savefig(D_FIG / "그림2_위험집단_행위_중대화율.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ---- 그림 4(언제): 사고시간대별 중대화율 ----
def g_time(v):
    v = str(v)
    if v == "체육":
        return "체육수업"
    if v in ("체육대회", "경기출전", "현장학습", "수련활동, 수학여행", "학교축제", "동아리활동",
             "자율활동", "봉사활동", "진로활동", "자유놀이활동시간", "신체활동, 게임", "기타 특별활동시간"):
        return "특별활동·행사"
    if v in ("쉬는시간", "식사시간(간식 포함)", "자습시간"):
        return "쉬는시간·식사"
    if v in ("방과후과정", "돌봄교실"):
        return "방과후·돌봄"
    if v in ("등교", "하교"):
        return "등·하교"
    if v in ("이론수업", "과학", "실과(기술·가정)", "기타(음악, 미술 등)", "언어활동", "음악, 미술",
             "수학, 과학활동", "요리활동", "실외활동(바깥놀이 포함)", "기타 활동시간"):
        return "정규수업(비체육)"
    return "기타"


df["시간대"] = df["사고시간"].map(g_time)
t_time = rate_table("시간대", minn=3000)
t_time.to_csv(D_DATA / "중대화율_사고시간대.csv", encoding="utf-8-sig")

fig, ax = plt.subplots(figsize=(10, 5.2))
hbar(ax, t_time, f"중대사고는 언제 발생하나 — 사고시간대별 중대화율 (점선=전체 평균 {overall:.1f})")
ax.set_xlabel("중대화율 (사고 1만 건당 중대 건수)", fontsize=9.5)
fig.text(0.01, 0.02, "자료: 학교안전공제 보상 데이터(2021~2025). 중대=중상·사망(급여 종류 기준). 표본 3,000건 이상 시간대만 표시.",
         fontsize=8, color="#888")
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(D_FIG / "그림4_언제_사고시간대별_중대화율.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print("완료. 폴더:", NEW)
print("데이터:", [p.name for p in sorted(D_DATA.glob('*.csv'))])
print("그래프:", [p.name for p in sorted(D_FIG.glob('*.png'))])
print(f"전체 평균 중대화율: {overall:.1f}/만건")
