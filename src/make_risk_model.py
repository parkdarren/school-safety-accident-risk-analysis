# -*- coding: utf-8 -*-
"""
중대사고(중상·사망) 위험요인 로지스틱 회귀 모델.

- 대상: 보상 데이터 지급건(보전비용 전용 제외). 중대(중상+사망)=1, 경상=0.
- 중대 566건뿐이므로 변수를 의미 단위로 묶어 과적합을 억제(사건 10건당 변수 1개 원칙).
- 결과: 각 요인의 오즈비(OR)+95% 신뢰구간+p값, 교차검증 AUC, 고위험 페르소나.
- 해석: 관찰데이터 기반 연관성(인과 아님). 지급된 사고 안에서의 모델(선택편향 존재).
- 변수 선택 근거: 선행연구(김위정·김진희, 2025)가 학교안전사고 심각도 예측에 사용한
  사고형태·장소·활동·부위 등을 참고.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

folder = Path(__file__).resolve().parent
data = folder / "data"
pay_file = data / "★2021-2025 학교안전사고 보상 데이터.xlsx"

cols = ["학교급", "사고자구분", "사고자성별", "사고시간", "사고장소", "사고부위", "사고형태",
        "요양급여", "장해급여", "간병급여", "유족급여", "장례비", "위로금", "보전비용"]
frames = [pd.read_excel(pay_file, sheet_name=y, usecols=cols) for y in ["2021", "2022", "2023", "2024", "2025"]]
df = pd.concat(frames, ignore_index=True)
for c in ["요양급여", "장해급여", "간병급여", "유족급여", "장례비", "위로금", "보전비용"]:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)


# 라벨: 중대(중상+사망)=1
def label(r):
    core = r["요양급여"] + r["장해급여"] + r["간병급여"] + r["유족급여"] + r["장례비"]
    if (r["보전비용"] > 0 or r["위로금"] > 0) and core == 0:
        return np.nan  # 보전비용·위로금 전용 제외
    if r["유족급여"] > 0 or r["장례비"] > 0:
        return 1
    if r["장해급여"] > 0 or r["간병급여"] > 0:
        return 1
    return 0


df["중대"] = df.apply(label, axis=1)
df = df[df["중대"].notna()].copy()
df["중대"] = df["중대"].astype(int)
df = df[df["학교급"] != "기타학교"]  # 이질적 catch-all 제외(소수)


# ---- 의미 단위 변수 묶기 ----
def g_form(v):
    if v == "넘어짐":
        return "넘어짐"
    if "떨어짐" in str(v):
        return "추락"
    if "부딪힘" in str(v):
        return "부딪힘"
    if "충격을 가함" in str(v):
        return "충격"
    return "기타형태"


def g_place(v):
    v = str(v)
    if v == "운동장":
        return "운동장"
    if "체육" in v or "강당" in v:
        return "체육시설"
    if "교실" in v:
        return "교실"
    if v in ("계단", "복도", "현관"):
        return "계단복도"
    return "기타장소"


head = ["두피", "뇌(두개내)", "이마", "눈", "코", "귀", "볼", "턱", "입술 및 구강", "치아", "목구멍"]
trunk = ["흉부", "복부", "내장기관", "등", "허리", "골반/엉덩이"]
arm = ["어깨", "위팔", "팔꿈치", "아래팔", "손목", "손", "손가락"]
leg = ["넓적다리(허벅지)", "무릎", "아래다리(종아리)", "발목", "발", "발가락"]


def g_part(v):
    if v in head:
        return "머리·얼굴"
    if v == "목":
        return "목"
    if v in trunk:
        return "몸통"
    if v in arm:
        return "팔·손"
    if v in leg:
        return "다리·발"
    if v == "복합부위":
        return "복합부위"
    return "기타부위"


df["형태"] = df["사고형태"].map(g_form)
df["장소"] = df["사고장소"].map(g_place)
df["부위"] = df["사고부위"].map(g_part)
df["방과후"] = df["사고시간"].isin(["방과후과정", "돌봄교실"]).astype(int)
df["체육특기"] = (df["사고자구분"] == "체육특기학생").astype(int)
df["남학생"] = (df["사고자성별"] == "남").astype(int)

# 참조범주(가장 흔한 값)를 기준으로 더미화
cats = {
    "학교급": "초등학교", "형태": "넘어짐", "장소": "운동장", "부위": "다리·발",
}
X_parts = [df[["방과후", "체육특기", "남학생"]]]
labels_map = {}
for col, ref in cats.items():
    d = pd.get_dummies(df[col], prefix=col)
    d = d.drop(columns=[f"{col}_{ref}"])
    X_parts.append(d.astype(int))
    labels_map[col] = ref
X = pd.concat(X_parts, axis=1)
y = df["중대"].values

print(f"표본: {len(df):,} | 중대 {int(y.sum())} | 경상 {int((y==0).sum())} | 변수 {X.shape[1]}개")
print("참조범주:", labels_map, "| 방과후·체육특기·남학생은 '아니오' 대비")

# ---- 로지스틱 회귀 (statsmodels: OR+신뢰구간+p) ----
Xc = sm.add_constant(X.astype(float))
res = sm.Logit(y, Xc).fit(disp=False, maxiter=200)
odds = pd.DataFrame({
    "오즈비": np.exp(res.params),
    "CI_low": np.exp(res.conf_int()[0]),
    "CI_high": np.exp(res.conf_int()[1]),
    "p값": res.pvalues,
}).drop("const")
odds = odds.sort_values("오즈비", ascending=False)
pd.set_option("display.width", 200)
print("\n=== 위험요인 오즈비 (중대화 = 경상 대비) ===")
for name, r in odds.iterrows():
    star = "***" if r["p값"] < 0.001 else "**" if r["p값"] < 0.01 else "*" if r["p값"] < 0.05 else ""
    print(f"  {name:14s} OR {r['오즈비']:5.2f}  (95%CI {r['CI_low']:4.2f}~{r['CI_high']:5.2f})  p={r['p값']:.4f} {star}")

# ---- 교차검증 AUC (sklearn) ----
clf = LogisticRegression(max_iter=1000, C=1.0)
auc = cross_val_score(clf, X.astype(float), y, cv=5, scoring="roc_auc")
print(f"\n5겹 교차검증 AUC: {auc.mean():.3f} (±{auc.std():.3f})")

# ---- 고위험 페르소나: 각 범주에서 '가장 위험한 수준 하나씩'만 선택 ----
profile = pd.Series(0, index=X.columns, dtype=float)
chosen = []
# 이진 변수: 유의(p<0.05)하게 위험(OR>1)이면 1
for b in ["방과후", "체육특기", "남학생"]:
    if odds.loc[b, "오즈비"] > 1 and odds.loc[b, "p값"] < 0.05:
        profile[b] = 1
        chosen.append(b)
# 범주형: 각 그룹에서 유의(p<0.05)·OR>1 중 최대 하나 (없으면 참조범주가 최고위험)
for col in cats:
    gcols = [c for c in X.columns if c.startswith(col + "_")]
    cand = odds.loc[[c for c in gcols if c in odds.index]]
    cand = cand[(cand["p값"] < 0.05) & (cand["오즈비"] > 1)]
    if len(cand):
        top = cand["오즈비"].idxmax()
        profile[top] = 1
        chosen.append(top)
    else:
        chosen.append(f"{col}={cats[col]}(참조=최고위험)")

prow = sm.add_constant(pd.DataFrame([profile]), has_constant="add")[Xc.columns]
p_persona = res.predict(prow)[0]
base = y.mean()
print(f"\n=== 고위험 페르소나 (각 항목 최고위험 1개씩) ===")
print("구성:", chosen)
print(f"예측 중대화 확률: {p_persona*100:.1f}%  |  전체 평균 {base*100:.3f}%  →  약 {p_persona/base:.0f}배")

odds.to_csv(data / "중대화_위험요인_오즈비.csv", encoding="utf-8-sig")
print("\n저장: data/중대화_위험요인_오즈비.csv")
