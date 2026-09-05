# -*- coding: utf-8 -*-
"""
지역별 중증도(경상/중상/사망) 사고 빈도 파일 생성 → data/사고빈도_지역_중증도.csv

- 사망/중상 = 보상 데이터 급여 유형 기준(보상_사망/중상_지역.csv, 지급 기준)
    · 사망 = 유족급여 OR 장례비
    · 중상 = 장해급여 OR 간병급여
- 경상 = 전체 사고 − 중상 − 사망 (사고빈도_지역_월별.csv, 발생 기준). 공제급여 미지급 사고 포함.
- 학생 1만 명당 비율은 5년 학생인년 기준.
- 기준 차이: 경상은 발생 기준, 중상·사망은 지급 기준(공통 사건키 없어 조합). 중상·사망이 소수라 경상엔 영향 미미.

선행 실행: make_frequency.py, make_severity.py
"""
from pathlib import Path

import pandas as pd

folder = Path(__file__).resolve().parent
data = folder / "data"
out_file = data / "사고빈도_지역_중증도.csv"

freq = pd.read_csv(data / "사고빈도_지역_월별.csv")
acc = freq[freq["지역"] != "전국"].groupby("지역")["사고건수"].sum()
중상 = pd.read_csv(data / "보상_중상_지역.csv").set_index("지역")
사망 = pd.read_csv(data / "보상_사망_지역.csv").set_index("지역")
students = 중상["학생수_5년인년"]

regions = sorted(acc.index)
rows = []
for region in regions + ["전국"]:
    if region == "전국":
        a = int(acc.sum())
        m = int(중상.loc[regions, "지급건수"].sum())
        d = int(사망.loc[regions, "지급건수"].sum())
        s = int(students.loc[regions].sum())
    else:
        a = int(acc[region]); m = int(중상.loc[region, "지급건수"])
        d = int(사망.loc[region, "지급건수"]); s = int(students[region])
    g = a - m - d  # 경상 = 전체 사고 − 중상 − 사망
    rows.append({
        "지역": region,
        "경상_건수": g, "중상_건수": m, "사망_건수": d, "전체사고": a,
        "학생수_5년인년": s,
        "경상_학생1만명당": round(g / s * 10000, 3) if s else 0.0,
        "중상_학생1만명당": round(m / s * 10000, 3) if s else 0.0,
        "사망_학생1만명당": round(d / s * 10000, 3) if s else 0.0,
    })

df = pd.DataFrame(rows)
df.to_csv(out_file, index=False, encoding="utf-8-sig")

nat = df[df["지역"] == "전국"].iloc[0]
print(f"저장: {out_file.name} ({df.shape[0]}행)")
print(f"전국: 경상 {nat['경상_건수']:,} / 중상 {nat['중상_건수']:,} / 사망 {nat['사망_건수']:,} / 전체사고 {nat['전체사고']:,}")
ok = (df["경상_건수"] + df["중상_건수"] + df["사망_건수"] == df["전체사고"]).all()
print("검산: 경상+중상+사망 = 전체사고 →", "모든 행 일치" if ok else "불일치!")
