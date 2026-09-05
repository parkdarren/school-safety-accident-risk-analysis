# -*- coding: utf-8 -*-
"""
사고 데이터 기반 지역별·월별 사고빈도 파일 생성.

- 빈도 = 사고 발생 건수(보상액 아님). 사고 데이터(★사고 데이터)에서 산출.
- 사고연월(YYYY-MM)에서 사고연도·사고월을 뽑고, 2021~2025년 발생분만 남긴다.
  (사고 데이터 시트는 접수연도 2021~2025이지만 사고연월은 2002~2025가 섞여 있어 발생월로 재필터.)
- 분모는 교육통계 학생수(해당 연도 인원, year.csv). 월별 학생수는 없으므로 연도 인원을 사용한다.
- 보상(중증도) 파일은 날짜 컬럼이 없어 '지급연도 2021~2025'가 경계이며 이미 준수한다.
"""
from pathlib import Path

import pandas as pd

folder = Path(__file__).resolve().parent
data_folder = folder / "data"
acc_file = data_folder / "★2021-2025 학교안전사고 데이터.xlsx"
year_file = data_folder / "year.csv"
out_file = data_folder / "사고빈도_지역_월별.csv"

sheets = ["2021", "2022", "2023", "2024", "2025"]

# 1) 사고 원자료 로드(지역·사고연월만)
frames = []
for s in sheets:
    df = pd.read_excel(acc_file, sheet_name=s, usecols=["지역", "사고연월"])
    df["접수연도"] = int(s)
    frames.append(df)
acc = pd.concat(frames, ignore_index=True)
total_raw = len(acc)

# 2) 사고연월 → 사고연도·사고월 파싱
ym = acc["사고연월"].astype(str).str.strip()
acc["사고연도"] = pd.to_numeric(ym.str[:4], errors="coerce")
acc["사고월"] = pd.to_numeric(ym.str[5:7], errors="coerce")

# 3) 2021~2025년 발생분만 남기기
before = len(acc)
mask = acc["사고연도"].between(2021, 2025) & acc["사고월"].between(1, 12)
acc = acc[mask].copy()
acc["사고연도"] = acc["사고연도"].astype(int)
acc["사고월"] = acc["사고월"].astype(int)
removed = before - len(acc)
print(f"전체 사고행 {total_raw:,} → 2021~2025 발생분 {len(acc):,} (제외 {removed:,}, {removed/total_raw*100:.2f}%)")

# 4) 지역 × 사고연도 × 사고월 집계
grp = (
    acc.groupby(["지역", "사고연도", "사고월"], as_index=False)
    .size()
    .rename(columns={"size": "사고건수"})
)

# 전국 = 모든 지역 합
nat = (
    grp.groupby(["사고연도", "사고월"], as_index=False)["사고건수"].sum()
)
nat.insert(0, "지역", "전국")
table = pd.concat([grp, nat], ignore_index=True)

# 5) 학생수(해당 연도) 분모 결합
year = pd.read_csv(year_file, encoding="utf-8-sig")
stu_region = year.set_index(["지역", "연도"])["학생수"]
stu_nation = year.groupby("연도")["학생수"].sum()


def students(region, y):
    if region == "전국":
        return int(stu_nation.get(y, 0))
    return int(stu_region.get((region, y), 0))


table["학생수_해당연도"] = table.apply(lambda r: students(r["지역"], r["사고연도"]), axis=1)
table["학생1만명당_사고건수"] = (
    table["사고건수"] / table["학생수_해당연도"].where(table["학생수_해당연도"] != 0) * 10000
).round(3)

# 6) 정렬(전국 먼저, 그다음 지역 가나다·연도·월) 후 저장
table["_ord"] = (table["지역"] != "전국").astype(int)
table = table.sort_values(["_ord", "지역", "사고연도", "사고월"]).drop(columns="_ord")
table = table[["지역", "사고연도", "사고월", "사고건수", "학생수_해당연도", "학생1만명당_사고건수"]]
table.to_csv(out_file, index=False, encoding="utf-8-sig")
print(f"저장: {out_file.name} ({len(table):,}행, 지역 {table['지역'].nunique()}개)")

# 7) 검산
nat_total = table[table["지역"] == "전국"]["사고건수"].sum()
region_total = table[table["지역"] != "전국"]["사고건수"].sum()
print(f"검산: 전국 합 {nat_total:,} = 지역 합 {region_total:,}",
      "(일치)" if nat_total == region_total else "(불일치!)")
print("월별 전국 사고건수(5년 합):")
by_month = table[table["지역"] == "전국"].groupby("사고월")["사고건수"].sum()
print("  ", {int(m): int(v) for m, v in by_month.items()})
