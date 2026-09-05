# -*- coding: utf-8 -*-
"""
급여 유형 기준 중증도 3분류(경상/중상/사망) 지역별 집계 파일 생성.

분류 근거(액수가 아니라 「학교안전사고 예방 및 보상에 관한 법률」의 급여 종류):
  - 사망: 유족급여(제39조) 또는 장례비(제40조)가 지급된 건. 명백한 사망 급여만 사용.
  - 중상: 장해급여(제37조, 치료 종료 후 장해 잔존) 또는 간병급여(제38조, 상시·수시 간병)가 지급된 건(사망 제외)
  - 경상: 위에 속하지 않는 나머지 전부(요양 종결형/일반 피해).
  - 제외: 보전비용(제48조, 교직원 지출)만 또는 위로금(제40조의2, 원인불명)만 있는 건 = 명확한 부상·사망 아님

왜 요양급여 액수로 나누지 않는가:
  - 법이 요양급여(치료비, 제36조)와 장해급여(치료 후 장해, 제37조)를 분리 정의한다.
  - 실제로 장해급여 건의 요양급여 중앙값이 0원이다(치료비는 다른 연도에 별도 지급). 액수는 중증도 지표가 아니다.

주의:
  - 보상 데이터에는 사고발생일이 없으므로 '2021~2025 지급 기준' 집계다.
  - 사고 데이터와 공통 사건키가 없어 사건 단위 연결은 하지 않는다.
  - 지급액은 의학적 중증도가 아니라 제도적 급여 종류로 해석한다.
  - 사망(전국 25건), 중상(541건)은 소표본이므로 5년 누적으로만 제시한다.
"""
from pathlib import Path

import pandas as pd

folder = Path(__file__).resolve().parent
data_folder = folder / "data"
out_folder = data_folder
pay_file = data_folder / "★2021-2025 학교안전사고 보상 데이터.xlsx"
year_file = data_folder / "year.csv"

years = ["2021", "2022", "2023", "2024", "2025"]
benefit_cols = ["요양급여", "장해급여", "간병급여", "유족급여", "장례비", "위로금", "보전비용"]

# 1) 보상 원자료 로드 및 연도 통합
frames = []
for y in years:
    df = pd.read_excel(pay_file, sheet_name=y, usecols=["지역", *benefit_cols])
    df["지급연도"] = int(y)
    frames.append(df)
pay = pd.concat(frames, ignore_index=True)
for c in benefit_cols:
    pay[c] = pd.to_numeric(pay[c], errors="coerce").fillna(0)
pay["총지급액"] = pay[benefit_cols].sum(axis=1)


# 2) 급여 유형 기준 중증도 분류
core_cols = ["요양급여", "장해급여", "간병급여", "유족급여", "장례비"]  # 부상·사망 판정 급여


def classify(row):
    # 보전비용(제48조, 교직원 지출) 또는 위로금(제40조의2, 원인불명)만 있는 건은
    # 명확한 학생 부상·사망으로 보기 어려워 제외한다.
    if (row["보전비용"] > 0 or row["위로금"] > 0) and sum(row[c] for c in core_cols) == 0:
        return "제외"
    if row["유족급여"] > 0 or row["장례비"] > 0:
        return "사망"
    if row["장해급여"] > 0 or row["간병급여"] > 0:
        return "중상"
    return "경상"


pay["중증도"] = pay.apply(classify, axis=1)
excluded = int((pay["중증도"] == "제외").sum())
pay = pay[pay["중증도"] != "제외"].copy()  # 보전비용 전용 건 제외

# 3) 학생수(5년 인년) 분모
year = pd.read_csv(year_file, encoding="utf-8-sig")
students = year.groupby("지역", as_index=True)["학생수"].sum()
total_students = students.sum()

regions = sorted(students.index.tolist())
labels = {"경상": "보상_경상_지역", "중상": "보상_중상_지역", "사망": "보상_사망_지역"}


def build_table(level):
    sub = pay[pay["중증도"] == level]
    rows = []
    for region in regions + ["전국"]:
        if region == "전국":
            part = sub
            denom = total_students
        else:
            part = sub[sub["지역"] == region]
            denom = students.get(region, 0)
        cnt = len(part)
        total_amt = part["총지급액"].sum()
        row = {
            "지역": region,
            "지급건수": cnt,
            "총지급액": int(total_amt),
            "학생수_5년인년": int(denom),
            "학생1만명당_지급건수": round(cnt / denom * 10000, 3) if denom else 0.0,
            "건당평균금액": int(round(total_amt / cnt)) if cnt else 0,
        }
        for c in benefit_cols:
            row[c] = int(part[c].sum())
        rows.append(row)
    return pd.DataFrame(rows)


print("=== 급여유형 기준 중증도 3분류 (2021~2025 지급) ===")
for level, name in labels.items():
    table = build_table(level)
    path = out_folder / f"{name}.csv"
    table.to_csv(path, index=False, encoding="utf-8-sig")
    nat = table[table["지역"] == "전국"].iloc[0]
    print(f"[{level}] {name}.csv 저장 | 전국 {nat['지급건수']:,}건, "
          f"{nat['총지급액']/1e8:.1f}억, 1만명당 {nat['학생1만명당_지급건수']}건, "
          f"건당 {nat['건당평균금액']:,}원")

# 4) 검산: 세 분류 합 + 제외 = 전체 지급건수
total_rows = len(pay)
by_level = pay["중증도"].value_counts()
print(f"\n보전비용 전용(제외): {excluded}건")
print("검산: 분류 합계 =", int(by_level.sum()), "+ 제외", excluded,
      "=", int(by_level.sum()) + excluded, "/ 원자료 528,503",
      "(일치)" if int(by_level.sum()) + excluded == 528503 else "(불일치!)")
print("분류별 건수:", by_level.to_dict())
