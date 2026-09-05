from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


base = Path(__file__).resolve().parent
data_dir = base / "data"
years = range(2021, 2026)
seed = 20260720

accident_file_name = "★2021-2025 학교안전사고 데이터.xlsx"
payment_file_name = "★2021-2025 학교안전사고 보상 데이터.xlsx"
standard_school_levels = ["유치원", "초등학교", "중학교", "고등학교", "특수학교"]
expected_regions = {
    "서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종", "경기",
    "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주",
}
other_school_levels = [
    "기타학교",
    "각종학교",
    "고등공민학교",
    "고등기술학교",
    "방송통신고등학교",
    "방송통신중학교",
]

pay_cols = [
    "요양급여",
    "장해급여",
    "간병급여",
    "유족급여",
    "장례비",
    "위로금",
    "보전비용",
]

heavy_cols = ["장해급여", "간병급여", "유족급여", "장례비"]

parts = {
    "사고시간": "사고시간",
    "사고장소": "사고장소",
    "사고부위": "사고부위",
    "사고형태": "사고형태",
    "사고당시활동": "사고당시활동",
}

type_order = ["복합위험형", "반복사고형", "숨은중대위험형", "상대적안정형"]


def divide(a, b):
    return np.where(np.asarray(b) != 0, np.asarray(a) / np.asarray(b), np.nan)


def risk_group(freq, sev):
    high_freq = freq >= 100
    high_sev = sev >= 100
    if high_freq and high_sev:
        return "복합위험형"
    if high_freq:
        return "반복사고형"
    if high_sev:
        return "숨은중대위험형"
    return "상대적안정형"


def most_common(x):
    count = Counter(x.dropna().astype(str))
    if not count:
        return "", 0
    order = {name: i for i, name in enumerate(type_order)}
    name, num = sorted(count.items(), key=lambda v: (-v[1], order.get(v[0], 99), v[0]))[0]
    return name, num


def school_level_group(value):
    name = str(value).strip()
    if name in standard_school_levels:
        return name
    if name in other_school_levels:
        return "기타학교"
    raise ValueError(f"알 수 없는 학교급 코드입니다: {name}")


def check_regions(frame, region_col, label):
    if frame[region_col].isna().any():
        raise ValueError(f"{label}에 지역 결측값이 있습니다.")
    found = set(frame[region_col].astype(str).str.strip().unique())
    if found != expected_regions:
        raise ValueError(
            f"{label}의 지역 구성이 예상과 다릅니다. "
            f"누락={sorted(expected_regions - found)}, 추가={sorted(found - expected_regions)}"
        )


def require_columns(frame, columns, label):
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise ValueError(f"{label}에 필수 열이 없습니다: {missing}")


def find_raw():
    accident_file = data_dir / accident_file_name
    payment_file = data_dir / payment_file_name
    missing = [str(file) for file in [accident_file, payment_file] if not file.exists()]
    if missing:
        raise FileNotFoundError(f"사고·보상 원자료를 찾지 못했습니다: {missing}")
    return accident_file, payment_file


def find_school(year):
    pattern = f"{year}년 유초중등 학교별 학년별 학생수 학급수 입학 졸업 교원 직원 면적*.xlsx"
    files = list(data_dir.glob(pattern))
    if len(files) != 1:
        raise FileNotFoundError(f"{year}년 학생수 파일을 찾지 못했습니다: {files}")
    return files[0]


def header_line(file, year):
    check = pd.read_excel(file, sheet_name=0, header=None, nrows=30, usecols=[0])
    target = f"{year}0401"
    for i, value in enumerate(check.iloc[:, 0]):
        if str(value).strip().startswith(target):
            return i - 1
    raise ValueError(f"{file.name}에서 표 제목을 찾지 못했습니다.")


def school_by_area(year):
    file = find_school(year)
    row = header_line(file, year)
    school = pd.read_excel(file, sheet_name=0, header=row)
    school.columns = school.columns.astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

    require_columns(
        school,
        ["시도", "학교급", "상태", "편성학급수_계", "학생수_총계_계", "교원수_총계_계", "교지면적"],
        f"{year}년 학생수 자료",
    )

    use = school[
        ["시도", "학교급", "상태", "편성학급수_계", "학생수_총계_계", "교원수_총계_계", "교지면적"]
    ].copy()
    use.columns = ["region", "school_level", "status", "classes", "students", "teachers", "site_area"]
    for col in ["classes", "students", "teachers", "site_area"]:
        use[col] = pd.to_numeric(use[col], errors="coerce").fillna(0)
    use = use[use["students"] > 0].copy()
    use["school_level_group"] = use["school_level"].map(school_level_group)

    grouped = (
        use.groupby("region", as_index=False)
        .agg(
            students=("students", "sum"),
            teachers=("teachers", "sum"),
            classes=("classes", "sum"),
            site_area=("site_area", "sum"),
            schools=("school_level", "size"),
        )
    )
    grouped["year"] = year
    check_regions(grouped, "region", f"{year}년 학생수 자료")

    level = (
        use.groupby(["region", "school_level_group"], as_index=False)
        .agg(
            students=("students", "sum"),
            teachers=("teachers", "sum"),
            classes=("classes", "sum"),
            site_area=("site_area", "sum"),
            schools=("school_level", "size"),
        )
    )
    level["year"] = year
    return grouped, level


def clean_pay(pay, year):
    pay = pay.copy()
    pay["school_level_group"] = pay["학교급"].map(school_level_group)
    for col in pay_cols:
        pay[col] = pd.to_numeric(pay[col], errors="coerce").fillna(0)
    if pay[pay_cols].lt(0).any().any():
        bad = pay_cols[pay[pay_cols].lt(0).any()].tolist()
        raise ValueError(f"{year}년 보상자료에 음수 지급액이 있습니다: {bad}")
    pay["payment_total"] = pay[pay_cols].sum(axis=1)
    if (pay["payment_total"] <= 0).any():
        count = int((pay["payment_total"] <= 0).sum())
        raise ValueError(f"{year}년 보상자료에 합계가 0원 이하인 행이 {count}건 있습니다.")
    pay["is_severe_benefit"] = pay[heavy_cols].gt(0).any(axis=1)

    cutoffs = {
        "p90": float(pay["payment_total"].quantile(0.90)),
        "p95": float(pay["payment_total"].quantile(0.95)),
        "p99": float(pay["payment_total"].quantile(0.99)),
    }
    for name, cut in cutoffs.items():
        pay[f"is_high_cost_{name}"] = pay["payment_total"] >= cut
    pay["payment_winsor_p99"] = pay["payment_total"].clip(upper=cutoffs["p99"])
    pay["year"] = year
    return pay, cutoffs


def pay_by_area(pay, cutoffs):
    area = (
        pay.groupby("지역", as_index=False)
        .agg(
            payment_rows=("payment_total", "size"),
            payment_total=("payment_total", "sum"),
            avg_payment_per_claim=("payment_total", "mean"),
            median_payment_per_claim=("payment_total", "median"),
            max_payment=("payment_total", "max"),
            severe_benefit_rows=("is_severe_benefit", "sum"),
            high_cost_rows_p90=("is_high_cost_p90", "sum"),
            high_cost_rows_p95=("is_high_cost_p95", "sum"),
            high_cost_rows_p99=("is_high_cost_p99", "sum"),
            winsor_payment_total_p99=("payment_winsor_p99", "sum"),
        )
        .rename(columns={"지역": "region"})
    )

    top_sum = (
        pay[pay["is_high_cost_p99"]]
        .groupby("지역")["payment_total"]
        .sum()
        .rename("top1pct_payment_total")
    )
    rest = (
        pay[~pay["is_high_cost_p99"]]
        .groupby("지역")
        .agg(
            payment_rows_excl_top1pct=("payment_total", "size"),
            payment_total_excl_top1pct=("payment_total", "sum"),
        )
    )
    area = area.merge(top_sum, left_on="region", right_index=True, how="left")
    area = area.merge(rest, left_on="region", right_index=True, how="left")
    area["top1pct_payment_total"] = area["top1pct_payment_total"].fillna(0)
    area[["payment_rows_excl_top1pct", "payment_total_excl_top1pct"]] = area[
        ["payment_rows_excl_top1pct", "payment_total_excl_top1pct"]
    ].fillna(0)

    area["avg_payment_excl_region_max"] = divide(
        area["payment_total"] - area["max_payment"],
        area["payment_rows"] - 1,
    )
    area["avg_payment_excl_top1pct"] = divide(
        area["payment_total_excl_top1pct"], area["payment_rows_excl_top1pct"]
    )
    area["avg_payment_winsor_p99"] = divide(
        area["winsor_payment_total_p99"], area["payment_rows"]
    )
    area["severe_benefit_rate"] = divide(area["severe_benefit_rows"], area["payment_rows"])
    for p in ["p90", "p95", "p99"]:
        area[f"high_cost_rate_{p}"] = divide(
            area[f"high_cost_rows_{p}"], area["payment_rows"]
        )
    area["top_claim_payment_share"] = divide(area["max_payment"], area["payment_total"])
    area["top1pct_payment_share"] = divide(area["top1pct_payment_total"], area["payment_total"])

    all_info = {
        "payment_rows": int(len(pay)),
        "payment_total": float(pay["payment_total"].sum()),
        "avg_payment_per_claim": float(pay["payment_total"].mean()),
        "median_payment_per_claim": float(pay["payment_total"].median()),
        "severe_benefit_rows": int(pay["is_severe_benefit"].sum()),
        "severe_benefit_rate": float(pay["is_severe_benefit"].mean()),
        "high_cost_rate_p90": float(pay["is_high_cost_p90"].mean()),
        "high_cost_rate_p95": float(pay["is_high_cost_p95"].mean()),
        "high_cost_rate_p99": float(pay["is_high_cost_p99"].mean()),
        "avg_payment_excl_region_max": float(
            (area["payment_total"].sum() - area["max_payment"].sum())
            / (area["payment_rows"].sum() - len(area))
        ),
        "avg_payment_excl_top1pct": float(
            pay.loc[~pay["is_high_cost_p99"], "payment_total"].mean()
        ),
        "avg_payment_winsor_p99": float(pay["payment_winsor_p99"].mean()),
        **{f"high_cost_cutoff_{name}": value for name, value in cutoffs.items()},
    }
    return area, all_info


def school_standard(acc, pay, school_level, standard_weight, year):
    acc = acc.copy()
    acc["school_level_group"] = acc["학교급"].map(school_level_group)

    school_core = school_level[
        school_level["school_level_group"].isin(standard_school_levels)
    ].copy()
    level_count = school_core.groupby("region")["school_level_group"].nunique()
    missing = level_count[level_count != len(standard_school_levels)]
    if not missing.empty:
        raise ValueError(f"{year}년 학교급 표준화에 필요한 층이 부족합니다: {missing.to_dict()}")

    acc_core = acc[acc["school_level_group"].isin(standard_school_levels)].copy()
    pay_core = pay[pay["school_level_group"].isin(standard_school_levels)].copy()
    acc_level = (
        acc_core.groupby(["지역", "school_level_group"], as_index=False)
        .size()
        .rename(columns={"지역": "region", "size": "accident_rows"})
    )
    pay_level = (
        pay_core.groupby(["지역", "school_level_group"], as_index=False)
        .agg(
            payment_rows=("payment_total", "size"),
            payment_total=("payment_total", "sum"),
        )
        .rename(columns={"지역": "region"})
    )

    level = school_core.merge(
        acc_level, on=["region", "school_level_group"], how="left"
    ).merge(pay_level, on=["region", "school_level_group"], how="left")
    for col in ["accident_rows", "payment_rows", "payment_total"]:
        level[col] = level[col].fillna(0)

    level["standard_weight"] = level["school_level_group"].map(standard_weight)
    level["accident_rate_per_10k"] = divide(level["accident_rows"], level["students"]) * 10_000
    level["payment_rate_per_10k"] = divide(level["payment_rows"], level["students"]) * 10_000
    level["payment_burden_per_10k"] = divide(level["payment_total"], level["students"]) * 10_000

    for name in ["accident_rate_per_10k", "payment_rate_per_10k", "payment_burden_per_10k"]:
        level[f"weighted_{name}"] = level[name] * level["standard_weight"]

    result = (
        level.groupby("region", as_index=False)
        .agg(
            std_accident_rate_per_10k=("weighted_accident_rate_per_10k", "sum"),
            std_payment_rate_per_10k=("weighted_payment_rate_per_10k", "sum"),
            std_payment_burden_per_10k=("weighted_payment_burden_per_10k", "sum"),
        )
    )
    result["std_avg_payment_per_record"] = divide(
        result["std_payment_burden_per_10k"], result["std_payment_rate_per_10k"]
    )

    national_level = (
        level.groupby("school_level_group", as_index=False)
        .agg(
            students=("students", "sum"),
            accident_rows=("accident_rows", "sum"),
            payment_rows=("payment_rows", "sum"),
            payment_total=("payment_total", "sum"),
        )
    )
    national_level["standard_weight"] = national_level["school_level_group"].map(
        standard_weight
    )
    national_level["accident_rate"] = divide(
        national_level["accident_rows"], national_level["students"]
    ) * 10_000
    national_level["payment_rate"] = divide(
        national_level["payment_rows"], national_level["students"]
    ) * 10_000
    national_level["payment_burden"] = divide(
        national_level["payment_total"], national_level["students"]
    ) * 10_000
    national_accident_rate = float(
        (national_level["accident_rate"] * national_level["standard_weight"]).sum()
    )
    national_payment_rate = float(
        (national_level["payment_rate"] * national_level["standard_weight"]).sum()
    )
    national_payment_burden = float(
        (national_level["payment_burden"] * national_level["standard_weight"]).sum()
    )
    national_avg_payment = national_payment_burden / national_payment_rate
    result["national_std_accident_rate_per_10k"] = national_accident_rate
    result["national_std_payment_rate_per_10k"] = national_payment_rate
    result["national_std_payment_burden_per_10k"] = national_payment_burden
    result["national_std_avg_payment_per_record"] = national_avg_payment
    result["frequency_index_school_standardized"] = (
        result["std_accident_rate_per_10k"] / national_accident_rate * 100
    )
    result["payment_rate_index_school_standardized"] = (
        result["std_payment_rate_per_10k"] / national_payment_rate * 100
    )
    result["severity_index_school_standardized"] = (
        result["std_avg_payment_per_record"] / national_avg_payment * 100
    )
    result["burden_index_school_standardized"] = (
        result["std_payment_burden_per_10k"] / national_payment_burden * 100
    )
    result["risk_type_school_standardized"] = [
        risk_group(f, s)
        for f, s in zip(
            result["frequency_index_school_standardized"],
            result["severity_index_school_standardized"],
        )
    ]

    all_students = float(school_core["students"].sum())
    all_school_students = float(school_level["students"].sum())
    result["standard_student_coverage"] = all_students / all_school_students
    result["standard_accident_coverage"] = len(acc_core) / len(acc)
    result["standard_payment_coverage"] = len(pay_core) / len(pay)
    result["year"] = year
    level["year"] = year
    return result, level


def make_year_data():
    acc_file, pay_file = find_raw()
    year_list = []
    trend_list = []
    standard_list = []
    acc_data = {}
    pay_data = {}

    school_data = {}
    school_level_data = {}
    school_level_all = []
    for year in years:
        school, school_level = school_by_area(year)
        school_data[year] = school
        school_level_data[year] = school_level
        school_level_all.append(school_level)
    school_level_all = pd.concat(school_level_all, ignore_index=True)
    school_level_all = school_level_all[
        school_level_all["school_level_group"].isin(standard_school_levels)
    ].copy()
    standard_students = school_level_all.groupby("school_level_group")["students"].sum()
    standard_weight = standard_students / standard_students.sum()
    if set(standard_weight.index) != set(standard_school_levels):
        raise ValueError(f"학교급 표준층이 예상과 다릅니다: {standard_weight.index.tolist()}")
    if not np.isclose(standard_weight.sum(), 1):
        raise ValueError("학교급 표준가중치 합이 1이 아닙니다.")

    for year in years:
        acc = pd.read_excel(acc_file, sheet_name=str(year))
        raw = pd.read_excel(pay_file, sheet_name=str(year))
        require_columns(acc, ["지역", "학교급", *parts.keys()], f"{year}년 사고자료")
        require_columns(raw, ["지역", "학교급", *pay_cols], f"{year}년 보상자료")
        check_regions(acc, "지역", f"{year}년 사고자료")
        check_regions(raw, "지역", f"{year}년 보상자료")
        pay, cutoffs = clean_pay(raw, year)
        acc_data[year] = acc.copy()
        pay_data[year] = pay.copy()

        acc_area = (
            acc.groupby("지역", as_index=False)
            .size()
            .rename(columns={"지역": "region", "size": "accident_rows"})
        )
        pay_area, pay_all = pay_by_area(pay, cutoffs)
        school = school_data[year]
        school_level = school_level_data[year]
        standard, standard_detail = school_standard(
            acc, pay, school_level, standard_weight, year
        )

        one = school.merge(acc_area, on="region", how="left").merge(
            pay_area, on="region", how="left"
        )
        one = one.merge(standard, on=["region", "year"], how="left")
        one["accident_rate_per_10k"] = divide(one["accident_rows"], one["students"]) * 10_000
        one["payment_rate_per_10k"] = divide(one["payment_rows"], one["students"]) * 10_000
        one["payment_burden_per_10k"] = divide(one["payment_total"], one["students"]) * 10_000
        one["students_per_teacher"] = divide(one["students"], one["teachers"])
        one["students_per_class"] = divide(one["students"], one["classes"])
        one["site_area_per_student"] = divide(one["site_area"], one["students"])

        all_students = float(one["students"].sum())
        all_acc = float(one["accident_rows"].sum())
        acc_rate = all_acc / all_students * 10_000
        pay_rate = pay_all["payment_rows"] / all_students * 10_000
        pay_load = pay_all["payment_total"] / all_students * 10_000

        one["national_accident_rate_per_10k"] = acc_rate
        one["national_payment_rate_per_10k"] = pay_rate
        one["national_avg_payment_per_claim"] = pay_all["avg_payment_per_claim"]
        one["national_payment_burden_per_10k"] = pay_load
        one["frequency_index"] = one["accident_rate_per_10k"] / acc_rate * 100
        one["payment_rate_index"] = one["payment_rate_per_10k"] / pay_rate * 100
        one["severity_index"] = one["avg_payment_per_claim"] / pay_all["avg_payment_per_claim"] * 100
        one["burden_index"] = one["payment_burden_per_10k"] / pay_load * 100
        one["severity_index_excl_region_max"] = (
            one["avg_payment_excl_region_max"] / pay_all["avg_payment_excl_region_max"] * 100
        )
        one["severity_index_excl_top1pct"] = (
            one["avg_payment_excl_top1pct"] / pay_all["avg_payment_excl_top1pct"] * 100
        )
        one["severity_index_winsor_p99"] = (
            one["avg_payment_winsor_p99"] / pay_all["avg_payment_winsor_p99"] * 100
        )
        one["high_cost_p99_index"] = one["high_cost_rate_p99"] / pay_all["high_cost_rate_p99"] * 100
        one["severe_benefit_index"] = one["severe_benefit_rate"] / pay_all["severe_benefit_rate"] * 100

        mid_acc = float(one["accident_rate_per_10k"].median())
        mid_pay = float(one["avg_payment_per_claim"].median())
        one["frequency_index_vs_region_median"] = one["accident_rate_per_10k"] / mid_acc * 100
        one["severity_index_vs_region_median"] = one["avg_payment_per_claim"] / mid_pay * 100

        one["risk_type_primary"] = [
            risk_group(f, s) for f, s in zip(one["frequency_index"], one["severity_index"])
        ]
        one["risk_type_region_median"] = [
            risk_group(f, s)
            for f, s in zip(
                one["frequency_index_vs_region_median"], one["severity_index_vs_region_median"]
            )
        ]
        one["risk_type_excl_region_max"] = [
            risk_group(f, s)
            for f, s in zip(one["frequency_index"], one["severity_index_excl_region_max"])
        ]
        one["risk_type_excl_top1pct"] = [
            risk_group(f, s)
            for f, s in zip(one["frequency_index"], one["severity_index_excl_top1pct"])
        ]
        one["risk_type_winsor_p99"] = [
            risk_group(f, s)
            for f, s in zip(one["frequency_index"], one["severity_index_winsor_p99"])
        ]
        one["risk_type_high_cost_p99"] = [
            risk_group(f, s)
            for f, s in zip(one["frequency_index"], one["high_cost_p99_index"])
        ]

        scenario_cols = [
            "risk_type_region_median",
            "risk_type_excl_region_max",
            "risk_type_excl_top1pct",
            "risk_type_winsor_p99",
            "risk_type_high_cost_p99",
        ]
        one["scenario_agreement_with_primary"] = one.apply(
            lambda row: np.mean([row[column] == row["risk_type_primary"] for column in scenario_cols]), axis=1
        )
        one["year"] = year

        trend_list.append(
            {
                "year": year,
                "students": all_students,
                "accident_rows": all_acc,
                "payment_rows": pay_all["payment_rows"],
                "payment_total": pay_all["payment_total"],
                "accident_rate_per_10k": acc_rate,
                "payment_rate_per_10k": pay_rate,
                "avg_payment_per_claim": pay_all["avg_payment_per_claim"],
                "median_payment_per_claim": pay_all["median_payment_per_claim"],
                "payment_burden_per_10k": pay_load,
                "severe_benefit_rows": pay_all["severe_benefit_rows"],
                "severe_benefit_rate": pay_all["severe_benefit_rate"],
                "high_cost_cutoff_p90": pay_all["high_cost_cutoff_p90"],
                "high_cost_cutoff_p95": pay_all["high_cost_cutoff_p95"],
                "high_cost_cutoff_p99": pay_all["high_cost_cutoff_p99"],
                "avg_payment_excl_region_max": pay_all["avg_payment_excl_region_max"],
                "avg_payment_excl_top1pct": pay_all["avg_payment_excl_top1pct"],
                "avg_payment_winsor_p99": pay_all["avg_payment_winsor_p99"],
            }
        )
        year_list.append(one)
        standard_list.append(standard_detail)

    year_data = pd.concat(year_list, ignore_index=True)
    trend = pd.DataFrame(trend_list)
    standard_detail = pd.concat(standard_list, ignore_index=True)
    return year_data, trend, acc_data, pay_data, standard_detail


def make_period_data(year_data):
    periods = [
        ("2021-2025 통합산출", 2021, 2025),
        ("2022-2025 통합산출", 2022, 2025),
    ]
    pieces = []
    for label, first_year, last_year in periods:
        use = year_data[(year_data["year"] >= first_year) & (year_data["year"] <= last_year)]
        all_students = float(use["students"].sum())
        all_accidents = float(use["accident_rows"].sum())
        all_payments = float(use["payment_rows"].sum())
        all_payment_total = float(use["payment_total"].sum())
        national_accident_rate = all_accidents / all_students * 10_000
        national_payment_rate = all_payments / all_students * 10_000
        national_avg_payment = all_payment_total / all_payments
        national_payment_burden = all_payment_total / all_students * 10_000

        one = (
            use.groupby("region", as_index=False)
            .agg(
                student_person_years=("students", "sum"),
                accident_rows=("accident_rows", "sum"),
                payment_rows=("payment_rows", "sum"),
                payment_total=("payment_total", "sum"),
            )
        )
        one["accident_rate_per_10k"] = divide(one["accident_rows"], one["student_person_years"]) * 10_000
        one["payment_rate_per_10k"] = divide(one["payment_rows"], one["student_person_years"]) * 10_000
        one["avg_payment_per_record"] = divide(one["payment_total"], one["payment_rows"])
        one["payment_burden_per_10k"] = divide(one["payment_total"], one["student_person_years"]) * 10_000
        one["frequency_index"] = one["accident_rate_per_10k"] / national_accident_rate * 100
        one["payment_rate_index"] = one["payment_rate_per_10k"] / national_payment_rate * 100
        one["severity_index"] = one["avg_payment_per_record"] / national_avg_payment * 100
        one["burden_index"] = one["payment_burden_per_10k"] / national_payment_burden * 100
        one["risk_type"] = [
            risk_group(f, s) for f, s in zip(one["frequency_index"], one["severity_index"])
        ]
        one["period"] = label
        one["first_year"] = first_year
        one["last_year"] = last_year
        one["national_accident_rate_per_10k"] = national_accident_rate
        one["national_payment_rate_per_10k"] = national_payment_rate
        one["national_avg_payment_per_record"] = national_avg_payment
        one["national_payment_burden_per_10k"] = national_payment_burden
        pieces.append(one)
    return pd.concat(pieces, ignore_index=True)


def spearman_corr(x, y):
    x_rank = pd.Series(np.asarray(x, dtype=float)).rank(method="average").to_numpy()
    y_rank = pd.Series(np.asarray(y, dtype=float)).rank(method="average").to_numpy()
    if np.std(x_rank) == 0 or np.std(y_rank) == 0:
        return np.nan
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def permutation_p(x, y, count=100_000, random_seed=seed):
    x_rank = pd.Series(np.asarray(x, dtype=float)).rank(method="average").to_numpy()
    y_rank = pd.Series(np.asarray(y, dtype=float)).rank(method="average").to_numpy()
    observed = float(np.corrcoef(x_rank, y_rank)[0, 1])
    rng = np.random.default_rng(random_seed)
    order = np.argsort(rng.random((count, len(y_rank))), axis=1)
    permuted = y_rank[order]
    x_centered = x_rank - x_rank.mean()
    y_centered = permuted - permuted.mean(axis=1, keepdims=True)
    denominator = np.sqrt(
        np.square(x_centered).sum() * np.square(y_centered).sum(axis=1)
    )
    values = y_centered @ x_centered / denominator
    p_value = (np.sum(np.abs(values) >= abs(observed)) + 1) / (count + 1)
    return observed, float(p_value)


def bootstrap_spearman_ci(x, y, count=20_000, random_seed=seed + 1):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rng = np.random.default_rng(random_seed)
    samples = rng.integers(0, len(x), size=(count, len(x)))
    values = []
    for rows in samples:
        value = spearman_corr(x[rows], y[rows])
        if np.isfinite(value):
            values.append(value)
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high), len(values)


def temporal_bootstrap(one, main_type, count=20_000, random_seed=seed + 2):
    freq = one.sort_values("year")["frequency_index"].to_numpy(dtype=float)
    sev = one.sort_values("year")["severity_index"].to_numpy(dtype=float)
    rng = np.random.default_rng(random_seed)
    rows = rng.integers(0, len(freq), size=(count, len(freq)))
    freq_mean = freq[rows].mean(axis=1)
    sev_mean = sev[rows].mean(axis=1)

    labels = np.full(count, "상대적안정형", dtype=object)
    labels[(freq_mean >= 100) & (sev_mean >= 100)] = "복합위험형"
    labels[(freq_mean >= 100) & (sev_mean < 100)] = "반복사고형"
    labels[(freq_mean < 100) & (sev_mean >= 100)] = "숨은중대위험형"
    probs = {name: float(np.mean(labels == name)) for name in type_order}
    freq_low, freq_high = np.quantile(freq_mean, [0.025, 0.975])
    sev_low, sev_high = np.quantile(sev_mean, [0.025, 0.975])
    return {
        "temporal_bootstrap_primary_probability": probs[main_type],
        "temporal_bootstrap_complex_probability": probs["복합위험형"],
        "temporal_bootstrap_repeat_probability": probs["반복사고형"],
        "temporal_bootstrap_hidden_probability": probs["숨은중대위험형"],
        "temporal_bootstrap_stable_probability": probs["상대적안정형"],
        "mean_frequency_index_ci_low": float(freq_low),
        "mean_frequency_index_ci_high": float(freq_high),
        "mean_severity_index_ci_low": float(sev_low),
        "mean_severity_index_ci_high": float(sev_high),
    }


def make_summary(year_data, period_data):
    all_pay = float(year_data["payment_rows"].sum())
    heavy_rate = float(year_data["severe_benefit_rows"].sum() / all_pay)
    high_rate = float(year_data["high_cost_rows_p95"].sum() / all_pay)
    tests = {
        "primary": ("frequency_index", "severity_index"),
        "region_median": ("frequency_index_vs_region_median", "severity_index_vs_region_median"),
        "excl_region_max": ("frequency_index", "severity_index_excl_region_max"),
        "excl_top1pct": ("frequency_index", "severity_index_excl_top1pct"),
        "winsor_p99": ("frequency_index", "severity_index_winsor_p99"),
        "high_cost_p99": ("frequency_index", "high_cost_p99_index"),
    }
    rows = []
    for area_num, (area, one) in enumerate(year_data.groupby("region", sort=True)):
        row = {
            "region": area,
            "student_person_years": one["students"].sum(),
            "accident_rows": one["accident_rows"].sum(),
            "payment_rows": one["payment_rows"].sum(),
            "payment_total_nominal": one["payment_total"].sum(),
            "accident_rate_per_10k_person_years": one["accident_rows"].sum() / one["students"].sum() * 10_000,
            "avg_nominal_payment_per_claim": one["payment_total"].sum() / one["payment_rows"].sum(),
            "payment_rate_per_10k_person_years": one["payment_rows"].sum() / one["students"].sum() * 10_000,
            "payment_burden_per_10k_person_years": one["payment_total"].sum() / one["students"].sum() * 10_000,
            "severe_benefit_rows": one["severe_benefit_rows"].sum(),
            "severe_benefit_rate": one["severe_benefit_rows"].sum() / one["payment_rows"].sum(),
            "severe_benefit_index_5yr": (
                one["severe_benefit_rows"].sum() / one["payment_rows"].sum()
            ) / heavy_rate * 100,
            "high_cost_rows_p95": one["high_cost_rows_p95"].sum(),
            "high_cost_rate_p95": one["high_cost_rows_p95"].sum() / one["payment_rows"].sum(),
            "high_cost_p95_index_5yr": (
                one["high_cost_rows_p95"].sum() / one["payment_rows"].sum()
            ) / high_rate * 100,
            "mean_frequency_index": one["frequency_index"].mean(),
            "mean_payment_rate_index": one["payment_rate_index"].mean(),
            "mean_severity_index": one["severity_index"].mean(),
            "mean_burden_index": one["burden_index"].mean(),
            "years_frequency_above_national": int((one["frequency_index"] >= 100).sum()),
            "years_severity_above_national": int((one["severity_index"] >= 100).sum()),
            "mean_annual_scenario_agreement": one["scenario_agreement_with_primary"].mean(),
        }
        common, common_years = most_common(one["risk_type_primary"])
        row["modal_annual_risk_type"] = common
        row["modal_annual_risk_type_years"] = common_years

        test_types = []
        for name, (freq_col, sev_col) in tests.items():
            result = risk_group(one[freq_col].mean(), one[sev_col].mean())
            row[f"risk_type_{name}"] = result
            test_types.append(result)
        main_type = str(row["risk_type_primary"])
        row["scenario_agreement_5yr"] = np.mean([v == main_type for v in test_types[1:]])
        row["frequency_threshold_distance"] = abs(row["mean_frequency_index"] - 100)
        row["severity_threshold_distance"] = abs(row["mean_severity_index"] - 100)
        row["minimum_threshold_distance"] = min(
            row["frequency_threshold_distance"], row["severity_threshold_distance"]
        )
        row["distance_from_national_100"] = float(
            np.hypot(row["mean_frequency_index"] - 100, row["mean_severity_index"] - 100)
        )
        row.update(temporal_bootstrap(one, main_type, random_seed=seed + 100 + area_num))
        row["borderline_flag"] = bool(
            row["minimum_threshold_distance"] <= 5
            or row["scenario_agreement_5yr"] < 0.8
            or row["temporal_bootstrap_primary_probability"] < 0.8
        )
        rows.append(row)

    summary = pd.DataFrame(rows)
    pooled = period_data[period_data["period"] == "2021-2025 통합산출"][
        ["region", "frequency_index", "payment_rate_index", "severity_index", "burden_index", "risk_type"]
    ].rename(
        columns={
            "frequency_index": "pooled_frequency_index",
            "payment_rate_index": "pooled_payment_rate_index",
            "severity_index": "pooled_severity_index",
            "burden_index": "pooled_burden_index",
            "risk_type": "risk_type_pooled",
        }
    )
    recent = (
        year_data[year_data["year"] >= 2022]
        .groupby("region", as_index=False)
        .agg(
            mean_frequency_index_2022_2025=("frequency_index", "mean"),
            mean_payment_rate_index_2022_2025=("payment_rate_index", "mean"),
            mean_severity_index_2022_2025=("severity_index", "mean"),
            mean_burden_index_2022_2025=("burden_index", "mean"),
        )
    )
    recent["risk_type_2022_2025"] = [
        risk_group(f, s)
        for f, s in zip(
            recent["mean_frequency_index_2022_2025"],
            recent["mean_severity_index_2022_2025"],
        )
    ]
    standard = (
        year_data.groupby("region", as_index=False)
        .agg(
            mean_frequency_index_school_standardized=("frequency_index_school_standardized", "mean"),
            mean_payment_rate_index_school_standardized=("payment_rate_index_school_standardized", "mean"),
            mean_severity_index_school_standardized=("severity_index_school_standardized", "mean"),
            mean_burden_index_school_standardized=("burden_index_school_standardized", "mean"),
        )
    )
    standard["risk_type_school_standardized"] = [
        risk_group(f, s)
        for f, s in zip(
            standard["mean_frequency_index_school_standardized"],
            standard["mean_severity_index_school_standardized"],
        )
    ]
    summary = summary.merge(pooled, on="region", how="left")
    summary = summary.merge(recent, on="region", how="left")
    summary = summary.merge(standard, on="region", how="left")
    summary["same_type_pooled"] = summary["risk_type_primary"] == summary["risk_type_pooled"]
    summary["same_type_2022_2025"] = summary["risk_type_primary"] == summary["risk_type_2022_2025"]
    summary["same_type_school_standardized"] = (
        summary["risk_type_primary"] == summary["risk_type_school_standardized"]
    )
    extended_cols = [
        "risk_type_region_median",
        "risk_type_excl_region_max",
        "risk_type_excl_top1pct",
        "risk_type_winsor_p99",
        "risk_type_high_cost_p99",
        "risk_type_pooled",
        "risk_type_2022_2025",
        "risk_type_school_standardized",
    ]
    summary["extended_agreement_5yr"] = summary.apply(
        lambda row: np.mean([row[col] == row["risk_type_primary"] for col in extended_cols]),
        axis=1,
    )
    summary["frequency_rank"] = summary["mean_frequency_index"].rank(ascending=False, method="min").astype(int)
    summary["severity_rank"] = summary["mean_severity_index"].rank(ascending=False, method="min").astype(int)
    summary["burden_rank"] = summary["mean_burden_index"].rank(ascending=False, method="min").astype(int)
    summary["rank_gap_frequency_minus_severity"] = summary["frequency_rank"] - summary["severity_rank"]
    return summary.sort_values(["risk_type_primary", "distance_from_national_100"], ascending=[True, False])


def pick_sample(summary):
    picked = []
    for name in type_order:
        check = summary[summary["risk_type_primary"] == name].copy()
        if check.empty:
            continue
        check = check.sort_values(
            [
                "extended_agreement_5yr",
                "temporal_bootstrap_primary_probability",
                "modal_annual_risk_type_years",
                "distance_from_national_100",
            ],
            ascending=[False, False, False, False],
        )
        picked.append(check.head(1))
    if not picked:
        return pd.DataFrame()
    return pd.concat(picked, ignore_index=True)


def make_uncertainty(year_data, summary, period_data):
    main = summary.sort_values("region")
    x = main["mean_frequency_index"].to_numpy(dtype=float)
    y = main["mean_severity_index"].to_numpy(dtype=float)
    observed, p_value = permutation_p(x, y)
    ci_low, ci_high, valid_bootstrap = bootstrap_spearman_ci(x, y)

    loo_rows = []
    for area in main["region"]:
        check = main[main["region"] != area]
        loo_rows.append(
            {
                "excluded_region": area,
                "spearman_rho": spearman_corr(
                    check["mean_frequency_index"], check["mean_severity_index"]
                ),
                "n_regions": len(check),
            }
        )
    loo = pd.DataFrame(loo_rows)

    rows = [
        {
            "analysis": "2021-2025 연평균 주분석",
            "spearman_rho": observed,
            "n_regions": len(main),
            "permutation_p_two_sided": p_value,
            "bootstrap_ci_low": ci_low,
            "bootstrap_ci_high": ci_high,
            "leave_one_out_min": loo["spearman_rho"].min(),
            "leave_one_out_max": loo["spearman_rho"].max(),
            "permutation_count": 100_000,
            "bootstrap_count": valid_bootstrap,
            "random_seed": seed,
            "note": "지역 재표본 불확실성 참고치이며 인과검정이 아님",
        }
    ]

    pooled = period_data[period_data["period"] == "2021-2025 통합산출"].sort_values("region")
    recent = main.sort_values("region")
    tests = [
        (
            "2021-2025 통합산출",
            pooled["frequency_index"],
            pooled["severity_index"],
        ),
        (
            "2022-2025 연평균(2021 제외)",
            recent["mean_frequency_index_2022_2025"],
            recent["mean_severity_index_2022_2025"],
        ),
        (
            "2021-2025 학교급 표준화 연평균",
            recent["mean_frequency_index_school_standardized"],
            recent["mean_severity_index_school_standardized"],
        ),
    ]
    for year in years:
        one = year_data[year_data["year"] == year].sort_values("region")
        tests.append(
            (
                f"{year} 연도별",
                one["frequency_index"],
                one["severity_index"],
            )
        )
    for label, freq, sev in tests:
        rows.append(
            {
                "analysis": label,
                "spearman_rho": spearman_corr(freq, sev),
                "n_regions": len(freq),
                "permutation_p_two_sided": np.nan,
                "bootstrap_ci_low": np.nan,
                "bootstrap_ci_high": np.nan,
                "leave_one_out_min": np.nan,
                "leave_one_out_max": np.nan,
                "permutation_count": 0,
                "bootstrap_count": 0,
                "random_seed": seed,
                "note": "비교용 기술적 상관",
            }
        )
    return pd.DataFrame(rows), loo


def make_detail(acc_data, pay_data):
    acc = pd.concat(
        [df.assign(year=year) for year, df in acc_data.items() if year >= 2023], ignore_index=True
    )
    pay = pd.concat(
        [df.assign(year=year) for year, df in pay_data.items() if year >= 2023], ignore_index=True
    )
    pieces = []

    for part, col in parts.items():
        acc_one = acc[["지역", col]].copy()
        pay_one = pay[["지역", col, "payment_total", "is_severe_benefit"]].copy()
        acc_one["category"] = acc_one[col].fillna("(결측)").astype(str)
        pay_one["category"] = pay_one[col].fillna("(결측)").astype(str)

        acc_area = (
            acc_one.groupby(["지역", "category"], as_index=False)
            .size()
            .rename(columns={"지역": "region", "size": "accident_rows"})
        )
        acc_all = (
            acc_one.groupby("category", as_index=False)
            .size()
            .rename(columns={"size": "national_accident_rows"})
        )
        pay_area = (
            pay_one.groupby(["지역", "category"], as_index=False)
            .agg(
                payment_rows=("payment_total", "size"),
                payment_total=("payment_total", "sum"),
                severe_benefit_rows=("is_severe_benefit", "sum"),
            )
            .rename(columns={"지역": "region"})
        )
        pay_all = (
            pay_one.groupby("category", as_index=False)
            .agg(
                national_payment_rows=("payment_total", "size"),
                national_payment_total=("payment_total", "sum"),
            )
        )

        one = acc_area.merge(pay_area, on=["region", "category"], how="outer")
        one = one.merge(acc_all, on="category", how="left").merge(
            pay_all, on="category", how="left"
        )
        for name in ["accident_rows", "payment_rows", "payment_total", "severe_benefit_rows"]:
            one[name] = one[name].fillna(0)

        area_acc = one.groupby("region")["accident_rows"].transform("sum")
        area_pay = one.groupby("region")["payment_total"].transform("sum")
        all_acc = float(acc_all["national_accident_rows"].sum())
        all_pay = float(pay_all["national_payment_total"].sum())

        one["region_accident_share"] = divide(one["accident_rows"], area_acc)
        one["national_accident_share"] = one["national_accident_rows"] / all_acc
        one["accident_share_diff_pp"] = (
            one["region_accident_share"] - one["national_accident_share"]
        ) * 100
        one["region_payment_share"] = divide(one["payment_total"], area_pay)
        one["national_payment_share"] = one["national_payment_total"] / all_pay
        one["payment_share_diff_pp"] = (
            one["region_payment_share"] - one["national_payment_share"]
        ) * 100
        one["dimension"] = part
        pieces.append(one)

    detail = pd.concat(pieces, ignore_index=True)
    enough = detail[(detail["accident_rows"] >= 10) & (detail["payment_rows"] >= 10)].copy()
    signal = (
        enough.sort_values(
            ["region", "dimension", "payment_share_diff_pp", "payment_total"],
            ascending=[True, True, False, False],
        )
        .groupby(["region", "dimension"], as_index=False)
        .head(1)
    )
    return detail, signal


ko_cols = {
    "region": "지역", "year": "연도", "students": "학생수", "teachers": "교원수",
    "classes": "학급수", "site_area": "교지면적", "schools": "학교수",
    "school_level_group": "학교급표준층",
    "accident_rows": "사고접수건수", "payment_rows": "보상지급건수", "payment_total": "총보상액",
    "avg_payment_per_claim": "건당평균보상액", "median_payment_per_claim": "건당중앙보상액",
    "max_payment": "최대보상액", "severe_benefit_rows": "중대급여건수",
    "high_cost_rows_p90": "고비용건수_상위10%", "high_cost_rows_p95": "고비용건수_상위5%",
    "high_cost_rows_p99": "고비용건수_상위1%", "winsor_payment_total_p99": "99분위조정총보상액",
    "top1pct_payment_total": "상위1%보상액", "payment_rows_excl_top1pct": "상위1%제외보상건수",
    "payment_total_excl_top1pct": "상위1%제외총보상액",
    "avg_payment_excl_region_max": "지역최대건제외평균보상액",
    "avg_payment_excl_top1pct": "상위1%제외평균보상액",
    "avg_payment_winsor_p99": "99분위조정평균보상액", "severe_benefit_rate": "중대급여비율",
    "high_cost_rate_p90": "고비용비율_상위10%", "high_cost_rate_p95": "고비용비율_상위5%",
    "high_cost_rate_p99": "고비용비율_상위1%", "top_claim_payment_share": "최대건보상비중",
    "top1pct_payment_share": "상위1%보상비중", "accident_rate_per_10k": "학생1만명당사고접수",
    "payment_rate_per_10k": "학생1만명당보상지급", "payment_burden_per_10k": "학생1만명당보상부담",
    "students_per_teacher": "교원1인당학생수", "students_per_class": "학급당학생수",
    "site_area_per_student": "학생1인당교지면적",
    "national_accident_rate_per_10k": "전국학생1만명당사고접수",
    "national_payment_rate_per_10k": "전국학생1만명당보상지급",
    "national_avg_payment_per_claim": "전국건당평균보상액",
    "national_payment_burden_per_10k": "전국학생1만명당보상부담",
    "frequency_index": "사고빈도지수", "payment_rate_index": "보상지급빈도지수",
    "severity_index": "건당심각도지수", "burden_index": "보상부담지수",
    "severity_index_excl_region_max": "지역최대건제외심각도지수",
    "severity_index_excl_top1pct": "상위1%제외심각도지수",
    "severity_index_winsor_p99": "99분위조정심각도지수", "high_cost_p99_index": "고비용1%지수",
    "severe_benefit_index": "중대급여지수", "frequency_index_vs_region_median": "지역중앙값기준빈도지수",
    "severity_index_vs_region_median": "지역중앙값기준심각도지수", "risk_type_primary": "주분류",
    "risk_type_region_median": "지역중앙값분류", "risk_type_excl_region_max": "최대건제외분류",
    "risk_type_excl_top1pct": "상위1%제외분류", "risk_type_winsor_p99": "99분위조정분류",
    "risk_type_high_cost_p99": "고비용1%분류", "scenario_agreement_with_primary": "주분류일치도",
    "high_cost_cutoff_p90": "고비용기준_상위10%", "high_cost_cutoff_p95": "고비용기준_상위5%",
    "high_cost_cutoff_p99": "고비용기준_상위1%", "student_person_years": "학생인년",
    "payment_total_nominal": "5년총보상액", "accident_rate_per_10k_person_years": "5년학생1만명당사고접수",
    "avg_nominal_payment_per_claim": "5년건당평균보상액",
    "payment_rate_per_10k_person_years": "5년학생1만명당보상지급",
    "payment_burden_per_10k_person_years": "5년학생1만명당보상부담",
    "severe_benefit_index_5yr": "5년중대급여지수", "high_cost_p95_index_5yr": "5년고비용5%지수",
    "mean_frequency_index": "5년평균사고빈도지수", "mean_payment_rate_index": "5년평균보상지급빈도지수",
    "mean_severity_index": "5년평균건당심각도지수", "mean_burden_index": "5년평균보상부담지수",
    "years_frequency_above_national": "빈도전국상회연도수",
    "years_severity_above_national": "심각도전국상회연도수",
    "mean_annual_scenario_agreement": "연평균기준일치도", "modal_annual_risk_type": "연도별최빈유형",
    "modal_annual_risk_type_years": "최빈유형연도수", "scenario_agreement_5yr": "5년기준일치도",
    "distance_from_national_100": "전국100과거리", "frequency_rank": "사고빈도순위",
    "frequency_threshold_distance": "빈도기준선거리",
    "severity_threshold_distance": "피해규모기준선거리",
    "minimum_threshold_distance": "최소기준선거리",
    "borderline_flag": "경계지역여부",
    "severity_rank": "건당심각도순위", "burden_rank": "보상부담순위",
    "rank_gap_frequency_minus_severity": "빈도심각도순위차", "category": "항목",
    "national_accident_rows": "전국사고접수건수", "national_payment_rows": "전국보상지급건수",
    "national_payment_total": "전국총보상액", "region_accident_share": "지역내사고비중",
    "national_accident_share": "전국사고비중", "accident_share_diff_pp": "사고비중차이_퍼센트포인트",
    "region_payment_share": "지역내보상비중", "national_payment_share": "전국보상비중",
    "payment_share_diff_pp": "보상비중차이_퍼센트포인트", "dimension": "구분",
    "std_accident_rate_per_10k": "학교급표준화_학생1만명당사고접수",
    "std_payment_rate_per_10k": "학교급표준화_학생1만명당보상지급",
    "std_payment_burden_per_10k": "학교급표준화_학생1만명당보상부담",
    "std_avg_payment_per_record": "학교급표준화_지급자료1건당평균액",
    "national_std_accident_rate_per_10k": "전국학교급표준화_학생1만명당사고접수",
    "national_std_payment_rate_per_10k": "전국학교급표준화_학생1만명당보상지급",
    "national_std_payment_burden_per_10k": "전국학교급표준화_학생1만명당보상부담",
    "national_std_avg_payment_per_record": "전국학교급표준화_지급자료1건당평균액",
    "frequency_index_school_standardized": "학교급표준화_사고빈도지수",
    "payment_rate_index_school_standardized": "학교급표준화_보상지급빈도지수",
    "severity_index_school_standardized": "학교급표준화_보상기반피해규모지수",
    "burden_index_school_standardized": "학교급표준화_보상부담지수",
    "risk_type_school_standardized": "학교급표준화분류",
    "standard_student_coverage": "학교급표준화_학생포함비율",
    "standard_accident_coverage": "학교급표준화_사고포함비율",
    "standard_payment_coverage": "학교급표준화_보상포함비율",
    "standard_weight": "전국학교급표준가중치",
    "weighted_accident_rate_per_10k": "가중사고접수율",
    "weighted_payment_rate_per_10k": "가중보상지급률",
    "weighted_payment_burden_per_10k": "가중보상부담",
    "period": "분석기간",
    "first_year": "시작연도",
    "last_year": "종료연도",
    "avg_payment_per_record": "지급자료1건당평균액",
    "national_avg_payment_per_record": "전국지급자료1건당평균액",
    "risk_type": "위험유형",
    "pooled_frequency_index": "5년통합사고빈도지수",
    "pooled_payment_rate_index": "5년통합보상지급빈도지수",
    "pooled_severity_index": "5년통합보상기반피해규모지수",
    "pooled_burden_index": "5년통합보상부담지수",
    "risk_type_pooled": "5년통합분류",
    "mean_frequency_index_2022_2025": "2021제외평균사고빈도지수",
    "mean_payment_rate_index_2022_2025": "2021제외평균보상지급빈도지수",
    "mean_severity_index_2022_2025": "2021제외평균보상기반피해규모지수",
    "mean_burden_index_2022_2025": "2021제외평균보상부담지수",
    "risk_type_2022_2025": "2021제외분류",
    "mean_frequency_index_school_standardized": "학교급표준화_5년평균사고빈도지수",
    "mean_payment_rate_index_school_standardized": "학교급표준화_5년평균보상지급빈도지수",
    "mean_severity_index_school_standardized": "학교급표준화_5년평균보상기반피해규모지수",
    "mean_burden_index_school_standardized": "학교급표준화_5년평균보상부담지수",
    "same_type_pooled": "5년통합주분류일치",
    "same_type_2022_2025": "2021제외주분류일치",
    "same_type_school_standardized": "학교급표준화주분류일치",
    "extended_agreement_5yr": "확장민감도일치도",
    "temporal_bootstrap_primary_probability": "연도재표본_주분류비율",
    "temporal_bootstrap_complex_probability": "연도재표본_복합위험형비율",
    "temporal_bootstrap_repeat_probability": "연도재표본_반복사고형비율",
    "temporal_bootstrap_hidden_probability": "연도재표본_숨은중대위험형비율",
    "temporal_bootstrap_stable_probability": "연도재표본_상대적안정형비율",
    "mean_frequency_index_ci_low": "연도재표본_빈도지수95참고하한",
    "mean_frequency_index_ci_high": "연도재표본_빈도지수95참고상한",
    "mean_severity_index_ci_low": "연도재표본_피해규모지수95참고하한",
    "mean_severity_index_ci_high": "연도재표본_피해규모지수95참고상한",
    "analysis": "분석구분",
    "spearman_rho": "스피어만상관",
    "n_regions": "지역수",
    "permutation_p_two_sided": "순열검정양측p값",
    "bootstrap_ci_low": "지역부트스트랩95하한",
    "bootstrap_ci_high": "지역부트스트랩95상한",
    "leave_one_out_min": "지역하나씩제외_최솟값",
    "leave_one_out_max": "지역하나씩제외_최댓값",
    "permutation_count": "순열반복수",
    "bootstrap_count": "부트스트랩유효반복수",
    "random_seed": "난수시드",
    "note": "해석주의",
    "excluded_region": "제외지역",
}


def save_csv(
    year_data,
    trend,
    summary,
    sample,
    detail,
    signal,
    standard_detail,
    period_data,
    uncertainty,
    loo,
):
    files = {
        "year.csv": year_data,
        "trend.csv": trend,
        "summary.csv": summary,
        "sample.csv": sample,
        "detail.csv": detail,
        "signal.csv": signal,
        "standard.csv": standard_detail,
        "period.csv": period_data,
        "uncertainty.csv": uncertainty,
        "loo.csv": loo,
    }
    for name, df in files.items():
        missing = [col for col in df.columns if col not in ko_cols]
        if missing:
            raise KeyError(f"한국어 이름이 없는 열: {missing}")
        out = df.rename(columns=ko_cols)
        out.to_csv(data_dir / name, index=False, encoding="utf-8-sig")
        print(name, out.shape)


def main():
    year_data, trend, acc_data, pay_data, standard_detail = make_year_data()
    period_data = make_period_data(year_data)
    summary = make_summary(year_data, period_data)
    sample = pick_sample(summary)
    uncertainty, loo = make_uncertainty(year_data, summary, period_data)
    detail, signal = make_detail(acc_data, pay_data)

    area_count = 17
    assert len(year_data) == len(years) * area_count
    assert year_data[["year", "region"]].duplicated().sum() == 0
    assert year_data.isna().sum().sum() == 0
    assert len(trend) == len(years)
    assert len(summary) == area_count
    assert set(summary["region"]) == expected_regions
    assert len(standard_detail) == len(years) * area_count * len(standard_school_levels)
    assert len(period_data) == area_count * 2
    assert len(uncertainty) == 9
    assert len(loo) == area_count
    assert np.allclose(
        year_data["burden_index"],
        year_data["payment_rate_index"] * year_data["severity_index"] / 100,
    )
    assert np.allclose(
        year_data["burden_index_school_standardized"],
        year_data["payment_rate_index_school_standardized"]
        * year_data["severity_index_school_standardized"]
        / 100,
    )
    assert year_data["risk_type_primary"].tolist() == [
        risk_group(f, s)
        for f, s in zip(year_data["frequency_index"], year_data["severity_index"])
    ]
    assert year_data["risk_type_school_standardized"].tolist() == [
        risk_group(f, s)
        for f, s in zip(
            year_data["frequency_index_school_standardized"],
            year_data["severity_index_school_standardized"],
        )
    ]
    assert period_data["risk_type"].tolist() == [
        risk_group(f, s)
        for f, s in zip(period_data["frequency_index"], period_data["severity_index"])
    ]
    assert summary["risk_type_primary"].tolist() == [
        risk_group(f, s)
        for f, s in zip(summary["mean_frequency_index"], summary["mean_severity_index"])
    ]

    save_csv(
        year_data,
        trend,
        summary,
        sample,
        detail,
        signal,
        standard_detail,
        period_data,
        uncertainty,
        loo,
    )

    print("\n대표 지역")
    cols = [
        "region", "risk_type_primary", "mean_frequency_index",
        "mean_severity_index", "mean_burden_index", "extended_agreement_5yr",
    ]
    print(sample[cols].rename(columns=ko_cols).to_string(index=False))


if __name__ == "__main__":
    main()
