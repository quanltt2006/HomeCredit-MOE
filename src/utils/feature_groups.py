# ============================================================
# NHÓM FEATURE CON CỦA MODALITY "application"
# ============================================================

APPLICATION_FEATURE_GROUPS = {
    "demographics_personal": [
        "CODE_GENDER", "CNT_CHILDREN", "CNT_FAM_MEMBERS", "DAYS_BIRTH",
    ],
    "financial_loan": [
        "NAME_CONTRACT_TYPE", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY",
        "AMT_GOODS_PRICE", "CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO",
        "CREDIT_TERM", "INCOME_PER_PERSON", "CREDIT_GOODS_RATIO",
    ],
    "employment": [
        "DAYS_EMPLOYED", "DAYS_EMPLOYED_BIRTH_RATIO",
    ],
    "assets_housing_type": [
        "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "OWN_CAR_AGE",
    ],
    "housing_building_details": [
        "APARTMENTS_AVG", "BASEMENTAREA_AVG", "YEARS_BEGINEXPLUATATION_AVG", "YEARS_BUILD_AVG",
        "COMMONAREA_AVG", "ELEVATORS_AVG", "ENTRANCES_AVG", "FLOORSMAX_AVG", "FLOORSMIN_AVG",
        "LANDAREA_AVG", "LIVINGAPARTMENTS_AVG", "LIVINGAREA_AVG", "NONLIVINGAPARTMENTS_AVG",
        "NONLIVINGAREA_AVG",
        "APARTMENTS_MODE", "BASEMENTAREA_MODE", "YEARS_BEGINEXPLUATATION_MODE", "YEARS_BUILD_MODE",
        "COMMONAREA_MODE", "ELEVATORS_MODE", "ENTRANCES_MODE", "FLOORSMAX_MODE", "FLOORSMIN_MODE",
        "LANDAREA_MODE", "LIVINGAPARTMENTS_MODE", "LIVINGAREA_MODE", "NONLIVINGAPARTMENTS_MODE",
        "NONLIVINGAREA_MODE",
        "APARTMENTS_MEDI", "BASEMENTAREA_MEDI", "YEARS_BEGINEXPLUATATION_MEDI", "YEARS_BUILD_MEDI",
        "COMMONAREA_MEDI", "ELEVATORS_MEDI", "ENTRANCES_MEDI", "FLOORSMAX_MEDI", "FLOORSMIN_MEDI",
        "LANDAREA_MEDI", "LIVINGAPARTMENTS_MEDI", "LIVINGAREA_MEDI", "NONLIVINGAPARTMENTS_MEDI",
        "NONLIVINGAREA_MEDI",
        "TOTALAREA_MODE",
    ],
    "external_scores": [
        "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
    ],
    "region_geo": [
        "REGION_POPULATION_RELATIVE", "REGION_RATING_CLIENT", "REGION_RATING_CLIENT_W_CITY",
        "REG_REGION_NOT_LIVE_REGION", "REG_REGION_NOT_WORK_REGION", "LIVE_REGION_NOT_WORK_REGION",
        "REG_CITY_NOT_LIVE_CITY", "REG_CITY_NOT_WORK_CITY", "LIVE_CITY_NOT_WORK_CITY",
    ],
    "contact_process": [
        "FLAG_MOBIL", "FLAG_EMP_PHONE", "FLAG_WORK_PHONE", "FLAG_CONT_MOBILE", "FLAG_PHONE",
        "FLAG_EMAIL", "DAYS_LAST_PHONE_CHANGE", "HOUR_APPR_PROCESS_START", "DAYS_REGISTRATION",
        "DAYS_ID_PUBLISH",
    ],
    "social_circle": [
        "OBS_30_CNT_SOCIAL_CIRCLE", "DEF_30_CNT_SOCIAL_CIRCLE",
        "OBS_60_CNT_SOCIAL_CIRCLE", "DEF_60_CNT_SOCIAL_CIRCLE",
    ],
    "document_flags": [f"FLAG_DOCUMENT_{i}" for i in range(2, 22)],
    "bureau_inquiry": [
        "AMT_REQ_CREDIT_BUREAU_HOUR", "AMT_REQ_CREDIT_BUREAU_DAY", "AMT_REQ_CREDIT_BUREAU_WEEK",
        "AMT_REQ_CREDIT_BUREAU_MON", "AMT_REQ_CREDIT_BUREAU_QRT", "AMT_REQ_CREDIT_BUREAU_YEAR",
    ],
}

APPLICATION_ONEHOT_PREFIX_TO_GROUP = {
    "NAME_TYPE_SUITE_": "demographics_personal",
    "NAME_FAMILY_STATUS_": "demographics_personal",
    "NAME_EDUCATION_TYPE_": "demographics_personal",
    "NAME_INCOME_TYPE_": "financial_loan",
    "OCCUPATION_TYPE_": "employment",
    "ORGANIZATION_TYPE_": "employment",
    "NAME_HOUSING_TYPE_": "assets_housing_type",
    "HOUSETYPE_MODE_": "assets_housing_type",
    "WALLSMATERIAL_MODE_": "assets_housing_type",
    "FONDKAPREMONT_MODE_": "assets_housing_type",
    "EMERGENCYSTATE_MODE_": "assets_housing_type",
    "WEEKDAY_APPR_PROCESS_START_": "contact_process",
}

def group_application_features(feature_names):
    """
    Map tên cột thực tế của X_application vào các nhóm ngữ nghĩa.
    Trả về dict: group_name -> list[int] (index cột).
    """
    groups = {name: [] for name in APPLICATION_FEATURE_GROUPS}
    groups["other"] = []

    exact_lookup = {}
    for group_name, cols in APPLICATION_FEATURE_GROUPS.items():
        for c in cols:
            exact_lookup[c] = group_name

    for idx, name in enumerate(feature_names):
        if name in exact_lookup:
            groups[exact_lookup[name]].append(idx)
            continue

        matched = False
        for prefix, group_name in APPLICATION_ONEHOT_PREFIX_TO_GROUP.items():
            if name.startswith(prefix):
                groups[group_name].append(idx)
                matched = True
                break

        if not matched:
            groups["other"].append(idx)

    return {k: v for k, v in groups.items() if len(v) > 0}