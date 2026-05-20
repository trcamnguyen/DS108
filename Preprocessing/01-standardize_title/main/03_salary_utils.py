import pandas as pd

THOA_THUAN_VALUES: set[str] = {
    "thỏa thuận",
    "thoả thuận",
    "thoa thuan",
    "negotiable",
}

INVALID_SALARY_VALUES: set[str] = {
    "thỏa thuận",
    "thoả thuận",
    "thoa thuan",
    "negotiable",
    "",
}


def add_salary_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Thêm các cột flag salary:
      - is_null_salary   : salary là NaN
      - is_thoa_thuan    : salary thuộc THOA_THUAN_VALUES
      - is_missing_salary: is_null_salary OR is_thoa_thuan

    Trả về DataFrame mới (không mutate input).
    """
    out = df.copy()
    salary_raw = out["salary"]
    salary_str = salary_raw.astype("string").str.strip().str.lower()

    out["is_null_salary"] = salary_raw.isna()
    out["is_thoa_thuan"] = salary_str.isin(THOA_THUAN_VALUES)
    out["is_missing_salary"] = out["is_null_salary"] | out["is_thoa_thuan"]
    return out


def get_valid_salary_mask(df: pd.DataFrame) -> pd.Series:
    """
    Trả về boolean mask: True nếu salary hợp lệ (có số, không phải thỏa thuận).
    """
    salary_raw = df["salary"]
    salary_str = salary_raw.astype("string").str.strip().str.lower()
    return ~(salary_raw.isna() | salary_str.isin(INVALID_SALARY_VALUES))
