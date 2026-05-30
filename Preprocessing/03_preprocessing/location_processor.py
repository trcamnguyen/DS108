import pandas as pd


def _clean_itviec(loc) -> str | None:
    """Lấy thành phố cuối chuỗi địa chỉ: '...Tan Binh, Ho Chi Minh' → 'Ho Chi Minh'."""
    if pd.isna(loc):
        return None
    return str(loc).split(",")[-1].split("-")[0].strip()


def _clean_topcv(loc) -> str:
    """Lấy phần sau dấu '-' cuối cùng, bỏ prefix 'Việc làm tại'."""
    if pd.isna(loc):
        return "Khác"
    cleaned = str(loc).split("-")[-1].replace("Việc làm tại", "").strip().rstrip(".,")
    return cleaned or "Khác"


def process_location(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    itviec = df["source"] == "itviec"
    topcv = df["source"] == "topcv"
    df.loc[itviec, "location"] = df.loc[itviec, "location"].apply(_clean_itviec)
    df.loc[topcv, "location"] = df.loc[topcv, "location"].apply(_clean_topcv)
    return df
