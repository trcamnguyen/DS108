import re
import pandas as pd


def process_level(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. TopCV: override job_level từ job_title nếu chứa intern signal
    topcv_mask = df["source"] == "topcv"
    intern_title = df["job_title"].astype(str).str.contains(
        r"(?:thực tập sinh|intern)", flags=re.IGNORECASE, regex=True, na=False
    )
    df.loc[topcv_mask & intern_title, "job_level"] = "Intern"

    # 2. Normalize "Thực tập sinh" → "Intern" toàn bộ (ITViec output từ _extract_level)
    df["job_level"] = df["job_level"].replace("Thực tập sinh", "Intern")

    return df
