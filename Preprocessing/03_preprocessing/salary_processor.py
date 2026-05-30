import re
import pandas as pd

# ITViec có nhiều dạng nhiễu tiếng Anh; TopCV chủ yếu "Thoả thuận"
_NOISE_PATTERN = (
    r"love it|sign in|negotiation|attractive|competitive|negotiable|"
    r"let's discuss|best in the market|open to negotiation|"
    r"thương lượng|thỏa thuận| thuận"
)

_USD_RATE = 25_000  # VND per USD, cố định


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df["salary"] = df["salary"].fillna("Thỏa thuận")
    mask = df["salary"].str.lower().str.contains(_NOISE_PATTERN, na=False)
    df.loc[mask, "salary"] = "Thỏa thuận"
    return df


def _parse_one(s: str) -> tuple[float, float, float]:
    if pd.isna(s) or s == "Thỏa thuận":
        return 0.0, 0.0, 0.0

    s = str(s).lower().strip().replace("–", "-").replace(",", "")
    is_usd = any(k in s for k in ["usd", "$"])

    # Xóa dấu chấm phân nghìn (dot theo sau đúng 3 chữ số: 15.000, 15.000.000)
    # Giữ dấu thập phân (dot theo sau 1–2 chữ số: 66.7, 15.5)
    s = re.sub(r"(?<=\d)\.(?=\d{3}(?!\d))", "", s)

    numbers = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", s)]
    if not numbers:
        return 0.0, 0.0, 0.0

    if "up to" in s or "upto" in s or "tới" in s:
        min_v, max_v = 0.0, numbers[0]
    elif "từ" in s or "from" in s:
        min_v, max_v = numbers[0], 0.0
    elif "-" in s and len(numbers) >= 2:
        min_v, max_v = numbers[0], numbers[1]
    else:
        min_v = max_v = numbers[0]

    def to_million(v: float) -> float:
        if v <= 0:
            return 0.0
        if is_usd:
            return v * _USD_RATE / 1_000_000
        if v >= 1_000_000:
            return v / 1_000_000
        return v  # đã là triệu (15m, 20tr…)

    lo = to_million(min_v)
    hi = to_million(max_v)
    avg = (lo + hi) / 2 if lo > 0 and hi > 0 else max(lo, hi)
    return round(lo, 2), round(hi, 2), round(avg, 2)


def _compute_flags(df: pd.DataFrame) -> pd.DataFrame:
    col = df["salary"].astype(str).str.lower().str.strip()
    df["is_null_salary"] = df["salary"].isna()
    df["is_thoa_thuan"] = col.str.contains("thỏa thuận", regex=True, na=False)
    df["is_missing_salary"] = (
        df["is_null_salary"]
        | df["is_thoa_thuan"]
        | col.isin(["", "nan", "none"])
    )
    return df


def process_salary(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = _normalize(df)
    parsed = df["salary"].apply(_parse_one)
    df[["min_salary", "max_salary", "avg_salary"]] = pd.DataFrame(
        parsed.tolist(), index=df.index
    )
    df = _compute_flags(df)
    return df
