import re
import pandas as pd

_USD_RATE = 25_000  # VND per USD, cố định

# Pattern nhận diện "thỏa thuận" và toàn bộ biến thể EN/VN.
# Sửa lỗi cũ:
#   - " thuận" (có space thừa) → \bthuận\b  (word boundary, không bắt "thuận" trong từ khác)
#   - "let's discuss" → let'?s\s+discuss  (bắt cả "lets discuss")
_THOA_THUAN_PATTERN = (
    r"love it|sign in|negotiation|attractive|competitive|negotiable|"
    r"let'?s\s+discuss|best in the market|open to negotiation|"
    r"thương lượng|thỏa thuận|\bthuận\b"
)


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bước 1: lưu salary_raw trước khi sửa (phục vụ audit và is_thoa_thuan).
    Bước 2: fill null → 'Thỏa thuận'.
    Bước 3: replace toàn bộ biến thể thỏa thuận → 'Thỏa thuận'.
    """
    df["salary_raw"] = df["salary"].copy()
    df["salary"] = df["salary"].fillna("Thỏa thuận")
    mask = df["salary"].str.contains(
        _THOA_THUAN_PATTERN, flags=re.IGNORECASE, na=False, regex=True
    )
    df.loc[mask, "salary"] = "Thỏa thuận"
    return df


def _parse_one(s: str) -> tuple[float, float, float]:
    """
    Parse một chuỗi salary thô → (min, max, avg) tính bằng triệu VND.
    Trả về (0.0, 0.0, 0.0) khi không parse được (thỏa thuận hoặc format lạ).
    """
    if pd.isna(s) or s == "Thỏa thuận":
        return 0.0, 0.0, 0.0

    s = str(s).lower().strip().replace("–", "-").replace(",", "")
    is_usd = any(k in s for k in ["usd", "$"])

    # Xóa dấu chấm phân nghìn: "15.000" → "15000"
    # Giữ dấu thập phân: "15.5" không bị chạm
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
            return round(v * _USD_RATE / 1_000_000, 2)
        if v >= 1_000_000:
            return round(v / 1_000_000, 2)
        return v  # đã là triệu (15, 20, 66.7…)

    lo = to_million(min_v)
    hi = to_million(max_v)
    avg = (lo + hi) / 2 if lo > 0 and hi > 0 else max(lo, hi)
    return round(lo, 2), round(hi, 2), round(avg, 2)


def _compute_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tính 3 flags dựa trên salary_raw (giá trị trước normalize) và avg_salary.

    is_thoa_thuan  : salary gốc là null HOẶC khớp pattern thỏa thuận/biến thể.
                     Không phân biệt nguồn gốc (null, "competitive", "thương lượng"…)
                     — dùng salary_raw để audit được tại sao flag = True.

    is_null_salary : parse thất bại — salary có text cụ thể (không phải thỏa thuận)
                     nhưng _parse_one không extract được số (avg_salary == 0).
                     Ví dụ: "lương hấp dẫn", "15 triệu" (thiếu đơn vị số).

    is_missing_salary : is_thoa_thuan | is_null_salary — dùng để filter
                        toàn bộ record không có thông tin lương số học.
    """
    # Kiểm tra trên salary_raw để giữ nguyên nguồn gốc
    raw_col = df["salary_raw"].astype(str).str.lower().str.strip()

    is_truly_null = df["salary_raw"].isna()
    is_negotiable_text = raw_col.str.contains(
        _THOA_THUAN_PATTERN, flags=re.IGNORECASE, na=False, regex=True
    )

    df["is_thoa_thuan"] = is_truly_null | is_negotiable_text

    # Parse fail: avg == 0 nhưng KHÔNG phải thỏa thuận
    # → salary_raw có text cụ thể nhưng regex không ra số
    df["is_null_salary"] = (df["avg_salary"] == 0) & ~df["is_thoa_thuan"]

    df["is_missing_salary"] = df["is_thoa_thuan"] | df["is_null_salary"]
    return df


def process_salary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline hoàn chỉnh:
      1. _normalize   → salary_raw, salary (canonical)
      2. _parse_one   → min_salary, max_salary, avg_salary (triệu VND)
      3. _compute_flags → is_thoa_thuan, is_null_salary, is_missing_salary
    """
    df = df.copy()
    df = _normalize(df)
    parsed = df["salary"].apply(_parse_one)
    df[["min_salary", "max_salary", "avg_salary"]] = pd.DataFrame(
        parsed.tolist(), index=df.index
    )
    df = _compute_flags(df)
    return df