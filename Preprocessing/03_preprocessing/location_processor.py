import unicodedata

import pandas as pd


def _normalize_key(text: str) -> str:
    """Lowercase + strip Vietnamese diacritics for fuzzy matching."""
    text = text.lower().strip()
    text = text.replace("đ", "d").replace("đ", "d")
    nfd = unicodedata.normalize("NFD", text)
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn")


# Canonical city name → list of alias keys (normalized, no-accent)
# The canonical form is the accented Vietnamese name.
_CANONICAL_ALIASES: dict[str, list[str]] = {
    "Hồ Chí Minh": ["ho chi minh", "hcm", "tp hcm", "tp.hcm", "saigon", "sai gon", "tphcm"],
    "Hà Nội": ["ha noi", "hanoi", "hn"],
    "Đà Nẵng": ["da nang", "danang"],
    "Cần Thơ": ["can tho", "cantho"],
    "Hải Phòng": ["hai phong", "haiphong"],
    "Bắc Ninh": ["bac ninh"],
    "Bình Dương": ["binh duong"],
    "Đồng Nai": ["dong nai"],
    "Long An": ["long an"],
    "Hưng Yên": ["hung yen"],
    "Hải Dương": ["hai duong"],
    "Khánh Hòa": ["khanh hoa", "nha trang"],
    "Đà Lạt": ["da lat", "dalat"],
    "Huế": ["hue"],
    "Vĩnh Phúc": ["vinh phuc"],
    "Quảng Ninh": ["quang ninh"],
    "Thái Nguyên": ["thai nguyen"],
    "Nghệ An": ["nghe an", "vinh"],
    "Thanh Hóa": ["thanh hoa"],
    "Bình Phước": ["binh phuoc"],
    "Bà Rịa - Vũng Tàu": ["ba ria vung tau", "ba ria - vung tau", "vung tau", "ba ria"],
    "Bình Thuận": ["binh thuan", "phan thiet"],
    "Lâm Đồng": ["lam dong"],
    "An Giang": ["an giang"],
    "Tiền Giang": ["tien giang"],
    "Foreign:Japan": ["japan", "nhat ban"],
    "Foreign:Singapore": ["singapore"],
    "Foreign:USA": ["usa", "united states", "my"],
    "Foreign:Korea": ["korea", "han quoc"],
    "Foreign:Australia": ["australia"],
}

# Build reverse lookup: normalized alias → canonical
_LOOKUP: dict[str, str] = {}
for _canonical, _aliases in _CANONICAL_ALIASES.items():
    # Add the canonical itself (normalized)
    _LOOKUP[_normalize_key(_canonical)] = _canonical
    for _alias in _aliases:
        _LOOKUP[_alias] = _canonical


def _to_canonical(raw: str | None) -> str:
    """Map a raw location string to its canonical city name, handling both accented and non-accented forms."""
    if not raw:
        return "Khác"
    key = _normalize_key(raw)
    if key in _LOOKUP:
        return _LOOKUP[key]
    # Partial match: check if any lookup key is contained in the normalized input
    for alias, canonical in _LOOKUP.items():
        if alias in key:
            return canonical
    return "Khác"


def _clean_itviec(loc) -> str | None:
    """Lấy thành phố cuối chuỗi địa chỉ: '...Tan Binh, Ho Chi Minh' → canonical city."""
    if pd.isna(loc):
        return None
    raw = str(loc).split(",")[-1].split("-")[0].strip()
    return _to_canonical(raw)


def _clean_topcv(loc) -> str:
    """Lấy phần sau dấu '-' cuối cùng, bỏ prefix 'Việc làm tại', map to canonical."""
    if pd.isna(loc):
        return "Khác"
    raw = str(loc).split("-")[-1].replace("Việc làm tại", "").strip().rstrip(".,")
    return _to_canonical(raw) if raw else "Khác"


def process_location(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    itviec = df["source"] == "itviec"
    topcv = df["source"] == "topcv"
    df.loc[itviec, "location"] = df.loc[itviec, "location"].apply(_clean_itviec)
    df.loc[topcv, "location"] = df.loc[topcv, "location"].apply(_clean_topcv)
    return df
