"""
Trích xuất job_level / experience / employment_type / education từ text.

ITViec: 4 cột này là null trong raw data → extract từ job_title + job_description + requirement.
TopCV:  4 cột đã có sẵn → pass-through, không ghi đè.
"""
import re
import pandas as pd


# ---------------------------------------------------------------------------
# employment_type
# ---------------------------------------------------------------------------

def _extract_employment_type(text: str) -> str:
    if not isinstance(text, str):
        return "Toàn thời gian"
    t = text.lower()

    if any(k in t for k in ["part-time", "bán thời gian", "part time", "parttime"]):
        return "Bán thời gian"

    if any(k in t for k in ["contract", "freelance", "thời vụ", "hợp đồng",
                             "hợp đồng ngắn hạn", "project based", "project-based", "consultant"]):
        return "Thời vụ"

    if re.search(r"\bhybrid\b", t):
        work_hints = [
            "hybrid working", "hybrid work", "làm việc hybrid", "hybrid remote",
            "hybrid model", "hybrid arrangement", "hybrid schedule", "hybrid onsite", "days hybrid",
        ]
        if any(p in t for p in work_hints) or re.search(r"\bhybrid\s*[\(\-\~\:\[]", t):
            return "Hybrid"
        tech_hints = [
            "hybrid system", "hybrid architecture", "hybrid app", "hybrid mobile",
            "hybrid cloud", "hybrid integration", "hybrid protocol", "hybrid infrastructure",
            "hybrid engine", "hybrid solution", "hybrid platform", "hybrid framework",
        ]
        if not any(p in t for p in tech_hints):
            return "Hybrid"

    if any(k in t for k in ["remote", "làm từ xa", "wfh", "work from home",
                             "fully remote", "remote only", "remote position", "remote work"]):
        return "Remote"

    return "Toàn thời gian"


# ---------------------------------------------------------------------------
# job_level
# ---------------------------------------------------------------------------

def _extract_level(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return "Nhân viên"
    t = text.lower()

    if any(k in t for k in ["manager", "trưởng phòng", "head of", "director",
                             "giám đốc", "vp", "cto", "ceo", "chief"]):
        return "Quản lý / Giám sát"
    if any(k in t for k in ["tech lead", "technical lead", "trưởng nhóm",
                             "team lead", "group lead"]) or re.search(r"\blead\b", t):
        return "Trưởng nhóm"
    if any(k in t for k in ["senior", "expert", "chuyên gia", "principal", "staff", "architect"]):
        return "Senior"
    if any(k in t for k in ["junior", "mid-level", "middle", " mid "]):
        return "Junior"
    if any(k in t for k in ["fresher", "mới tốt nghiệp", "entry level", "entry-level"]):
        return "Fresher"
    if any(k in t for k in ["intern", "thực tập", "thực tập sinh"]):
        return "Thực tập sinh"
    return "Nhân viên"


# ---------------------------------------------------------------------------
# experience
# ---------------------------------------------------------------------------

def _extract_experience(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return "Không yêu cầu"
    t = text.lower()

    if any(k in t for k in ["fresh", "mới tốt nghiệp", "no experience",
                             "không yêu cầu", "không cần kinh nghiệm", "0 năm"]):
        return "Dưới 1 năm"

    is_age = any(p in t for p in ["years old", "year old", "tuổi",
                                   "under 30", "over 30", "below 30"])

    # "2+ years", "2 + year"
    m = re.search(r"(\d+)\s*\+\s*(?:năm|year|years|yr)", t)
    if m:
        y = int(m.group(1))
        return "Dưới 1 năm" if y == 0 else f"Trên {y} năm"

    # "1-3 years", "2–5 năm"
    m = re.search(r"(\d+)\s*[-–~]\s*(\d+)\s*(?:năm|year|years)", t)
    if m and not is_age:
        y = int(m.group(1))
        if y <= 20:
            return "Dưới 1 năm" if y == 0 else f"{y} năm"

    patterns = [
        r"(?:ít nhất|tối thiểu|minimum|from|at least|trên|over)\s*(\d+)\s*(?:năm|year|years)",
        r"(\d+)\s*(?:năm|year|years)\s*(?:kinh nghiệm|experience|exp)?",
        r"(?:kinh nghiệm|experience|exp)[:\s-]*(\d+)",
        r"(\d+)\s*(?:năm|year|years)",
    ]
    for pat in patterns:
        m = re.search(pat, t)
        if m:
            y = int(m.group(1))
            if is_age and y > 18:
                continue
            return "Dưới 1 năm" if y == 0 else f"{y} năm"

    if any(k in t for k in ["senior", "nhiều năm", "expert", "chuyên gia", "lead"]):
        return "Trên 3 năm"
    return "Không yêu cầu"


# ---------------------------------------------------------------------------
# education
# ---------------------------------------------------------------------------

def _extract_education(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return "Không yêu cầu"
    t = text.lower()

    edu_kw = [
        "thạc sĩ", "master", "msc", "postgraduate", "cao học",
        "đại học", "university", "bachelor", "cử nhân", "bsc",
        "tốt nghiệp đại học", "graduate", "degree", "tốt nghiệp",
        "cao đẳng", "college", "associate",
    ]
    if not any(k in t for k in edu_kw):
        return "Không yêu cầu"

    if any(k in t for k in ["thạc sĩ", "master", "msc", "cao học"]):
        return "Thạc sĩ trở lên"
    if any(k in t for k in ["đại học", "university", "bachelor", "cử nhân",
                             "tốt nghiệp đại học", "graduate", "degree"]):
        return "Đại học trở lên"
    if any(k in t for k in ["cao đẳng", "college"]):
        return "Cao đẳng trở lên"
    # fallback "tốt nghiệp" không kèm từ khóa trình độ rõ ràng
    if "tốt nghiệp" in t or "graduate" in t:
        return "Đại học trở lên"
    return "Không yêu cầu"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_TEXT_COLS = ["job_title", "job_description", "requirement"]


def _combined(row) -> str:
    return " ".join(str(row[c]) for c in _TEXT_COLS if pd.notna(row.get(c)))


def process_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Chỉ xử lý các dòng ITViec (source == 'itviec').
    TopCV giữ nguyên vì 4 cột đã có sẵn từ platform.
    """
    df = df.copy()
    mask = df["source"] == "itviec"
    if not mask.any():
        return df

    itviec_df = df[mask]
    combined = itviec_df.apply(_combined, axis=1)

    df.loc[mask, "job_level"] = combined.apply(_extract_level)
    df.loc[mask, "experience"] = combined.apply(_extract_experience)
    df.loc[mask, "employment_type"] = combined.apply(_extract_employment_type)
    df.loc[mask, "education"] = combined.apply(_extract_education)

    return df
