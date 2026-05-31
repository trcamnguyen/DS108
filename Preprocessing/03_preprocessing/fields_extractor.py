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

import re

# Part-time / full-time detection helpers
_FT_SIGNAL_RE = re.compile(r"\bfull[\s\-]?time\b")
_PT_SIGNAL_RE = re.compile(r"\bpart[\s\-]?time\b")
_PT_PEOPLE_SUFFIX_RE = re.compile(
    r"\s+(?:contributor|contributors|employee|employees|staff|resource|resources"
    r"|worker|workers|member|members|team|hire|hires|person|people)\b"
)
_PT_NEG_PREFIX_RE = re.compile(r"\b(?:không|ko|no|not|without|non)\b")


def _affirmative_parttime(text: str) -> bool:
    """True nếu text có 'part-time' mang nghĩa affirmative (không bị phủ định,
    không mô tả người được quản lý)."""
    for m in _PT_SIGNAL_RE.finditer(text):
        if _PT_PEOPLE_SUFFIX_RE.match(text[m.end():m.end() + 40]):
            continue
        if _PT_NEG_PREFIX_RE.search(text[max(0, m.start() - 40):m.start()]):
            continue
        return True
    return False


# Pre-compile word-boundary regex for terms that can be embedded in other words
# (e.g., "contractor" inside "subcontractor", "contract" inside "contractual")
_TEMP_BOUNDARY_RE = re.compile(
    r"\b(?:"
    r"freelancer"
    r"|freelance\s+(?:position|role|job|opportunity)"
    r"|contractor"
    r"|contract\s+(?:position|role|hire|opportunity)"
    r"|contract-based"
    r"|fixed[\s\-]?term\s+contract"
    r"|short[\s\-]?term\s+contract"
    r"|temporary\s+(?:contract|position)"
    r"|contracted\s+through"
    r"|thời\s+vụ"
    r"|hợp\s+đồng\s+(?:ngắn\s+hạn|thời\s+vụ|có\s+thời\s+hạn)"
    r"|project[\s\-]?based"
    r")\b"
)
# "Contract Type: Contractor" / "Contract Type: Timed"
_CONTRACT_TYPE_RE = re.compile(
    r"\bcontract[\s_]*type[\s:]+(?:contractor|timed|temporary|freelance)\b"
)


def _extract_employment_type(job_title: str = "", job_description: str = "") -> str:
    """
    Phân loại hình thức làm việc (employment_type) từ job_title + job_description.

    Bối cảnh dataset:
        Cột employment_type gốc của ITViec là 100% null → trường này được suy
        ra từ text mô tả. TopCV có thể có metadata khác (cần kiểm tra riêng).

    Output (5 nhóm):
        "Toàn thời gian" | "Bán thời gian" | "Thời vụ" | "Hybrid" | "Remote"

    Thứ tự ưu tiên (dừng ở match đầu tiên):
        1. Bán thời gian   — explicit part-time keywords
        2. Thời vụ         — strong contract/freelance signals
        3. Title brackets  — [Remote], [Hybrid], (Remote)…
        4. Strong Remote   — "100% remote", "fully remote", "remote-first"…
        5. Hybrid          — chỉ với explicit work mode (loại tech: hybrid
                             cloud, hybrid app, hybrid search…)
        6. Weak Remote     — "remote work", "WFH", "làm từ xa"
        7. Default: Toàn thời gian

    Design notes (đã verify trên 751 records ITViec):
        - Dùng \\b word boundary cho "contractor" → tránh FP với
          "subcontractor". Tương tự cho "contract" trong "contractual",
          "API contracts", "data contracts", "labor contracts"...
        - "consultant" alone không đủ → là job title (SAP Consultant, ERP
          Consultant), không phải hình thức làm việc.
        - "hợp đồng" alone không đủ → có thể là HR doc (hợp đồng lao động
          chính thức, hợp đồng kinh tế với khách hàng…).
        - "hybrid" không default về Hybrid → quá nhiều tech context
          (hybrid cloud, hybrid mobile app, hybrid search trong RAG,
          hybrid communication system, public/private/hybrid…).
        - "remote" alone không đủ → tech context (remote lock/wipe,
          Firebase Remote Config, remote management tools, remote edge
          devices, remote site protection…).
    """
    title = job_title.lower() if isinstance(job_title, str) else ""
    desc = job_description.lower() if isinstance(job_description, str) else ""
    full = (title + " " + desc).strip()

    if not full:
        return "Toàn thời gian"

    # ─── 1. Bán thời gian (Part-time) ────────────────────────────────────────
    # Nếu full-time cũng xuất hiện (vd. career path listing) → ưu tiên Toàn thời gian
    if (_affirmative_parttime(full) or "parttime" in full or "bán thời gian" in full) \
            and not _FT_SIGNAL_RE.search(full):
        return "Bán thời gian"

    # ─── 2. Thời vụ (Contract / Freelance) ───────────────────────────────────
    # 2a. Numeric duration + contract: "2-month contract", "6 Month Contract"
    if re.search(r"\b\d+[\s\-]+(?:month|months|year|years|tháng|năm)[\s\-]+contract\b", full):
        return "Thời vụ"
    if re.search(r"\bcontract[\s:\-]+\d+[\s\-]+(?:month|months|year|years|tháng|năm)\b", full):
        return "Thời vụ"
    # Word-number duration: "one-year contract", "six-month contract"
    if re.search(r"\b(one|two|three|four|five|six|nine|twelve)[\s\-]+(month|year)[\s\-]+contract\b", full):
        return "Thời vụ"

    # 2b. Explicit freelance/contractor employment status (word-boundary safe)
    if _TEMP_BOUNDARY_RE.search(full):
        return "Thời vụ"
    if _CONTRACT_TYPE_RE.search(full):
        return "Thời vụ"

    # 2c. Title brackets: [Freelancer], (Contractor), /Freelancer
    if re.search(r"[\[\(\/]\s*(?:freelancer|contractor)\b", title):
        return "Thời vụ"

    # ─── 3. Title bracket signals — highly reliable ──────────────────────────
    # [Remote], (Remote), [HCM-Remote], [Remote - Mid Level], (Remote cho ...)
    if re.search(r"[\[\(][^\]\)]*\bremote\b[^\]\)]*[\]\)]", title):
        return "Remote"
    # "Remote " at very start of title ("Remote AI Engineer", "Remote -Sr. …")
    if re.match(r"^\s*remote\b[\s\-:]", title):
        return "Remote"
    if re.search(r"[\[\(][^\]\)]*\bhybrid\b[^\]\)]*[\]\)]", title):
        return "Hybrid"

    # ─── 4. Strong Remote phrases ────────────────────────────────────────────
    strong_remote_phrases = (
        "100% remote", "fully remote", "remote-first", "remote first",
        "remote only", "remote-only", "remote position", "remote role",
        "work location: remote", "location: remote",
        "làm việc từ xa", "làm từ xa",
    )
    if any(p in full for p in strong_remote_phrases):
        return "Remote"
    # "Employment Type: ... Remote" (catches "Full-time, Remote")
    if re.search(r"\b(?:employment|job|work)\s*type[\s:]+[^.\n]{0,40}?\bremote\b", full):
        return "Remote"

    # ─── 5. Hybrid — explicit work mode only ─────────────────────────────────
    if re.search(r"\bhybrid\b", full):
        hybrid_work_phrases = (
            "hybrid working", "hybrid work", "hybrid model", "hybrid arrangement",
            "hybrid schedule", "hybrid setup", "hybrid role", "hybrid environment",
            "hybrid position", "hybrid remote", "hybrid onsite",
            "working model: hybrid", "work model: hybrid", "model: hybrid",
            "co-located hybrid", "làm việc hybrid",
        )
        if any(p in full for p in hybrid_work_phrases):
            return "Hybrid"
        # "Hybrid with N days" / "N (office) days hybrid"
        if re.search(r"\bhybrid\s+(?:with\s+)?\d+\s+(?:office\s+)?days?\b", full):
            return "Hybrid"
        if re.search(r"\b\d+\s+(?:office\s+)?days?\s+(?:per\s+week\s+)?hybrid\b", full):
            return "Hybrid"

    # ─── 6. Weak Remote signals ──────────────────────────────────────────────
    weak_remote_phrases = (
        "remote work", "remote working", "work from home", "wfh",
        "accept remote",
    )
    if any(p in full for p in weak_remote_phrases):
        return "Remote"

    # ─── 7. Default ──────────────────────────────────────────────────────────
    return "Toàn thời gian"


# ---------------------------------------------------------------------------
# job_level
# ---------------------------------------------------------------------------

import re

def _extract_level(text: str) -> str | None:
    """Extract level từ text, return None nếu không có signal."""
    if not isinstance(text, str) or not text.strip():
        return None
    t = text.lower()
    def has(p): return bool(re.search(p, t))

    if has(r'\b(manager|director|chief|cto|ceo|vp)\b') \
       or has(r'\bhead of\b') or has(r'trưởng phòng|giám đốc'):
        return "Quản lý / Giám sát"
    if has(r'\b(tech|technical|team|group)\s+lead\b') \
       or has(r'\blead\b') or has(r'trưởng nhóm'):
        return "Trưởng nhóm"
    if has(r'\b(senior|expert|principal|staff|architect)\b') or has(r'chuyên gia'):
        return "Senior"
    if has(r'\b(junior|mid-level|middle)\b') or has(r'\bmid\b'):
        return "Junior"
    if has(r'\b(fresher|entry[- ]level)\b') or has(r'mới tốt nghiệp'):
        return "Fresher"
    if has(r'\bintern(?:ship|s)?\b') or has(r'thực tập sinh'):
        return "Thực tập sinh"
    return None


# Pattern khai báo role tường minh — chỉ match level keyword TRONG các pattern này
# Ý tưởng: extract phần ngay sau "position:", "we are hiring", v.v.
# rồi áp _extract_level lên đó (cấu trúc gọn — chỉ match khi explicit declared)
ROLE_DECLARATION_PATTERNS = [
    # English
    r'\bposition\s*[:\-]\s*([^\n.]{1,80})',
    r'\bjob\s*title\s*[:\-]\s*([^\n.]{1,80})',
    r'\brole\s*[:\-]\s*([^\n.]{1,80})',
    r'\bwe\s+(?:are|\'re)\s+(?:looking\s+for|hiring|seeking)\s+(?:an?\s+)?([^\n.]{1,80})',
    # Vietnamese
    r'vị\s*trí\s*[:\-]\s*([^\n.]{1,80})',
    r'chức\s*danh\s*[:\-]\s*([^\n.]{1,80})',
    r'chúng\s*tôi\s+(?:đang\s+)?(?:tìm\s+kiếm|tuyển\s+dụng)\s+(?:một\s+)?([^\n.]{1,80})',
]


def extract_level(job_title: str,
                  job_description: str = '',
                  requirement: str = '') -> str:
    """
    Title-first extraction với fallback có kiểm soát:
      1. Title (signal cao nhất)
      2. Fallback: chỉ match level keyword trong các pattern role declaration
         tường minh (vd. "position: Senior X", "we are looking for a Lead Y")
      3. Default: "Nhân viên"

    KHÔNG match keyword trong prose ngẫu nhiên của JD/req để tránh FP
    kiểu "report to manager", "lead team là lợi thế".
    """
    # Step 1: title
    level = _extract_level(job_title)
    if level is not None:
        return level

    # Step 2: explicit role declaration trong JD/req
    combined = (str(job_description or '') + ' ' + str(requirement or '')).lower()
    for pat in ROLE_DECLARATION_PATTERNS:
        for m in re.finditer(pat, combined):
            declared_text = m.group(1)
            lvl = _extract_level(declared_text)
            if lvl is not None:
                return lvl

    # Step 3: default
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
# TopCV employment_type reconciliation
# ---------------------------------------------------------------------------

_PT_SIGNALS = (
    "part-time", "part time", "parttime", "bán thời gian",
    "theo ca", "theo giờ", "flexible hours", "làm nửa ngày",
    "4 tiếng/ngày", "4h/ngày", "20 tiếng/tuần", "20h/tuần",
)
_TV_SIGNALS = (
    "contract", "freelance", "freelancer", "contractor",
    "thời vụ", "hợp đồng ngắn hạn", "hợp đồng thời vụ",
    "project-based", "project based",
    "tháng hợp đồng", "hợp đồng có thời hạn",
)


def reconcile_topcv_employment_type(
    metadata_value: str,
    job_title: str,
    job_description: str,
    requirement: str = "",
) -> tuple[str, str]:
    """
    Cross-validate TopCV's employment_type metadata against text content.
    Returns (final_value, audit_flag).
    """
    text_for_extract = " ".join(filter(None, [job_description or "", requirement or ""]))
    text_extracted = _extract_employment_type(job_title, text_for_extract)

    meta = metadata_value.strip() if isinstance(metadata_value, str) else ""
    full_lower = " ".join(filter(None, [
        (job_title or "").lower(),
        (job_description or "").lower(),
        (requirement or "").lower(),
    ]))

    # Case 0: "Thực tập" là platform mistag → override bằng text extraction
    if meta == "Thực tập":
        return text_extracted, "meta_intern_mistag_overridden"

    # Case 1: Metadata empty → fall back to text extraction
    if not meta or meta.lower() in ("nan", "none"):
        return text_extracted, "meta_empty"

    # Case 2: Agreement → high confidence
    if meta == text_extracted:
        return meta, "agree"

    # Case 3: Metadata = Bán thời gian — require text confirmation
    if meta == "Bán thời gian":
        if any(s in full_lower for s in _PT_SIGNALS):
            return "Bán thời gian", "meta_PT_confirmed"
        return "Toàn thời gian", "meta_PT_overridden_no_text_signal"

    # Case 4: Metadata = Thời vụ — same asymmetric trust
    if meta == "Thời vụ":
        if any(s in full_lower for s in _TV_SIGNALS):
            return "Thời vụ", "meta_TV_confirmed"
        return "Toàn thời gian", "meta_TV_overridden_no_text_signal"

    # Case 5: Metadata = Toàn thời gian — trust metadata,
    # nhưng enrich nếu text có explicit work mode (Remote/Hybrid)
    if meta == "Toàn thời gian":
        if text_extracted in ("Remote", "Hybrid"):
            return text_extracted, f"meta_FT_enriched_to_{text_extracted}"
        return "Toàn thời gian", "meta_FT_kept"

    # Case 6: Metadata khác (Remote/Hybrid tag từ TopCV, hiếm) — trust metadata
    return meta, "meta_other_kept"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_TEXT_COLS = ["job_title", "job_description", "requirement"]


def _combined(row) -> str:
    return " ".join(str(row[c]) for c in _TEXT_COLS if pd.notna(row.get(c)))


def process_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    ITViec: extract 4 cột từ text (job_level, experience, employment_type, education).
    TopCV: reconcile employment_type metadata với text, ghi kết quả vào
           employment_type_final + employment_type_audit.
    """
    df = df.copy()
    itviec_mask = df["source"] == "itviec"

    # ── ITViec ────────────────────────────────────────────────────────────────
    if itviec_mask.any():
        itviec_df = df[itviec_mask]
        combined = itviec_df.apply(_combined, axis=1)

        df.loc[itviec_mask, "job_level"] = itviec_df.apply(
            lambda row: extract_level(
                str(row.get("job_title") or ""),
                str(row.get("job_description") or ""),
                str(row.get("requirement") or ""),
            ),
            axis=1,
        )
        df.loc[itviec_mask, "experience"] = combined.apply(_extract_experience)
        df.loc[itviec_mask, "employment_type"] = combined.apply(_extract_employment_type)
        df.loc[itviec_mask, "education"] = combined.apply(_extract_education)

    # ── TopCV ─────────────────────────────────────────────────────────────────
    topcv_mask = df["source"] == "topcv"
    if topcv_mask.any():
        topcv_df = df[topcv_mask]
        result = topcv_df.apply(
            lambda r: reconcile_topcv_employment_type(
                r.get("employment_type", ""),
                str(r.get("job_title") or ""),
                str(r.get("job_description") or ""),
                str(r.get("requirement") or ""),
            ),
            axis=1,
        )
        df.loc[topcv_mask, "employment_type"] = result.map(lambda x: x[0])

    return df
