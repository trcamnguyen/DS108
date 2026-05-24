"""
Filter job titles không thuộc scope IT salary prediction.

Sửa từ phiên bản trước:
- BUG FIX: Missing comma giữa "illustrator" và "media" trong DROP_KEYWORDS
  → Python silent string concatenation thành "illustratormedia" (keyword chết).
- BUG FIX: Substring matching không có word boundary
  → "cnc", "seo", "plc", "3d", "2d" có thể match nhầm substring trong các title
    legitimate. Đổi sang \b...\b (non-capturing) để chỉ match nguyên token.
- BUG FIX: "marketing" keyword cũ không bắt "Marketer" → mất Mobile App Marketer.
  Thêm cả "marketing" và "marketer" làm 2 keyword riêng (KHÔNG dùng prefix
  match "market\\w*" vì sẽ false-positive với "Marketplace Engineer").
- BUG FIX: "graphic" keyword cũ không bắt "Graphics Designer" (plural).
  Thêm "graphics" làm keyword bổ sung.
- Bỏ comment "intentional – matches notebook behaviour" vì che giấu bug.
- Thêm sanity check `audit_dead_keywords`: in cảnh báo cho mọi rule không
  match input nào (giúp phát hiện rule chết / typo).
- Thêm cột `drop_reason` trong df_dropped để audit downstream.

Lưu ý design:
- Các rule trong EXACT_DROP_TITLES và DROP_KEYWORDS được giữ defensive ngay cả
  khi không match dataset hiện tại (audit sẽ in WARN). Lý do: các rule này
  cover các pattern có thể xuất hiện ở lần re-crawl sau (Frame 1 — robustness).
- Trong final report, cần log rõ số jobs bị drop và phân phối drop_reason
  trong Datasheets §Threats to Validity.
"""

import re

import pandas as pd

# ---------------------------------------------------------------------------
# EXACT_DROP_TITLES — match toàn bộ chuỗi (sau lowercase + strip).
# Dùng cho các title cụ thể không nằm trong scope IT salary prediction,
# nhưng có thể không bị catch bởi DROP_KEYWORDS (ví dụ "designer" trần,
# "system analyst" — analyst nghiệp vụ, không phải IT system analyst).
# ---------------------------------------------------------------------------
EXACT_DROP_TITLES: set[str] = {
    "business architect",
    "sales consultant",
    "it procurement",
    "product marketer",
    "monetization specialist",
    "ad monetization specialist",
    "cad designer",
    "media designer",
    "photo editor",
    "web designer",
    "motion graphics designer",
    "digital artist",
    "illustrator",
    "content moderator",
    "game operator",
    "data operator",
    "maintenance technician",
    "electrical engineer",
    "technical support",
    "designer",
    "game operations",
    "user acquisition specialist",
    "it communicator",
    "product operations",
    "project coordinator",
    "customer success",
    "it compliance specialist",
    "system analyst",
    "data annotator",
    "cae engineer",
    "art director",
    "image editor",
    "mechanical engineer",
}

# ---------------------------------------------------------------------------
# DROP_KEYWORDS — match như TOKEN (word boundary \b...\b). Nếu title chứa
# bất kỳ keyword nào dưới đây như 1 token độc lập, drop title đó.
#
# Plural/derivative phải list riêng (graphic + graphics; marketing + marketer)
# vì word-boundary regex KHÔNG xử lý morphology.
# ---------------------------------------------------------------------------
DROP_KEYWORDS: list[str] = [
    "graphic",       # graphic designer, graphic artist
    "graphics",      # graphics designer (plural form; \bgraphic\b không match "graphics")
    "artist",        # game artist, 3d artist, vfx artist, ...
    "3d",            # 3d modeler, 3d animator, ... — token, không match "3dimensional"
    "2d",
    "multimedia",
    "illustrator",
    "media",         # media designer/editor/specialist (multimedia có exact match riêng)
    "business development",   # multi-word phrase
    "video editor",           # multi-word phrase
    "animator",
    "marketing",     # marketing executive, marketing manager
    "marketer",      # mobile app marketer (root khác với "marketing", phải list riêng)
    "data entry",             # multi-word phrase
    "content",                # content moderator/writer/creator
    "cnc",                    # cnc programmer/operator
    "seo",
    "trainer",
    "writer",
    "plc",                    # plc engineer/programmer (industrial automation, ko phải IT)
]

# Tất cả keywords match dạng TOKEN (word-boundary \b...\b).
# Multi-word phrases ("data entry") vẫn match đúng vì \b nằm 2 đầu cụm.
# Lưu ý: "market" KHÔNG được dùng làm prefix vì sẽ false-positive
# với "Marketplace Engineer". Phải list explicit "marketing", "marketer".


def _build_keyword_pattern(keywords: list[str]) -> str:
    """
    Build regex pattern với word boundary cho mỗi keyword.

    Multi-word keywords (chứa space) được giữ nguyên dạng phrase;
    word boundary \b vẫn áp dụng ở 2 đầu cụm.

    Dùng non-capturing group (?:...) để tránh pandas UserWarning
    "pattern has match groups" khi gọi str.contains.

    Ví dụ:
    - "3d" → \b(?:3d)\b → match "3d modeler", KHÔNG match "3dimensional"
    - "data entry" → \b(?:data entry)\b → match cụm "data entry"
    """
    escaped = [re.escape(kw) for kw in keywords]
    return r"\b(?:" + "|".join(escaped) + r")\b"


def filter_by_standardized_title(
    df: pd.DataFrame,
    audit_dead_keywords: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Lọc DataFrame dựa trên cột standardized_title (sau LLM).

    Parameters
    ----------
    df : DataFrame có cột 'standardized_title'
    audit_dead_keywords : nếu True, in cảnh báo cho mọi keyword
                          không match bất kỳ row nào (giúp phát hiện
                          rule chết / typo trong DROP_KEYWORDS).

    Returns
    -------
    df_keep    : các row được giữ lại
    df_dropped : các row bị loại (để audit, có cột 'drop_reason')
    """
    title_series = df["standardized_title"].astype(str).str.strip().str.lower()

    # --- Mask 1: keyword match (word-boundary regex) ---
    keyword_pattern = _build_keyword_pattern(DROP_KEYWORDS)
    mask_keywords = title_series.str.contains(
        keyword_pattern, case=False, na=False, regex=True
    )

    # --- Mask 2: exact title match ---
    mask_exact_titles = title_series.isin(EXACT_DROP_TITLES)

    # --- Mask 3: sales-related, trừ presales / sales engineer (IT pre-sales) ---
    mask_sales = (
        title_series.str.contains(r"\bsales\b", case=False, na=False, regex=True)
        & ~title_series.str.contains(
            r"\b(?:presales|pre-sales|sales engineer)\b",
            case=False, na=False, regex=True,
        )
    )

    # --- Mask 4: account manager (sales role, không phải IT) ---
    mask_account_manager = title_series.eq("account manager")

    drop_mask = mask_keywords | mask_exact_titles | mask_sales | mask_account_manager

    df_keep = df[~drop_mask].copy()
    df_dropped = df[drop_mask].copy()

    # --- Annotate drop reason để audit ---
    if len(df_dropped) > 0:
        reasons = pd.Series(index=df_dropped.index, dtype=str)
        reasons.loc[mask_keywords[drop_mask]] = "keyword"
        reasons.loc[mask_exact_titles[drop_mask]] = "exact_title"
        reasons.loc[mask_sales[drop_mask]] = "sales_non_it"
        reasons.loc[mask_account_manager[drop_mask]] = "account_manager"
        df_dropped["drop_reason"] = reasons

    # --- Sanity check: phát hiện keywords / titles không match gì ---
    if audit_dead_keywords:
        _audit_dead_rules(title_series, DROP_KEYWORDS, EXACT_DROP_TITLES)

    return df_keep, df_dropped


def _audit_dead_rules(
    title_series: pd.Series,
    keywords: list[str],
    exact_titles: set[str],
) -> None:
    """In cảnh báo cho mọi rule không match bất kỳ title nào trong input."""
    dead_keywords = []
    for kw in keywords:
        pat = r"\b" + re.escape(kw) + r"\b"
        if not title_series.str.contains(pat, case=False, na=False, regex=True).any():
            dead_keywords.append(kw)

    dead_exacts = [t for t in exact_titles if not title_series.eq(t).any()]

    if dead_keywords:
        print(f"[WARN] DROP_KEYWORDS không match title nào: {dead_keywords}")
    if dead_exacts:
        print(f"[WARN] EXACT_DROP_TITLES không match title nào: {dead_exacts}")
    if not dead_keywords and not dead_exacts:
        print("[OK] Tất cả filter rules đều có match.")