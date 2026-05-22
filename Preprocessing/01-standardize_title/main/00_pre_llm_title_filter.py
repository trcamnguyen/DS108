"""
00_pre_llm_title_filter.py
==========================
Lọc bỏ các job posting KHÔNG thuộc lĩnh vực IT/Tech TRƯỚC KHI đưa vào
LLM để chuẩn hóa title — giảm chi phí token và noise.

Hỗ trợ 2 dataset với 2 logic filter riêng biệt:

    TOPCV  → Logic GỐC (strict)
             - Filter chặt hơn, drop nhiều case borderline
             - Phù hợp khi data TopCV có nhiều noise non-IT

    ITVIEC → Logic MỚI (conservative)
             - Filter lỏng hơn, chỉ drop case 100% chắc chắn non-IT
             - Để LLM + post-processing handle các case borderline
             - Phù hợp khi data ITViec đã được pre-curate, ít noise

Usage:
    python 00_pre_llm_title_filter.py --dataset topcv
    python 00_pre_llm_title_filter.py --dataset itviec
    python 00_pre_llm_title_filter.py --dataset all
"""

import argparse
import re
import pandas as pd
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════
# ╔═══════════════════════════════════════════════════════════════════╗
# ║                                                                   ║
# ║   TOPCV PATTERNS (LOGIC GỐC - STRICT)                             ║
# ║                                                                   ║
# ╚═══════════════════════════════════════════════════════════════════╝
# ═══════════════════════════════════════════════════════════════════════

TOPCV_SAFEGUARD_PATTERNS: list[str] = [
    # ── UI/UX Design (IT product design role) ──────────────────
    r"\bui[-/\s]*ux\b",
    r"\bux[-/\s]*ui\b",
    r"\buiux\b",
    r"\bui\s+designer\b",
    r"\bux\s+designer\b",
    r"\bhead\s+of\s+ux\b",

    # ── Presales kỹ thuật ──────────────────────────────────────
    r"presales",
    r"pre[\s-]sales",

    # ── Solution roles ─────────────────────────────────────────
    r"solution\s+(architect|engineer|developer|consultant|lead)",
    r"(network|security|it|cloud|infrastructure)\s+solution",
    r"(kỹ\s+sư|kỹ\s+thuật)\s+giải\s+pháp",

    # ── Product & Game Designer ────────────────────────────────
    r"product\s+design(er)?",
    r"game\s+design(er)?",
    r"game\s+level\s+design",
    r"thiết\s+kế\s+game",
    r"thiết\s+kế\s+game\s+mobile",

    # ── GUI Designer ───────────────────────────────────────────
    r"\bgui\s+design",

    # ── Automation / Nocode trong IT context ───────────────────
    r"automation\s+solution",
    r"nocode.*solution",
    r"nocode\s+.*(developer|engineer)",

    # ── Embedded / Circuit Design ──────────────────────────────
    r"thiết\s+kế\s+(mạch|điện\s+tử|nhúng|iot)",

    # ── IT Technical staff ─────────────────────────────────────
    r"kỹ\s+thuật\s+it",
    r"biết\s+thiết\s+kế\s+cơ\s+bản",

    # ── Application Engineer ───────────────────────────────────
    r"application\s+engineer",

    # ── Marketing với AI/Automation context ────────────────────
    r"(ai|automation|platform)\s+.*marketing",
    r"marketing.*\b(ai|automation)\b",
    r"marketing\s+(analytics?|data\s+analytics?)",
    r"kỹ\s+sư\s+ai\s+marketing",
]


TOPCV_BLOCKLIST_GROUPS: dict[str, list[str]] = {

    "graphic_visual": [
        r"\bgraphic\b",
        r"\billustrat",
        r"\bmotion\s+graphic",
        r"\bvideo\s+edit",
        r"\bphoto\s+edit",
        r"\bnhân\s+viên\s+design\b",
        r"\bthực\s+tập\s+sinh\s+design\b",
        r"\bdesigner\b(?!.*\b(ui|ux|game|product|gui)\b)",
        r"\bsenior\s+designer\b(?!.*\b(ui|ux|game|product)\b)",
        r"\blead\s+design\b(?!.*\b(ui|ux|game|product)\b)",
        r"\bthiết\s+kế\b(?!\s+(game|ui|ux|web|app|mạch|phần\s+mềm|hệ\s+thống|nhúng|iot|điện\s+tử))",
    ],

    "industrial_engineering": [
        r"\bplc\b",
        r"\bcnc\b",
        r"\bcad/cam\b",
        r"lập\s+trình\s+plc",
        r"kỹ\s+sư\s+điện\b(?!\s*(tử|nhúng|iot))",
        r"máy\s+gia\s+công",
        r"tự\s+động\s+hóa\s+plc",
        r"\bcơ\s+điện\s+tử\b",
        r"thiết\s+kế\s+bản\s+vẽ\s+điện",
        r"điện\s+plc\b",
        r"plc\s+servo",
    ],

    "marketing_sales": [
        r"\bmarketing\b(?!.*\b(automation|data|ai|platform|tech|mobile\s+app)\b)",
        r"\bseo\b(?!\s+developer)",
        r"\bads?\s+specialist\b",
        r"\buser\s+acquisition\b(?!.*mobile\s+app)",
        r"performance\s+marketing",

        r"\baccount\s+executive\b",
        r"\baccount\s+manager\b",
        r"\bsales\s+executive\b",
        r"\bsales\s+manager\b",
        r"\bsales\s+consultant\b",

        r"\bbusiness\s+development\b",

        r"\btelesales?\b",

        r"\bit\s+sales\b",
        r"\bsales\s+it\b",

        r"\bsales\s+(phần\s+mềm|b2b|engineer|admin|representative|support|development)\b",
        r"\bproduct\s+sales\b",
        r"\bsales\s+giải\s+pháp\b",

        r"\bnhân\s+viên\s+sales\b",
        r"\bchuyên\s+viên\s+sales\b",

        r"\bnhân\s+viên\s+kinh\s+doanh\b",
        r"\bchuyên\s+viên\s+kinh\s+doanh\b",
        r"\bchuyên\s+viên\s+phát\s+triển\s+kinh\s+doanh\b",
        r"\bnhân\s+viên\s+phát\s+triển\s+kinh\s+doanh\b",
        r"\btrưởng\s+(nhóm|phòng)\s+kinh\s+doanh\b",
        r"\bgiám\s+đốc\s+kinh\s+doanh\b",
        r"\bthực\s+tập\s+sinh\s+kinh\s+doanh\b",
        r"\bphát\s+triển\s+kinh\s+doanh\b",
        r"\btư\s+vấn\s+kinh\s+doanh\b",
        r"\bkinh\s+doanh\s+(giải\s+pháp|dự\s+án|mảng|phân\s+phối|b2b|phần\s+mềm)\b",
    ],

    "content_media": [
        r"\bcontent\s+(writer|creator|specialist|manager|marketing)\b",
        r"\bcopywriter\b",
        r"\bmedia\s+designer\b",
        r"\bvideo\s+producer\b",
        r"\banima[ot]",
        r"\b3d\s+artist\b",
        r"\b2d\s+artist\b",
        r"\bconcept\s+artist\b(?!.*ui)",
    ],

    "cad_mechanical": [
        r"\bsolidworks\b(?!.*\b(developer|engineer)\b)",
        r"\bcad\s+designer\b",
        r"\bcad\s+engineer\b(?!.*software)",
        r"kỹ\s+sư\s+cad\b",
        r"lập\s+trình\s+cad/cam",
    ],

    "non_tech_roles": [
        r"\btrainer\b(?!.*technical)",
        r"\bdata\s+entry\b",
        r"\bdata\s+annotator\b",
        r"\bphoto\s+editor\b",
        r"\bit\s+communicat",
        r"\bcustomer\s+success\b",
        r"\bit\s+compliance\b",
        r"\bcae\s+engineer\b",
        r"\belectrical\s+engineer\b(?!.*software)",
    ],
}


# ═══════════════════════════════════════════════════════════════════════
# ╔═══════════════════════════════════════════════════════════════════╗
# ║                                                                   ║
# ║   ITVIEC PATTERNS (LOGIC MỚI - CONSERVATIVE)                      ║
# ║                                                                   ║
# ╚═══════════════════════════════════════════════════════════════════╝
# ═══════════════════════════════════════════════════════════════════════

ITVIEC_SAFEGUARD_PATTERNS: list[str] = [

    # ── UI/UX & Product Design ─────────────────────────────────
    r"\bui[-/\s]*ux\b",
    r"\bux[-/\s]*ui\b",
    r"\buiux\b",
    r"\bui\s+designer\b",
    r"\bux\s+designer\b",
    r"\bhead\s+of\s+ux\b",
    r"\bproduct\s+design(er)?\b",
    r"\bgame\s+design(er)?\b",
    r"\bgame\s+level\s+design\b",
    r"\bthiết\s+kế\s+game\b",
    r"\bgui\s+design",
    r"\bweb\s+design(er)?\b",

    # ── Solution & Architecture roles (ALL types) ──────────────
    r"\bsolution\s+(architect|engineer|developer|consultant|lead|designer|specialist|analyst)\b",
    r"\b(network|security|it|cloud|infrastructure|software|system|technical)\s+solution\b",
    r"\b(kỹ\s+sư|kỹ\s+thuật|chuyên\s+viên)\s+giải\s+pháp\b",
    r"\bsystem\s+design(er)?\b",
    r"\barchitect\b",

    # ── Presales / Sales Engineer ──────────────────────────────
    r"\bpresales?\b",
    r"\bpre[\s-]sales?\b",
    r"\bsales\s+engineer\b",
    r"\bsolution\s+sales\b",

    # ── Embedded / IoT / Hardware-Software ─────────────────────
    r"\bembedded\b",
    r"\b(thiết\s+kế|lập\s+trình)\s+(mạch|điện\s+tử|nhúng|iot|fpga|chip)\b",
    r"\bhardware\s+engineer\b",
    r"\bfirmware\b",

    # ── IT Technical / Support / Operations ────────────────────
    r"\bit\s+(staff|support|helpdesk|operations|ops|admin|technical|infrastructure)\b",
    r"\b(kỹ\s+thuật|nhân\s+viên)\s+it\b",
    r"\bhelpdesk\b",
    r"\bhelp\s+desk\b",

    # ── Vietnam-specific IT roles ──────────────────────────────
    r"\bbrse\b",
    r"\bbridge\s+(system\s+)?(engineer|se)\b",
    r"\bkỹ\s+sư\s+cầu\s+nối\b",
    r"\bit\s+comm?(unica|tor)",
    r"\bcomtor\b",
    r"\bbiết\s+thiết\s+kế\s+cơ\s+bản\b",

    # ── Application / Software Engineer roles ──────────────────
    r"\bapplication\s+engineer\b",
    r"\bsoftware\s+(engineer|developer|architect)\b",
    r"\bweb\s+(developer|engineer)\b",

    # ── AI/ML/Data trong mọi context ───────────────────────────
    r"\b(ai|ml|machine\s+learning|deep\s+learning)\s+(engineer|developer|specialist|scientist|architect)\b",
    r"\b(prompt|llm)\s+engineer\b",
    r"\bdata\s+(engineer|scientist|analyst|architect)\b",
    r"\bbig\s+data\b",
    r"\b(business|data|systems?)\s+analyst\b",

    # ── Marketing với IT/Tech context ──────────────────────────
    r"\b(ai|automation|platform|tech|product)\s+.*marketing\b",
    r"\bmarketing\s+(analytics?|data\s+analytics?|automation|technology|operations)\b",
    r"\bgrowth\s+(engineer|hacker|analyst)\b",
    r"\bkỹ\s+sư\s+ai\s+marketing\b",
    r"\bmarketing\s+technologist\b",
    r"\bmartech\b",

    # ── IT Sales (giữ lại để post-processing handle) ───────────
    r"\bit\s+sales\b",
    r"\bsales\s+it\b",
    r"\btech(nical)?\s+sales\b",
    r"\bsales\s+(phần\s+mềm|software|solution)\b",

    # ── Automation / Low-code / No-code ────────────────────────
    r"\bautomation\s+(engineer|developer|specialist|solution)\b",
    r"\bnocode\b",
    r"\blow[\s-]?code\b",
    r"\brpa\b",

    # ── Core IT keywords (catch-all an toàn) ───────────────────
    r"\b(backend|frontend|fullstack|full[\s-]stack|back[\s-]end|front[\s-]end)\b",
    r"\b(devops|sre|site\s+reliability)\b",
    r"\b(qa|qc|tester|test\s+engineer|automation\s+test)\b",
    r"\b(scrum\s+master|product\s+owner|product\s+manager)\b",
    r"\b(blockchain|crypto)\s+(developer|engineer)\b",
    r"\bmobile\s+(developer|engineer|app)\b",
    r"\b(ios|android|flutter|react\s+native)\s+(developer|engineer)\b",
    r"\b(lập\s+trình\s+viên|kỹ\s+sư\s+phần\s+mềm)\b",
]


ITVIEC_BLOCKLIST_GROUPS: dict[str, list[str]] = {

    "graphic_visual": [
        r"\bgraphic\s+design(er)?\b",
        r"\bvisual\s+design(er)?\b",
        r"\bmotion\s+(graphic|design(er)?)\b",
        r"\billustrat(or|ion)\b",
        r"\bvideo\s+edit(or|ing)?\b",
        r"\bphoto\s+edit(or|ing)?\b",
        r"\bnhân\s+viên\s+design\b",
        r"\bthực\s+tập\s+sinh\s+design\b",
        r"\bthiết\s+kế\s+đồ\s+họa\b",
        r"\bthiết\s+kế\s+(banner|poster|logo|brochure)\b",
        r"\bbrand\s+design(er)?\b",
        r"\bcreative\s+design(er)?\b(?!.*\b(ui|ux|product|game)\b)",
    ],

    "industrial_engineering": [
        r"\bplc\b",
        r"\bcnc\b",
        r"\bcad[/\s]cam\b",
        r"\blập\s+trình\s+plc\b",
        r"\bkỹ\s+sư\s+điện\b(?!\s*(tử|nhúng|iot))",
        r"\bmáy\s+gia\s+công\b",
        r"\btự\s+động\s+hóa\s+plc\b",
        r"\bcơ\s+điện\s+tử\b",
        r"\bthiết\s+kế\s+bản\s+vẽ\s+điện\b",
        r"\bđiện\s+plc\b",
        r"\bplc\s+servo\b",
        r"\bbản\s+vẽ\s+2d\b",
        r"\bbản\s+vẽ\s+3d\b",
        r"\bdraftsman\b",
        r"\bcad\s+(designer|drafter)\b",
        r"\bsolidworks\b(?!.*\b(developer|engineer|software)\b)",
        r"\bcae\s+engineer\b",
        r"\bautocad\b(?!.*\b(developer|engineer|software)\b)",
    ],

    "marketing_pure": [
        r"\bmarketing\s+(specialist|executive|coordinator|intern|assistant|associate)\b",
        r"\bdigital\s+marketing\b(?!.*\b(automation|platform|tech)\b)",
        r"\bcontent\s+marketing\b",
        r"\bseo\s+(specialist|executive|manager|analyst)\b",
        r"\bsem\s+(specialist|executive|manager)\b",
        r"\bads?\s+specialist\b",
        r"\bperformance\s+marketing\b",
        r"\bbrand\s+(manager|specialist|executive)\b",
        r"\bpr\s+(specialist|executive|manager)\b",
        r"\bnhân\s+viên\s+marketing\b",
        r"\bchuyên\s+viên\s+marketing\b",
        r"\btrưởng\s+phòng\s+marketing\b",
    ],

    "sales_pure": [
        r"\baccount\s+(executive|manager)\b(?!.*\b(tech|it|software|solution)\b)",
        r"\bsales\s+(executive|manager|representative|admin|associate)\b(?!.*\b(engineer|tech|it|software|solution)\b)",
        r"\bsales\s+consultant\b(?!.*\b(tech|it|software|solution)\b)",
        r"\bbusiness\s+development\b(?!.*\b(tech|it|software|engineer|platform)\b)",
        r"\btelesales?\b",
        r"\btelemarketing\b",
        r"\b(nhân|chuyên)\s+viên\s+kinh\s+doanh\b(?!.*\b(phần\s+mềm|công\s+nghệ|it|software|giải\s+pháp|saas)\b)",
        r"\b(trưởng|giám\s+đốc)\s+(nhóm|phòng)?\s*kinh\s+doanh\b(?!.*\b(phần\s+mềm|công\s+nghệ|it|software|saas)\b)",
        r"\bthực\s+tập\s+sinh\s+kinh\s+doanh\b",
        r"\btư\s+vấn\s+kinh\s+doanh\b(?!.*\b(phần\s+mềm|công\s+nghệ|it|saas)\b)",
        r"\bcustomer\s+success\b(?!.*\b(engineer|tech|saas|platform)\b)",
    ],

    "content_media": [
        r"\bcontent\s+(writer|creator|specialist|manager|editor)\b",
        r"\bcopywriter\b",
        r"\bcopy\s+writer\b",
        r"\bmedia\s+(designer|planner|specialist)\b",
        r"\bvideo\s+(producer|editor)\b",
        r"\banima[ot]",
        r"\b3d\s+artist\b",
        r"\b2d\s+artist\b",
        r"\bconcept\s+artist\b(?!.*\b(ui|game)\b)",
        r"\bjournalist\b",
        r"\bbiên\s+tập\s+viên\b",
        r"\bphóng\s+viên\b",
    ],

    "non_tech_roles": [
        r"\btrainer\b(?!.*\b(technical|tech|it|software)\b)",
        r"\bdata\s+entry\b",
        r"\bnhập\s+liệu\b",
        r"\bphoto\s+editor\b",
        r"\bkế\s+toán\b",
        r"\baccountant\b",
        r"\bbảo\s+vệ\b",
        r"\btài\s+xế\b",
        r"\bdriver\b",
        r"\breceptionist\b",
        r"\blễ\s+tân\b",
        r"\bhr\s+(specialist|executive|coordinator)\b(?!.*\b(it|tech)\b)",
        r"\bnhân\s+viên\s+nhân\s+sự\b",
        r"\bchuyên\s+viên\s+nhân\s+sự\b",
        r"\bnhân\s+viên\s+bán\s+hàng\b",
        r"\belectrical\s+engineer\b(?!.*\b(software|firmware|embedded)\b)",
    ],
}


# ═══════════════════════════════════════════════════════════════════════
# COMPILE PATTERNS (một lần khi import)
# ═══════════════════════════════════════════════════════════════════════

def _compile(patterns: list[str]) -> re.Pattern:
    combined = "|".join(f"(?:{p})" for p in patterns)
    return re.compile(combined, flags=re.IGNORECASE)


# TopCV compiled patterns
_TOPCV_SAFEGUARD_RE = _compile(TOPCV_SAFEGUARD_PATTERNS)
_TOPCV_BLOCKLIST_RE = {
    group: _compile(pats) for group, pats in TOPCV_BLOCKLIST_GROUPS.items()
}

# ITViec compiled patterns
_ITVIEC_SAFEGUARD_RE = _compile(ITVIEC_SAFEGUARD_PATTERNS)
_ITVIEC_BLOCKLIST_RE = {
    group: _compile(pats) for group, pats in ITVIEC_BLOCKLIST_GROUPS.items()
}


# Bảng tra cứu pattern theo dataset
_PATTERN_REGISTRY = {
    "topcv":  (_TOPCV_SAFEGUARD_RE,  _TOPCV_BLOCKLIST_RE),
    "itviec": (_ITVIEC_SAFEGUARD_RE, _ITVIEC_BLOCKLIST_RE),
}


# ═══════════════════════════════════════════════════════════════════════
# CORE CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════

def classify_title(title: str, dataset: str) -> tuple[str, str | None]:
    """
    Phân loại một raw job title theo 3 lớp, dùng logic riêng cho từng dataset.

    Parameters
    ----------
    title   : raw job title string
    dataset : "topcv" hoặc "itviec"

    Returns
    -------
    (decision, reason)
        decision : "keep" | "drop" | "review"
        reason   : tên blocklist group nếu drop, "empty_title" nếu review,
                   hoặc None nếu keep
    """
    if dataset not in _PATTERN_REGISTRY:
        raise ValueError(f"Unknown dataset: {dataset}. Use 'topcv' or 'itviec'.")

    safeguard_re, blocklist_re = _PATTERN_REGISTRY[dataset]

    if not isinstance(title, str) or not title.strip():
        return "review", "empty_title"

    t = title.strip()

    # Layer 1: Safeguard whitelist
    if safeguard_re.search(t):
        return "keep", None

    # Layer 2: Blocklist
    for group, pattern in blocklist_re.items():
        if pattern.search(t):
            return "drop", group

    # Layer 3: Default keep (conservative)
    return "keep", None


# ═══════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════

def filter_non_it_jobs(
    df: pd.DataFrame,
    dataset: str,
    title_col: str = "job_title",
    drop_ambiguous: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Lọc DataFrame theo raw job title, dùng logic phù hợp với dataset.

    Parameters
    ----------
    df             : DataFrame đầu vào (raw, chưa qua LLM)
    dataset        : "topcv" hoặc "itviec" - quyết định dùng logic nào
    title_col      : tên cột raw job title
    drop_ambiguous : True  -> gộp 'review' vào dropped
                     False -> giữ review riêng để human check

    Returns
    -------
    df_keep    : rows được giữ -> đưa vào LLM pipeline
    df_dropped : rows bị loại -> lưu để audit
    df_review  : rows cần human review
    """
    results = df[title_col].apply(lambda t: classify_title(t, dataset))

    out = df.copy()
    out["_filter_decision"] = results.apply(lambda x: x[0])
    out["_filter_reason"]   = results.apply(lambda x: x[1])

    df_keep    = out[out["_filter_decision"] == "keep"].copy()
    df_dropped = out[out["_filter_decision"] == "drop"].copy()
    df_review  = out[out["_filter_decision"] == "review"].copy()

    if drop_ambiguous:
        df_dropped = pd.concat([df_dropped, df_review], ignore_index=True)
        df_review  = df_review.iloc[0:0]

    # Xóa cột debug khỏi df_keep (clean output vào LLM)
    df_keep = df_keep.drop(columns=["_filter_decision", "_filter_reason"])

    return df_keep, df_dropped, df_review


def print_summary(dataset: str, df_raw, df_keep, df_dropped, df_review):
    """In summary report ra stdout."""
    n = len(df_raw)
    logic_label = "STRICT (logic gốc)" if dataset == "topcv" else "CONSERVATIVE (logic mới)"

    print(f"\n{'='*58}")
    print(f"  Pre-LLM Title Filter — Summary [{dataset.upper()}]")
    print(f"  Logic: {logic_label}")
    print(f"{'='*58}")
    print(f"  Original rows  : {n:>6,}")
    print(f"  -> KEEP (-> LLM) : {len(df_keep):>6,}  ({len(df_keep)/n*100:.1f}%)")
    print(f"  -> DROPPED      : {len(df_dropped):>6,}  ({len(df_dropped)/n*100:.1f}%)")
    print(f"  -> REVIEW       : {len(df_review):>6,}  ({len(df_review)/n*100:.1f}%)")
    print(f"{'='*58}")
    print(f"  Token saved ~{len(df_dropped) + len(df_review):,} rows x avg_tokens_per_row\n")

    if len(df_dropped) > 0:
        print("  Drop reason breakdown:")
        counts = df_dropped["_filter_reason"].value_counts()
        for reason, count in counts.items():
            print(f"    {reason:<25} {count:>4,}")
        print()


# ═══════════════════════════════════════════════════════════════════════
# DROPPED STATS REPORT
# ═══════════════════════════════════════════════════════════════════════

def save_dropped_stats(
    dataset: str,
    df_dropped: pd.DataFrame,
    df_review: pd.DataFrame,
    output_path: Path,
    title_col: str = "job_title",
):
    """
    Xuất file thống kê các job title bị loại bỏ.

    Nội dung gồm 3 phần:
        1. Thống kê tổng quan
        2. Danh sách đầy đủ job title bị DROP kèm lý do
        3. Danh sách job title cần REVIEW
    """
    lines: list[str] = []
    logic_label = "STRICT (logic gốc)" if dataset == "topcv" else "CONSERVATIVE (logic mới)"

    # ── Phần 1: Tổng quan ─────────────────────────────────────
    total_removed = len(df_dropped) + len(df_review)
    lines.append("=" * 60)
    lines.append(f"  JOB TITLE FILTER — REMOVED TITLES REPORT [{dataset.upper()}]")
    lines.append(f"  Logic: {logic_label}")
    lines.append("=" * 60)
    lines.append(f"  Tổng bị loại (DROP + REVIEW) : {total_removed:,}")
    lines.append(f"    -> DROP   : {len(df_dropped):,}")
    lines.append(f"    -> REVIEW : {len(df_review):,}")
    lines.append("")

    if len(df_dropped) > 0:
        lines.append("  Breakdown theo nhóm (DROP):")
        counts = df_dropped["_filter_reason"].value_counts()
        for reason, count in counts.items():
            pct = count / len(df_dropped) * 100
            lines.append(f"    {reason:<28} {count:>4,}  ({pct:.1f}%)")
        lines.append("")

    # ── Phần 2: Danh sách DROP ────────────────────────────────
    lines.append("=" * 60)
    lines.append("  DROPPED TITLES (theo nhóm)")
    lines.append("=" * 60)
    if len(df_dropped) > 0:
        for group in df_dropped["_filter_reason"].unique():
            subset = df_dropped[df_dropped["_filter_reason"] == group][title_col]
            lines.append(f"\n[{group}]  ({len(subset):,} titles)")
            for title in sorted(subset.dropna().unique()):
                lines.append(f"  - {title}")
    else:
        lines.append("  (không có title nào bị DROP)")

    # ── Phần 3: Danh sách REVIEW ──────────────────────────────
    lines.append("")
    lines.append("=" * 60)
    lines.append("  REVIEW TITLES (title rỗng / cần kiểm tra)")
    lines.append("=" * 60)
    if len(df_review) > 0:
        for title in df_review[title_col].tolist():
            lines.append(f"  - {repr(title)}")
    else:
        lines.append("  (không có title nào cần REVIEW)")

    output_path.write_text("\n".join(lines), encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

SOURCE_CONFIG: dict[str, dict] = {
    "topcv": {
        "input":   "data/raw/00-topcv_raw.csv",
        "keep":    "00-topcv_filtered.csv",
        "dropped": "00-topcv_dropped.csv",
        "review":  "00-topcv_review.csv",
        "stats":   "00-topcv_dropped_stats.txt",
    },
    "itviec": {
        "input":   "data/raw/00-itviec_raw.csv",
        "keep":    "00-itviec_filtered.csv",
        "dropped": "00-itviec_dropped.csv",
        "review":  "00-itviec_review.csv",
        "stats":   "00-itviec_dropped_stats.txt",
    },
}


def run_source(dataset: str, root: Path, output_dir: Path) -> None:
    cfg = SOURCE_CONFIG[dataset]

    input_path   = root / cfg["input"]
    keep_path    = output_dir / cfg["keep"]
    dropped_path = output_dir / cfg["dropped"]
    review_path  = output_dir / cfg["review"]
    stats_path   = output_dir / cfg["stats"]

    print(f"\n[{dataset.upper()}] Reading: {input_path}")
    df_raw = pd.read_csv(input_path)
    print(f"  -> {len(df_raw):,} rows loaded")

    df_keep, df_dropped, df_review = filter_non_it_jobs(
        df_raw,
        dataset=dataset,
        title_col="job_title",
        drop_ambiguous=False,
    )

    df_keep.to_csv(keep_path,       index=False, encoding="utf-8-sig")
    df_dropped.to_csv(dropped_path, index=False, encoding="utf-8-sig")
    df_review.to_csv(review_path,   index=False, encoding="utf-8-sig")
    save_dropped_stats(dataset, df_dropped, df_review, stats_path, title_col="job_title")

    print_summary(dataset, df_raw, df_keep, df_dropped, df_review)
    print(f"  Saved:")
    print(f"    {keep_path}")
    print(f"    {dropped_path}")
    print(f"    {review_path}")
    print(f"    {stats_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-LLM job title filter (TopCV / ITViec) - tách 2 logic"
    )
    parser.add_argument(
        "--dataset",
        choices=["topcv", "itviec", "all"],
        default="all",
        help="Dataset to filter (default: all)",
    )
    args = parser.parse_args()

    ROOT       = Path(__file__).resolve().parents[3]   # DS108/
    OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    datasets = ["topcv", "itviec"] if args.dataset == "all" else [args.dataset]
    for ds in datasets:
        run_source(ds, ROOT, OUTPUT_DIR)