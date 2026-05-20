"""
00_pre_llm_title_filter.py
==========================
Lọc bỏ các job posting KHÔNG thuộc lĩnh vực IT/Tech TRƯỚC KHI đưa vào
LLM để chuẩn hóa title — giảm chi phí token và noise.

Strategy 3 lớp:
    LAYER 1 — SAFEGUARD (whitelist)
        Nếu title match → KEEP ngay, bỏ qua blocklist.
        Bảo vệ các borderline IT roles: UI/UX, presales kỹ thuật,
        solution engineer, game designer, embedded design, application engineer.

    LAYER 2 — BLOCKLIST (nhóm keyword)
        Nếu title match bất kỳ nhóm nào → DROP, ghi nhận lý do.
        Chia theo nhóm để dễ audit và tune về sau.

    LAYER 3 — DEFAULT
        Không match cả hai → KEEP.
        Conservative: tránh mất data do filter quá chặt.

Output:
    topcv_pre_llm_keep.csv    ← đưa vào LLM standardization
    topcv_pre_llm_dropped.csv ← audit / confirm
    topcv_pre_llm_review.csv  ← human review (title rỗng, edge case)

Tuân thủ DS108 Guidelines:
    ✓ Không can thiệp thủ công (tất cả qua code)
    ✓ Không Data Leakage (filter chỉ dùng raw title, không dùng features khác)
    ✓ Reproducible: chạy lại cho cùng kết quả
"""

import re
import pandas as pd
from pathlib import Path


# ═══════════════════════════════════════════════════════════════
# LAYER 1 — SAFEGUARD PATTERNS (whitelist)
# Ưu tiên kiểm tra TRƯỚC blocklist.
# Nếu match → KEEP, không drop dù blocklist cũng match.
# ═══════════════════════════════════════════════════════════════
SAFEGUARD_PATTERNS: list[str] = [

    # # ── UI/UX Design (IT product design role) ──────────────────
    # r"ui\s*/?\s*ux",
    # r"ux\s*/?\s*ui",
    # r"\buiux\b",

    # ── Presales kỹ thuật (IT/Network/Security pre-sales) ──────
    r"presales",
    r"pre[\s-]sales",

    # ── Solution roles (IT infrastructure & software) ──────────
    r"solution\s+(architect|engineer|developer|consultant|lead)",
    r"(network|security|it|cloud|infrastructure)\s+solution",
    r"giải\s+pháp\s+(mạng|bảo\s*mật|cntt|phần\s*mềm|hạ\s*tầng)",

    # ── Product & Game Designer (IT creative roles) ────────────
    r"product\s+design(er)?",
    r"game\s+design(er)?",
    r"game\s+level\s+design",
    r"thiết\s+kế\s+game",
    r"thiết\s+kế\s+game\s+mobile",

    # ── GUI Designer (software interface) ──────────────────────
    r"\bgui\s+design",

    # ── Automation / Nocode trong IT context ───────────────────
    r"automation\s+solution",
    r"nocode.*solution",
    r"nocode\s+.*(developer|engineer)",

    # ── Embedded / Circuit Design (IoT, hardware-software) ─────
    r"thiết\s+kế\s+(mạch|điện\s+tử|nhúng|iot)",

    # ── IT Technical staff có "biết thiết kế" ──────────────────
    r"kỹ\s+thuật\s+it",
    r"biết\s+thiết\s+kế\s+cơ\s+bản",

    # ── Application Engineer (software AE, not CAD/mechanical) ─
    r"application\s+engineer",

    # ── Marketing với AI/Data/Automation context ───────────────
    r"(ai|data|automation|platform)\s+.*marketing",
    r"marketing.*\b(ai|data|automation)\b",
    r"kỹ\s+sư\s+ai\s+marketing",
]


# ═══════════════════════════════════════════════════════════════
# LAYER 2 — BLOCKLIST GROUPS
# Dict: group_name → [regex patterns]
# Group name được lưu vào cột _filter_reason để audit sau.
# ═══════════════════════════════════════════════════════════════
BLOCKLIST_GROUPS: dict[str, list[str]] = {

    # ── Đồ họa / Visual design (không phải UI/UX software) ─────
    "graphic_visual": [
        r"\bgraphic\b",
        r"\billustrat",                          # illustrator, illustration
        r"\bmotion\s+graphic",
        r"\bvideo\s+edit",
        r"\bphoto\s+edit",
        r"\bnhân\s+viên\s+design\b",            # "Nhân Viên Design" bare
        r"\bthực\s+tập\s+sinh\s+design\b",
        # "designer" bare (không có ui/ux/game/product/gui sau)
        r"\bdesigner\b(?!.*\b(ui|ux|game|product|gui)\b)",
        r"\bsenior\s+designer\b(?!.*\b(ui|ux|game|product)\b)",
        r"\blead\s+design\b(?!.*\b(ui|ux|game|product)\b)",
        # "thiết kế" không theo sau bởi các context IT
        r"\bthiết\s+kế\b(?!\s+(game|ui|ux|web|app|mạch|phần\s+mềm|hệ\s+thống|nhúng|iot|điện\s+tử))",
    ],

    # ── Kỹ thuật điện / PLC / CNC (industrial, không phải software) ─
    "industrial_engineering": [
        r"\bplc\b",
        r"\bcnc\b",
        r"\bcad/cam\b",
        r"lập\s+trình\s+plc",
        r"kỹ\s+sư\s+điện\b(?!\s*(tử|nhúng|iot))",   # "kỹ sư điện tử/nhúng" vẫn giữ
        r"máy\s+gia\s+công",
        r"tự\s+động\s+hóa\s+plc",
        r"\bcơ\s+điện\s+tử\b",
        r"thiết\s+kế\s+bản\s+vẽ\s+điện",
        r"điện\s+plc\b",
        r"plc\s+servo",
    ],

    # ── Marketing / Sales thuần (không phải IT sales) ──────────
    "marketing_sales": [
        # marketing không có AI/data/automation/mobile app context (đã lọc ở safeguard)
        r"\bmarketing\b(?!.*\b(automation|data|ai|platform|tech|mobile\s+app)\b)",
        r"\bseo\b(?!\s+developer)",
        r"\bads?\s+specialist\b",
        r"\buser\s+acquisition\b(?!.*mobile\s+app)",  # UA non-app context
        r"performance\s+marketing",
        r"\baccount\s+executive\b",
        r"\bsales\s+executive\b",
        r"\bsales\s+manager\b",
        r"\bsales\s+consultant\b",
        r"\bnhân\s+viên\s+kinh\s+doanh\b",
    ],

    # ── Content / Media / Creative (không phải tech) ───────────
    "content_media": [
        r"\bcontent\s+(writer|creator|specialist|manager|marketing)\b",
        r"\bcopywriter\b",
        r"\bmedia\s+designer\b",
        r"\bvideo\s+producer\b",
        r"\banima[ot]",                          # animator, animation
        r"\b3d\s+artist\b",
        r"\b2d\s+artist\b",
        r"\bconcept\s+artist\b(?!.*ui)",
    ],

    # ── CAD / SolidWorks thuần cơ khí (không phải software dev) ─
    "cad_mechanical": [
        r"\bsolidworks\b(?!.*\b(developer|engineer)\b)",
        r"\bcad\s+designer\b",
        r"\bcad\s+engineer\b(?!.*software)",
        r"kỹ\s+sư\s+cad\b",
        r"lập\s+trình\s+cad/cam",
    ],

    # ── Vai trò non-tech khác ───────────────────────────────────
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


# ═══════════════════════════════════════════════════════════════
# COMPILE PATTERNS (một lần khi import)
# ═══════════════════════════════════════════════════════════════

def _compile(patterns: list[str]) -> re.Pattern:
    combined = "|".join(f"(?:{p})" for p in patterns)
    return re.compile(combined, flags=re.IGNORECASE)


_SAFEGUARD_RE: re.Pattern = _compile(SAFEGUARD_PATTERNS)

_BLOCKLIST_RE: dict[str, re.Pattern] = {
    group: _compile(pats)
    for group, pats in BLOCKLIST_GROUPS.items()
}


# ═══════════════════════════════════════════════════════════════
# CORE CLASSIFIER
# ═══════════════════════════════════════════════════════════════

def classify_title(title: str) -> tuple[str, str | None]:
    """
    Phân loại một raw job title theo 3 lớp.

    Returns
    -------
    (decision, reason)
        decision : "keep" | "drop" | "review"
        reason   : tên blocklist group nếu drop, "empty_title" nếu review,
                   hoặc None nếu keep
    """
    if not isinstance(title, str) or not title.strip():
        return "review", "empty_title"

    t = title.strip()

    # Layer 1: Safeguard whitelist
    if _SAFEGUARD_RE.search(t):
        return "keep", None

    # Layer 2: Blocklist
    for group, pattern in _BLOCKLIST_RE.items():
        if pattern.search(t):
            return "drop", group

    # Layer 3: Default keep (conservative)
    return "keep", None


# ═══════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════

def filter_non_it_jobs(
    df: pd.DataFrame,
    title_col: str = "job_title",
    drop_ambiguous: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Lọc DataFrame theo raw job title.

    Parameters
    ----------
    df            : DataFrame đầu vào (raw, chưa qua LLM)
    title_col     : tên cột raw job title
    drop_ambiguous: True  → gộp 'review' vào dropped
                    False → giữ review riêng để human check

    Returns
    -------
    df_keep    : rows được giữ → đưa vào LLM pipeline
    df_dropped : rows bị loại → lưu để audit
    df_review  : rows cần human review
    """
    results = df[title_col].apply(classify_title)

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


def print_summary(df_raw, df_keep, df_dropped, df_review):
    """In summary report ra stdout."""
    n = len(df_raw)
    print(f"\n{'='*58}")
    print(f"  Pre-LLM Title Filter — Summary")
    print(f"{'='*58}")
    print(f"  Original rows  : {n:>6,}")
    print(f"  → KEEP (→ LLM) : {len(df_keep):>6,}  ({len(df_keep)/n*100:.1f}%)")
    print(f"  → DROPPED      : {len(df_dropped):>6,}  ({len(df_dropped)/n*100:.1f}%)")
    print(f"  → REVIEW       : {len(df_review):>6,}  ({len(df_review)/n*100:.1f}%)")
    print(f"{'='*58}")
    print(f"  Token saved ≈ {len(df_dropped) + len(df_review):,} rows × avg_tokens_per_row\n")

    if len(df_dropped) > 0:
        print("  Drop reason breakdown:")
        counts = df_dropped["_filter_reason"].value_counts()
        for reason, count in counts.items():
            print(f"    {reason:<25} {count:>4,}")
        print()


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def save_dropped_stats(df_dropped: pd.DataFrame, df_review: pd.DataFrame, output_path: Path):
    """
    Xuất file thống kê các job title bị loại bỏ.

    Nội dung gồm 3 phần:
        1. Thống kê tổng quan (tổng dropped, review, breakdown theo nhóm)
        2. Danh sách đầy đủ job title bị DROP kèm lý do
        3. Danh sách job title cần REVIEW (title rỗng / edge case)
    """
    lines: list[str] = []

    # ── Phần 1: Tổng quan ─────────────────────────────────────
    total_removed = len(df_dropped) + len(df_review)
    lines.append("=" * 60)
    lines.append("  JOB TITLE FILTER — REMOVED TITLES REPORT")
    lines.append("=" * 60)
    lines.append(f"  Tổng bị loại (DROP + REVIEW) : {total_removed:,}")
    lines.append(f"    → DROP   : {len(df_dropped):,}")
    lines.append(f"    → REVIEW : {len(df_review):,}")
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
            subset = df_dropped[df_dropped["_filter_reason"] == group]["job_title"]
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
        for title in df_review["job_title"].tolist():
            lines.append(f"  - {repr(title)}")
    else:
        lines.append("  (không có title nào cần REVIEW)")

    output_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    # ── Paths ─────────────────────────────────────────────────
    ROOT        = Path(__file__).resolve().parents[3]   # DS108/
    INPUT_PATH  = ROOT / "data" / "raw" / "00-topcv_raw.csv"
    OUTPUT_DIR  = Path(__file__).resolve().parents[1] / "output"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    KEEP_PATH    = OUTPUT_DIR / "00-topcv_filtered.csv"
    DROPPED_PATH = OUTPUT_DIR / "00-topcv_dropped.csv"
    REVIEW_PATH  = OUTPUT_DIR / "00-topcv_review.csv"
    STATS_PATH   = OUTPUT_DIR / "00-topcv_dropped_stats.txt"

    # ── Load ──────────────────────────────────────────────────
    print(f"Reading: {INPUT_PATH}")
    df_raw = pd.read_csv(INPUT_PATH)
    print(f"  → {len(df_raw):,} rows loaded")

    # ── Filter ────────────────────────────────────────────────
    df_keep, df_dropped, df_review = filter_non_it_jobs(
        df_raw,
        title_col="job_title",
        drop_ambiguous=False,
    )

    # ── Save ──────────────────────────────────────────────────
    df_keep.to_csv(KEEP_PATH,    index=False, encoding="utf-8-sig")
    df_dropped.to_csv(DROPPED_PATH, index=False, encoding="utf-8-sig")
    df_review.to_csv(REVIEW_PATH,   index=False, encoding="utf-8-sig")
    save_dropped_stats(df_dropped, df_review, STATS_PATH)

    # ── Report ────────────────────────────────────────────────
    print_summary(df_raw, df_keep, df_dropped, df_review)

    print(f"  Saved:")
    print(f"    {KEEP_PATH}")
    print(f"    {DROPPED_PATH}")
    print(f"    {REVIEW_PATH}")
    print(f"    {STATS_PATH}")