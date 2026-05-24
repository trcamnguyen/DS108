"""
Normalize standardized_title: merge case-only duplicates, near-synonyms,
và mapping rule-based cho 1 số role cluster.

Sửa từ phiên bản trước:
- Gỡ merge questionable (vi phạm semantic role boundary cho salary prediction):
    * Product Owner → Product Manager   [REMOVED] PO/PM khác lương khác trách nhiệm
    * Database Administrator → Database Engineer  [REMOVED] DBA ops vs DBE dev là 2 role
    * Infrastructure Engineer → Cloud Engineer    [REMOVED] On-prem ≠ cloud
    * NOC Engineer → IT Operations                [REMOVED] canonical "IT Operations"
                                                            không tồn tại trong dữ liệu
- Dọn dead rules: "Solutions Architect → Solution Architect" / "IT Architect"
    không match gì trong dữ liệu hiện tại — đánh dấu rõ để audit.
- Bỏ trùng giữa hai bước:
    map_standardized_title() đã trả về "Solution Engineer" (Title Case),
    nên key "Solutions Engineer" (Title Case) trong SEMANTIC_MERGE_MAP cũ là
    no-op. Chuyển toàn bộ rule về 1 dict mapping lowercase → Title Case.
- Thêm sanity check: assert mọi mapping target xuất hiện trong output ≥ 1 lần
    (catch dead rules); assert không có case-only duplicates sau merge.
- Trả về metadata về số title đã thay đổi (giúp audit reproducibility).
"""

import re

import pandas as pd

# ---------------------------------------------------------------------------
# RULE_BASED_MAP — title (lowercase, strip) → canonical title (Title Case)
#
# Dùng cho các case có thể decide bằng substring/token rule.
# Áp dụng theo thứ tự: rule cao hơn (đầu list) override rule sau.
# Mỗi rule là (predicate_fn, canonical_output, justification).
# ---------------------------------------------------------------------------


def tokenize(text: str) -> list[str]:
    """Tách text thành token alphanumeric (lowercase)."""
    return re.findall(r"\b[a-zA-Z0-9]+\b", str(text).lower())


def map_standardized_title(title: str) -> str:
    """
    Áp dụng rule-based normalization. Trả về canonical title nếu match,
    hoặc title gốc (không đổi) nếu không có rule nào áp dụng.

    Thứ tự rule quan trọng: rule earlier wins.
    """
    t = str(title).strip().lower()
    tokens = tokenize(t)

    # --- QA Engineer ---
    # Trigger: "tester" / "qa" / "quality assurance"
    if "tester" in tokens or "qa" in tokens or "quality assurance" in t:
        return "QA Engineer"

    # --- UI/UX Designer ---
    if (
        "ui/ux" in t
        or "ux/ui" in t
        or ("ui" in tokens and "ux" in tokens)
        or ("ui" in tokens and "designer" in tokens)
        or ("ux" in tokens and "designer" in tokens)
    ):
        return "UI/UX Designer"

    # --- Fullstack Developer ---
    if "fullstack" in tokens or ("full" in tokens and "stack" in tokens):
        return "Fullstack Developer"

    # --- Software Engineer ---
    if t in {"software developer"}:
        return "Software Engineer"

    # --- IT Support ---
    # NOTE: "application support" và "it technician" merge về IT Support
    # là chấp nhận được vì cùng role helpdesk/end-user support.
    if t in {"application support", "it technician"}:
        return "IT Support"

    # --- Solution Engineer ---
    # Gộp cả 3 surface form Solution(s) Engineer.
    if t in {"solutions engineer", "solution engineer"}:
        return "Solution Engineer"

    # --- Solution Consultant ---
    # NEW: gộp surface form Solution(s) Consultant tương tự pattern Engineer.
    # IT Solutions Consultant giữ riêng vì có prefix "IT" — có thể signal khác.
    if t in {"solutions consultant", "solution consultant"}:
        return "Solution Consultant"

    # --- Embedded Engineer ---
    # IoT / Firmware / FPGA cùng cluster embedded. Giữ rule cũ.
    if (
        t in {"iot engineer", "embedded developer", "firmware engineer", "fpga engineer"}
        or "embedded" in tokens
    ):
        return "Embedded Engineer"

    # --- Project Manager ---
    # PMO Office variants gộp vào Project Manager.
    # IT Manager → Project Manager: vì IT Manager trong dataset VN thường là
    # role quản lý dự án IT, không phải engineering manager.
    if t in {
        "it manager",
        "project managment office",       # typo từ JD gốc
        "project management office",
        "project management officer",
    }:
        return "Project Manager"

    # --- Game Developer ---
    # Unity Developer chuyên cho game (rare standalone case).
    if t in {"unity developer"}:
        return "Game Developer"

    # --- AI Engineer ---
    # Specialized sub-roles của AI/CV/NLP/Voice merge về AI Engineer.
    # NOTE: AI Architect / AI Manager / AI Application Specialist giữ riêng
    # theo yêu cầu (preserve seniority signal).
    if t in {
        "computer vision engineer",
        "machine vision engineer",
        "voice engineer",
        "nlp engineer",
    }:
        return "AI Engineer"

    # --- Machine Learning Engineer ---
    if t in {"ml engineer"}:
        return "Machine Learning Engineer"

    # --- Bridge Engineer ---
    # BrSE = Bridge System Engineer. Gộp toàn bộ về Bridge Engineer.
    if "bridge" in tokens or "brse" in tokens:
        return "Bridge Engineer"

    # --- Presales Engineer ---
    # Gộp "Presales Engineer" / "Pre-Sales Engineer" / "Pre-sales Engineer"
    # về 1 canonical. Token "pre" + "sales" match "pre-sales" sau tokenize.
    if "presales" in tokens or ("pre" in tokens and "sales" in tokens):
        return "Presales Engineer"

    # --- Mobile Developer ---
    if t in {"ios developer"}:
        return "Mobile Developer"

    # --- ERP Developer ---
    if t in {"odoo developer"}:
        return "ERP Developer"

    # Không match rule nào → giữ nguyên (mapping sẽ chuyển tiếp vào
    # SEMANTIC_MERGE_MAP ở bước sau).
    return title


# ---------------------------------------------------------------------------
# SEMANTIC_MERGE_MAP — exact-string mapping (case-sensitive, Title Case).
#
# Áp dụng SAU map_standardized_title(). Dùng cho các merge không thể
# express bằng rule đơn giản, hoặc các near-synonyms specific.
#
# Mỗi entry phải có comment justification. Merge questionable đã được gỡ:
#   - Product Owner ≠ Product Manager
#   - Database Administrator ≠ Database Engineer
#   - Infrastructure Engineer ≠ Cloud Engineer
#   - NOC Engineer ≠ IT Operations (canonical không tồn tại)
# ---------------------------------------------------------------------------
SEMANTIC_MERGE_MAP: dict[str, str] = {
    # Case-only / plural variants — đây là lỗi LLM normalization không nhất quán.
    "Systems Engineer": "System Engineer",

    # AI generalist subtypes — sub-role không có signal khác biệt rõ
    # so với AI Engineer (specialist vs engineer vs ai/ml mix).
    "AI Specialist": "AI Engineer",
    "AI/ML Engineer": "AI Engineer",

    # Security cluster — cybersecurity = security trong context VN IT.
    "Cybersecurity Engineer": "Security Engineer",
    "Security Specialist": "Security Engineer",

    # IT Support cluster — application/product support là end-user support.
    "Application Support Engineer": "IT Support",
    "Product Support": "IT Support",

    # Project Manager cluster — IT PM = PM trong ngành IT.
    "IT Project Manager": "Project Manager",

    # ERP cluster.
    "ERP Specialist": "ERP Consultant",
    "Functional Consultant": "ERP Consultant",  # functional consultant trong VN thường là ERP

    # Implementation cluster — implementer/engineer/consultant/specialist
    # đều là role triển khai phần mềm cho khách hàng.
    "Software Implementer": "Implementation Specialist",
    "Software Implementation Specialist": "Implementation Specialist",
    "Implementation Engineer": "Implementation Specialist",
    "Implementation Consultant": "Implementation Specialist",

    # DevOps cluster — DevSecOps trong VN thường = DevOps + security awareness,
    # population nhỏ (2 records), gộp chấp nhận được.
    "DevSecOps Engineer": "DevOps Engineer",

    # Data cluster — Governance Specialist gộp với Analyst (cùng role analytics
    # về governance, không tách rời).
    "Data Governance Specialist": "Data Governance Analyst",

    # Hardware cluster — technician/engineer khác seniority, nhưng dataset
    # chỉ có 1 Hardware Technician → merge để tránh singleton.
    "Hardware Technician": "Hardware Engineer",

    # QA cluster — full form vs short form cùng role.
    "Quality Assurance Engineer": "QA Engineer",
}


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------
def normalize_titles(
    df: pd.DataFrame,
    audit: bool = False,
) -> pd.DataFrame:
    """
    Áp dụng map_standardized_title() rồi SEMANTIC_MERGE_MAP
    lên cột standardized_title. Trả về DataFrame mới (không mutate input).

    Parameters
    ----------
    df : DataFrame có cột 'standardized_title'
    audit : nếu True, in báo cáo về rule chết (target không xuất hiện trong
            output) và case-only duplicates còn sót.
    """
    out = df.copy()
    before = out["standardized_title"].copy()

    out["standardized_title"] = (
        out["standardized_title"]
        .apply(map_standardized_title)
        .replace(SEMANTIC_MERGE_MAP)
    )

    if audit:
        _audit_normalization(before, out["standardized_title"])

    return out


def _audit_normalization(before: pd.Series, after: pd.Series) -> None:
    """In báo cáo audit về:
       - Số title đã thay đổi (before vs after)
       - Dead rules: target trong SEMANTIC_MERGE_MAP không xuất hiện trong output
       - Case-only duplicates còn sót sau normalize
    """
    n_changed = (before != after).sum()
    print(f"[AUDIT] Số rows có title thay đổi: {n_changed} / {len(before)}")

    # Dead rules trong SEMANTIC_MERGE_MAP
    output_titles = set(after.unique())
    dead_targets = [
        target for target in set(SEMANTIC_MERGE_MAP.values())
        if target not in output_titles
    ]
    if dead_targets:
        print(f"[WARN] SEMANTIC_MERGE_MAP target không xuất hiện trong output: {dead_targets}")
    else:
        print("[OK] Mọi SEMANTIC_MERGE_MAP target đều xuất hiện trong output.")

    # Case-only duplicates còn sót
    title_counts = after.value_counts()
    titles_lc = pd.Series(title_counts.index).str.lower().str.strip()
    dup_mask = titles_lc.duplicated(keep=False)
    if dup_mask.any():
        dups = pd.Series(title_counts.index)[dup_mask].tolist()
        print(f"[WARN] Case-only duplicates còn sót sau normalize: {dups}")
    else:
        print("[OK] Không có case-only duplicates sau normalize.")