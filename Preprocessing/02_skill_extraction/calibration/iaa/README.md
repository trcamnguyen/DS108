# IAA Framework — DS108 Skill Annotation

Tính Inter-Annotator Agreement giữa Human A, Human B và LLM trên 3 sub-tasks:
- **T1** Extraction — Macro F1
- **T2** Label — Cohen's κ (binary) + Fleiss' κ (3 raters)
- **T3** Category — Cohen's κ (11-class)

---

## Cách chạy

```bash
cd Preprocessing/02_skill_extraction/calibration

python -m iaa.iaa_framework --human-a annotated_skills_cnguyen.json --human-b annotated_skills_kngoc.json --llm output/few_shot_parsed.csv --output-dir ./iaa_results
```

`--human-b` là tùy chọn. Nếu bỏ qua, chỉ tính `llm_vs_human_a` (không có Fleiss' κ).

**Format file đầu vào được hỗ trợ:**

| File | Format | Ghi chú |
|------|--------|---------|
| `annotated_skills_*.json` | JSON array | Output của app.py annotation, cho phép `NaN` |
| `output/few_shot_parsed.csv` | CSV flat (1 row/skill) | Output của 02_llm_skill_extraction.py |
| `annotations.jsonl` | Unified JSONL | `{record_id, annotator, skills}` mỗi dòng |

---

## Output

Tất cả ghi vào `--output-dir` (mặc định `./iaa_results`):

| File | Nội dung |
|------|---------|
| `iaa_report.json` | Toàn bộ metrics, CI 95%, PASS/FAIL |
| `label_confusion_llm_vs_a.png` | Confusion matrix 2×2 nhãn (LLM vs Human A) |
| `category_confusion_llm_vs_a.png` | Confusion matrix N×N category (LLM vs Human A) |
| `error_distribution.png` | Bar chart phân phối loại lỗi |
| `error_cases.csv` | Chi tiết từng disagreement |

**Ngưỡng PASS:**

| Metric | Cặp | Ngưỡng |
|--------|-----|--------|
| F1 | LLM vs Human | ≥ 0.82 |
| F1 | Human A vs B | ≥ 0.85 |
| Cohen's κ label | LLM vs Human | ≥ 0.70 |
| Cohen's κ label | Human A vs B | ≥ 0.75 |
| Cohen's κ category | LLM vs Human A | ≥ 0.60 |
| Cohen's κ category | Human A vs B | ≥ 0.65 |
| Fleiss' κ (3 raters) | A + B + LLM | ≥ 0.70 |
| Bootstrap CI lower | tất cả κ | ≥ 0.60 |

---

## Mô tả từng file

| File | Chức năng |
|------|-----------|
| `normalization.py` | `normalize_skill()` — 3 bước chuẩn hóa tên skill trước khi so sánh; `normalize_category()` — chuẩn hóa khoảng trắng quanh `/` |
| `matching.py` | `greedy_bipartite_match()` — fuzzy match bằng `SequenceMatcher`, threshold 0.80, greedy assignment; `three_way_match()` — tìm triple (A, B, LLM) khớp cả 3 |
| `metrics.py` | `compute_macro_f1()` — macro P/R/F1 qua 50 records; `cohen_kappa_label/category()` — Cohen's κ dùng sklearn; `fleiss_kappa()` — Fleiss' κ cho 3 raters |
| `bootstrap.py` | `bootstrap_f1_ci()` — CI bằng resampling records; `bootstrap_kappa_ci()` — CI bằng resampling matched pairs; seed=42, 1000 iterations |
| `loader.py` | Đọc cả 3 format (JSON array, CSV flat, unified JSONL); tự sửa `NaN` → `null`; group by record_id |
| `visualization.py` | 3 plot: confusion matrix label, confusion matrix category, bar chart error type |
| `export.py` | `collect_errors()` — phân loại lỗi (hallucination, omission, label/category disagreement); `export_error_cases()` → CSV |
| `iaa_framework.py` | Orchestration chính + CLI; gọi tất cả module theo thứ tự; in summary ra console |
