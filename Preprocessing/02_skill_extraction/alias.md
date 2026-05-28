# Alias Bootstrap via Gemini API

## Mục tiêu
Gọi Gemini 2.5 Pro để group các skill variants về canonical name, output `aliases.yaml` dùng trong clustering pipeline.

## Input
File `skill_distribution_after.csv` với schema:
```
final_canonical, dominant_category, total_count, n_raw_variants, sample_variants
machine learning, AI/ML/Data, 189, 12, ml | mô hình học máy | ai machine learning
```

## Output
File `aliases.yaml`:
```yaml
aliases:
  - canonical: machine learning
    variants:
      - ml
      - mô hình học máy
      - ai machine learning
      - ml (machine learning)
  - canonical: natural language processing
    variants:
      - nlp
```

---

## Implementation

### Bước 1 — Prepare input

Đọc `skill_distribution_after.csv`, lọc skill theo rule:

- **GIỮ LẠI:** skill có `total_count >= 5` (bao gồm các skill trong `<Category> Other` nếu count >= 5)
- **LOẠI BỎ:** skill có `total_count < 5` (long-tail noise, không cần normalize)

Rule này đồng bộ với threshold filter ở salary modeling (cùng count >= 5) — alias dictionary cover đúng phạm vi skill sẽ thực sự được dùng làm feature.

Format mỗi skill thành 1 dòng:
```
{skill_name} | count={total_count} | category={dominant_category}
```

Sort by `dominant_category` ASC rồi `total_count` DESC — group cùng category cạnh nhau giúp LLM nhận ra alias nhanh hơn. Các skill thuộc `<Category> Other` được sort như một category bình thường (theo tên).

Chia thành batch **200 skill/batch** (Gemini 2.5 Pro context lớn, batch lớn hơn vẫn xử lý tốt). Nếu tổng < 200 thì 1 batch.

### Bước 2 — Gọi API

Dùng official `google-genai` SDK (mới, recommended) hoặc `google-generativeai` (legacy). Ví dụ dưới dùng `google-genai`:

```python
from google import genai
from google.genai import types
import yaml, os

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

SYSTEM_PROMPT = """Bạn là expert IT skill taxonomy. Nhiệm vụ: nhóm skill cùng concept thành alias groups.
Output YAML THUẦN TÚY — không markdown fence, không giải thích, không preamble."""

USER_PROMPT_TEMPLATE = """Cho danh sách IT skill dưới đây, group các skill là CÙNG MỘT CONCEPT thành alias.

QUY TẮC GROUP HỢP LỆ:
- Acronym ↔ full form: "ml" ↔ "machine learning", "nlp" ↔ "natural language processing"
- EN ↔ VN: "statistics" ↔ "thống kê", "tiếng anh" ↔ "english"
- Lemma variant: "analytics" ↔ "analysis", "warehousing" ↔ "warehouse", "pipelines" ↔ "pipeline"
- Typo/format: "powerbi" ↔ "power bi", "etl etl" ↔ "etl"
- Cert → skill: "ielts" → "english", "toeic" → "english"

KHÔNG GROUP:
- Skill liên quan nhưng khác cấp: "deep learning" ≠ "machine learning"
- Sub-concept: "supervised learning" ≠ "machine learning"
- Nếu không chắc → bỏ qua

PICK CANONICAL:
- Ưu tiên tiếng Anh > tiếng Việt
- Ưu tiên full form > acronym (ngoại lệ: acronym phổ biến hơn nhiều như "sql", "api" → giữ acronym)
- Ưu tiên entry count cao nhất trong group

CHỈ output các group có ít nhất 1 variant (ngoài canonical). Không output group chỉ có mình canonical.

OUTPUT FORMAT (YAML thuần túy, đúng cấu trúc này):
aliases:
  - canonical: <tên canonical>
    variants:
      - <variant 1>
      - <variant 2>

DANH SÁCH SKILL:
{skill_list}"""

def call_gemini(skill_list_str: str) -> list[dict]:
    """Gọi API 1 batch, parse YAML, return list alias groups."""
    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=USER_PROMPT_TEMPLATE.format(skill_list=skill_list_str),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.2,        # thấp để output deterministic
            max_output_tokens=8192,
        ),
    )
    raw = response.text.strip()

    # Strip markdown fences nếu Gemini vẫn output (defensive)
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw
        if raw.endswith("```"):
            raw = raw.rsplit("```", 1)[0]
        raw = raw.strip()

    # Parse YAML — nếu fail thì log và return []
    try:
        parsed = yaml.safe_load(raw)
        return parsed.get("aliases", [])
    except yaml.YAMLError as e:
        print(f"  [WARN] YAML parse failed: {e}")
        print(f"  Raw output (first 300 chars): {raw[:300]}")
        return []
```

**Lưu ý:** Gemini 2.5 Pro đôi khi vẫn wrap output trong markdown fence dù prompt cấm — code defensive strip fence trước khi parse YAML.

### Bước 3 — Validate output

Với mỗi alias group trả về:

1. **Canonical phải tồn tại trong input list** (không hallucinate tên mới). Nếu không → drop group, log warning.
2. **Mỗi variant phải tồn tại trong input list**. Variant không có trong input → drop variant đó, giữ group nếu còn ≥ 1 variant hợp lệ.
3. **Không circular**: variant không được là canonical của group khác. Nếu conflict → ưu tiên group có canonical count cao hơn.
4. **Dedupe**: nếu cùng variant xuất hiện ở 2 group, chỉ giữ 1 (log warning).

### Bước 4 — Merge và output

Merge tất cả batch → dedupe → sort canonical alphabetically → ghi `aliases.yaml`.

Thêm `metadata` block ở đầu file để traceable:
```yaml
metadata:
  generated_at: "2026-05-27T10:00:00"
  model: "gemini-2.5-pro"
  input_file: "skill_distribution_after.csv"
  filter_rule: "total_count >= 5 (including <Category> Other)"
  n_input_skills: 532
  n_batches: 3
  batch_size: 200
  n_alias_groups: 95
  n_variants_total: 187

aliases:
  - canonical: ...
```

---

## CLI

```bash
python bootstrap_aliases.py \
    --input outputs/skill_distribution_after.csv \
    --output outputs/aliases.yaml \
    --min-count 5 \
    --batch-size 200
```

**Mặc định:**
- `--min-count 5` — filter skill có total_count >= 5
- `--batch-size 200` — phù hợp context Gemini 2.5 Pro

## Sanity check sau khi chạy

In ra console:
```
Input skills (after filter count >= 5): 532
  Including <Category> Other: 47 skills
Batches: 3 (200 + 200 + 132)
Alias groups generated: 95
Variants mapped: 187 (35.2% of input)
Groups with cross-category variants: 14  ← cần review thủ công

Top 20 alias groups by variant count:
  machine learning (8 variants): ml | mô hình học máy | ...
  english (6 variants): tiếng anh | ielts | toeic | ...
  ...
```

---

## Lưu ý

- **GEMINI_API_KEY** phải có trong env trước khi chạy (lấy từ Google AI Studio).
- Dependencies: `google-genai>=0.3`, `pyyaml`, `pandas`.
- Nếu batch nào YAML parse fail → log rõ batch đó, skip, tiếp tục batch khác. Không crash toàn bộ.
- `aliases.yaml` là file để **user review trước khi dùng**. Sau khi chạy script, user đọc qua file, xóa/sửa group nào sai, rồi mới feed vào clustering pipeline.