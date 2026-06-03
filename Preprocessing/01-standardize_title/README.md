# 01 — Standardize Job Title

Chuẩn hóa cột `job_title` của TopCV và ITViec bằng LLM (Gemini via Vertex AI), sau đó filter và normalize title để tạo ra dataset sạch cho các bước preprocessing tiếp theo.

---

## Luồng xử lý tổng quan

```
data/raw/00-{dataset}_raw.csv
        │
        ▼
[Bước 0] 00_pre_llm_title_filter.py --dataset {topcv|itviec|all}
        │   Lọc bỏ job không thuộc IT trước khi gửi LLM (tiết kiệm token)
        │   TopCV: logic strict | ITViec: logic conservative
        │
        ▼  output/00-{dataset}_filtered.csv
        │
[Bước 1] 01_process_job_title.py --dataset {topcv|itviec}
        │   Gọi Gemini standardize title theo batch, checkpoint sau mỗi batch
        │
        ▼  output/{dataset}_job_title_full.json
        │
[Bước 2] 02_merge_job_title.py --dataset {topcv|itviec|all}
        │   Merge kết quả JSON vào CSV đã filter theo url, drop row is_valid_job=False
        │
        ▼  output/01-{dataset}_llm_standardized.csv
        │
[Bước 3] 01_standardize_title.ipynb
        │   Load cả hai dataset, merge, post-filter + normalize, lưu
        │
        ▼
data/interim/01-standardized_title.csv   (topcv + itviec, có cột source)
```

---

## Cấu trúc thư mục

```
01-standardize_title/
├── main/
│   ├── 00_pre_llm_title_filter.py  # Bước 0 — pre-LLM filter (TopCV + ITViec)
│   ├── 01_process_job_title.py     # Bước 1 — LLM standardization
│   ├── 02_merge_job_title.py       # Bước 2 — merge JSON → CSV
│   ├── 03_title_filter.py          # Module: post-LLM title filter (dùng bởi notebook)
│   ├── 03_title_normalizer.py      # Module: title normalization (dùng bởi notebook)
│   └── 03_salary_utils.py          # Module: salary flag utilities (dùng bởi notebook)
├── output/                          # Intermediate outputs (không commit)
├── prompt/
│   └── prompt_job_title.txt         # System prompt cho LLM
├── .env                             # Credentials Vertex AI (không commit)
└── 01_standardize_title.ipynb       # Bước 3 — notebook chính
```

---

## Thiết lập môi trường

Tạo file `.env` trong thư mục gốc repo `DS108/DS108/`:

```env
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1
GEMINI_MODEL=gemini-2.0-flash-001
GOOGLE_APPLICATION_CREDENTIALS=credentials/service-account.json
```

Đặt file service account tại `DS108/DS108/credentials/service-account.json`.

---

## Cách chạy

### Bước 0 — Pre-LLM filter (TopCV + ITViec)

Lọc bỏ job không thuộc IT/Tech dựa trên raw `job_title` trước khi gửi LLM. Hỗ trợ cả hai dataset với 2 logic filter riêng biệt.

```bash
# Chạy cả hai dataset (default)
python main/00_pre_llm_title_filter.py

# Hoặc từng dataset
python main/00_pre_llm_title_filter.py --dataset topcv
python main/00_pre_llm_title_filter.py --dataset itviec
```

| | |
|---|---|
| **Input** | `data/raw/00-{dataset}_raw.csv` |
| **Output** | `output/00-{dataset}_filtered.csv` — gửi vào LLM |
| | `output/00-{dataset}_dropped.csv` — bị loại |
| | `output/00-{dataset}_review.csv` — cần human review |
| | `output/00-{dataset}_dropped_stats.txt` — báo cáo lý do loại |

> **TopCV** dùng logic **strict** (filter chặt hơn, drop nhiều case borderline).  
> **ITViec** dùng logic **conservative** (chỉ drop case 100% chắc chắn non-IT, để LLM + post-processing handle borderline).

---

### Bước 1 — LLM Standardization

Gọi Gemini qua Vertex AI để standardize title. Đọc từ `output/00-{dataset}_filtered.csv`. Tự động resume từ checkpoint nếu bị ngắt giữa chừng.

```bash
# TopCV
python main/01_process_job_title.py --dataset topcv

# ITViec
python main/01_process_job_title.py --dataset itviec
```

| | |
|---|---|
| **Input** | `output/00-{dataset}_filtered.csv` |
| **Prompt** | `prompt/prompt_job_title.txt` |
| **Output** | `output/{dataset}_job_title_full.json` (checkpoint per batch) |

**Output mỗi row trong JSON:**
```json
{
  "id": "https://...",
  "original": "Kỹ Sư Phần Mềm Senior",
  "is_valid_job": true,
  "standardized_title": "Software Engineer"
}
```

> `id` là URL của job posting (dùng làm key để merge ở Bước 2).

---

### Bước 2 — Merge JSON → CSV

Merge kết quả LLM vào CSV đã filter theo `url`, thêm cột `standardized_title` và **loại bỏ** các row có `is_valid_job = False`.

```bash
# Cả hai dataset (default)
python main/02_merge_job_title.py

# Hoặc từng dataset
python main/02_merge_job_title.py --dataset topcv
python main/02_merge_job_title.py --dataset itviec
```

| | |
|---|---|
| **Input** | `output/00-{dataset}_filtered.csv` |
| | `output/{dataset}_job_title_full.json` |
| **Output** | `output/01-{dataset}_llm_standardized.csv` |

Cột được thêm vào:
- `standardized_title`: title đã được chuẩn hóa bởi LLM

> Row với `is_valid_job = False` (LLM đánh giá không phải IT) bị **drop** khỏi output, không giữ lại.

---

### Bước 3 — Notebook: Post-process + Save

Mở và chạy toàn bộ `01_standardize_title.ipynb`. Notebook thực hiện:

1. Đọc raw data, thống kê missing (TopCV + ITViec)
2. Load `output/01-topcv_llm_standardized.csv` và `output/01-itviec_llm_standardized.csv`
3. Thêm cột `source`, merge cả hai thành một DataFrame
4. Lưu thống kê title distribution trước khi filter → `output/job_title_stats.csv`
5. **Filter** bằng `filter_by_standardized_title()` — loại bỏ title không phải IT
6. **Normalize** bằng `normalize_titles()` — gộp các variant title về canonical name
7. Lưu thống kê title distribution sau khi filter → `output/job_title_stats_filtered.csv`
8. Lưu dataset cuối cùng (merged, có cột `source`) → `data/interim/01-standardized_title.csv`

| | |
|---|---|
| **Input** | `output/01-topcv_llm_standardized.csv` |
| | `output/01-itviec_llm_standardized.csv` |
| **Output** | `data/interim/01-standardized_title.csv` |
| | `output/job_title_stats.csv` (thống kê trước filter) |
| | `output/job_title_stats_filtered.csv` (thống kê sau filter + normalize) |

---

## Chi tiết từng file

### `main/00_pre_llm_title_filter.py`

Lọc non-IT jobs từ raw CSV **trước khi gửi LLM** để tiết kiệm token. Hỗ trợ cả hai dataset với 2 bộ pattern riêng biệt.

**Chiến lược 3 lớp (áp dụng cho cả hai dataset):**
- **Layer 1 — Safeguard (whitelist)**: Match → giữ ngay, bỏ qua blocklist. Bảo vệ borderline IT roles.
- **Layer 2 — Blocklist**: Match bất kỳ nhóm → DROP.
- **Layer 3 — Default keep**: Không match cả hai → giữ (conservative, tránh mất data).

**Logic theo dataset:**
- **TopCV (strict)**: Blocklist groups: `graphic_visual`, `industrial_engineering`, `marketing_sales`, `content_media`, `cad_mechanical`, `non_tech_roles`.
- **ITViec (conservative)**: Blocklist groups: `graphic_visual`, `industrial_engineering`, `marketing_pure`, `sales_pure`, `content_media`, `non_tech_roles`. Safeguard rộng hơn (bao gồm cả IT Support, AI/ML/Data, Backend/Frontend keywords...).

**API công khai:**
```python
from main.pre_llm_title_filter import filter_non_it_jobs, classify_title

df_keep, df_dropped, df_review = filter_non_it_jobs(df, dataset="topcv", title_col="job_title")
decision, reason = classify_title("Senior DevOps Engineer", dataset="topcv")  # ("keep", None)
```

**Flag bổ sung:**
```bash
# Chỉ cập nhật filtered CSV với dữ liệu mới từ brand_recrawled rows (không chạy lại toàn bộ)
python main/00_pre_llm_title_filter.py --patch-recrawled --dataset topcv
```

---

### `main/01_process_job_title.py`

Gọi Gemini qua Vertex AI để standardize `job_title`, chạy theo batch với checkpoint.

**Tham số dòng lệnh:**
```
--dataset {topcv,itviec}   dataset cần xử lý (default: topcv)
--recrawled-only           chỉ xử lý các row có brand_recrawled=True
```

**Cơ chế resume**: Sau mỗi batch, kết quả được ghi vào `output/{dataset}_job_title_full.json`. Nếu script bị ngắt, lần chạy kế tiếp tự động skip các row đã có (match theo `url`).

---

### `main/02_merge_job_title.py`

Merge JSON output của LLM vào CSV đã filter theo `url`.

**Tham số dòng lệnh:**
```
--dataset {topcv,itviec,all}   dataset cần merge (default: all)
--recrawled-only               chỉ patch is_valid_job + standardized_title cho brand_recrawled rows
```

Hành vi merge:
- Row có `is_valid_job = False` → **bị drop** (không xuất hiện trong output)
- Row không có kết quả LLM (missing/error) → giữ lại với `standardized_title = ""`
- Row có `job_title` rỗng → bị skip

---

### `main/03_title_filter.py`

Module được import bởi notebook. Lọc `df` theo `standardized_title` **sau khi LLM đã standardize**.

```python
from main.title_filter import filter_by_standardized_title

df_keep, df_dropped = filter_by_standardized_title(df)
```

4 mask filter:
- `mask_keywords`: token-level match với `DROP_KEYWORDS` (word boundary `\b...\b`)
- `mask_exact_titles`: exact match với `EXACT_DROP_TITLES`
- `mask_sales`: chứa `\bsales\b` nhưng không phải `presales` / `sales engineer`
- `mask_account_manager`: exact match `"account manager"`

`df_dropped` có thêm cột `drop_reason` để audit (`keyword` / `exact_title` / `sales_non_it` / `account_manager`).

---

### `main/03_title_normalizer.py`

Module được import bởi notebook. Chuẩn hóa các variant title về canonical name.

```python
from main.title_normalizer import normalize_titles

df = normalize_titles(df)
```

**2 lớp mapping:**
1. `map_standardized_title()`: Rule-based theo token (e.g., `"tester"` / `"qa"` → `"QA Engineer"`, `"brse"` / `"bridge"` → `"Bridge Engineer"`)
2. `SEMANTIC_MERGE_MAP`: Dict exact-match (e.g., `"Systems Engineer"` → `"System Engineer"`, `"AI Specialist"` → `"AI Engineer"`)

---

### `main/03_salary_utils.py`

Module được import bởi notebook. Thêm flag phân tích salary.

```python
from main.salary_utils import add_salary_flags, get_valid_salary_mask

df = add_salary_flags(df)          # thêm: is_null_salary, is_thoa_thuan, is_missing_salary
mask = get_valid_salary_mask(df)   # True nếu salary có giá trị số hợp lệ
```

> Các cột flag này được drop trước khi lưu file cuối cùng.

---

## Output cuối cùng

| File | Mô tả |
|------|-------|
| `data/interim/01-standardized_title.csv` | TopCV + ITViec sau filter + normalize, có cột `source` phân biệt nguồn |
| `output/job_title_stats.csv` | Thống kê phân bố `standardized_title` trước khi filter |
| `output/job_title_stats_filtered.csv` | Thống kê phân bố `standardized_title` sau filter + normalize |

File `01-standardized_title.csv` chứa toàn bộ cột gốc của hai dataset cộng thêm `standardized_title` và `source`. Các cột `brand_recrawled`, `is_null_salary`, `is_thoa_thuan`, `is_missing_salary` được drop trước khi lưu.
