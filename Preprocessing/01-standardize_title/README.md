# 01 — Standardize Job Title

Chuẩn hóa cột `job_title` của TopCV và ITViec bằng LLM (Gemini via Vertex AI), sau đó filter và normalize title để tạo ra dataset sạch cho các bước preprocessing tiếp theo.

---

## Luồng xử lý tổng quan

```
data/raw/00-{dataset}_raw.csv
        │
        ▼
[Bước 0] 00_job_filter_topCV.py        ← chỉ chạy cho TopCV
        │   Lọc bỏ job không thuộc IT trước khi gửi LLM (tiết kiệm token)
        │
        ▼  output/00-topcv_filtered.csv
        │
[Bước 1] 01_process_job_title.py --dataset {topcv|itviec}
        │   Gọi Gemini standardize title theo batch, checkpoint sau mỗi batch
        │
        ▼  output/{dataset}_job_title_full.json
        │
[Bước 2] 02_merge_job_title.py --dataset {topcv|itviec}
        │   Merge kết quả JSON vào CSV gốc theo id
        │
        ▼  output/01-{dataset}_llm_standardized.csv
        │
[Bước 3] 01_standardize_title.ipynb
        │   Load cả hai dataset, merge, post-filter + normalize, split và save
        │
        ▼
data/interim/02-topcv_standardized_title.csv
data/interim/02-itviec_standardized_title.csv
```

---

## Cấu trúc thư mục

```
01-standardize_title/
├── main/
│   ├── 00_job_filter_topCV.py     # Bước 0 — pre-LLM filter (chỉ TopCV)
│   ├── 01_process_job_title.py    # Bước 1 — LLM standardization
│   ├── 02_merge_job_title.py      # Bước 2 — merge JSON → CSV
│   ├── 03_title_filter.py         # Module: post-LLM title filter (dùng bởi notebook)
│   ├── 03_title_normalizer.py     # Module: title normalization (dùng bởi notebook)
│   └── 03_salary_utils.py         # Module: salary flag utilities (dùng bởi notebook)
├── output/                         # Intermediate outputs (không commit)
├── prompt/
│   └── prompt_job_title.txt        # System prompt cho LLM
├── .env                            # Credentials Vertex AI (không commit)
└── 01_standardize_title.ipynb      # Bước 3 — notebook chính
```

---

## Thiết lập môi trường

Tạo file `.env` trong thư mục `01-standardize_title/`:

```env
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1
GEMINI_MODEL=gemini-2.0-flash-001
GOOGLE_APPLICATION_CREDENTIALS=credentials/service-account.json
```

Đặt file service account tại `01-standardize_title/credentials/service-account.json`.

---

## Cách chạy

### Bước 0 — Pre-LLM filter (chỉ TopCV)

Lọc bỏ job không thuộc IT/Tech dựa trên raw `job_title` trước khi gửi LLM.

```bash
python main/00_job_filter_topCV.py
```

| | |
|---|---|
| **Input** | `data/raw/00-topcv_raw.csv` |
| **Output** | `output/00-topcv_filtered.csv` — gửi vào LLM |
| | `output/00-topcv_dropped.csv` — bị loại |
| | `output/00-topcv_review.csv` — cần human review |
| | `output/00-topcv_dropped_stats.txt` — báo cáo lý do loại |

> ITViec không có bước này — tất cả rows được gửi thẳng vào LLM.

---

### Bước 1 — LLM Standardization

Gọi Gemini qua Vertex AI để standardize title. Tự động resume từ checkpoint nếu bị ngắt giữa chừng.

```bash
# TopCV
python main/01_process_job_title.py --dataset topcv

# ITViec (khi có data)
python main/01_process_job_title.py --dataset itviec
```

| | |
|---|---|
| **Input** | `data/raw/00-{dataset}_raw.csv` |
| **Prompt** | `prompt/prompt_job_title.txt` |
| **Output** | `output/{dataset}_job_title_full.json` (checkpoint per batch) |

> **Lưu ý**: Nếu có file checkpoint cũ từ lần chạy trước (`output/job_title_full.json`), đổi tên thành `output/topcv_job_title_full.json` trước khi chạy.

---

### Bước 2 — Merge JSON → CSV

Merge kết quả LLM vào CSV gốc dựa trên row index, thêm 2 cột `is_valid_job` và `standardized_title`.

```bash
# TopCV
python main/02_merge_job_title.py --dataset topcv

# ITViec (khi có data)
python main/02_merge_job_title.py --dataset itviec
```

| | |
|---|---|
| **Input** | `data/raw/00-{dataset}_raw.csv` |
| | `output/{dataset}_job_title_full.json` |
| **Output** | `output/01-{dataset}_llm_standardized.csv` |

---

### Bước 3 — Notebook: Post-process + Save

Mở và chạy toàn bộ `01_standardize_title.ipynb`. Notebook thực hiện:

1. Đọc raw data, thống kê missing (TopCV + ITViec nếu có)
2. Load `output/01-topcv_llm_standardized.csv` và `output/01-itviec_llm_standardized.csv`
3. Thêm cột `source`, merge cả hai thành một DataFrame
4. **Filter** bằng `filter_by_standardized_title()` — loại bỏ title không phải IT
5. **Normalize** bằng `normalize_titles()` — gộp các variant title về canonical name
6. Thống kê, visualize phân bố title và salary
7. Split theo `source`, loại bỏ `is_valid_job = False`, lưu riêng

| | |
|---|---|
| **Input** | `output/01-topcv_llm_standardized.csv` |
| | `output/01-itviec_llm_standardized.csv` (nếu có) |
| **Output** | `data/interim/02-topcv_standardized_title.csv` |
| | `data/interim/02-itviec_standardized_title.csv` |
| | `data/interim/job_title_stats.csv` (thống kê title distribution) |

---

## Chi tiết từng file

### `main/00_job_filter_topCV.py`

Lọc non-IT jobs từ raw TopCV **trước khi gửi LLM** để tiết kiệm token.

**Chiến lược 3 lớp:**
- **Layer 1 — Safeguard (whitelist)**: Match → giữ ngay, bỏ qua blocklist. Bảo vệ borderline IT roles: presales kỹ thuật, solution engineer, game designer, embedded design.
- **Layer 2 — Blocklist**: Match bất kỳ nhóm → DROP. Nhóm: `graphic_visual`, `industrial_engineering`, `marketing_sales`, `content_media`, `cad_mechanical`, `non_tech_roles`.
- **Layer 3 — Default keep**: Không match cả hai → giữ (conservative, tránh mất data).

**API công khai:**
```python
from main.job_filter_topCV import filter_non_it_jobs, classify_title

df_keep, df_dropped, df_review = filter_non_it_jobs(df, title_col="job_title")
decision, reason = classify_title("Senior DevOps Engineer")  # ("keep", None)
```

---

### `main/01_process_job_title.py`

Gọi Gemini qua Vertex AI để standardize `job_title`, chạy theo batch với checkpoint.

**Tham số dòng lệnh:**
```
--dataset {topcv,itviec}   dataset cần xử lý (default: topcv)
```

**Cơ chế resume**: Sau mỗi batch, kết quả được ghi vào `output/{dataset}_job_title_full.json`. Nếu script bị ngắt, lần chạy kế tiếp tự động skip các row đã có.

**Output mỗi row trong JSON:**
```json
{
  "id": 42,
  "original": "Kỹ Sư Phần Mềm Senior",
  "is_valid_job": true,
  "standardized_title": "Software Engineer"
}
```

---

### `main/02_merge_job_title.py`

Merge JSON output của LLM vào CSV gốc theo `id` (row index).

**Tham số dòng lệnh:**
```
--dataset {topcv,itviec}   dataset cần merge (default: topcv)
```

Thêm 2 cột vào CSV gốc:
- `is_valid_job`: `true` / `false` — LLM đánh giá có phải job IT không
- `standardized_title`: title đã được chuẩn hóa

---

### `main/03_title_filter.py`

Module được import bởi notebook. Lọc `df` theo `standardized_title` **sau khi LLM đã standardize**.

```python
from main.title_filter import filter_by_standardized_title

df_keep, df_dropped = filter_by_standardized_title(df)
```

Sử dụng 2 danh sách:
- `EXACT_DROP_TITLES`: set các title cần loại chính xác (e.g., `"designer"`, `"customer success"`)
- `DROP_KEYWORDS`: keyword pattern (e.g., `"graphic"`, `"marketing"`, `"content"`)
- Logic đặc biệt: loại `sales` nhưng giữ `presales` / `sales engineer`

---

### `main/03_title_normalizer.py`

Module được import bởi notebook. Chuẩn hóa các variant title về canonical name.

```python
from main.title_normalizer import normalize_titles

df = normalize_titles(df)
```

**2 lớp mapping:**
1. `map_standardized_title()`: Rule-based theo token (e.g., `"tester"` / `"qa"` → `"QA Engineer"`)
2. `SEMANTIC_MERGE_MAP`: Dict exact-match (e.g., `"Systems Engineer"` → `"System Engineer"`, `"AI Specialist"` → `"AI Engineer"`)

---

### `main/03_salary_utils.py`

Module được import bởi notebook. Thêm flag phân tích salary.

```python
from main.salary_utils import add_salary_flags, get_valid_salary_mask

df = add_salary_flags(df)          # thêm: is_null_salary, is_thoa_thuan, is_missing_salary
mask = get_valid_salary_mask(df)   # True nếu salary có giá trị số hợp lệ
```

---

## Output cuối cùng

| File | Mô tả |
|------|-------|
| `data/interim/02-topcv_standardized_title.csv` | TopCV sau khi filter + normalize title, bỏ job không hợp lệ |
| `data/interim/02-itviec_standardized_title.csv` | ITViec tương tự (tạo khi có data) |
| `data/interim/job_title_stats.csv` | Thống kê phân bố `standardized_title` trên cả hai dataset |

Cả hai file đều giữ nguyên toàn bộ cột gốc của dataset nguồn, không có cột `source` (đã drop trước khi save).
