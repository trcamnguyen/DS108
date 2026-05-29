# 02 Skill Extraction — Tài liệu Pipeline

Giai đoạn này nhận dataset job posting đã chuẩn hóa tiêu đề, trích xuất skill từ mô tả việc làm bằng LLM, canonical hóa các biến thể, và xuất ra hai file parquet sẵn sàng cho modeling.

---

## Tổng quan flow

```
data/interim/01-standardized_title.csv   (3,181 jobs)
        │
        ▼
[Bước 0] calibration/           ← IAA calibration (50 samples, 3 annotators)
        │
        ▼
[Bước 1] 02_full_extraction.py  ← Gọi Gemini API, ghi checkpoint JSONL
        │
        ▼   output_full/full_raw.jsonl
        │
[Bước 2] 03_parse_jsonl.py      ← Parse checkpoint → CSV (chạy độc lập để re-parse)
        │
        ▼   output_full/full_parsed.csv  (56,000+ skill rows)
        │
[Bước 3] bootstrap_aliases.py   ← Gọi Gemini tạo aliases.yaml (optional, 1 lần)
        │
        ▼   outputs/aliases.yaml
        │
[Bước 4] 04_cluster_skills.py   ← Embedding + clustering → canonical mapping
        │
        ▼   outputs/annotations_with_canonical.csv
        │
[Bước 5] 05_build_parquets.py   ← Gắn job_id, xuất parquet cuối
        │
        ▼
data/interim/02-skill_extracted/
    jobs.parquet    (1 row/job)
    skills.parquet  (1 row/skill mention)
```

---

## Cấu trúc thư mục

```
02_skill_extraction/
├── calibration/                    # Bước 0 — IAA
│   ├── 01_sample_calibration.py    # Lấy 50 mẫu ngẫu nhiên từ dataset
│   ├── 02_llm_skill_extraction.py  # Chạy LLM trên 50 mẫu calibration
│   ├── 03_visualize_results.py     # Visualize kết quả IAA
│   ├── calibration_dataset.csv     # 50 mẫu calibration
│   ├── annotated_skills_kngoc.json # Annotation Human B
│   ├── iaa/                        # IAA framework (kappa, matching, bootstrap)
│   ├── iaa_results/                # Kết quả IAA (confusion matrix, report JSON)
│   └── output/
│       ├── annotated_skills_cnguyen.json  # Annotation Human A / LLM
│       ├── few_shot_parsed.csv
│       └── few_shot_raw.jsonl
│
├── prompt/
│   ├── prompt_skill_extraction.txt # Prompt v4 — LOCKED (system + few-shot)
│   ├── prompt_skill_extraction_v1.txt
│   ├── prompt_skill_extraction_v2.txt
│   └── prompt_taxonomy.txt         # Prompt cho bootstrap_aliases.py
│
├── output_full/                    # Output của bước extraction
│   ├── full_raw.jsonl              # Checkpoint: 1 line/job (raw API response)
│   ├── full_parsed.csv             # Flat: 1 row/skill mention
│   ├── full_errors.jsonl           # Jobs thất bại sau 3 retry
│   └── extraction.log
│
├── outputs/                        # Output của bước clustering
│   ├── aliases.yaml                # Alias map từ bootstrap_aliases.py
│   ├── full_parsed_aliased.csv     # full_parsed + cột skill_aliased (audit)
│   ├── clusters_review.csv         # 1 row/cluster — file audit chính
│   ├── canonical_mapping.csv       # 1 row/(raw_skill, category) — file override
│   ├── skill_distribution_after.csv# 1 row/final_canonical
│   ├── cross_category_review.csv   # Filter các cluster cross-category
│   ├── cross_category_ties.csv     # Cluster cần tie-break thủ công
│   ├── annotations_with_canonical.csv  # full_parsed + cột final_canonical
│   └── clustering_report.json      # Metadata run
│
├── cache/
│   ├── embeddings.npy              # Cache embedding (sha256-keyed)
│   └── embeddings_key.txt          # Key để invalidate cache
│
├── category_skill_statistics/      # CSV thống kê per-category (từ KiemTraData.ipynb)
│
├── 02_full_extraction.py           # Bước 1 — LLM extraction
├── 03_parse_jsonl.py               # Bước 2 — Parse checkpoint
├── bootstrap_aliases.py            # Bước 3 — Tạo aliases.yaml
├── 04_cluster_skills.py            # Bước 4 — Clustering
├── 05_build_parquets.py            # Bước 5 — Xuất parquet
├── bootstrap_aliases.py
├── role_block.txt                  # Danh sách job title bị loại khỏi skill
├── category_overide.yaml           # Override category cho cross-category ties
├── cluster.md                      # Spec clustering (chi tiết)
├── alias.md                        # Spec bootstrap aliases
└── KiemTraData.ipynb               # Notebook kiểm tra data
```

---

## Bước 0 — Calibration IAA

**Mục đích:** Đo inter-annotator agreement trước khi annotation toàn dataset.

**Setup:** 3 annotator: Human A, Human B, Few-shot LLM. 50 samples ngẫu nhiên từ dataset.

**Target:** Cohen's Kappa ≥ 0.75 (Human vs Human), ≥ 0.70 (LLM vs Human).

```bash
# Lấy mẫu calibration
python calibration/01_sample_calibration.py

# Chạy LLM trên 50 mẫu (cần .env với GOOGLE_CLOUD_PROJECT)
python calibration/02_llm_skill_extraction.py

# Visualize kết quả
python calibration/03_visualize_results.py
```

**Output:** `calibration/iaa_results/iaa_report.json`, confusion matrix PNG.

---

## Bước 1 — Full Extraction (`02_full_extraction.py`)

**Mục đích:** Chạy LLM extraction toàn bộ 3,181 jobs. Output là checkpoint JSONL để resume nếu bị ngắt.

### Input
- `data/interim/01-standardized_title.csv` — 3,181 rows, cột `requirement` là text đầu vào
- `prompt/prompt_skill_extraction.txt` — system prompt + few-shot examples (phân tách bởi `===FEW_SHOT===`)

### Output (`output_full/`)
| File | Mô tả |
|------|-------|
| `full_raw.jsonl` | Checkpoint: 1 line/job. Key: `row_id` (= URL), `raw_text`, `error`, `source`, `url` |
| `full_parsed.csv` | Flat: 1 row/skill. Tạo từ `full_raw.jsonl` qua `_flush_parsed_csv()` |
| `full_errors.jsonl` | Jobs thất bại sau 3 retry |
| `extraction.log` | Log đầy đủ |

### Cách chạy

```bash
# Cần file .env ở project root với:
#   GOOGLE_CLOUD_PROJECT=your-project-id
#   GOOGLE_CLOUD_LOCATION=us-central1
#   GEMINI_MODEL=gemini-2.5-pro
#   GOOGLE_APPLICATION_CREDENTIALS=credentials/service-account.json

python Preprocessing/02_skill_extraction/02_full_extraction.py
```

### Logic chính

**Gemini Context Caching:** System prompt + few-shot examples được cache 1 lần qua `CachedContent` (TTL 24h). Mỗi API call chỉ gửi input mới, tiết kiệm ~39M tokens/run.

**Token Bucket Rate Limiter:** `_TokenBucket(TARGET_RPM=25)` — thread-safe, giới hạn 25 requests/phút. Mỗi worker `acquire()` trước khi gọi API.

**Producer-Consumer Pattern:**
- `ThreadPoolExecutor(max_workers=2)` — worker thread gọi API song song
- `queue.Queue` — worker đẩy kết quả vào queue
- 1 writer thread đọc queue, ghi file tuần tự (tránh race condition trên file I/O)

**Resume từ checkpoint:** `_load_processed_ids()` đọc `full_raw.jsonl`, collect các `row_id` đã thành công. Re-run chỉ xử lý các row chưa có hoặc có lỗi.

**JSON Repair pipeline:**
```
API response
  → json.loads()          ← parse thẳng
  → nếu fail: _repair_truncated_json()   ← strip markdown, fix trailing comma, thử cắt tại } cuối
  → nếu fail: _coerce_skills()           ← ép label/level về valid values
  → nếu fail sau 3 retry → ghi full_errors.jsonl
```

**Schema validation:** Pydantic `SkillExtractionOutput` — 6 fields: `skill_name`, `label`, `category`, `min_years`, `level`, `source_text`.

**Empty requirement handling:** Job có `requirement` rỗng → ghi thẳng vào checkpoint với `error="empty_requirement"`, không submit worker.

---

## Bước 2 — Parse Checkpoint (`03_parse_jsonl.py`)

**Mục đích:** Parse lại `full_raw.jsonl` → `full_parsed.csv` độc lập với extraction script. Hữu ích khi cần re-parse sau khi fix schema hoặc khi extraction bị interrupt.

```bash
python Preprocessing/02_skill_extraction/03_parse_jsonl.py
```

Logic giống hệt `_flush_parsed_csv()` trong `02_full_extraction.py`. Job có `error` nhưng vẫn có `raw_text` → thử parse lại (vd `min_years=0.5` float có thể được coerce).

---

## Bước 3 — Bootstrap Aliases (`bootstrap_aliases.py`)

**Mục đích:** Gọi Gemini 2.5 Pro để tự động phát hiện các skill variants cùng concept và tạo `aliases.yaml`. Chạy **1 lần** trước clustering.

### Input
`outputs/skill_distribution_after.csv` — filter `total_count >= 5`

### Output
`outputs/aliases.yaml` — alias map dạng:
```yaml
aliases:
  - canonical: machine learning
    variants: [ml, mô hình học máy, ai machine learning]
```

```bash
python Preprocessing/02_skill_extraction/bootstrap_aliases.py \
    --input outputs/skill_distribution_after.csv \
    --output outputs/aliases.yaml \
    --min-count 5 \
    --batch-size 200
```

### Logic

1. Filter skill `total_count >= 5`, sort theo `category ASC, count DESC`
2. Chia batch 200 skill/batch, format mỗi skill: `{name} | count={n} | category={cat}`
3. Gọi Gemini → parse YAML response
4. **Validate:** canonical phải tồn tại trong input (không hallucinate), variant phải trong input, không circular
5. **Auto-promote:** nếu canonical bị hallucinate → promote variant có count cao nhất làm canonical
6. **Merge batches:** union variants, resolve circular (ưu tiên canonical count cao hơn), dedupe

---

## Bước 4 — Clustering (`04_cluster_skills.py`)

**Mục đích:** Canonical hóa ~56,000 raw skill mentions thành tập canonical nhỏ hơn bằng multilingual embeddings + agglomerative clustering.

### Input
- `output_full/full_parsed.csv` — raw skill mentions từ LLM
- `outputs/aliases.yaml` — alias map (optional)
- `role_block.txt` — danh sách job title bị loại (optional)
- `category_overide.yaml` — override category cho cross-category ties (optional)

### Output (`outputs/`)

| File | Mô tả |
|------|-------|
| `annotations_with_canonical.csv` | Deliverable chính: full_parsed + `skill_normalized` + `final_canonical` |
| `clusters_review.csv` | 1 row/cluster — audit toàn bộ clustering |
| `canonical_mapping.csv` | 1 row/(raw_skill, category) — **file user override** |
| `skill_distribution_after.csv` | 1 row/final_canonical — thống kê sau gom |
| `cross_category_review.csv` | Filter cluster cross-category |
| `cross_category_ties.csv` | Cluster có tie (cần tie-break thủ công) |
| `clustering_report.json` | Metadata: threshold, model, count, runtime, config_hash |
| `full_parsed_aliased.csv` | Audit: full_parsed sau áp aliases (trước clustering) |

### Cách chạy

```bash
# Run đầu tiên (không có aliases, không override)
python Preprocessing/02_skill_extraction/04_cluster_skills.py \
    --input Preprocessing/02_skill_extraction/output_full/full_parsed.csv \
    --output-dir Preprocessing/02_skill_extraction/outputs/ \
    --job-id-col row_id

# Run với aliases + role_block + category override (run thực tế)
python Preprocessing/02_skill_extraction/04_cluster_skills.py \
    --input Preprocessing/02_skill_extraction/output_full/full_parsed.csv \
    --output-dir Preprocessing/02_skill_extraction/outputs/ \
    --aliases Preprocessing/02_skill_extraction/outputs/aliases.yaml \
    --save-aliased Preprocessing/02_skill_extraction/outputs/full_parsed_aliased.csv \
    --min-keep-count 5 \
    --short-threshold 7 \
    --role-block Preprocessing/02_skill_extraction/role_block.txt \
    --category-override Preprocessing/02_skill_extraction/category_overide.yaml \
    --job-id-col row_id

# Re-run với manual override (sau khi user sửa canonical_mapping.csv)
python Preprocessing/02_skill_extraction/04_cluster_skills.py \
    --input Preprocessing/02_skill_extraction/output_full/full_parsed.csv \
    --output-dir Preprocessing/02_skill_extraction/outputs/ \
    --apply-overrides Preprocessing/02_skill_extraction/outputs/canonical_mapping.csv \
    --job-id-col row_id
```

### Logic chi tiết

**Pre-normalization (`normalize_skill`):**
- Lowercase, strip, bỏ prefix tiếng Việt (`kỹ năng `, `kinh nghiệm `, `hiểu biết về `, ...)
- Replace `/`, `_` → space; `-` giữa alphanumeric → space
- Xử lý `.`: giữ nếu giữa 2 chữ cái VÀ độ dài chuỗi < 6 (vd `a.i`); còn lại replace → space

**Short-string bypass (CRITICAL):**
Skills có `len(skill_normalized) <= short_threshold` (default=7) **không đi qua embedding** vì multilingual MiniLM tạo vector nhiễu cho string ngắn, gây "catch-all cluster" gom các acronym không liên quan.
- Short pool → exact-match consolidation (cùng normalized string = 1 cluster)
- Long pool → embedding + agglomerative clustering

**Embedding:**
- Model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (~120MB)
- `normalize_embeddings=True` (L2-norm, để cosine = dot product)
- Cache: `cache/embeddings.npy` + key file sha256(`sorted(skills) + model_name`). Cache hit → skip re-encode

**Agglomerative Clustering:**
- `metric="cosine"`, `linkage="average"`, `distance_threshold=0.15`
- `n_clusters=None` (số cluster tự xác định theo threshold)
- Average linkage cân bằng giữa single (chain effect) và complete (quá chặt)

**Aggregation:** Key là `(skill_normalized, category)` — giữ category vì Codebook §4.1 quy định một số skill có nghĩa khác theo category (vd MLOps ở Infra ≠ MLOps ở AI/ML/Data).

**Pick canonical:**
1. Ưu tiên member chỉ chứa ASCII (EC-07: ưu tiên EN khi song ngữ)
2. Trong pool ASCII, chọn `total_count` cao nhất
3. Tie-break: string ngắn hơn → alphabetical

**Long-tail bucket:** Singleton cluster (`n_members == 1`) với `total_count < min_keep_count` → `final_canonical = "{Category} Other"`. Threshold tunable qua `--min-keep-count`.

**Cross-category clusters:** Cluster có skill từ nhiều category → `is_cross_category = True`. Default behavior: merge theo dominant category (dominant vote). Các cluster có tie (top 2 category có count bằng nhau) → ghi vào `cross_category_ties.csv`, user resolve qua `category_overide.yaml`.

**Role-block filter:** Sau clustering, drop cluster có `final_canonical` match với entry trong `role_block.txt` (job title không phải skill, vd "software engineer", "data analyst").

**Sanity checks (assert):**
1. Mention conservation: `sum(total_count)` trước = sau clustering
2. Mapping completeness: mọi `(raw_skill_name, category)` trong input đều có row trong `canonical_mapping.csv`
3. `final_canonical` không null/empty
4. Category column trong `annotations_with_canonical.csv` phải giữ nguyên so với input

### Workflow override

User muốn sửa canonical thủ công:
1. Mở `outputs/canonical_mapping.csv`
2. Fill cột `override_canonical` cho các row muốn thay đổi
3. Re-run với `--apply-overrides outputs/canonical_mapping.csv`

Embedding đã cache → re-run < 30 giây.

---

## Bước 5 — Build Parquets (`05_build_parquets.py`)

**Mục đích:** Gắn `job_id` integer (từ raw dataset) vào skills, xuất 2 file parquet sẵn sàng cho modeling.

### Input
- `data/interim/01-standardized_title.csv` — raw dataset
- `outputs/annotations_with_canonical.csv` — skills đã canonical hóa

### Output (`data/interim/02-skill_extracted/`)

**`jobs.parquet`** — 1 row/job, 18 cột:

| Cột | Mô tả |
|-----|-------|
| `job_id` | Integer, tăng dần (1-based), tạo mới |
| `url` | URL gốc |
| `job_title`, `company`, `location`, `salary`, ... | Các cột job-level từ raw |
| `job_level` | Đổi tên từ `level` (tránh trùng với `level` trong skills) |
| `platform_required_skills` | Đổi tên từ `required_skills` |
| `platform_preferred_skills` | Đổi tên từ `preferred_skills` |
| `standardized_title`, `source` | Từ pipeline bước 1 |

**`skills.parquet`** — 1 row/skill mention, 8 cột:

| Cột | Mô tả |
|-----|-------|
| `job_id` | FK tới `jobs.parquet` (join qua URL) |
| `skill_name` | Raw từ LLM |
| `final_canonical` | Sau clustering |
| `label` | `required_skill` / `preferred_skill` |
| `category` | 1 trong 15 category |
| `level` | `expert` / `intermediate` / `basic` / null |
| `min_years` | Nullable integer (Float64) |
| `source_text` | Đoạn JD gốc |

### Cách chạy

```bash
# Chạy với default paths (từ project root)
python Preprocessing/02_skill_extraction/05_build_parquets.py

# Chỉ định path thủ công
python Preprocessing/02_skill_extraction/05_build_parquets.py \
    --raw-csv data/interim/01-standardized_title.csv \
    --skills-csv Preprocessing/02_skill_extraction/outputs/annotations_with_canonical.csv \
    --output-dir data/interim/02-skill_extracted
```

### Logic

1. Đọc raw CSV → dedup theo URL (keep last) → gán `job_id` sequential 1-based
2. Rename các cột (`level` → `job_level`, `required/preferred_skills` → `platform_*`)
3. Assert `job_id` unique và không null
4. Đọc skills CSV → drop `job_id` cũ → map URL → `job_id` mới từ jobs
5. Assert tất cả `job_id` trong skills tồn tại trong jobs
6. Chỉ giữ 8 cột cần thiết trong skills

**13 jobs không có skill extraction:** Những job này có URL trong `jobs.parquet` nhưng không có row trong `skills.parquet` (requirement rỗng hoặc extraction thất bại). Xem notebook cell "Jobs không có skill extraction" trong `KiemTraData.ipynb`.

---

## Dependencies

```
google-genai>=0.3          # Gemini API (Vertex AI)
pydantic>=2.0              # Schema validation
sentence-transformers>=2.7 # Multilingual MiniLM embedding
scikit-learn>=1.3          # AgglomerativeClustering
pandas>=2.0
numpy>=1.24
pyyaml                     # YAML parsing (aliases.yaml)
tqdm                       # Progress bar
pyarrow                    # Parquet I/O
```

Lần đầu chạy `04_cluster_skills.py` sẽ download model ~120MB từ HuggingFace. Cache local sau đó.

---

## Environment

File `.env` ở project root:
```
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1
GEMINI_MODEL=gemini-2.5-pro
GOOGLE_APPLICATION_CREDENTIALS=credentials/service-account.json
```

---

## Các quyết định thiết kế quan trọng

| Quyết định | Lý do |
|------------|-------|
| Dùng URL làm `row_id` trong checkpoint | Stable qua re-sort; không cần tạo ID trước khi chạy |
| Tách short-string (≤7 chars) khỏi embedding | Run pilot phát hiện MiniLM tạo catch-all cluster cho acronym ngắn |
| Key aggregation là `(skill_normalized, category)` | Codebook §4.1: một số skill có nghĩa khác theo category |
| Canonical pick ưu tiên ASCII | EC-07: ưu tiên EN khi song ngữ |
| Default merge cross-category | ~80% cross-cat là LLM mis-categorize; merge theo dominant vote là đúng |
| `job_id` trong `05_build_parquets.py` tạo lại từ raw | job_id trong skills CSV là từ bước extraction cũ; tạo lại đảm bảo consistent với jobs.parquet |
| `min_years` kiểu Float64 (nullable) | `convert_dtypes()` trên pandas 1.x không cast float64→Int64 trực tiếp |
