# Skill Canonical Clustering — Implementation Spec

**Project:** DS108 — IT Job Salary Prediction
**Task:** Canonical hóa raw skill_name từ LLM extraction (post-processing)
**Codebook reference:** v1.5 — EC-07 (raw extraction), EC-07 NGOẠI LỆ 2 (ưu tiên EN), §4.1 (boundary rules)
**Status:** Spec finalized — sẵn sàng để implement

---

## 1. Mục tiêu

Sau khi LLM extract ~17.000–33.000 skill mentions từ 3.344 JD theo Codebook v1.5 (raw, không canonical hóa), bộ skill có **phân phối Zipf long-tail nặng**: ~69% skill chỉ xuất hiện 1 lần, biến thể EN/VN trộn, viết tắt, plural, suffix lặt vặt.

Pipeline này gom các biến thể của *cùng một skill* về một canonical name duy nhất, phục vụ:

- **Salary prediction model:** giảm sparsity của one-hot feature.
- **EDA thị trường:** thống kê demand chính xác.

Đây là **bước post-processing** mà Codebook EC-07 đã reserve — không thay đổi raw extraction.

---

## 2. Input

### 2.1 File CSV gốc

Một file CSV (path do user cung cấp khi chạy, mặc định `./data/skills_raw.csv`) với schema mỗi dòng = một skill mention đã extract từ một JD:

| Column | Type | Mô tả |
|---|---|---|
| `job_id` | str | ID của job posting |
| `skill_name` | str | Raw skill name từ LLM (KHÔNG normalize) |
| `label` | str | `required_skill` hoặc `preferred_skill` |
| `category` | str | Một trong 14 category của Codebook §4 |
| `min_years` | int / null | Optional |
| `level` | str / null | Optional |
| `source_text` | str | Đoạn JD source — KHÔNG dùng trong pipeline này |

**Lưu ý:** file có thể có thêm column khác (vd `dataset_source`: topcv/itviec); script phải tolerant — chỉ require 4 column cốt lõi `job_id`, `skill_name`, `label`, `category`.

### 2.2 Filter

- Drop dòng nào `skill_name` null hoặc empty sau strip.
- Drop dòng nào `category` null. (Theo Codebook §9, category bắt buộc — nếu vẫn có null là lỗi extraction cần báo cáo, không impute.)
- **KHÔNG filter theo label** — cluster cả required + preferred chung.
- **KHÔNG filter theo category** — cluster tất cả 14 category cùng một lần (strategy B đã chốt).

### 2.3 Sanity check input

Trước khi xử lý, script in ra:

- Tổng số dòng (mention) đầu vào.
- Số job_id unique.
- Số skill_name unique (sau strip whitespace cơ bản).
- Phân phối theo category (count mention + count unique skill).
- Số dòng bị drop và lý do.

---

## 3. Pre-normalization

Mục tiêu: chuẩn hóa **bề mặt** trước khi embed để giảm biến thể không mang nghĩa. **GIỮ tiếng Việt có dấu** — multilingual model handle được, và xóa dấu sẽ phá semantic.

### 3.1 Quy tắc normalize (theo thứ tự áp dụng)

Trên `skill_name`:

1. **Lowercase** toàn bộ string.
2. **Strip whitespace** đầu/cuối.
3. **Strip prefix descriptor** (EC-07 NGOẠI LỆ 3):
   - Đầu chuỗi nếu match một trong các prefix sau (case-insensitive sau bước 1) → bỏ:
     - `kỹ năng `, `khả năng `, `kĩ năng `
     - `kinh nghiệm `, `có kinh nghiệm với `, `có kinh nghiệm `
     - `hiểu biết về `, `kiến thức về `, `am hiểu `
   - Áp dụng *một lần*, không loop. Nếu chuỗi rỗng sau khi strip → giữ nguyên skill gốc, không strip.
4. **Replace separator** thành space:
   - `/`, `_`, `.` (giữa từ) → space
   - `-` giữa 2 ký tự alphanumeric → space (vd `fine-tuning` → `fine tuning`, nhưng `c++` không bị ảnh hưởng)
   - Giữ `.` chỉ khi là phần của tên acronym ngắn không có space xung quanh (vd `a.i`, `node.js`). Đơn giản nhất: chỉ replace `.` khi có space hoặc digit liền kề; với pattern `\w\.\w` ngắn, có thể giữ. **Quyết định an toàn:** thay tất cả `.` trừ trường hợp giữa 2 chữ cái và độ dài chuỗi < 6 ký tự (vd `a.i`, `c.s` giữ, nhưng `node.js` → `node js`).
   - Strip `.` ở cuối chuỗi.
5. **Collapse whitespace** — nhiều space → 1 space; strip lại đầu/cuối.

### 3.2 Ví dụ kết quả normalize

| Raw skill_name | Sau normalize |
|---|---|
| `Machine Learning` | `machine learning` |
| `Kỹ năng giao tiếp` | `giao tiếp` |
| `ReactJS` | `reactjs` |
| `React.js` | `react js` |
| `kinh nghiệm với Python` | `python` |
| `AI/ML` | `ai ml` |
| `fine-tuning` | `fine tuning` |
| `a.i` | `a.i` (giữ — chuỗi ngắn, giữa 2 chữ cái) |
| `Node.js` | `node js` |
| `C++` | `c++` |
| `Tiếng Anh` | `tiếng anh` |

### 3.3 Lưu ý quan trọng

- **KHÔNG strip suffix** (vd `framework`, `library`, `js`, plural `s`). Để embedding model handle — strip suffix bằng rule cứng dễ gây false merge (`flask` ≠ `flasks` nếu chỉ strip `s`).
- **KHÔNG remove dấu tiếng Việt.** Multilingual MiniLM được train trên cả tiếng Việt có dấu.
- **KHÔNG tự dịch EN/VN** ở bước này. Việc bridge `học máy` ↔ `machine learning` thuộc về embedding clustering (bước 5).
- Pre-normalize có thể tạo duplicate sau normalize (vd `AI/ML` và `AI ML` đều thành `ai ml`). Sau bước này, **dedupe** theo `(skill_normalized, category)` rồi tổng count — KHÔNG cluster các string identical lặp lại nhiều lần.

---

## 4. Aggregation trước khi embed

Sau pre-normalize, group dữ liệu thành:

```
KEY: (skill_normalized, category)
VALUE:
  - total_count: số mention
  - n_required: count với label = required_skill
  - n_preferred: count với label = preferred_skill
  - raw_variants: list các raw skill_name ban đầu đã collapse vào đây
  - job_ids: set các job_id (để debug, optional)
```

**Lý do (skill_normalized, category) là composite key:**

Codebook §4.1 BOUNDARY RULES quy định một số skill có vị trí phụ thuộc category — `MLOps` ở `AI/ML/Data` có thể khác nghĩa với `MLOps` ở `Infrastructure & DevOps` (training vs deployment). Giữ category là metadata cho phép phát hiện cross-category clusters (bước 7).

Mỗi `(skill_normalized, category)` unique → một row sẽ được embed.

---

## 4.5 Short-string bypass (CRITICAL — fix failure mode đã observe)

### 4.5.1 Background

Run trước đó với threshold 0.15 chặt vẫn tạo ra "short-string catch-all clusters" — cluster #69 gom 209 acronym ngắn không liên quan (`java`, `iot`, `siem`, `vb`, `qa`, `ui`...). Nguyên nhân: embedding multilingual MiniLM trên string cực ngắn (≤5 ký tự, thường là acronym kỹ thuật không phải từ tiếng Anh thật) tạo ra **vector nhiễu dồn cụm trong vùng nhỏ của embedding space** → cosine distance giữa chúng đều < 0.15 dù ngữ nghĩa hoàn toàn khác nhau.

### 4.5.2 Rule bypass

**Skill có `len(skill_normalized) <= 5` được TÁCH RIÊNG, KHÔNG đi qua embedding pipeline.**

Cách xử lý:

```
SHORT_THRESHOLD = 5

input_pool = aggregated_skills  # từ bước 4
short_pool = [s for s in input_pool if len(s.skill_normalized) <= SHORT_THRESHOLD]
long_pool  = [s for s in input_pool if len(s.skill_normalized) > SHORT_THRESHOLD]
```

### 4.5.3 Xử lý long_pool (≥ 6 ký tự)

Đi qua embedding + agglomerative clustering như Section 5-6 hiện tại. Không thay đổi.

### 4.5.4 Xử lý short_pool (≤ 5 ký tự) — exact-match consolidation

Short strings KHÔNG cluster bằng embedding, mà:

1. **Group by `skill_normalized` (case-insensitive exact match).** Cùng string sau normalize → cùng cluster.
2. **Mỗi unique short string = 1 cluster riêng.** Cluster_id được assign sequentially sau khi clustering long_pool xong.
3. **Canonical name = skill_normalized.** Không pick canonical phức tạp (chỉ có 1 string).
4. **Category handling:** nếu cùng short string xuất hiện ở nhiều category (vd `react` ở `Framework / Library` và `Tool & Platform`), gom thành 1 cluster, mark `is_cross_category = True` như cluster long thông thường. Suggested action vẫn tính bình thường.

**Ví dụ:**

Input short_pool:
```
("react", Framework, count=80)
("react", Tool, count=2)
("vue", Framework, count=45)
("redis", Database, count=30)
("siem", Tool, count=8)
("siem", Infrastructure, count=3)
```

→ Output clusters:

```
Cluster A: canonical="react",  members=[(react,Framework,80), (react,Tool,2)],   total=82,  cross_cat=True
Cluster B: canonical="vue",    members=[(vue,Framework,45)],                      total=45,  cross_cat=False
Cluster C: canonical="redis",  members=[(redis,Database,30)],                     total=30,  cross_cat=False
Cluster D: canonical="siem",   members=[(siem,Tool,8), (siem,Infrastructure,3)], total=11,  cross_cat=True
```

### 4.5.5 Trade-off đã chấp nhận

- ❌ **Mất khả năng gom `react` ↔ `reactjs`** (5 vs 7 ký tự — `reactjs` ở long_pool, `react` ở short_pool, không bao giờ cùng cluster).
- ❌ **Mất khả năng gom `vue` ↔ `vuejs`**, `node` ↔ `nodejs`, `next` ↔ `nextjs`.
- ✅ **Bù trừ:** trong long_pool, embedding vẫn gom được `reactjs` ↔ `react.js` ↔ `react native` (đều ≥ 6 chars). Trong short_pool, exact match vẫn dedupe đúng `react` → 1 cluster.
- ✅ **Loại bỏ catch-all disaster.** Cluster #69 thảm họa không bao giờ tái phát.

### 4.5.6 Mitigation cho trade-off (optional, recommended)

Để bù mất gom `react` ↔ `reactjs`, thêm **post-clustering manual alias step** qua override file:

User mở `canonical_mapping.csv`, fill override để merge thủ công các cặp short ↔ long mà mình muốn:

```csv
raw_skill_name,category,skill_normalized,auto_canonical,override_canonical
react,Framework / Library,react,react,reactjs
react.js,Framework / Library,react js,reactjs,reactjs
reactjs,Framework / Library,reactjs,reactjs,reactjs
```

Top alias candidate cần merge (theo observation từ run trước):

| Short | Long candidate |
|---|---|
| `react` | `reactjs` |
| `vue` | `vuejs` / `vue js` |
| `node` | `nodejs` / `node js` |
| `next` | `nextjs` / `next js` |
| `nuxt` | `nuxtjs` |
| `nest` | `nestjs` |
| `sql` | `mysql`, `postgresql`... (KHÔNG merge — `sql` là concept, các DB engine là tool riêng) |

Đây là step **user thực hiện sau khi review run output đầu tiên**, không phải step tự động. Spec không enforce — chỉ document recommendation.

### 4.5.7 Cấu hình threshold

`SHORT_THRESHOLD` phải là CLI argument:

```bash
python cluster_skills.py --short-threshold 5
```

Default = 5. User có thể giảm xuống 3 (loose) hoặc tăng lên 6-7 nếu muốn experiment.

### 4.5.8 Logging

Trong `clustering_report.json`:

```json
"short_string_bypass": {
  "threshold": 5,
  "n_short_skills_separated": 1234,
  "n_short_clusters_created": 567,
  "n_long_skills_clustered": 14000,
  "rationale": "Short strings (<=5 chars) routed to exact-match consolidation to avoid catch-all clusters from noisy embeddings."
}
```

Console output:

```
[Short bypass] 1234 skills (<=5 chars) → 567 exact-match clusters
[Embedding] 14000 skills (>5 chars) → 3257 semantic clusters
[Total] 3824 clusters
```

---

## 5. Embedding

### 5.1 Model

- **Name:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **Reason:** Paraphrase task fit với matching skill variants; multilingual cover VN; size nhẹ (~120MB) cho iteration nhanh; không có prefix gotcha.

### 5.2 Code reference

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
embeddings = model.encode(
    skill_strings,
    normalize_embeddings=True,   # quan trọng: để dùng cosine distance như dot product
    show_progress_bar=True,
    batch_size=64,
)
```

### 5.3 Yêu cầu

- `normalize_embeddings=True` (chuẩn hóa L2-norm — bắt buộc để cosine distance hoạt động đúng).
- Cache embedding ra file (`./cache/embeddings.npy`) để không phải re-embed khi tune threshold. Cache key bao gồm hash của list skill strings + model name.

---

## 6. Clustering

### 6.1 Algorithm

**Agglomerative clustering** với:

- `metric = "cosine"`
- `linkage = "average"`
- `distance_threshold = 0.15` (chặt — đã chốt)
- `n_clusters = None`

```python
from sklearn.cluster import AgglomerativeClustering
clusterer = AgglomerativeClustering(
    n_clusters=None,
    distance_threshold=0.15,
    metric="cosine",
    linkage="average",
)
labels = clusterer.fit_predict(embeddings)
```

### 6.2 Vì sao `average` linkage thay vì `single` hay `complete`

- `single` linkage gây chain effect (gom dài) — risk với threshold 0.15 chặt: có thể tạo cluster lớn không mong muốn.
- `complete` linkage quá chặt — risk split các variant hợp lệ.
- `average` cân bằng — phù hợp với skill variants có distance distribution không đều.

### 6.3 Threshold làm tham số config

Threshold 0.15 là default. Script phải expose qua CLI argument hoặc config:

```bash
python cluster_skills.py --distance-threshold 0.15 --input data/skills_raw.csv
```

Lý do: user có thể re-run với threshold khác (vd 0.20) để compare. Embedding đã cache nên re-cluster rất nhanh.

---

## 7. Cross-category cluster detection & handling

### 7.1 Detection

Sau clustering, **với mỗi cluster_id**, check:

```python
unique_categories_in_cluster = set(category for member in cluster)
is_cross_category = len(unique_categories_in_cluster) > 1
```

### 7.2 Phân loại 4 kịch bản

Cross-category cluster có thể rơi vào 4 trường hợp thực tế:

| Kịch bản | Mô tả | Frequency | Action |
|---|---|---|---|
| **K1 — LLM mis-categorize** | Cùng skill, LLM gán nhầm category ở 1 số JD. Vd `docker` (Infra=150) + `docker` (AI/ML/Data=3) | ~80% | **MERGE** thành 1 canonical, dùng dominant category |
| **K2 — Homonym theo §4.1** | Cùng tên, khác nghĩa theo Codebook BOUNDARY RULES. Vd `mlops` (Infra=20) + `mlops` (AI/ML/Data=15) | ~10% | **SPLIT** — 2 canonical riêng, kèm category suffix |
| **K3 — Skill cùng họ** | Embedding gom skill related nhưng không phải cùng skill. Hiếm với threshold 0.15. | ~7% | **REVIEW** — quyết định case-by-case |
| **K4 — Homonym khác hoàn toàn** | Vd `go` (Programming) + `go` (Soft Skill) | ~3% | **SPLIT** hoàn toàn |

### 7.3 Auto-suggest action

Với mỗi cross-category cluster, tính `suggested_action` theo rule:

```python
def suggest_action(cluster):
    total = cluster["total_count"]
    if total < 5:
        return "IGNORE"   # noise, để Other bucket xử lý
    
    cat_dist = cluster["category_distribution"]  # dict {category: count}
    sorted_pct = sorted(cat_dist.values(), reverse=True)
    dominant_pct = sorted_pct[0] / total
    
    if dominant_pct >= 0.90:
        return "MERGE"          # likely K1
    elif dominant_pct >= 0.70:
        return "MERGE_REVIEW"   # nghiêng MERGE nhưng có doubt
    else:
        return "REVIEW_SPLIT"   # likely K2/K4, cần user quyết định
```

### 7.4 Default behavior (KHI USER KHÔNG OVERRIDE)

Đây là điểm quan trọng để bạn chạy thử ngay mà không cần review hết:

- `MERGE` & `MERGE_REVIEW` & `IGNORE` → **tự động merge** theo `auto_canonical`, dùng category dominant. Không cần user can thiệp.
- `REVIEW_SPLIT` → **tự động merge** với warning trong log. User có thể override sau.

→ **Tóm lại: default = merge tất cả.** User chỉ cần review file `cross_category_review.csv` nếu thấy số `REVIEW_SPLIT` đáng kể (vd >10) hoặc thấy cluster cụ thể nào sai.

### 7.5 Cơ chế split qua override (khi user quyết định split)

Vì `canonical_mapping.csv` có composite key `(raw_skill_name, category)`, user fill `override_canonical` khác nhau cho 2 row của cùng skill ở 2 category → tool tự nhiên split:

```csv
raw_skill_name,category,auto_canonical,override_canonical,final_canonical
mlops,Infrastructure & DevOps,mlops,mlops,mlops
mlops,AI/ML/Data,mlops,mlops_training,mlops_training
```

Sau khi re-run với `--apply-overrides`, 2 row trên sẽ có `final_canonical` khác nhau → cluster gốc bị tách thành 2 canonical riêng biệt trong `skill_distribution_after.csv`.

**Convention naming cho split:** dùng suffix mô tả ngắn gọn (`mlops_training`, `mlops_deployment`) hoặc `_<category_short>` (`mlops_aiml`, `mlops_infra`). Tool không enforce convention — user tự đặt tên miễn unique.

### 7.6 Logging

`clustering_report.json` phải có field:

```json
"cross_category_summary": {
  "total_cross_category_clusters": 45,
  "action_breakdown": {
    "MERGE": 38,
    "MERGE_REVIEW": 5,
    "REVIEW_SPLIT": 2,
    "IGNORE": 0
  },
  "warning": "2 clusters marked REVIEW_SPLIT — manual review recommended. See cross_category_review.csv."
}
```

Console output cuối run:

```
[Cross-category] 45 cross-cat clusters detected
  → 38 auto-merged (dominant category >= 90%)
  → 5 auto-merged with warning (dominant 70-90%)
  → 2 require manual review (see cross_category_review.csv)
```

---

## 8. Pick canonical name

### 8.1 Rule per cluster

Cho mỗi cluster, chọn canonical_skill theo thứ tự ưu tiên:

1. **Ưu tiên member chỉ chứa ASCII** (EC-07 NGOẠI LỆ 2: ưu tiên EN khi song ngữ).
   - Cách check: string có encode được sang ASCII không. Nếu cluster có ≥ 1 member ASCII, pool candidate = các ASCII member. Nếu không có member ASCII nào, pool = toàn bộ member.
2. **Trong pool, chọn member có `total_count` cao nhất.**
3. **Tie-break 1:** member ngắn hơn (số ký tự `skill_normalized` nhỏ hơn).
4. **Tie-break 2:** alphabetical (deterministic).

Cross-category cluster: pick canonical theo rule trên (không quan tâm category) — user sẽ override sau nếu cần split.

### 8.2 Ví dụ

Cluster với members:
```
- ai (count=163, category=AI/ML/Data)
- trí tuệ nhân tạo (count=5, category=AI/ML/Data)
- a.i (count=2, category=AI/ML/Data)
- ai technologies (count=2, category=AI/ML/Data)
```

→ Pool ASCII: `ai`, `a.i`, `ai technologies`. Max count: `ai` (163). **Canonical = `ai`.**

### 8.3 Edge case: singleton cluster

Cluster chỉ có 1 member → canonical = member đó. Apply bucket rule (bước 9.1).

---

## 9. Long-tail bucket — "Other"

### 9.1 Rule áp dụng bucket

Một cluster được gộp vào bucket `<Category> Other` (vd `AI/ML/Data Other`) nếu:

- `n_members == 1` (singleton) **AND**
- `total_count < MIN_KEEP_COUNT` (default = 2)

**Cluster có ≥ 2 members KHÔNG bị gộp Other** — nó là gom hợp lệ dù count thấp.

Bucket name: dùng category dominant của cluster + ` Other`. Nếu cluster là cross-category, dùng category có total_count cao nhất trong cluster.

### 9.2 Threshold làm tham số

```bash
python cluster_skills.py --min-keep-count 2
```

User có thể tăng lên 3 hoặc 5 sau khi review output đầu tiên (xem distribution → quyết định cut-off).

---

## 10. Output

Tạo thư mục `./outputs/` với các file sau:

### 10.1 `clusters_review.csv` — file chính để audit

Mỗi dòng = 1 cluster. Sort theo `total_count` giảm dần.

| Column | Mô tả |
|---|---|
| `cluster_id` | Integer unique |
| `auto_canonical` | Canonical name tự động pick (bước 8) |
| `override_canonical` | **Empty** — để user fill nếu muốn override |
| `final_canonical` | Sẽ là `auto_canonical` nếu `override_canonical` empty, ngược lại = `override_canonical`. Cập nhật khi re-run với mapping override. |
| `n_members` | Số (skill_normalized, category) unique trong cluster |
| `total_count` | Tổng mention |
| `n_required` | Mention với label = required |
| `n_preferred` | Mention với label = preferred |
| `dominant_category` | Category có count cao nhất trong cluster |
| `dominant_category_pct` | Tỷ lệ % của dominant category trên tổng (vd 0.95) |
| `is_cross_category` | Boolean |
| `categories_involved` | Comma-separated list các category trong cluster (nếu cross) |
| `category_distribution` | JSON string `{"Infra & DevOps": 150, "AI/ML/Data": 3}` — phân phối count theo category |
| `suggested_action` | `MERGE` / `MERGE_REVIEW` / `REVIEW_SPLIT` / `IGNORE` / `N/A` (cho non-cross-cat) — xem §7.3 |
| `all_members` | Pipe-separated list các `skill_normalized` trong cluster |
| `all_members_raw` | Pipe-separated list các `skill_name` raw gốc (trace ngược) |
| `applied_other_bucket` | Boolean — cluster có bị gộp Other không |

### 10.2 `canonical_mapping.csv` — mapping per raw skill

Mỗi dòng = 1 (raw skill_name, category) unique. **Đây là file user override** khi muốn map manually.

| Column | Mô tả |
|---|---|
| `raw_skill_name` | Raw từ LLM extraction |
| `category` | Category của skill |
| `skill_normalized` | Sau pre-normalize |
| `cluster_id` | Cluster gốc |
| `auto_canonical` | Theo bước 8 |
| `override_canonical` | **Empty** — user fill |
| `final_canonical` | Effective canonical (auto hoặc override) |
| `raw_count` | Tổng mention của raw skill này |

User chỉnh `override_canonical` trong file này → re-run script với flag `--apply-overrides outputs/canonical_mapping.csv` để áp dụng.

### 10.3 `skill_distribution_after.csv` — phân phối sau gom

Mỗi dòng = 1 `final_canonical`. Sort theo `total_count` giảm dần.

| Column | Mô tả |
|---|---|
| `final_canonical` | Canonical name (bao gồm cả các `<Category> Other`) |
| `dominant_category` | Category chính |
| `total_count` | Tổng mention |
| `n_raw_variants` | Số raw skill_name khác nhau đã gộp vào |
| `n_required` | Mention required |
| `n_preferred` | Mention preferred |
| `sample_variants` | 5 ví dụ raw skill_name |

### 10.4 `cross_category_review.csv` — flag cross-cat

Filter từ `clusters_review.csv` với `is_cross_category = True`. **Đây là file user cần review nhất khi tune.**

Sort: theo `suggested_action` (REVIEW_SPLIT trước, rồi MERGE_REVIEW, rồi MERGE, rồi IGNORE), trong cùng action sort theo `total_count` giảm dần.

→ User mở file này, scan top vài chục dòng. Nếu thấy ít REVIEW_SPLIT và MERGE_REVIEW (vd <10 mỗi loại) → có thể skip review, default merge đã đủ. Nếu nhiều → review từng cluster, fill `override_canonical` trong `canonical_mapping.csv` theo cơ chế §7.5.

### 10.5 `annotations_with_canonical.csv` — apply mapping vào dataset gốc

File CSV gốc + 2 column thêm: `skill_normalized`, `final_canonical`. Đây là **deliverable cuối** dùng cho salary model và EDA.

### 10.6 `clustering_report.json` — metadata run

```json
{
  "matching_version": "v3_embedding_clustering",
  "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
  "distance_threshold": 0.15,
  "linkage": "average",
  "min_keep_count": 2,
  "input_rows": 25000,
  "input_unique_skills": 4500,
  "n_clusters": 1800,
  "n_clusters_cross_category": 45,
  "cross_category_summary": {
    "total_cross_category_clusters": 45,
    "action_breakdown": {
      "MERGE": 38,
      "MERGE_REVIEW": 5,
      "REVIEW_SPLIT": 2,
      "IGNORE": 0
    },
    "warning": "2 clusters marked REVIEW_SPLIT — manual review recommended. See cross_category_review.csv."
  },
  "n_clusters_in_other_bucket": 1200,
  "skills_after_canonical": 600,
  "total_mentions_preserved": 25000,
  "runtime_seconds": 95,
  "timestamp": "2026-05-27T10:00:00",
  "config_hash": "sha256-abc123..."
}
```

---

## 11. Sanity checks (bắt buộc trong script)

Script phải **assert** các invariant sau và fail loud nếu vi phạm:

1. **Mention conservation:** `sum(total_count)` trước và sau clustering phải bằng nhau. Không skill nào được "biến mất" hay "tạo ra".
2. **Mapping completeness:** mỗi `(raw_skill_name, category)` ở input phải có đúng 1 row trong `canonical_mapping.csv`.
3. **Final canonical non-null:** không cluster nào có `final_canonical` null hay empty.
4. **Category preservation:** category column ở output `annotations_with_canonical.csv` phải bằng category ở input cùng row.
5. **No collision** (warning, không fail): nếu 2 cluster khác nhau có cùng `auto_canonical` (rare nhưng possible với singleton clusters cùng tên ở 2 category khác nhau) → cảnh báo trong log.

---

## 12. CLI interface

```bash
python cluster_skills.py \
    --input data/skills_raw.csv \
    --output-dir outputs/ \
    --distance-threshold 0.15 \
    --short-threshold 5 \
    --min-keep-count 2 \
    --apply-overrides outputs/canonical_mapping.csv \
    --cache-dir cache/ \
    --seed 42
```

**Mặc định:**
- `--distance-threshold 0.15`
- `--short-threshold 5` (skill ≤5 ký tự bypass embedding — xem §4.5)
- `--min-keep-count 2`
- `--apply-overrides` không set → ignore override column
- `--cache-dir cache/` → tự tạo nếu không có
- `--seed 42` → fixed cho reproducibility

---

## 13. Workflow user

Pipeline được thiết kế cho 2-3 vòng iteration:

**Vòng 1 — Initial run (no review needed):**

```bash
python cluster_skills.py --input data/skills_raw.csv
```

→ Output đầy đủ trong `outputs/`. **Tất cả cross-category cluster đã được auto-merge theo dominant category.** User có thể dùng `annotations_with_canonical.csv` ngay cho EDA/model nếu không muốn review.

**Vòng 2 — Quick scan (recommended):**

Mở 3 file theo thứ tự:

1. **`clustering_report.json`** → xem `cross_category_summary`. Nếu `REVIEW_SPLIT = 0` và `MERGE_REVIEW < 10`, có thể skip phần lớn review.

2. **`cross_category_review.csv`** → scan top dòng có `suggested_action = REVIEW_SPLIT` trước, sau đó `MERGE_REVIEW`. Mỗi dòng quyết định:
   - Đồng ý merge (đa số case) → không làm gì
   - Cần split → ghi nhớ cluster_id để override ở bước sau

3. **`clusters_review.csv` top 50** → kiểm tra `auto_canonical` của các cluster lớn nhất có hợp lý không. Nếu thấy `auto_canonical = "ml"` mà bạn muốn `"machine learning"` → fill `override_canonical = "machine learning"` cho TẤT CẢ row của cluster đó trong `canonical_mapping.csv`.

4. **`skill_distribution_after.csv`** → xem có quá nhiều `<Category> Other` không. Nếu Other chiếm >25% mentions → tăng `--min-keep-count` lên 3 hoặc 5; nếu Other chiếm <5% → có thể giảm xuống 1.

**Vòng 3 — Re-run với override (nếu có):**

```bash
python cluster_skills.py \
    --input data/skills_raw.csv \
    --apply-overrides outputs/canonical_mapping.csv
```

→ Output cập nhật với override. Embedding cache giúp re-run < 30s.

**Khi cần tune threshold:**

```bash
# Thử threshold lỏng hơn (gom mạnh hơn)
python cluster_skills.py --input data/skills_raw.csv --distance-threshold 0.20

# So sánh số cluster, số cross-cat, distribution Other giữa các threshold
# để pick value phù hợp nhất với dataset thực tế của bạn.
```

---

## 14. Deliverable cho phase tiếp theo

Sau khi user review xong và satisfied:

- **Salary model:** dùng `annotations_with_canonical.csv` — column `final_canonical` thay cho `skill_name` khi build feature.
- **EDA:** dùng `skill_distribution_after.csv` cho thống kê demand.
- **Reproducibility:** lưu `clustering_report.json` + final `canonical_mapping.csv` vào repo. Version control bắt buộc.
- **IAA report (Methodology §3):** ghi rõ `matching_version = "v3_embedding_clustering"`, thông số (model, threshold, min_keep_count) vào báo cáo IAA cuối cùng.

---

## 15. Threats to validity (cần ghi trong report)

Theo Methodology §7, các limitation sau cần document:

1. **Embedding bias trên string ngắn:** Run pilot phát hiện multilingual MiniLM tạo embedding nhiễu cho string ≤5 ký tự (catch-all cluster với 200+ acronym không liên quan). Mitigation đã áp dụng: Section 4.5 — short-string bypass với exact-match consolidation.

2. **Trade-off của short-string bypass:** Mất khả năng gom semantic giữa `react` ↔ `reactjs`, `vue` ↔ `vuejs`, `node` ↔ `nodejs`. Mitigation: post-clustering manual alias step qua override file (Section 4.5.6). Recommended alias list documented.

3. **Generic word catch-all (Type-2 BROKEN — observed but not fixed):** Một số cluster có canonical là từ chung tiếng Anh (vd `project management`, `automation`, `mobile app`, `security`, `risk management`) gom 20-30 variants đúng concept nhưng `dominant_category_pct` thấp (40-70%). **Nguyên nhân: LLM extraction gán category KHÔNG NHẤT QUÁN cho cùng concept**, không phải clustering sai. Vd cùng skill `project management` được LLM gán vào 5 category: Engineering Concepts, Domain Knowledge, Soft Skill, Tool & Platform, Other. Quyết định: KHÔNG fix ở clustering layer — đây là insight về annotation quality cần báo cáo riêng trong IAA report. Có thể addressing sau bằng category re-assignment step nếu cần.

4. **Cross-category collision (legitimate cases):** Codebook §4.1 BOUNDARY RULES specify một số skill có nghĩa khác nhau theo category (vd MLOps training vs deployment). Flag để user review qua `cross_category_review.csv`, không tự động merge/split.

5. **Cut-off Other bucket:** Singleton cluster với count < `min_keep_count` được gộp `<Category> Other`. Mất một phần granularity cho long-tail. Mitigation: `--min-keep-count` tunable (default = 2). Trong pilot run, 4,422 cluster (7.8% mentions) rơi vào Other — chấp nhận được.

6. **Preprocessing fit on full dataset (data leakage caveat):** Canonical clustering được fit trên toàn bộ dataset (train + test), tương tự cách dùng pre-trained word embedding. **Đây không phải target leakage** — target `salary` không tham gia clustering, pipeline chỉ học từ skill text. Cluster boundary có thể bị ảnh hưởng nhẹ bởi distribution test data, nhưng trong context dataset skill IT VN có vocabulary stable, effect này là negligible. Workflow đúng: canonical hóa trên toàn bộ data → split train/test SAU. Document rõ trong methodology report.

7. **Reproducibility:** Embedding deterministic với fixed model version + seed (`--seed 42`). Pin `sentence-transformers` version trong `requirements.txt`. Re-run với cùng config + cùng input phải cho cùng output (`config_hash` verify).

---

## 16. Dependencies

```
python >= 3.10
sentence-transformers >= 2.7
scikit-learn >= 1.3
pandas >= 2.0
numpy >= 1.24
```

Lần đầu chạy sẽ download model ~120MB từ HuggingFace. Cache local sau lần đầu.

---

## 17. NOT in scope

Spec này KHÔNG bao gồm:

- Tự động re-categorize skill (vd phát hiện `docker` ở AI/ML/Data và move sang Infra). User quyết định qua override.
- Xử lý `min_years` và `level` — 2 field này pass-through, không clustering.
- Validation Codebook §4.1 BOUNDARY RULES tự động. Cross-category flag là cơ chế surface vấn đề; quyết định cuối là của user.
- IAA computation. Pipeline này tạo `final_canonical` để feed vào IAA pipeline riêng (theo `iaa_matching_patch.md`).