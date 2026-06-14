# Data Codebook — DS108

Mô tả hai file dữ liệu chính trong thư mục `data/processed/`.

**Cập nhật lần cuối**: 2026-06-14  
**Nguồn dữ liệu**: TopCV (topcv) và ITViec (itviec)

---

## 1. `jobs_cleaned.parquet`

Bảng tin tuyển dụng sau khi làm sạch và parse.

- **Số dòng**: 3,181
- **Số cột**: 27
- **Đơn vị lương**: triệu VND (VND million)
- **Ghi chú**: 1,705 dòng có `is_missing_salary = True` (thỏa thuận hoặc null) → giá trị lương = 0, không dùng cho regression.

### 1.1 Cột định danh & metadata gốc

| Cột | Kiểu | Null | Mô tả |
|-----|------|------|-------|
| `job_id` | int64 | 0 | ID duy nhất của tin tuyển dụng. Primary key. |
| `url` | str | 0 | URL gốc của tin tuyển dụng trên platform. |
| `source` | str | 0 | Nguồn dữ liệu: `topcv` (2,440 dòng) hoặc `itviec` (741 dòng). |
| `job_title` | str | 0 | Tiêu đề công việc gốc từ platform (chưa chuẩn hóa). |
| `company` | str | 0 | Tên công ty tuyển dụng. |
| `location` | str | 4 | Địa điểm làm việc sau khi chuẩn hóa. Xem danh sách giá trị ở mục 1.4. |
| `industry` | str | 69 | Ngành nghề của tin tuyển dụng. 70 giá trị duy nhất. Null cho 69 dòng. |
| `employment_type` | str | 0 | Hình thức làm việc. Xem mục 1.4. |
| `education` | str | 0 | Yêu cầu học vấn. 9 giá trị (có biến thể chữ hoa/thường, xem mục 1.4). |
| `experience` | str | 0 | Mô tả kinh nghiệm gốc từ platform (raw text). |
| `job_level` | str | 0 | Cấp bậc công việc gốc từ platform. 9 giá trị. |
| `salary` | str | 0 | Chuỗi lương gốc hiển thị trên platform. |
| `specialization` | str | 741 | Chuyên môn (chỉ có ở TopCV). Null = ITViec. |
| `platform_required_skills` | str | 0 | Kỹ năng bắt buộc theo tag platform (raw). |
| `platform_preferred_skills` | str | 741 | Kỹ năng ưu tiên theo tag platform (chỉ có ở TopCV). |
| `job_description` | str | 0 | Mô tả công việc đầy đủ (raw text). Không dùng cho skill extraction (EC-08). |
| `requirement` | str | 0 | Phần yêu cầu tuyển dụng — **cột chính** cho skill extraction. |

### 1.2 Cột được tạo ra trong quá trình preprocessing

| Cột | Kiểu | Null | Mô tả |
|-----|------|------|-------|
| `standardized_title` | str | 0 | Job title đã chuẩn hóa. 148 giá trị. Top 5: Backend Developer (344), Business Analyst (269), QA Engineer (231), Fullstack Developer (189), IT Support (168). |
| `job_level_tier` | str | 0 | Cấp bậc đã chuẩn hóa thành 7 tier: `Mid`, `Lead`, `Senior`, `Manager`, `Intern`, `Junior`, `Fresher`. |
| `experience_years` | float64 | 0 | Số năm kinh nghiệm yêu cầu (parse từ cột `experience`). Khoảng 0–13. 0 = không yêu cầu hoặc không xác định. |
| `salary_raw` | str | 0 | Chuỗi lương đã normalize sơ bộ (trước khi parse số). |
| `min_salary` | float64 | 0 | Mức lương tối thiểu (triệu VND). 0 nếu không có lương số. |
| `max_salary` | float64 | 0 | Mức lương tối đa (triệu VND). 0 nếu không có lương số. |
| `avg_salary` | float64 | 0 | Mức lương trung bình = (min + max) / 2 (triệu VND). 0 nếu không có lương số. Dùng làm target cho regression. |
| `is_thoa_thuan` | bool | 0 | `True` nếu lương ghi "Thỏa thuận" (1,703 dòng). |
| `is_null_salary` | bool | 0 | `True` nếu trường lương bị null/rỗng hoàn toàn (2 dòng). |
| `is_missing_salary` | bool | 0 | `True` nếu không có lương số = `is_thoa_thuan OR is_null_salary` (1,705 dòng). Dùng để lọc tập modeling. |

### 1.3 Thống kê lương (chỉ 1,476 dòng có lương số)

| Thống kê | min_salary | max_salary | avg_salary |
|----------|-----------|-----------|-----------|
| 25th pct | 8 triệu | 17 triệu | 15 triệu |
| Median | 15 triệu | 25 triệu | 22.5 triệu |
| 75th pct | 20 triệu | 37.5 triệu | 32.5 triệu |
| Max | 250,000 | 300,000 | 275,000 |

> **Lưu ý**: Các giá trị max cực lớn (>200 triệu) là outlier, có thể do lỗi parse lương USD (chưa chia 1,000). Tập modeling nên lọc thêm theo ngưỡng hợp lý.

### 1.4 Giá trị categorical

**`source`**: `topcv` · `itviec`

**`location`** (19 giá trị):

| Giá trị | Số dòng |
|---------|---------|
| Hà Nội | 1,981 |
| Hồ Chí Minh | 924 |
| Đà Nẵng | 81 |
| Khác | 59 |
| Hải Phòng | 23 |
| Nghệ An | 18 |
| Hưng Yên | 15 |
| Cần Thơ | 15 |
| Đồng Nai | 12 |
| Thanh Hóa | 11 |
| Khác (các tỉnh) | <10 mỗi |
| Foreign:Japan | 8 |

**`job_level_tier`** (7 giá trị):

| Giá trị | Số dòng |
|---------|---------|
| Mid | 1,933 |
| Lead | 603 |
| Senior | 253 |
| Manager | 219 |
| Intern | 141 |
| Junior | 28 |
| Fresher | 4 |

**`employment_type`** (7 giá trị):
`Toàn thời gian` (3,123) · `Remote` (20) · `Hybrid` (19) · `Thời vụ` (10) · `Khác` (5) · `Bán thời gian` (3) · `Làm tại nhà` (1)

**`education`** (9 giá trị — có biến thể chữ hoa/thường chưa được deduplicate):
- Đại Học trở lên / Đại học trở lên
- Cao Đẳng trở lên / Cao đẳng trở lên
- Cao học trở lên / Thạc sĩ trở lên
- Trung cấp trở lên
- Trung học phổ thông (Cấp 3) trở lên
- Không yêu cầu

---

## 2. `skills.parquet`

Bảng kỹ năng trích xuất từ tin tuyển dụng (one row per skill per job).

- **Số dòng**: 56,335
- **Số cột**: 8
- **Số job có kỹ năng**: 3,168 / 3,181 (13 job không có kỹ năng nào)
- **Trung bình kỹ năng / job**: 17.8 (min 1, max 86)

### 2.1 Mô tả cột

| Cột | Kiểu | Null | Mô tả |
|-----|------|------|-------|
| `job_id` | int64 | 0 | Foreign key → `jobs_cleaned.job_id`. |
| `skill_name` | str | 0 | Tên kỹ năng **giữ nguyên raw** từ JD (15,987 giá trị duy nhất). VD: "lập trình Python", "ReactJS", "mô hình học máy". |
| `final_canonical` | str | 0 | Tên kỹ năng đã chuẩn hóa về dạng canonical (2,762 giá trị duy nhất). VD: "Python", "React", "Machine Learning". |
| `label` | str | 0 | Nhãn mức độ yêu cầu. Xem mục 2.2. |
| `category` | str | 0 | Danh mục kỹ năng. Xem mục 2.3. |
| `level` | str | 45,342 | Mức độ thành thạo (nếu JD đề cập). `null` cho 80.5% dòng. Xem mục 2.2. |
| `min_years` | Float64 | 52,522 | Số năm kinh nghiệm tối thiểu yêu cầu cho kỹ năng (nếu JD đề cập). `null` cho 93.2% dòng. Khoảng 0–13. |
| `source_text` | str | 0 | Đoạn trích nguyên văn từ JD làm cơ sở trích xuất kỹ năng (32,577 đoạn duy nhất). |

### 2.2 Giá trị categorical

**`label`** (2 giá trị):

| Giá trị | Số dòng | Tỷ lệ |
|---------|---------|-------|
| `required_skill` | 43,978 | 78.1% |
| `preferred_skill` | 12,357 | 21.9% |

**`level`** (3 giá trị, null = 80.5%):

| Giá trị | Mô tả |
|---------|-------|
| `expert` | Yêu cầu trình độ chuyên sâu / senior |
| `intermediate` | Yêu cầu trình độ trung cấp |
| `basic` | Quen thuộc cơ bản |
| `null` | JD không đề cập mức độ (45,342 dòng) |

### 2.3 Danh mục kỹ năng (`category`) — 15 giá trị

| Danh mục | Số dòng | Tỷ lệ | Ví dụ |
|----------|---------|-------|-------|
| Soft Skill | 8,740 | 15.5% | Giao tiếp, Teamwork, Tiếng Anh |
| Infrastructure & DevOps | 8,023 | 14.2% | AWS, Docker, Kubernetes, CI/CD |
| Engineering Concepts & Methodologies | 7,822 | 13.9% | OOP, Agile, REST API, Microservices |
| Tool & Platform | 5,601 | 9.9% | Git, Jira, Linux, Postman |
| Domain Knowledge | 5,026 | 8.9% | FinTech, eCommerce, ERP |
| Programming Language | 3,296 | 5.9% | Python, Java, C++, JavaScript |
| Framework / Library | 3,229 | 5.7% | React, Laravel, Spring Boot |
| Database | 2,867 | 5.1% | MySQL, PostgreSQL, MongoDB |
| AI/ML/Data | 2,625 | 4.7% | Machine Learning, NLP, TensorFlow |
| Testing & QA | 2,574 | 4.6% | Unit Test, Selenium, Test Plan |
| Other | 2,259 | 4.0% | Kỹ năng không thuộc 14 nhóm trên |
| Data Engineering & Analytics | 1,870 | 3.3% | ETL, Airflow, Power BI, Spark |
| Design & UX | 1,092 | 1.9% | Figma, UI/UX, Wireframe |
| Embedded & Firmware | 693 | 1.2% | RTOS, C Embedded, PLC, Arduino |
| IT Support & Hardware | 618 | 1.1% | Troubleshooting, Network, Hardware |

---

## 3. Quan hệ giữa hai bảng

```
jobs_cleaned.parquet          skills.parquet
─────────────────────         ──────────────────────
job_id (PK)        ◄──────── job_id (FK)
...                           skill_name
                              final_canonical
                              label
                              category
                              level
                              min_years
                              source_text
```

- Quan hệ **1 : nhiều** (1 job → nhiều kỹ năng, trung bình ~17.8).
- 13 job trong `jobs_cleaned` không có dòng tương ứng trong `skills` (không trích xuất được kỹ năng).

---

## 4. Lưu ý chất lượng dữ liệu

| Vấn đề | Cột bị ảnh hưởng | Mức độ | Ghi chú |
|--------|-----------------|--------|---------|
| Outlier lương cực lớn | `min_salary`, `max_salary`, `avg_salary` | Nhỏ | Giá trị >200 triệu có thể là lỗi parse USD. Cần lọc trước khi train. |
| Biến thể chữ hoa/thường | `education` | Nhỏ | "Đại Học" và "Đại học" là cùng giá trị. Cần deduplicate khi dùng. |
| Null có chủ đích | `level`, `min_years` | Bình thường | Null = JD không đề cập, không phải lỗi. Xử lý bằng imputation hoặc binary flag. |
| Null `location` | `location` | Rất nhỏ | 4 dòng. Có thể fill "Khác". |
| Null `industry` | `industry` | Nhỏ | 69 dòng (~2.2%). Cần xử lý khi dùng làm feature. |
| `platform_preferred_skills`, `specialization` | — | Có hệ thống | Null cho toàn bộ ITViec (741 dòng) — thiếu theo thiết kế, không phải lỗi. |
