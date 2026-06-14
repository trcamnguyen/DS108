# DS108 — Phân Tích Thị Trường Tuyển Dụng IT & Dự Đoán Lương

Đồ án xây dựng pipeline thu thập, làm sạch, trích xuất kỹ năng và dự đoán mức lương cho tin tuyển dụng IT tại Việt Nam từ hai nguồn **TopCV** và **ITViec**.

---

## Mục tiêu

- Thu thập tin tuyển dụng IT từ TopCV và ITViec.
- Trích xuất và chuẩn hoá kỹ năng từ mô tả công việc bằng LLM (few-shot annotation).
- Phân tích thị trường: phân phối lương, tỷ lệ kỹ năng, tech stack theo vai trò.
- Xây dựng mô hình dự đoán mức lương dựa trên đặc trưng metadata và kỹ năng.
- Dashboard trực quan hoá kết quả phân tích.

---

## Dataset

| Nguồn   | Số bản ghi | Cột chính                                  |
|---------|-----------|---------------------------------------------|
| TopCV   | 2,406     | `requirement`                               |
| ITViec  | 938       | `consolidated_text` (requirement + experience + preferred_skills) |
| **Tổng**| **3,344** |                                             |

- ~1,277 bản ghi có lương số (dùng cho regression).
- ~2,067 bản ghi lương "Thỏa thuận" — loại khỏi tập modeling (không có giá trị lương để train).

---

## Cấu trúc thư mục

```
DS108/
├── Crawler/
│   ├── Crawl_TopCV/        # Crawler cho TopCV
│   └── Crawl_ITViec/       # Crawler cho ITViec
├── Preprocessing/
│   ├── 01-standardize_title/   # Chuẩn hoá job title
│   ├── 02_skill_extraction/    # Trích xuất kỹ năng bằng LLM
│   └── 03_preprocessing/       # Parse lương, kinh nghiệm, địa điểm, cấp bậc
├── EDA/
│   ├── EDA_01_Salary.ipynb
│   ├── EDA_02_Skill_Required_Ratio.ipynb
│   ├── EDA_03_CoreStack_per_Role.ipynb
│   └── EDA_04_Missingness_Analysis.ipynb
├── Model/
│   ├── Salary_Prediction_Final.ipynb   # Notebook huấn luyện chính
│   ├── src/                            # Module feature engineering & modeling
│   └── MODELING_GUIDE.md
├── Dashboard/
│   ├── app.py                  # Streamlit app
│   ├── pages/
│   └── utils/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
└── requirements.txt
```

---

## Pipeline

```
Thu thập dữ liệu  →  Làm sạch & Parse  →  Trích xuất kỹ năng (LLM)
      ↓                     ↓                        ↓
  Crawler/             Preprocessing/          02_skill_extraction/
                            ↓                        ↓
                          EDA/              Chuẩn hoá canonical (D4)
                                                     ↓
                                               Model/  →  Dashboard/
```

### Các giai đoạn chính

| Giai đoạn | Mô tả |
|-----------|-------|
| **Crawling** | Scrape tin tuyển dụng từ TopCV và ITViec |
| **Preprocessing** | Parse lương (VND/USD), kinh nghiệm, địa điểm, cấp bậc |
| **Skill Extraction** | LLM few-shot trích xuất kỹ năng theo 11 danh mục, 2 nhãn |
| **EDA** | Phân tích phân phối lương, kỹ năng phổ biến, tech stack theo vai trò |
| **Modeling** | Dự đoán lương với Feature Set A (metadata) và B (metadata + kỹ năng) |
| **Dashboard** | Streamlit dashboard trực quan hoá kết quả |

---

## Skill Annotation Schema

Mỗi kỹ năng trích xuất gồm 6 trường:

```json
{
  "skill_name": "giữ nguyên raw từ JD",
  "label": "required_skill | preferred_skill",
  "category": "một trong 11 danh mục",
  "min_years": 0–20 | null,
  "level": "expert | intermediate | basic | null",
  "source_text": "đoạn trích nguyên văn từ JD"
}
```

**14 danh mục kỹ năng**: Programming Language · Framework / Library · Database · Infrastructure & DevOps · AI/ML/Data · Data Engineering & Analytics · Engineering Concepts & Methodologies · Tool & Platform · Soft Skill · Testing & QA · Domain Knowledge · IT Support & Hardware · Embedded & Firmware · Other

---

## Modeling

Bài toán **regression** dự đoán mức lương trên ~1,277 bản ghi có lương số. Tin "Thỏa thuận" bị loại khỏi tập training.

Biến mục tiêu: `log_salary = log(1 + avg_salary)` (giảm right-skewed).

Hai bộ đặc trưng được so sánh:
- **Feature Set A**: metadata (nguồn, địa điểm, ngành, cấp bậc, kinh nghiệm, job title).
- **Feature Set B**: Feature Set A + skill aggregates (số kỹ năng, số kỹ năng theo từng danh mục, v.v.).

**Kết quả tốt nhất**: Ridge Regression với Feature Set B — MAE = 0.2686, RMSE = 0.3629 (cải thiện có ý nghĩa thống kê so với Feature Set A, p = 0.024).

Các mô hình đã thử: Global Median Baseline, Median-by-Title Baseline, Ridge Regression, LightGBM.

---

## Cài đặt

```bash
pip install -r requirements.txt
```

Chạy Dashboard:

```bash
streamlit run Dashboard/app.py
```

---

## Công nghệ sử dụng

- **Crawling**: Selenium, BeautifulSoup, Requests
- **Preprocessing**: Pandas, Regex, scikit-learn
- **Skill Extraction**: Claude API (few-shot LLM annotation)
- **Modeling**: scikit-learn, XGBoost, LightGBM
- **Dashboard**: Streamlit, Plotly
- **Versioning**: Git, DVC (data)

---

## Nhóm

**DS108** — HK2 2026–2027, TP.HCM
