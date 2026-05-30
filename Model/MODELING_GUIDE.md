# DS108 — Salary Prediction Model: Gợi ý Training

> Task: train model dự đoán lương từ `jobs.parquet` + `skills.parquet`.

---

## Input → Output

**Input đã có:**
- `data/processed/jobs.parquet` — 1 row/job (salary, location, title, source, ...)
- `data/processed/skills.parquet` — 1 row/skill (`job_id`, `final_canonical`, `category`, `label`, `level`, `min_years`)

**Cần build thêm:** `features_wide.parquet` — pivot wide, 1 row/job, columns gồm top-K skill one-hot + aggregate features.

**Deliverables cuối:**
- Notebook `modeling.ipynb` end-to-end chạy được
- Pipeline file `.joblib` (preprocessing + model gói chung)
- Bảng kết quả: 4 baselines × MAE/RMSE × CV mean ± std
- Feature importance plot (top 30) + residual analysis plot

---

## 1. Xử lý target (salary)

Lương TopCV/ITViec rất bẩn. Chốt policy và **document trong notebook**:

- **Target column**: đề xuất `log(salary_mid)` với `salary_mid = (salary_min + salary_max) / 2`. Lương long-tail phải → log giúp model học ổn định hơn.
- **"Thoả thuận" / "negotiable"**: drop khỏi modeling. PHẢI đếm % drop và check distribution drop vs giữ lại có khác nhau không (đây là MNAR — bias này cần báo cáo trong Datasheet).
- **Outliers**: winsorize ở P99, không drop (drop = can thiệp dữ liệu thô).

---

## 2. ⚠️ CRITICAL: Zero Data Leakage

Rubric DS108 ghi rõ "Yêu cầu tuyệt đối: ZERO DATA LEAKAGE" cho mức A.

**Nguyên tắc**: split train/test TRƯỚC, mọi `fit()` chỉ chạy trên train.

```python
# ❌ SAI
scaler.fit(X)
X_train, X_test = split(X_scaled)

# ✓ ĐÚNG
X_train, X_test = split(X)
scaler.fit(X_train)
X_test_scaled = scaler.transform(X_test)   # transform thôi, KHÔNG fit
```

**Khuyến nghị mạnh**: gói preprocessing + model vào `sklearn.Pipeline` + `ColumnTransformer`. Pipeline tự đảm bảo CV không bị leak.

**Phải fit-on-train-only**: scaler, target encoder, imputer, top-K skill selection (nếu chọn theo frequency).

**Đã làm trên full data và OK**: skill clustering (không dùng salary → không leak — luận điểm này nên ghi vào report).

---

## 3. Feature engineering — gợi ý


- **Top-K skill one-hot** (K = 100–200 cho tree, K = 50–100 cho linear)
- **Skill counts**: `n_required`, `n_preferred`, `n_total` — simple nhưng signal mạnh
- **Category aggregates**: count skill per 14 category — tận dụng taxonomy
- **Experience**: `max_min_years`, `has_year_req` (tương quan cao — linear drop một, tree giữ cả)
- **Level aggregates**: count skill có level=expert/intermediate/basic

**Job-level** từ `jobs.parquet`:
- Location (normalize HCM/HCMC/Sài Gòn về 1 form TRƯỚC khi encode)
- Standardized_title
- Source (TopCV/ITViec) — bắt buộc có, lý do ở §4

---

## 4. Multi-source bias

TopCV và ITViec **không cùng distribution**:
- ITViec skew toward foreign/senior → lương cao hơn
- TopCV broader, có cả non-IT companies tuyển IT

Gợi ý:
- **Include `source` as feature**
- **Stratified split theo source + salary quartile** để đảm bảo CV folds đại diện

Bias này phải nêu rõ trong **Datasheet for Datasets** — rubric mức A đánh giá cao phần self-evaluation về bias.

---

## 5. Model choice

Default đề xuất: **LightGBM** hoặc **XGBoost** — handle missing tự nhiên, không cần scaling, tolerate sparse one-hot tốt, fast iteration.

Thêm **Ridge regression** làm linear baseline để có coefficients diễn giải được.

Không khuyến khích deep learning với n ≈ 3k — overkill và khó justify với scope project.

---

## 6. Evaluation 

Metric: **MAE trên log(salary)** + **MAE thô (triệu VND)** để diễn giải nghiệp vụ.

**Baselines bắt buộc** (đây là argument vàng cho report):

1. Predict median salary (constant) — sàn tuyệt đối
2. Predict median salary per `standardized_title` — sàn realistic
3. Model với chỉ job-level features (không skill)
4. Model với full features (job + skill)

**Gap (4) − (3)** = evidence cho value của skill extraction pipeline. Nhấn mạnh số này trong báo cáo.

CV: 5-fold stratified theo salary quartile + source. Không dùng single train/test split.

---

## 7. Reproducibility checklist

- [ ] Fix `random_state=42` ở mọi chỗ (split, model, bootstrap)
- [ ] Lưu pipeline bằng `joblib.dump()` — preprocessing + model gói chung
- [ ] Snapshot hash của `features_wide.parquet` đã dùng (ghi vào log)
- [ ] Notebook chạy end-to-end từ parquet → metrics, không phụ thuộc thứ tự cell
- [ ] Update `requirements.txt`

---

## Quyết định nên hỏi owner trước khi tự chốt

- Drop policy cho "thoả thuận" (drop hết hay impute?)
- USD/VND handling (convert hay drop?)
- Top-K threshold (K = ?)
- Có gộp TopCV + ITViec train chung không, hay train riêng