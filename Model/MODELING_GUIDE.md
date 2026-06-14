# BÁO CÁO XÂY DỰNG MÔ HÌNH DỰ ĐOÁN LƯƠNG (SALARY PREDICTION)

Dự án này xây dựng mô hình Machine Learning nhằm dự đoán mức lương trong lĩnh vực Công nghệ Thông tin dựa trên thông tin tuyển dụng và các đặc trưng kỹ năng được trích xuất từ mô tả công việc.

## 1. Xử lý dữ liệu và định nghĩa biến mục tiêu

### 1.1 Biến mục tiêu (Target Variable)

Biến mục tiêu được sử dụng là:

**`log_salary = log(1 + avg_salary)`**

Trong đó:
- `avg_salary` là mức lương trung bình của tin tuyển dụng.
- Phép biến đổi log giúp giảm hiện tượng phân phối lệch phải (right-skewed distribution).
- Giảm ảnh hưởng của các mức lương rất cao.
- Giúp các mô hình hồi quy học ổn định hơn.

### 1.2 Xử lý tin tuyển dụng "Thỏa thuận"

Các tin tuyển dụng có mức lương "Thỏa thuận" (Negotiable) không cung cấp giá trị lương số nên không thể sử dụng làm nhãn huấn luyện.

Do đó:
- Các tin tuyển dụng này được loại khỏi tập modeling.
- Đây là một nguồn sai lệch dữ liệu (selection bias) vì nhóm tin tuyển dụng này xuất hiện nhiều hơn ở một số phân khúc senior và quản lý.
- Hạn chế này được ghi nhận trong quá trình phân tích dữ liệu.

### 1.3 Xử lý Outlier

Sau khi kiểm tra phân phối lương và loại bỏ các lỗi parse rõ ràng, các giá trị lương cao còn lại được đánh giá là hợp lý.

Kết quả cho thấy:
- P99 ≈ 75 triệu VND
- Max ≈ 125 triệu VND

Khoảng cách này được xem là chấp nhận được đối với thị trường tuyển dụng IT.

Vì vậy:
- Không thực hiện winsorization.
- Không loại bỏ thêm outlier.

## 2. Thiết kế đặc trưng (Feature Engineering)

### 2.1 Feature Set A (Metadata Only)

Feature Set A chỉ sử dụng thông tin metadata của tin tuyển dụng:
- `source`
- `location`
- `industry` (gốc — **không dùng industry_cleaned**)
- `education`
- `job_level`
- `standardized_title`
- `experience_years`

**Lý do giữ industry gốc**: Thí nghiệm cho thấy `industry_cleaned` không cải thiện hiệu năng so với phiên bản gốc (xem Experiment_Industry_Cleaned.ipynb).

Mục tiêu của Feature Set A là đánh giá mức độ dự đoán lương chỉ từ thông tin tuyển dụng cơ bản.

### 2.2 Feature Set B (Metadata + Skill Aggregates)

Feature Set B bao gồm toàn bộ Feature Set A và bổ sung các đặc trưng tổng hợp từ kỹ năng:
- Tổng số kỹ năng
- Số kỹ năng bắt buộc
- Số kỹ năng ưu tiên
- Số kỹ năng theo từng nhóm chuyên môn
- Các đặc trưng tổng hợp liên quan đến taxonomy kỹ năng

Mục tiêu là đo lường giá trị mà pipeline trích xuất kỹ năng mang lại cho bài toán dự đoán lương.

#### 2.2.1 Mối liên hệ giữa kỹ năng và mức lương

Nhóm thực hiện phân tích tương quan giữa các đặc trưng kỹ năng tổng hợp và biến mục tiêu log_salary.

Kết quả cho thấy nhiều đặc trưng kỹ năng có tương quan dương với mức lương, bao gồm:

- n_skills (r = 0.423)
- n_required (r = 0.412)
- has_year_requirement (r = 0.334)
- cat_engineering_concepts_&_methodologies (r = 0.332)

Điều này cho thấy các đặc trưng kỹ năng chứa thông tin liên quan đến mức lương và có khả năng bổ sung tín hiệu ngoài metadata truyền thống.

### 2.3 Thử nghiệm Top-K Skills

Ngoài hai bộ đặc trưng chính, một nhóm thí nghiệm riêng được thực hiện với Top-K Skill One-Hot Encoding.

Các giá trị `K` được thử nghiệm:
- K = 25, 50, 75, 100, 150, 200

Soft Skills được loại khỏi tập Top-K nhằm tập trung vào các kỹ năng kỹ thuật.

**Kết quả:**
- Top-25 là cấu hình tốt nhất trong nhóm Top-K.
- Khi K tăng, hiệu năng mô hình giảm dần.
- Các đặc trưng tổng hợp của Feature Set B vẫn vượt trội hơn mọi biến thể Top-K.

Do đó mô hình cuối cùng **không sử dụng** Top-K Skills.

### 2.4 Thử nghiệm Ordinal Encoding

Đã thử nghiệm sử dụng Ordinal Encoding cho `job_level` và `education`.

**Kết quả**: Hiệu năng **tệ hơn** so với One-Hot Encoding (xem Experiment_Ordinal_Encoding.ipynb).

Do đó, **One-Hot Encoding** tiếp tục được giữ nguyên cho hai biến này.

## 3. Thiết lập đánh giá mô hình

### 3.1 Cross Validation

Toàn bộ thực nghiệm sử dụng:
- **5-Fold Stratified Cross Validation**

Stratification được thực hiện theo các nhóm lương (`salary_bin`).

### 3.2 Chống Data Leakage

Toàn bộ quá trình preprocessing được đóng gói trong `ColumnTransformer` + `Pipeline`. Nhờ đó:
- Scaler chỉ được fit trên training fold.
- Encoder chỉ được fit trên training fold.
- Không xảy ra data leakage giữa train và validation.

## 4. Các mô hình được đánh giá

**Baseline 1 – Global Median**  
Dự đoán cùng một mức lương cho mọi tin tuyển dụng.

| Metric | Value  |
|--------|--------|
| MAE    | 0.4932 |
| RMSE   | 0.6606 |

**Baseline 2 – Median by Title**  
Dự đoán mức lương trung vị theo chức danh công việc.

| Metric | Value  |
|--------|--------|
| MAE    | 0.4343 |
| RMSE   | 0.6130 |

**Ridge Regression – Feature Set A**  
Sử dụng metadata của tin tuyển dụng.

| Metric | Value  |
|--------|--------|
| MAE    | 0.2724 |
| RMSE   | 0.3678 |

**Ridge Regression – Feature Set B** (Mô hình tốt nhất)  
Sử dụng metadata và các đặc trưng tổng hợp từ kỹ năng.

| Metric | Value  |
|--------|--------|
| MAE    | 0.2686 |
| RMSE   | 0.3629 |

**LightGBM – Feature Set A**

| Metric | Value  |
|--------|--------|
| MAE    | 0.2726 |
| RMSE   | 0.3765 |

**LightGBM – Feature Set B**

| Metric | Value  |
|--------|--------|
| MAE    | 0.2784 |
| RMSE   | 0.3773 |

## 5. So sánh kết quả

| Model                  | MAE    |
|------------------------|--------|
| Ridge B                | 0.2686 |
| Ridge A                | 0.2724 |
| LightGBM A             | 0.2726 |
| LightGBM B             | 0.2784 |
| Baseline Title         | 0.4343 |
| Baseline Median        | 0.4932 |

### 5.1 Kiểm định ý nghĩa thống kê

Để đánh giá liệu việc bổ sung Skill Aggregate Features có thực sự cải thiện mô hình hay chỉ là dao động ngẫu nhiên của Cross-Validation, nhóm thực hiện Paired t-test trên kết quả 5 fold giữa Ridge A và Ridge B.

| Metric      | T-statistic | P-value     |
|-------------|-------------|-------------|
| MAE         | 3.53        | 0.024       |
| RMSE        | 3.53        | 0.024       |

Kết quả cho thấy p-value < 0.05 đối với cả MAE và RMSE.

Do đó có thể bác bỏ giả thuyết không (H0) rằng hai mô hình có hiệu năng tương đương.

Việc bổ sung Skill Aggregate Features mang lại cải thiện có ý nghĩa thống kê và không chỉ là dao động ngẫu nhiên của quá trình Cross-Validation.

## 6. Kết luận chính

- Metadata chứa phần lớn tín hiệu dự đoán lương.
- Việc bổ sung các đặc trưng tổng hợp từ kỹ năng giúp cải thiện Ridge Regression từ **0.2724 → 0.2686**.
- Ridge Regression khai thác hiệu quả hơn LightGBM trong thiết lập hiện tại.
- Ridge Regression với **Feature Set B** đạt kết quả tốt nhất và được lựa chọn làm mô hình triển khai cuối cùng.
- `industry_cleaned` và Ordinal Encoding **không được sử dụng** trong mô hình cuối.

## 7. Diễn giải mô hình

### Feature Importance
Các đặc trưng có ảnh hưởng mạnh nhất đến dự đoán lương bao gồm:
- Job Level
- Experience Years
- Education
- Location
- Standardized Title
- Industry

Kết quả cho thấy mô hình sử dụng đồng thời thông tin về cấp bậc công việc, chức danh, kinh nghiệm, địa điểm, học vấn và ngành nghề để dự đoán mức lương. Trong đó, cấp bậc công việc và chức danh công việc là những nguồn tín hiệu nổi bật nhất.

### Skill Feature Analysis

Để đánh giá vai trò của các đặc trưng kỹ năng trong mô hình, nhóm phân tích tương quan giữa các đặc trưng kỹ năng tổng hợp và biến mục tiêu `log_salary`.

Kết quả cho thấy nhiều đặc trưng kỹ năng có tương quan dương với mức lương:

- `n_skills` (r ≈ 0.42)
- `n_required` (r ≈ 0.41)
- `has_year_requirement` (r ≈ 0.33)
- `cat_engineering_concepts_&_methodologies` (r ≈ 0.33)
- `cat_infrastructure_&_devops` (r ≈ 0.24)
- `level_expert` (r ≈ 0.21)

Điều này cho thấy các vị trí có nhiều kỹ năng yêu cầu hơn, yêu cầu kinh nghiệm rõ ràng hơn hoặc đòi hỏi các nhóm kỹ năng kỹ thuật chuyên sâu thường đi kèm mức lương cao hơn.

Kết quả này cung cấp thêm bằng chứng rằng các đặc trưng kỹ năng tổng hợp thực sự mang tín hiệu liên quan đến mức lương và không chỉ đơn thuần lặp lại thông tin từ metadata của tin tuyển dụng.

### Residual Analysis
- Phân phối phần dư tập trung quanh giá trị 0.
- Không quan sát thấy xu hướng dự đoán cao hơn hoặc thấp hơn một cách có hệ thống.
- Phần lớn quan sát tập trung gần đường phần dư bằng 0, cho thấy sai số dự đoán nhìn chung được kiểm soát ở mức hợp lý.
- Vẫn tồn tại một số điểm ngoại lệ (outliers) với sai số lớn, cho thấy còn có những yếu tố ảnh hưởng đến lương chưa được mô hình hóa đầy đủ.
- Nhìn chung, mô hình hoạt động ổn định và không xuất hiện dấu hiệu vi phạm nghiêm trọng các giả định cơ bản của hồi quy.

## 8. Tái lập kết quả (Reproducibility)

Các biện pháp đảm bảo khả năng tái lập:
- Sử dụng `random_state = 42`
- Toàn bộ preprocessing được đóng gói trong Pipeline
- Lưu mô hình bằng Joblib
- Snapshot MD5 của dữ liệu đầu vào được lưu lại

## 9. Deliverables

- `Salary_Prediction_Final.ipynb`
- `Experiment_Industry_Cleaned.ipynb`
- `Experiment_Ordinal_Encoding.ipynb`
- `salary_predictor.joblib`
- `features_wide.parquet`
- `requirements.txt`

---

Dự án đáp ứng đầy đủ yêu cầu huấn luyện, đánh giá và tái lập mô hình dự đoán lương cho môn DS108.