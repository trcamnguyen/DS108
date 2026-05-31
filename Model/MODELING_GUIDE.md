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
- `industry`
- `education`
- `job_level`
- `standardized_title`
- `experience_years`

Mục tiêu của Feature Set A là đánh giá mức độ dự đoán lương chỉ từ thông tin tuyển dụng cơ bản.

### 2.2 Feature Set B (Metadata + Skill Aggregates)

Feature Set B bao gồm toàn bộ Feature Set A và bổ sung các đặc trưng tổng hợp từ kỹ năng:
- Tổng số kỹ năng
- Số kỹ năng bắt buộc
- Số kỹ năng ưu tiên
- Số kỹ năng theo từng nhóm chuyên môn
- Các đặc trưng tổng hợp liên quan đến taxonomy kỹ năng

Mục tiêu là đo lường giá trị mà pipeline trích xuất kỹ năng mang lại cho bài toán dự đoán lương.

### 2.3 Thử nghiệm Top-K Skills

Ngoài hai bộ đặc trưng chính, một nhóm thí nghiệm riêng được thực hiện với Top-K Skill One-Hot Encoding.

Các giá trị `K` được thử nghiệm:
- K = 25, 50, 75, 100, 150, 200

Soft Skills được loại khỏi tập Top-K nhằm tập trung vào các kỹ năng kỹ thuật có khả năng mang tín hiệu lương mạnh hơn.

**Kết quả:**
- Top-25 là cấu hình tốt nhất trong nhóm Top-K.
- Khi K tăng, hiệu năng mô hình giảm dần.
- Các đặc trưng tổng hợp của Feature Set B vẫn vượt trội hơn mọi biến thể Top-K.

Do đó mô hình cuối cùng **không sử dụng** Top-K Skills.

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
| MAE    | 0.2727 |
| RMSE   | 0.3735 |

**Ridge Regression – Feature Set B** (Mô hình tốt nhất)  
Sử dụng metadata và các đặc trưng tổng hợp từ kỹ năng.

| Metric | Value  |
|--------|--------|
| MAE    | 0.2678 |
| RMSE   | 0.3631 |

**LightGBM – Feature Set A**

| Metric | Value  |
|--------|--------|
| MAE    | 0.2710 |
| RMSE   | 0.3732 |

**LightGBM – Feature Set B**

| Metric | Value  |
|--------|--------|
| MAE    | 0.2758 |
| RMSE   | 0.3745 |

## 5. So sánh kết quả

| Model                  | MAE    |
|------------------------|--------|
| Ridge B                | 0.2678 |
| LightGBM A             | 0.2710 |
| Ridge A                | 0.2727 |
| LightGBM B             | 0.2758 |
| Baseline Title         | 0.4343 |
| Baseline Median        | 0.4932 |

## 6. Kết luận chính

- Metadata chứa phần lớn tín hiệu dự đoán lương.
- Việc bổ sung các đặc trưng tổng hợp từ kỹ năng giúp cải thiện Ridge Regression từ **0.2727 → 0.2678**.
- Ridge Regression với Feature Set B đạt kết quả tốt nhất và được lựa chọn làm mô hình triển khai cuối cùng.

## 7. Diễn giải mô hình

### Feature Importance
Các đặc trưng có ảnh hưởng mạnh nhất đến dự đoán lương bao gồm:
- Job Level
- Experience Years
- Source
- Location
- Standardized Title
- Industry

### Residual Analysis
- Phân phối phần dư tập trung quanh giá trị 0.
- Phần lớn sai số nằm trong khoảng ±0.2 đơn vị log-salary.
- Mô hình có xu hướng đánh giá thấp các mức lương rất cao và đánh giá cao các mức lương rất thấp (hiện tượng thường gặp ở Ridge Regression).

## 8. Tái lập kết quả (Reproducibility)

Các biện pháp đảm bảo khả năng tái lập:
- Sử dụng `random_state = 42`
- Toàn bộ preprocessing được đóng gói trong Pipeline
- Lưu mô hình bằng Joblib
- Snapshot MD5 của dữ liệu đầu vào được lưu lại

**MD5 của features_wide.parquet**:  
`16a7ce26b2c87309a014cf5d4bd88b67`

## 9. Deliverables

- `modeling_GPT.ipynb`
- `salary_predictor.joblib`
- `features_wide.parquet`
- `requirements.txt`

---

Dự án đáp ứng đầy đủ yêu cầu huấn luyện, đánh giá và tái lập mô hình dự đoán lương cho môn DS108.