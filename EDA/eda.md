# EDA — Missingness Analysis: `is_thoa_thuan` vs `job_level`

## Mục tiêu

Kiểm định cơ chế missing của nhãn lương ("Thỏa thuận") có phụ thuộc vào
`job_level` không — tức là xác định missing mechanism là **MAR** (Missing
At Random, phụ thuộc vào biến quan sát được) hay **MCAR** (Missing
Completely At Random).

Kết quả sẽ được dùng để biện luận chiến lược xử lý missing trong báo cáo
IEEE (Section Preprocessing + EDA).

---

## Input

File: `data/interim/02-skill_extracted/jobs.parquet`

Các cột cần dùng:
- `salary` — chuỗi lương gốc
- `job_level` — cấp bậc sau harmonization
- `source` — `topcv` hoặc `itviec`
- `sal_mid` — giá trị lương trung điểm (nếu đã có), hoặc tự tính

> Nếu chưa có cột `is_thoa_thuan`, tự tạo:
> ```python
> NEG_PATTERNS = [
>     "thỏa thuận", "thoả thuận", "thoa thuan",
>     "negotiable", "competitive", "you'll love it",
>     "sign in to view", "very attractive", "thương lượng"
> ]
> df["is_thoa_thuan"] = df["salary"].str.lower().str.strip().apply(
>     lambda x: any(p in str(x) for p in NEG_PATTERNS)
>         if pd.notna(x) else True  # null cũng tính là không có lương
> )
> ```

---

## Các bước thực hiện

### Bước 1 — Kiểm tra phân phối `is_thoa_thuan` theo `job_level`

```python
# Tính tỷ lệ thỏa thuận theo job_level, split theo source
# Output: DataFrame với columns [job_level, source, pct_thoa_thuan, count]
# Sort theo pct_thoa_thuan descending
```

Plot: **bar chart** tỷ lệ % thỏa thuận theo `job_level`, màu split theo
`source`. Tương tự plot đã có nhưng thêm hue=source.

Lưu: `eda_outputs/missingness_by_level.png`

---

### Bước 2 — Chi-square test: `is_thoa_thuan` ~ `job_level`

```python
from scipy.stats import chi2_contingency

# Contingency table: job_level × is_thoa_thuan
# Chỉ dùng các level có count >= 10 để test có ý nghĩa
contingency = pd.crosstab(df["job_level"], df["is_thoa_thuan"])
chi2, p, dof, expected = chi2_contingency(contingency)

print(f"Chi-square = {chi2:.4f}")
print(f"p-value    = {p:.4e}")
print(f"df         = {dof}")
```

**Diễn giải:**
- p < 0.05 → reject H0 → `is_thoa_thuan` **không độc lập** với
  `job_level` → **MAR** (missing phụ thuộc vào biến quan sát được)
- p ≥ 0.05 → không đủ bằng chứng bác bỏ MCAR

---

### Bước 3 — Chi-square test: `is_thoa_thuan` ~ `source`

```python
contingency_source = pd.crosstab(df["source"], df["is_thoa_thuan"])
chi2_src, p_src, dof_src, _ = chi2_contingency(contingency_source)

print(f"Chi-square (source) = {chi2_src:.4f}")
print(f"p-value    (source) = {p_src:.4e}")
```

---

### Bước 4 — Cramér's V: effect size

```python
import numpy as np

def cramers_v(chi2, n, r, k):
    """r = số hàng, k = số cột"""
    return np.sqrt(chi2 / (n * (min(r, k) - 1)))

n = contingency.sum().sum()
r, k = contingency.shape
v = cramers_v(chi2, n, r, k)
print(f"Cramér's V = {v:.4f}")
# 0.1–0.3: weak, 0.3–0.5: moderate, >0.5: strong
```

---

### Bước 5 — Heatmap tỷ lệ thỏa thuận theo `job_level × source`

```python
# Pivot table: rows=job_level, cols=source, values=mean(is_thoa_thuan)
pivot = df.groupby(["job_level", "source"])["is_thoa_thuan"].mean().unstack()

# Plot heatmap với annot=True, fmt=".0%"
# Lưu: eda_outputs/missingness_heatmap.png
```

---

## Output cần xuất

| File | Mô tả |
|------|-------|
| `eda_outputs/missingness_by_level.png` | Bar chart tỷ lệ thỏa thuận theo job_level × source |
| `eda_outputs/missingness_heatmap.png` | Heatmap job_level × source |
| `eda_outputs/missingness_stats.json` | Chi-square, p-value, dof, Cramér's V cho cả 2 test |

Format `missingness_stats.json`:
```json
{
  "job_level": {
    "chi2": ...,
    "p_value": ...,
    "dof": ...,
    "cramers_v": ...
  },
  "source": {
    "chi2": ...,
    "p_value": ...,
    "dof": ...,
    "cramers_v": ...
  }
}
```

---

## Lưu ý

- **Không hardcode** tên cột — tự inspect `df.columns` trước khi chạy
- **Không drop** rows có `job_level` null trước khi báo cáo — ghi nhận
  riêng số lượng null và loại khỏi test sau khi đã report
- Các level có `count < 10` → loại khỏi chi-square test nhưng vẫn giữ
  trong plot (annotate count nhỏ bằng màu khác hoặc hatch)
- Plot dùng cùng color palette với các EDA plot khác trong project để
  nhất quán visual