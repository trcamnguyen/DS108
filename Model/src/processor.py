import pandas as pd
import numpy as np
import re

def parse_experience(exp):
    """
    Chuyển đổi nhãn kinh nghiệm (text) sang số năm kinh nghiệm (float).
    Ví dụ: '1 năm' -> 1.0, 'Trên 3 năm' -> 3.5, 'Không yêu cầu' -> 0.0
    """
    exp = str(exp).strip().lower()

    if exp == "không yêu cầu":
        return 0.0

    if "dưới 1 năm" in exp:
        return 0.5

    # Tìm số năm trong chuỗi
    m = re.search(r"(\d+)", exp)
    if m:
        value = int(m.group(1))
        # Nếu có chữ "trên", cộng thêm 0.5 để phản ánh mức kinh nghiệm cao hơn mốc đó
        if "trên" in exp:
            return value + 0.5
        return float(value)

    return np.nan

def group_rare_categories(series, min_freq=10):
    """
    Gộp các giá trị xuất hiện ít hơn min_freq vào nhóm 'Other'.
    Giúp giảm độ thưa (sparsity) cho các biến One-Hot Encoding.
    """
    counts = series.value_counts()
    keep = counts[counts >= min_freq].index
    return series.where(series.isin(keep), "Other")

def clean_industry(industry_series):
    """
    Xử lý cơ bản cột Industry: viết thường, bỏ khoảng trắng thừa.
    """
    return industry_series.fillna("Unknown").str.strip()