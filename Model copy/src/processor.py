import pandas as pd
import numpy as np
import re

def group_rare_categories(series, min_freq=10):
    """
    Gộp các giá trị xuất hiện ít hơn min_freq vào nhóm 'Other'.
    Giúp giảm độ thưa (sparsity) cho các biến One-Hot Encoding.
    """
    counts = series.value_counts()
    keep = counts[counts >= min_freq].index
    return series.where(series.isin(keep), "Other")