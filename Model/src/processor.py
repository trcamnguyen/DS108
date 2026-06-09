import pandas as pd
import numpy as np
import re

def calculate_skill_coverage(jobs_df,skills_df):
    total_jobs = jobs_df["job_id"].nunique()
    jobs_with_skills = (skills_df["job_id"].nunique())
    return jobs_with_skills / total_jobs

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