import pandas as pd
import numpy as np

def build_skill_aggregates(skills_df):
    """
    Chuyển đổi DataFrame skills (nhiều dòng cho 1 job) thành 
    Aggregate Features (1 dòng cho 1 job).
    Gồm: n_skills, category counts, level counts, experience requirements.
    """
    # 1. Tổng số kỹ năng
    skill_counts = skills_df.groupby("job_id").size().reset_index(name="n_skills")

    # 2. Đếm theo nhãn (Required/Preferred)
    label_counts = pd.crosstab(skills_df["job_id"], skills_df["label"]).reset_index()
    label_counts.columns = ["job_id", "n_preferred", "n_required"]

    # 3. Đếm theo Category (AI, Backend, v.v.)
    category_counts = pd.crosstab(skills_df["job_id"], skills_df["category"]).reset_index()
    category_counts.columns = ["job_id"] + [f"cat_{c.lower().replace(' ', '_').replace('/', '_')}" 
                                           for c in category_counts.columns[1:]]

    # 4. Đếm theo Level (Expert, Basic, v.v.)
    level_counts = pd.crosstab(skills_df["job_id"], skills_df["level"]).reset_index()
    level_counts.columns = ["job_id"] + [f"level_{c}" for c in level_counts.columns[1:]]

    # 5. Đặc trưng kinh nghiệm yêu cầu trong skill
    exp_features = skills_df.groupby("job_id").agg(
        required_experience_level=("min_years", "max")
    ).reset_index()
    exp_features["has_year_requirement"] = exp_features["required_experience_level"].notna().astype(int)

    # Merge tất cả lại
    skill_agg = (
        skill_counts
        .merge(label_counts, on="job_id", how="left")
        .merge(category_counts, on="job_id", how="left")
        .merge(level_counts, on="job_id", how="left")
        .merge(exp_features, on="job_id", how="left")
    ).fillna(0)
    skill_agg["has_year_requirement"] = skill_agg["has_year_requirement"].astype(int)
    
    return skill_agg

def get_skill_matrix(skills_df, k=100, exclude_categories=None):
    """
    Tạo ma trận One-Hot cho Top-K kỹ năng phổ biến nhất.
    """
    df = skills_df.copy()
    if exclude_categories:
        df = df[~df["category"].isin(exclude_categories)]
    
    # Lấy danh sách k kỹ năng xuất hiện nhiều nhất
    topk = df["final_canonical"].value_counts().head(k).index.tolist()
    
    # Tạo ma trận chéo (pivot table)
    skill_matrix = pd.crosstab(df["job_id"], df["final_canonical"])
    
    # Chỉ giữ lại các cột thuộc topk và reset index để có cột job_id
    skill_matrix = skill_matrix[topk].reset_index()
    
    return skill_matrix, topk