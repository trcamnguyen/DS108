import pandas as pd
import streamlit as st


def render_sidebar(jobs_df: pd.DataFrame, skills_long_df: pd.DataFrame):
    """Render sidebar filters and return (filtered_jobs, filtered_skills)."""
    with st.sidebar:
        st.title("Bộ lọc")

        source_opts = sorted(jobs_df["source"].dropna().unique().tolist())
        sources = st.multiselect(
            "Nguồn dữ liệu", source_opts, default=source_opts, key="sb_source"
        )

        cat_opts = sorted(jobs_df["category"].dropna().unique().tolist())
        categories = st.multiselect(
            "Ngành nghề", cat_opts, default=cat_opts, key="sb_cat"
        )

        _exp_order = ["Junior", "Middle", "Senior", "Lead", "Unknown"]
        exp_vals = set(jobs_df["experience_level"].dropna().unique())
        exp_opts = [e for e in _exp_order if e in exp_vals]
        exp_levels = st.multiselect(
            "Cấp độ kinh nghiệm", exp_opts, default=exp_opts, key="sb_exp"
        )

        sal_range = st.slider(
            "Mức lương (triệu VND)", 0, 100, (0, 100), step=5, key="sb_sal"
        )

        include_negotiable = st.checkbox(
            "Bao gồm việc lương thỏa thuận", value=True, key="sb_nego"
        )

        min_skills = st.number_input(
            "Số kỹ năng tối thiểu / JD", min_value=1, max_value=30, value=5, key="sb_msk"
        )

    # Compute skill count per job and join
    skill_counts = (
        skills_long_df.groupby("job_id").size().reset_index(name="skill_count")
    )
    base = jobs_df.drop(columns=["skill_count"], errors="ignore")
    fj = base.merge(skill_counts, on="job_id", how="left")
    fj["skill_count"] = fj["skill_count"].fillna(0).astype(int)

    if sources:
        fj = fj[fj["source"].isin(sources)]
    if categories:
        fj = fj[fj["category"].isin(categories)]
    if exp_levels:
        fj = fj[fj["experience_level"].isin(exp_levels)]

    salary_mask = (fj["salary_mid"] >= sal_range[0]) & (fj["salary_mid"] <= sal_range[1])
    if include_negotiable:
        fj = fj[salary_mask | fj["salary_mid"].isna()]
    else:
        fj = fj[salary_mask]
    fj = fj[fj["skill_count"] >= min_skills]

    valid_ids = set(fj["job_id"])
    fs = skills_long_df[skills_long_df["job_id"].isin(valid_ids)]

    return fj.reset_index(drop=True), fs.reset_index(drop=True)
