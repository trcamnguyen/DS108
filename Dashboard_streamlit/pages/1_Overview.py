import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st

from utils.chart_helpers import salary_boxplot, salary_histogram, salary_scatter, source_bar
from utils.data_loader import load_all
from utils.sidebar import render_sidebar

st.set_page_config(page_title="Tổng quan", layout="wide")

jobs_df, skills_long_df, skill_salary_df, iaa, using_sample = load_all()
if using_sample:
    st.warning("⚠️ Running with SAMPLE DATA — replace with real processed files in data/processed/")

filtered_jobs, filtered_skills = render_sidebar(jobs_df, skills_long_df)

st.title("Tổng quan dữ liệu")

if filtered_jobs.empty:
    st.warning("Không có dữ liệu với bộ lọc hiện tại — hãy nới lỏng bộ lọc ở thanh bên.")
    st.stop()

# KPI cards
c1, c2, c3, c4 = st.columns(4)
c1.metric("Tổng việc làm", f"{len(filtered_jobs):,}")
c2.metric("Lương median", f"{filtered_jobs['salary_mid'].dropna().median():.1f}M VND")
c3.metric("Lương trung bình", f"{filtered_jobs['salary_mid'].dropna().mean():.1f}M VND")
c4.metric("Kỹ năng duy nhất", f"{filtered_skills['skill_name'].nunique():,}")

st.divider()

# Salary histogram
st.subheader("Phân phối lương")
st.info(
    "💡 Phân phối lương cho thấy mặt bằng chung của thị trường — "
    "phần lớn tập trung ở đâu, và có bao nhiêu vị trí high-end."
)
bin_size = st.slider("Khoảng chia (triệu VND)", min_value=2, max_value=20, value=5)
st.plotly_chart(salary_histogram(filtered_jobs, bin_size), use_container_width=True)

st.divider()

ca, cb = st.columns(2)
with ca:
    st.subheader("Số việc làm theo nguồn")
    st.plotly_chart(source_bar(filtered_jobs), use_container_width=True)
with cb:
    st.subheader("Lương theo cấp độ kinh nghiệm")
    st.info("💡 So sánh lương theo cấp độ giúp ứng viên định vị mức kỳ vọng phù hợp với kinh nghiệm.")
    st.plotly_chart(salary_scatter(filtered_jobs), use_container_width=True)

st.divider()

st.subheader("Phân phối lương theo cấp độ (Box Plot)")
st.info("💡 Box plot cho thấy độ chênh lệch lương trong cùng cấp độ — Senior và Lead có khoảng dao động rất rộng.")
st.plotly_chart(salary_boxplot(filtered_jobs), use_container_width=True)
