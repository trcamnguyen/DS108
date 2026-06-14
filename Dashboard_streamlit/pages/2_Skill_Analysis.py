import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import plotly.express as px
import streamlit as st

from utils.chart_helpers import COLOR_MAP, skill_hbar, skill_treemap
from utils.data_loader import SKILL_CATEGORIES, load_all
from utils.sidebar import render_sidebar

st.set_page_config(page_title="Phân tích kỹ năng", layout="wide")

jobs_df, skills_long_df, skill_salary_df, iaa, using_sample = load_all()
if using_sample:
    st.warning("⚠️ Running with SAMPLE DATA — replace with real processed files in data/processed/")

filtered_jobs, filtered_skills = render_sidebar(jobs_df, skills_long_df)

st.title("Phân tích kỹ năng")
st.info(
    "💡 Khám phá kỹ năng nào được tuyển dụng nhiều nhất, "
    "và kỹ năng nào thực sự bắt buộc vs chỉ là lợi thế."
)

if filtered_skills.empty:
    st.warning("Không có dữ liệu kỹ năng với bộ lọc hiện tại.")
    st.stop()

# Controls row
ctrl1, ctrl2, ctrl3 = st.columns(3)
with ctrl1:
    cat_options = ["Tất cả"] + SKILL_CATEGORIES
    selected_cat = st.selectbox("Danh mục kỹ năng", cat_options, key="skill_cat")
with ctrl2:
    top_n = st.slider("Top N kỹ năng", 5, 50, 20, key="skill_topn")
with ctrl3:
    show_pct = st.toggle("Hiển thị % việc làm", value=False, key="skill_pct")

# Filter by category
plot_skills = (
    filtered_skills[filtered_skills["category"] == selected_cat]
    if selected_cat != "Tất cả"
    else filtered_skills
)

n_jobs_total = max(filtered_jobs["job_id"].nunique(), 1)

skill_counts = (
    plot_skills.groupby(["skill_name", "category"])["job_id"]
    .nunique()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
)
skill_counts["pct"] = (skill_counts["count"] / n_jobs_total * 100).round(1)

st.divider()

# Horizontal bar chart
st.subheader(f"Top {top_n} kỹ năng theo tần suất")
y_col = "pct" if show_pct else "count"
st.plotly_chart(
    skill_hbar(skill_counts, y_col=y_col, top_n=top_n, title=f"Top {top_n} kỹ năng"),
    use_container_width=True,
)

st.divider()

# Stacked bar: required vs preferred
st.subheader("Tỷ lệ Required vs Preferred (Top 15)")
st.info("💡 Kỹ năng có tỷ lệ required cao là tiêu chuẩn tuyển dụng thực sự — nhà tuyển dụng không linh hoạt với những kỹ năng này.")

top15_names = skill_counts.head(15)["skill_name"].tolist()
label_data = (
    filtered_skills[filtered_skills["skill_name"].isin(top15_names)]
    .groupby(["skill_name", "label"])["job_id"]
    .nunique()
    .reset_index(name="count")
)

if not label_data.empty:
    fig_stack = px.bar(
        label_data, x="skill_name", y="count", color="label",
        barmode="stack",
        title="Required vs Preferred — Top 15 kỹ năng",
        labels={"skill_name": "Kỹ năng", "count": "Số việc làm", "label": "Nhãn"},
        color_discrete_map={"required_skill": "#636EFA", "preferred_skill": "#FFA15A"},
    )
    fig_stack.update_xaxes(tickangle=-30)
    st.plotly_chart(fig_stack, use_container_width=True)

st.divider()

# Treemap
st.subheader("Bản đồ kỹ năng theo danh mục (Treemap)")
st.info("💡 Treemap cho thấy danh mục kỹ năng nào đang được thị trường cần nhiều nhất.")
st.plotly_chart(skill_treemap(filtered_skills), use_container_width=True)
