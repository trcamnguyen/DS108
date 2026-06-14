import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import plotly.express as px
import streamlit as st

from utils.data_loader import load_all
from utils.sidebar import render_sidebar

st.set_page_config(page_title="Kỹ năng & Lương", layout="wide")

_BRACKETS = [(0, 20, "0–20M"), (20, 40, "20–40M"), (40, 60, "40–60M"), (60, 200, "60M+")]
_BRACKET_ORDER = [b[2] for b in _BRACKETS]


def _bracket_label(s: float) -> str:
    for lo, hi, lbl in _BRACKETS:
        if lo <= s < hi:
            return lbl
    return "60M+"


jobs_df, skills_long_df, skill_salary_df, iaa, using_sample = load_all()
if using_sample:
    st.warning("⚠️ Running with SAMPLE DATA — replace with real processed files in data/processed/")

filtered_jobs, filtered_skills = render_sidebar(jobs_df, skills_long_df)

st.title("Kỹ năng & Lương — Deep Dive")
st.info(
    "💡 Kỹ năng nào thực sự mang lại mức lương cao hơn trên thị trường tuyển dụng IT Việt Nam?"
)

if filtered_jobs.empty or filtered_skills.empty:
    st.warning("Không có dữ liệu với bộ lọc hiện tại.")
    st.stop()

skills_with_salary = filtered_skills.merge(
    filtered_jobs[["job_id", "salary_mid"]], on="job_id", how="inner"
)

# --- Section 1: Single skill box plot ---
st.divider()
st.subheader("So sánh lương: Có vs Không có kỹ năng")
st.info("💡 Kỹ năng tạo ra khoảng cách lương rõ rệt là những kỹ năng đáng đầu tư nhất.")

skill_freq = (
    filtered_skills.groupby("skill_name")["job_id"].nunique()
    .sort_values(ascending=False)
)
all_skills_sorted = skill_freq.index.tolist()

if all_skills_sorted:
    selected_skill = st.selectbox("Chọn kỹ năng", all_skills_sorted, key="sel_skill")
    jobs_with = set(filtered_skills[filtered_skills["skill_name"] == selected_skill]["job_id"])

    box_df = filtered_jobs[["job_id", "salary_mid"]].copy()
    box_df["Nhóm"] = box_df["job_id"].apply(
        lambda x: f"Có  {selected_skill}" if x in jobs_with else f"Không có  {selected_skill}"
    )
    n_with = len(jobs_with & set(filtered_jobs["job_id"]))
    n_without = len(filtered_jobs) - n_with
    st.caption(f"Có: {n_with} JD | Không có: {n_without} JD")

    fig_box = px.box(
        box_df, x="Nhóm", y="salary_mid",
        title=f"Phân phối lương — {selected_skill}",
        labels={"salary_mid": "Lương (triệu VND)"},
        color="Nhóm",
        color_discrete_sequence=["#636EFA", "#AAAAAA"],
    )
    st.plotly_chart(fig_box, use_container_width=True)

# --- Section 2: Skill × Salary Bracket Heatmap ---
st.divider()
st.subheader("Heatmap: Top 20 kỹ năng × Nhóm lương")
st.info("💡 Heatmap cho thấy kỹ năng nào chủ yếu xuất hiện ở mức lương cao và kỹ năng nào phổ biến ở mọi phân khúc.")

if not skills_with_salary.empty:
    sws = skills_with_salary.copy()
    sws["bracket"] = sws["salary_mid"].apply(_bracket_label)

    top20 = skill_freq.head(20).index.tolist()
    heat_data = (
        sws[sws["skill_name"].isin(top20)]
        .groupby(["skill_name", "bracket"])["job_id"]
        .nunique()
        .reset_index(name="count")
    )
    present_brackets = [b for b in _BRACKET_ORDER if b in heat_data["bracket"].unique()]
    heat_pivot = (
        heat_data.pivot_table(
            index="skill_name", columns="bracket", values="count", fill_value=0
        )
        .reindex(columns=present_brackets, fill_value=0)
    )
    fig_heat = px.imshow(
        heat_pivot,
        title="Top 20 kỹ năng × Nhóm lương (số JD)",
        labels={"x": "Nhóm lương", "y": "Kỹ năng", "color": "Số JD"},
        color_continuous_scale="Blues",
        aspect="auto",
    )
    st.plotly_chart(fig_heat, use_container_width=True)

# --- Section 3: Top 10 highest median salary skills ---
st.divider()
st.subheader("Top 10 kỹ năng có lương median cao nhất")
st.info("💡 Chỉ tính kỹ năng xuất hiện trong ít nhất 5 JD để đảm bảo kết quả đại diện.")

if not skills_with_salary.empty:
    top_sal = (
        skills_with_salary.groupby("skill_name")["salary_mid"]
        .agg(median_salary="median", job_count="count")
        .reset_index()
        .query("job_count >= 5")
        .sort_values("median_salary", ascending=False)
        .head(10)
    )
    top_sal["median_salary"] = top_sal["median_salary"].round(1)
    st.dataframe(
        top_sal.rename(columns={
            "skill_name":     "Kỹ năng",
            "median_salary":  "Lương Median (M VND)",
            "job_count":      "Số JD",
        }),
        use_container_width=True,
        hide_index=True,
    )

# --- Section 4: Co-occurrence Heatmap ---
st.divider()
st.subheader("Ma trận đồng xuất hiện (Top 15 kỹ năng)")
st.info("💡 Kỹ năng thường xuất hiện cùng nhau phản ánh tech stack phổ biến trên thị trường.")

top15 = skill_freq.head(15).index.tolist()

if len(top15) >= 2:
    piv = (
        filtered_skills[filtered_skills["skill_name"].isin(top15)]
        .pivot_table(index="job_id", columns="skill_name", values="label", aggfunc="count")
        .fillna(0)
        .clip(upper=1)
    )
    cooc = piv.T.dot(piv).astype(int)

    fig_cooc = px.imshow(
        cooc,
        title="Ma trận đồng xuất hiện kỹ năng (số JD cùng xuất hiện)",
        color_continuous_scale="Viridis",
        labels={"color": "Số JD"},
        aspect="auto",
    )
    fig_cooc.update_xaxes(tickangle=-35)
    st.plotly_chart(fig_cooc, use_container_width=True)
