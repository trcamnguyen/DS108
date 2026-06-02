import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from utils.data_loader import load_all

st.set_page_config(page_title="Dự đoán Lương IT", layout="wide")

# ── Model & data ──────────────────────────────────────────────────────────────
_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_MODEL_PATH = os.path.join(_PROJECT_DIR, "Model", "salary_predictor_1.joblib")
_SKILLS_PATH = os.path.join(_PROJECT_DIR, "data", "processed", "skills.parquet")


@st.cache_resource
def _load_model():
    return joblib.load(_MODEL_PATH)


@st.cache_data
def _load_skill_options():
    if not os.path.exists(_SKILLS_PATH):
        return pd.DataFrame(columns=["skill_name", "category"])
    raw = pd.read_parquet(_SKILLS_PATH, columns=["final_canonical", "category"])
    raw = raw.rename(columns={"final_canonical": "skill_name"})
    raw = raw[~raw["skill_name"].str.endswith(" Other", na=False)]
    raw = raw.dropna(subset=["skill_name", "category"])
    deduped = (
        raw.groupby("skill_name")["category"]
        .agg(lambda x: x.mode().iloc[0])
        .reset_index()
    )
    return deduped.sort_values("skill_name")


# ── Constants ─────────────────────────────────────────────────────────────────
ROLES = [
    "AI Application Specialist", "AI Automation Engineer", "AI Data Annotator",
    "AI Engineer", "App Monetization Specialist", "Automation Developer",
    "Automation Engineer", "Automation Test Engineer", "Backend Developer",
    "Blockchain Developer", "Bridge Engineer", "Business Analyst",
    "Chief Technology Officer", "Cloud Engineer", "Data Analyst",
    "Data Analyst Training Manager", "Data Architect", "Data Engineer",
    "Data Governance Analyst", "Data Labeling Coordinator",
    "Data Management Specialist", "Data Scientist", "Database Administrator",
    "Desktop Application Developer", "DevOps Engineer",
    "Digital Transformation Manager", "Digitalization Manager",
    "E-commerce Administrator", "ERP Consultant", "Embedded Engineer",
    "Engineering Manager", "Enterprise Architect", "Frontend Developer",
    "Fullstack Developer", "Functional Safety Engineer", "GIS Developer",
    "Game Analyst", "Game Designer", "Game Developer", "Game Level Designer",
    "Game Operations Specialist", "Game Producer", "Growth Advisor",
    "Hardware Test Engineer", "IT Account Manager", "IT Auditor",
    "IT Governance", "IT Instructor", "IT Procurement Specialist",
    "IT Project Coordinator", "IT Recruiter", "IT Sales Engineer",
    "IT Solutions Consultant", "IT Support", "IT Teacher", "IT Translator",
    "Implementation Specialist", "Infrastructure Engineer", "Level Designer",
    "Machine Learning Engineer", "Mainframe Engineer", "Mobile Developer",
    "Monetization Optimizer", "Network Engineer", "Network Security Engineer",
    "Outsystem Developer", "Power Apps Developer", "Power Platform Developer",
    "Presales Engineer", "Product Configuration Specialist", "Product Manager",
    "Product Operations Executive", "Product Owner", "Programming Tutor",
    "Project Manager", "Python Instructor", "QA Engineer", "Renewal Specialist",
    "SDK Developer", "Sales Engineer", "Scrum Master", "Security Engineer",
    "Shopify Developer", "Simulation Engineer", "Software Architect",
    "Software Consultant", "Software Engineer", "Software Renewal Specialist",
    "Solution Engineer", "System Administrator", "System Engineer",
    "System Integration Engineer", "Technical Consultant", "Technical Lead",
    "Technical Project Coordinator", "UI/UX Designer", "Website Administrator",
]

JOB_LEVELS = [
    "Intern", "Fresher", "Junior", "Nhân viên", "Senior",
    "Trưởng nhóm", "Trưởng/Phó phòng", "Quản lý / Giám sát", "Giám đốc",
]

LOCATIONS = [
    "Hồ Chí Minh", "Hà Nội", "Đà Nẵng", "Bình Dương", "Hải Phòng",
    "Bắc Ninh", "Cần Thơ", "Đồng Nai", "Hưng Yên", "Khánh Hòa",
    "Thái Nguyên", "An Giang", "Huế", "Nghệ An", "Quảng Ninh",
    "Thanh Hóa", "Foreign:Japan", "Khác", "Unknown",
]

INDUSTRIES = [
    "IT - Phần mềm", "Software Products and Web Services",
    "IT Services and IT Consulting", "IT - Phần cứng",
    "Thương mại điện tử", "Tài chính", "Financial Services",
    "Giáo dục / Đào tạo", "Sản xuất", "Viễn thông",
    "Marketing / Truyền thông / Quảng cáo",
    "Dược phẩm / Y tế / Công nghệ sinh học",
    "Bán lẻ - Hàng tiêu dùng - FMCG", "Other", "Khác",
]

EDUCATIONS = [
    "Đại học trở lên", "Đại Học trở lên", "Cao Đẳng trở lên",
    "Thạc sĩ trở lên", "Cao học trở lên", "Trung cấp trở lên",
    "Trung học phổ thông (Cấp 3) trở lên", "Không yêu cầu",
]

CAT_FEATURES = [
    "cat_ai_ml_data", "cat_data_engineering_&_analytics", "cat_database",
    "cat_design_&_ux", "cat_domain_knowledge", "cat_embedded_&_firmware",
    "cat_engineering_concepts_&_methodologies", "cat_framework___library",
    "cat_it_support_&_hardware", "cat_infrastructure_&_devops",
    "cat_other", "cat_programming_language", "cat_soft_skill",
    "cat_testing_&_qa", "cat_tool_&_platform",
]


def _cat_to_feature(cat: str) -> str:
    return "cat_" + cat.lower().replace(" ", "_").replace("/", "_")


def _build_input(role, exp_years, job_level, selected_skills, skill_df,
                 source, location, industry, education):
    cat_counts = {f: 0 for f in CAT_FEATURES}
    for skill in selected_skills:
        rows = skill_df[skill_df["skill_name"] == skill]
        if not rows.empty:
            feat = _cat_to_feature(rows.iloc[0]["category"])
            if feat in cat_counts:
                cat_counts[feat] += 1
            else:
                cat_counts["cat_other"] += 1

    n_skills = len(selected_skills)
    row = {
        "source":            source,
        "location":          location,
        "industry_cleaned":  industry,
        "education":         education,
        "job_level":         job_level,
        "standardized_title": role,
        "experience_years":  exp_years,
        "n_skills":          n_skills,
        "n_preferred":       0,
        "n_required":        n_skills,
        **cat_counts,
        "level_basic":       0,
        "level_expert":      0,
        "level_intermediate": 0,
        "has_year_requirement": int(exp_years > 0),
    }
    return pd.DataFrame([row])


# ── Page layout ───────────────────────────────────────────────────────────────
st.title("Dự đoán Lương IT")
st.info(
    "Chọn vai trò, số năm kinh nghiệm, level và kỹ năng để nhận ước tính mức lương "
    "trên thị trường IT Việt Nam."
)

model = _load_model()
skill_df = _load_skill_options()

jobs_df, skills_long_df, _, _, using_sample = load_all()

if "skill_basket" not in st.session_state:
    st.session_state.skill_basket = set()

# ── Inputs ────────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    role = st.selectbox(
        "Vai trò (Role) *",
        ROLES,
        index=ROLES.index("Software Engineer"),
    )

with col2:
    exp_years = st.slider("Số năm kinh nghiệm *", min_value=0, max_value=15, value=2, step=1)

with col3:
    job_level = st.selectbox(
        "Level *",
        JOB_LEVELS,
        index=JOB_LEVELS.index("Junior"),
    )

# ── Skill picker (category → skills) ─────────────────────────────────────────
st.markdown("**Kỹ năng**")
CATEGORIES = sorted(skill_df["category"].unique().tolist())

sk_col1, sk_col2 = st.columns([1, 2])

with sk_col1:
    sel_cat = st.selectbox(
        "Danh mục",
        options=["(Tất cả)"] + CATEGORIES,
        key="sel_category",
    )

if sel_cat == "(Tất cả)":
    cat_skills = skill_df["skill_name"].sort_values().tolist()
else:
    cat_skills = (
        skill_df[skill_df["category"] == sel_cat]["skill_name"]
        .sort_values()
        .tolist()
    )

cat_skills_set = set(cat_skills)
default_for_cat = sorted(st.session_state.skill_basket & cat_skills_set)
ms_key = f"_skill_ms_{sel_cat}"

def _update_basket():
    new_sel = set(st.session_state[ms_key])
    outside_cat = st.session_state.skill_basket - cat_skills_set
    st.session_state.skill_basket = outside_cat | new_sel

with sk_col2:
    st.multiselect(
        f"Kỹ năng trong danh mục ({len(cat_skills)})",
        options=cat_skills,
        default=default_for_cat,
        key=ms_key,
        on_change=_update_basket,
        placeholder="Chọn kỹ năng...",
    )

# ── Skill basket ──────────────────────────────────────────────────────────────
basket = sorted(st.session_state.skill_basket)
if basket:
    basket_cols = st.columns([4, 1])
    with basket_cols[0]:
        skill_tags = "  ".join(f"`{s}`" for s in basket)
        st.caption(f"**{len(basket)} kỹ năng đã chọn:** {skill_tags}")
    with basket_cols[1]:
        if st.button("Xóa tất cả", key="clear_basket"):
            st.session_state.skill_basket = set()
            st.rerun()
else:
    st.caption("Chưa chọn kỹ năng nào — bạn vẫn có thể dự đoán dựa trên vai trò và level.")

selected_skills = basket

with st.expander("Tuỳ chọn nâng cao", expanded=False):
    adv1, adv2, adv3 = st.columns(3)
    with adv1:
        location = st.selectbox("Địa điểm", LOCATIONS, index=0)
    with adv2:
        industry = st.selectbox("Ngành", INDUSTRIES, index=0)
    with adv3:
        source = st.selectbox("Nguồn dữ liệu", ["topcv", "itviec"], index=0)
        education = st.selectbox("Học vấn", EDUCATIONS, index=0)

st.divider()

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("Dự đoán lương", type="primary", use_container_width=True):
    X = _build_input(
        role, exp_years, job_level, selected_skills, skill_df,
        source, location, industry, education,
    )
    log_pred = float(model.predict(X)[0])
    pred = float(np.exp(log_pred))
    pred = max(pred, 0.0)

    lo = pred * 0.80
    hi = pred * 1.25

    st.success(f"### Lương ước tính: **{pred:.1f} triệu VND / tháng**")

    m1, m2, m3 = st.columns(3)
    m1.metric("Thấp (–20%)", f"{lo:.1f}M")
    m2.metric("Ước tính", f"{pred:.1f}M", delta=None)
    m3.metric("Cao (+25%)", f"{hi:.1f}M")

    st.caption(
        "Khoảng ước tính dựa trên độ biến động lương thị trường ±20–25%. "
        "Mô hình được huấn luyện trên dữ liệu TopCV + ITViec (2025 Q2)."
    )

    # ── Skill breakdown ────────────────────────────────────────────────────────
    if selected_skills:
        st.divider()
        st.subheader("Phân bổ kỹ năng đã chọn")
        cat_data = []
        for skill in selected_skills:
            rows = skill_df[skill_df["skill_name"] == skill]
            cat = rows.iloc[0]["category"] if not rows.empty else "Other"
            cat_data.append({"Kỹ năng": skill, "Danh mục": cat})

        cat_df = pd.DataFrame(cat_data)
        cat_count = cat_df["Danh mục"].value_counts().reset_index()
        cat_count.columns = ["Danh mục", "Số lượng"]

        fig = px.bar(
            cat_count,
            x="Danh mục", y="Số lượng",
            title="Số kỹ năng theo danh mục",
            color="Danh mục",
            text="Số lượng",
        )
        fig.update_layout(showlegend=False, xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(cat_df, use_container_width=True, hide_index=True)

    # ── Market comparison ──────────────────────────────────────────────────────
    st.divider()
    st.subheader("So sánh với thị trường")

    has_salary = jobs_df.dropna(subset=["salary_mid"])
    role_jobs = has_salary[
        (has_salary["job_title"] == role) |
        (has_salary.get("category", pd.Series(dtype=str)) == role)
    ]

    if len(role_jobs) >= 5:
        market_med = role_jobs["salary_mid"].median()
        market_min = role_jobs["salary_mid"].quantile(0.25)
        market_max = role_jobs["salary_mid"].quantile(0.75)

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Median thị trường (vai trò này)", f"{market_med:.1f}M")
        mc2.metric("Tứ vị 25%", f"{market_min:.1f}M")
        mc3.metric("Tứ vị 75%", f"{market_max:.1f}M")
        mc4.metric("Số JD có lương", len(role_jobs))

        fig2 = px.histogram(
            role_jobs, x="salary_mid", nbins=20,
            title=f"Phân phối lương — {role}",
            labels={"salary_mid": "Lương (triệu VND)"},
            color_discrete_sequence=["#636EFA"],
        )
        fig2.add_vline(
            x=pred, line_dash="dash", line_color="red",
            annotation_text=f"Dự đoán: {pred:.1f}M",
            annotation_position="top right",
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info(
            f"Không đủ dữ liệu lương cho vai trò **{role}** để so sánh thị trường "
            f"(cần ≥ 5 JD có lương, hiện có: {len(role_jobs)})."
        )

    # ── Overall market by level ────────────────────────────────────────────────
    if "experience_level" in has_salary.columns:
        level_map = {
            "Intern": "Intern", "Fresher": "Fresher", "Junior": "Junior",
            "Nhân viên": "Middle", "Senior": "Senior",
            "Trưởng nhóm": "Lead", "Trưởng/Phó phòng": "Lead",
            "Quản lý / Giám sát": "Lead", "Giám đốc": "Principal",
        }
        user_level_key = level_map.get(job_level, job_level)
        level_jobs = has_salary[has_salary["experience_level"] == user_level_key]
        if len(level_jobs) >= 10:
            level_med = level_jobs["salary_mid"].median()
            delta = pred - level_med
            st.metric(
                f"Median lương toàn thị trường — level {user_level_key}",
                f"{level_med:.1f}M",
                delta=f"{delta:+.1f}M so với dự đoán của bạn",
            )
