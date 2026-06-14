import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

from utils.data_loader import load_all

st.set_page_config(
    page_title="DS108 — Thị trường IT Việt Nam",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

jobs_df, skills_long_df, skill_salary_df, iaa, using_sample = load_all()

if using_sample:
    st.warning(
        "⚠️ Running with SAMPLE DATA — đặt file thực vào `Dashboard/data/processed/` để xem dữ liệu thật."
    )

st.title("DS108 — Khám phá thị trường IT Việt Nam")

st.markdown(
    """
Phân tích xu hướng tuyển dụng IT Việt Nam theo kỹ năng, vai trò và mức lương.

| Nguồn   | Số bản ghi |
|---------|-----------|
| TopCV   | 2,396     |
| ITViec  | 938       |
| **Tổng** | **3,334** |

---

### Điều hướng — chọn trang trong thanh bên trái:

| Trang | Nội dung |
|-------|---------|
| **1 Tổng quan** | Thống kê cơ bản, phân phối lương, so sánh nguồn |
| **2 Phân tích kỹ năng** | Kỹ năng phổ biến, required vs preferred, treemap |
| **3 Kỹ năng & Lương** | So sánh lương theo kỹ năng, heatmap, tech stack phổ biến |
"""
)

st.divider()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Tổng việc làm", f"{len(jobs_df):,}")
c2.metric("Kỹ năng duy nhất", f"{skills_long_df['skill_name'].nunique():,}")
c3.metric("Lương median", f"{jobs_df['salary_mid'].dropna().median():.1f}M VND")

c4.metric("Danh mục ngành nghề", f"{jobs_df['category'].nunique():,}")
