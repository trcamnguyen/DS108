Build a multi-page Streamlit EDA dashboard for DS108 IT salary prediction project.

# Project context
- Data scraped from TopCV (2396 records) and ITViec (938 records) in Vietnam
- LLM-extracted skill annotations (~14 skill categories, required/preferred labels)
- Goal: interactive EDA for salary prediction feature analysis

# File structure expected
data/processed/
  jobs.parquet          # columns: source, job_title, salary_min, salary_max, salary_mid, experience_level, category
  skills_long_df.csv   # columns: job_id, skill_name, label, category, source
  skill_salary_df.csv  # columns: skill_name, category, median_salary, mean_salary, job_count, label_ratio_required
iaa_report.json        # { f1_llm_a, f1_llm_b, kappa_label_llm_a, kappa_label_ab, kappa_cat_llm_a, coverage_ok_pct, coverage_broken_pct }

# App structure
app.py                 # main entry, st.set_page_config, sidebar filters
pages/
  1_Overview.py
  2_Skill_Analysis.py
  3_Skill_Salary.py
  4_Data_Quality.py
utils/
  data_loader.py       # all @st.cache_data loading functions
  chart_helpers.py     # reusable plotly figure builders

# Sidebar filters (applied globally via st.session_state)
- Source: multiselect [TopCV, ITViec, Both]
- Job category: multiselect from unique values
- Experience level: multiselect [Junior, Middle, Senior, Lead]
- Salary range: slider (0 - 100M VND)
- Min skill count per job: number_input default=5

# Page 1 — Overview EDA
- 4 KPI metric cards: total jobs, median salary, mean salary, unique skills
- Histogram: salary distribution with bin width slider
- Side-by-side bar: job count by source (TopCV vs ITViec)
- Scatter plot: salary_mid vs experience years (color by source)
- Box plot: salary distribution by experience level

# Page 2 — Skill Analysis  
- Dropdown: select skill category (all 14 categories + "All")
- Slider: Top N skills to display (5-50, default 20)
- Horizontal bar chart: top N skills by frequency
- Stacked bar: required vs preferred ratio per skill (top 15)
- Treemap: all skills colored by category, sized by frequency
- Toggle: show raw counts vs % of jobs

# Page 3 — Skill–Salary Deep Dive
- Selectbox: pick any skill → box plot salary distribution for jobs requiring that skill vs not
- Heatmap: top 20 skills × salary bracket (0-20M, 20-40M, 40-60M, 60M+)
- Ranked table: top 10 highest median salary skills
- Co-occurrence heatmap: top 15 skills × top 15 skills (how often they appear together)

# Page 4 — Data Quality & IAA
- IAA score cards: F1 (LLM vs Human A/B), Cohen κ label, Cohen κ category — color coded green/yellow/red vs thresholds
- Cluster coverage donut chart: OK vs BROKEN cluster %
- Bar chart: skill count distribution per job (with threshold line at min_count=5)
- Missing data heatmap: % null per column in jobs_df

# Technical requirements
- All charts: plotly (not matplotlib) for interactivity
- Color scheme: consistent across pages — use a single COLOR_MAP dict for 14 categories
- @st.cache_data on all data loading functions
- st.session_state for sidebar filters shared across pages
- requirements.txt: streamlit, plotly, pandas, numpy
- Add st.info() boxes explaining each chart's relevance to the salary prediction goal
- Vietnamese labels where appropriate (axis titles, chart titles)
- Handle missing iaa_report.json gracefully with st.warning()

# Sample data generation
In data_loader.py, add a generate_sample_data() function that creates realistic fake data so the dashboard runs immediately without real data files. Flag it with a banner: st.warning("⚠️ Running with SAMPLE DATA — replace with real processed files")