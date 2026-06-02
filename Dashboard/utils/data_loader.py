import json
import os

import numpy as np
import pandas as pd
import streamlit as st

SKILL_CATEGORIES = [
    "Programming Language", "Framework / Library", "Database",
    "Infrastructure & DevOps", "AI/ML/Data", "Data Engineering & Analytics",
    "Engineering Concepts & Methodologies", "Tool & Platform",
    "Soft Skill", "Domain Knowledge", "Testing & QA",
    "Design & UX", "Embedded & Firmware", "IT Support & Hardware", "Other",
]

_DASHBOARD_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJECT_DIR = os.path.dirname(_DASHBOARD_DIR)
DATA_DIR = os.path.join(_DASHBOARD_DIR, "data", "processed")

_REAL_JOBS_PATH   = os.path.join(_PROJECT_DIR, "data", "processed", "jobs_cleaned.parquet")
_REAL_SKILLS_PATH = os.path.join(_PROJECT_DIR, "data", "processed", "skills.parquet")

_LEVEL_MAP = {
    "Junior": "Junior", "Mid": "Middle", "Senior": "Senior",
    "Lead": "Lead", "Manager": "Lead", "Unknown": "Unknown",
}

_IAA_SEARCH_PATHS = [
    os.path.join(_DASHBOARD_DIR, "iaa_report.json"),
    os.path.join(
        _PROJECT_DIR,
        "Preprocessing", "02_skill_extraction",
        "calibration", "iaa_results", "iaa_report.json",
    ),
]


def _load_real_data():
    """Load and transform real project parquet files into dashboard format."""
    if not (os.path.exists(_REAL_JOBS_PATH) and os.path.exists(_REAL_SKILLS_PATH)):
        return None, None, None

    raw_jobs = pd.read_parquet(_REAL_JOBS_PATH)

    # Salary: set thoa_thuan rows to NaN, cap extreme outliers
    salary_mid = raw_jobs["avg_salary"].where(raw_jobs["avg_salary"] > 0).clip(upper=200)
    salary_min = raw_jobs["min_salary"].where(raw_jobs["min_salary"] > 0)
    salary_max = raw_jobs["max_salary"].where(raw_jobs["max_salary"] > 0)

    jobs_df = pd.DataFrame({
        "job_id":           raw_jobs["job_id"],
        "source":           raw_jobs["source"].map({"topcv": "TopCV", "itviec": "ITViec"}),
        "job_title":        raw_jobs["standardized_title"],
        "salary_min":       salary_min,
        "salary_max":       salary_max,
        "salary_mid":       salary_mid,
        "experience_level": raw_jobs["job_level_tier"].map(_LEVEL_MAP),
        "category":         raw_jobs["standardized_title"],
    }).reset_index(drop=True)

    raw_skills = pd.read_parquet(_REAL_SKILLS_PATH)
    src_map = dict(zip(raw_jobs["job_id"],
                       raw_jobs["source"].map({"topcv": "TopCV", "itviec": "ITViec"})))

    skills_long_df = raw_skills[["job_id", "final_canonical", "label", "category"]].copy()
    skills_long_df = skills_long_df.rename(columns={"final_canonical": "skill_name"})
    skills_long_df = skills_long_df[~skills_long_df["skill_name"].str.endswith(" Other")]
    skills_long_df["source"] = skills_long_df["job_id"].map(src_map)
    skills_long_df = skills_long_df.reset_index(drop=True)

    merged = skills_long_df.merge(
        jobs_df[["job_id", "salary_mid"]].dropna(subset=["salary_mid"]),
        on="job_id", how="inner",
    )
    skill_salary_df = (
        merged.groupby(["skill_name", "category"])
        .agg(
            median_salary=("salary_mid", "median"),
            mean_salary=("salary_mid", "mean"),
            job_count=("job_id", "nunique"),
            label_ratio_required=("label", lambda x: (x == "required_skill").mean()),
        )
        .reset_index()
        .round(2)
    )

    return jobs_df, skills_long_df, skill_salary_df


def generate_sample_data():
    rng = np.random.default_rng(42)
    n_jobs = 650

    sources = rng.choice(["TopCV", "ITViec"], n_jobs, p=[0.72, 0.28])
    title_pool = [
        "Backend Developer", "Frontend Developer", "Full Stack Developer",
        "Data Engineer", "Data Scientist", "ML Engineer", "DevOps Engineer",
        "Mobile Developer", "QA Engineer", "System Analyst",
        "Embedded Engineer", "Security Engineer", "iOS Developer",
        "Android Developer", "Cloud Architect",
    ]
    job_titles = rng.choice(title_pool, n_jobs)
    exp_levels = rng.choice(
        ["Junior", "Middle", "Senior", "Lead"], n_jobs, p=[0.25, 0.40, 0.25, 0.10]
    )
    base_sal = {"Junior": 12.0, "Middle": 22.0, "Senior": 38.0, "Lead": 55.0}
    salary_mid = np.array(
        [float(base_sal[e]) * float(rng.lognormal(0.0, 0.25)) for e in exp_levels]
    )
    salary_mid = np.clip(salary_mid, 5.0, 100.0).round(1)
    salary_min = (salary_mid * rng.uniform(0.70, 0.90, n_jobs)).round(1)
    salary_max = (salary_mid * rng.uniform(1.10, 1.40, n_jobs)).round(1)
    cat_pool = ["Backend", "Frontend", "Data", "Mobile", "DevOps", "QA", "Full Stack", "Embedded", "Security"]
    categories = rng.choice(cat_pool, n_jobs)

    jobs_df = pd.DataFrame({
        "job_id": range(n_jobs),
        "source": sources,
        "job_title": job_titles,
        "salary_min": salary_min,
        "salary_max": salary_max,
        "salary_mid": salary_mid,
        "experience_level": exp_levels,
        "category": categories,
    })

    skills_pool = {
        "Programming Language": ["Python", "Java", "JavaScript", "TypeScript", "Go", "C++", "Kotlin", "Swift", "PHP", "Rust"],
        "Framework":            ["React", "Vue", "Angular", "Spring Boot", "Django", "FastAPI", "Laravel", "Flutter", "Next.js", "Express"],
        "Database":             ["MySQL", "PostgreSQL", "MongoDB", "Redis", "Elasticsearch", "Oracle", "Cassandra", "DynamoDB"],
        "Cloud & DevOps":       ["AWS", "Docker", "Kubernetes", "CI/CD", "Azure", "GCP", "Jenkins", "Terraform", "Linux", "Ansible"],
        "AI/ML/Data":           ["Machine Learning", "Deep Learning", "NLP", "TensorFlow", "PyTorch", "Scikit-learn", "Computer Vision", "LLM"],
        "Data Engineering":     ["Apache Kafka", "Airflow", "Spark", "ETL", "Power BI", "Tableau", "dbt", "Hadoop"],
        "Methodology":          ["OOP", "Agile", "REST API", "Microservices", "SOLID", "Design Patterns", "TDD", "Scrum"],
        "Tool & Platform":      ["Git", "Jira", "Linux", "Postman", "VS Code", "Confluence", "Figma", "MATLAB"],
        "Soft Skill":           ["Communication", "Teamwork", "Problem Solving", "English", "Leadership", "Work Ethic", "Japanese"],
        "Domain Knowledge":     ["FinTech", "eCommerce", "Embedded Systems", "Healthcare IT", "ERP", "Logistics"],
        "Other":                ["Algorithm", "Data Structures", "System Design", "OOP Design"],
    }
    all_skills = [(s, c) for c, ss in skills_pool.items() for s in ss]

    rows = []
    for _, job in jobs_df.iterrows():
        n_skills = int(rng.integers(5, 18))
        chosen = rng.choice(len(all_skills), min(n_skills, len(all_skills)), replace=False)
        for idx in chosen:
            skill_name, cat = all_skills[int(idx)]
            label = "required_skill" if rng.random() > 0.28 else "preferred_skill"
            rows.append({
                "job_id": int(job["job_id"]),
                "skill_name": skill_name,
                "label": label,
                "category": cat,
                "source": str(job["source"]),
            })

    skills_long_df = pd.DataFrame(rows)

    merged = skills_long_df.merge(jobs_df[["job_id", "salary_mid"]], on="job_id")
    skill_salary_df = (
        merged.groupby(["skill_name", "category"])
        .agg(
            median_salary=("salary_mid", "median"),
            mean_salary=("salary_mid", "mean"),
            job_count=("job_id", "nunique"),
            label_ratio_required=("label", lambda x: (x == "required_skill").mean()),
        )
        .reset_index()
        .round(2)
    )

    return jobs_df, skills_long_df, skill_salary_df


def _load_iaa_raw():
    for path in _IAA_SEARCH_PATHS:
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return json.load(f)
    return None


def parse_iaa(raw):
    """Normalise nested IAA report to flat dict consumed by the dashboard."""
    if raw is None:
        return None
    if "t1_extraction" not in raw:
        return raw  # already flat format
    t1 = raw["t1_extraction"]
    t2 = raw["t2_label"]
    t3 = raw["t3_category"]
    return {
        "f1_llm_a":          t1["llm_vs_human_a"]["f1"],
        "f1_llm_b":          t1["llm_vs_human_b"]["f1"],
        "f1_human_a_b":      t1["human_a_vs_human_b"]["f1"],
        "kappa_label_llm_a": t2["llm_vs_human_a"]["kappa"],
        "kappa_label_llm_b": t2["llm_vs_human_b"]["kappa"],
        "kappa_label_ab":    t2["human_a_vs_human_b"]["kappa"],
        "fleiss_kappa":      t2["fleiss_kappa_all"]["kappa"],
        "kappa_cat_llm_a":   t3["llm_vs_human_a"]["kappa"],
        "kappa_cat_llm_b":   t3["llm_vs_human_b"]["kappa"],
        "kappa_cat_ab":      t3["human_a_vs_human_b"]["kappa"],
        "overall_status":    raw.get("overall_status", "UNKNOWN"),
        "n_error_cases":     raw.get("n_error_cases", 0),
        "_ci": {
            "f1_llm_a":          t1["llm_vs_human_a"]["ci_95"],
            "kappa_label_llm_a": t2["llm_vs_human_a"]["ci_95"],
            "kappa_cat_llm_a":   t3["llm_vs_human_a"]["ci_95"],
            "kappa_label_ab":    t2["human_a_vs_human_b"]["ci_95"],
        },
    }


@st.cache_data
def load_all():
    # Priority 1: real project parquet files
    jobs_df, skills_long_df, skill_salary_df = _load_real_data()
    using_sample = False

    if jobs_df is None:
        # Priority 2: pre-exported CSV files in Dashboard/data/processed/
        jobs_path   = os.path.join(DATA_DIR, "jobs_df.csv")
        skills_path = os.path.join(DATA_DIR, "skills_long_df.csv")
        sal_path    = os.path.join(DATA_DIR, "skill_salary_df.csv")
        if os.path.exists(jobs_path) and os.path.exists(skills_path) and os.path.exists(sal_path):
            jobs_df         = pd.read_csv(jobs_path)
            skills_long_df  = pd.read_csv(skills_path)
            skill_salary_df = pd.read_csv(sal_path)
        else:
            # Priority 3: generated sample data
            jobs_df, skills_long_df, skill_salary_df = generate_sample_data()
            using_sample = True

    iaa = parse_iaa(_load_iaa_raw())
    return jobs_df, skills_long_df, skill_salary_df, iaa, using_sample
