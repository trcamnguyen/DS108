import pandas as pd
import plotly.express as px

COLOR_MAP = {
    "Programming Language":                 "#636EFA",
    "Framework / Library":                  "#EF553B",
    "Database":                             "#00CC96",
    "Infrastructure & DevOps":              "#AB63FA",
    "AI/ML/Data":                           "#FFA15A",
    "Data Engineering & Analytics":         "#19D3F3",
    "Engineering Concepts & Methodologies": "#FF6692",
    "Tool & Platform":                      "#B6E880",
    "Soft Skill":                           "#FF97FF",
    "Domain Knowledge":                     "#FECB52",
    "Testing & QA":                         "#72B7B2",
    "Design & UX":                          "#F28E2B",
    "Embedded & Firmware":                  "#8B6914",
    "IT Support & Hardware":                "#2CA02C",
    "Other":                                "#AAAAAA",
}

IAA_THRESHOLDS = {
    "f1_llm_a":          {"warn": 0.70, "ok": 0.78, "label": "F1 (LLM vs A)"},
    "f1_llm_b":          {"warn": 0.70, "ok": 0.78, "label": "F1 (LLM vs B)"},
    "f1_human_a_b":      {"warn": 0.75, "ok": 0.85, "label": "F1 (A vs B)"},
    "kappa_label_llm_a": {"warn": 0.70, "ok": 0.80, "label": "κ Label (LLM vs A)"},
    "kappa_label_llm_b": {"warn": 0.70, "ok": 0.80, "label": "κ Label (LLM vs B)"},
    "kappa_label_ab":    {"warn": 0.75, "ok": 0.85, "label": "κ Label (A vs B)"},
    "fleiss_kappa":      {"warn": 0.70, "ok": 0.80, "label": "Fleiss κ (All 3)"},
    "kappa_cat_llm_a":   {"warn": 0.60, "ok": 0.70, "label": "κ Category (LLM vs A)"},
    "kappa_cat_llm_b":   {"warn": 0.60, "ok": 0.70, "label": "κ Category (LLM vs B)"},
    "kappa_cat_ab":      {"warn": 0.65, "ok": 0.80, "label": "κ Category (A vs B)"},
}


def kpi_color(value: float, warn: float, ok: float) -> str:
    if value >= ok:
        return "green"
    if value >= warn:
        return "orange"
    return "red"


def salary_histogram(df: pd.DataFrame, bin_size: int = 5):
    fig = px.histogram(
        df, x="salary_mid",
        nbins=max(1, int(100 / bin_size)),
        title="Phân phối mức lương (triệu VND)",
        labels={"salary_mid": "Mức lương (triệu VND)", "count": "Số việc làm"},
        color_discrete_sequence=["#636EFA"],
    )
    fig.update_layout(bargap=0.05)
    return fig


def source_bar(df: pd.DataFrame):
    counts = df["source"].value_counts().reset_index()
    counts.columns = ["source", "count"]
    return px.bar(
        counts, x="source", y="count",
        title="Số việc làm theo nguồn",
        labels={"source": "Nguồn", "count": "Số việc làm"},
        color="source",
        color_discrete_map={"TopCV": "#636EFA", "ITViec": "#EF553B"},
    )


def salary_scatter(df: pd.DataFrame):
    order = ["Junior", "Middle", "Senior", "Lead"]
    df2 = df.copy()
    df2["experience_level"] = pd.Categorical(df2["experience_level"], categories=order, ordered=True)
    return px.scatter(
        df2.sort_values("experience_level"),
        x="experience_level", y="salary_mid",
        color="source",
        title="Lương theo cấp độ kinh nghiệm",
        labels={"experience_level": "Cấp độ", "salary_mid": "Lương (triệu VND)", "source": "Nguồn"},
        opacity=0.55,
        color_discrete_map={"TopCV": "#636EFA", "ITViec": "#EF553B"},
    )


def salary_boxplot(df: pd.DataFrame):
    order = ["Junior", "Middle", "Senior", "Lead"]
    df2 = df.copy()
    df2["experience_level"] = pd.Categorical(df2["experience_level"], categories=order, ordered=True)
    return px.box(
        df2.sort_values("experience_level"),
        x="experience_level", y="salary_mid",
        title="Phân phối lương theo cấp độ",
        labels={"experience_level": "Cấp độ", "salary_mid": "Lương (triệu VND)"},
        color="experience_level",
    )


def skill_hbar(df: pd.DataFrame, y_col: str = "count", top_n: int = 20, title: str = "Top kỹ năng"):
    data = df.head(top_n)
    # Dynamic height: 28px per bar, minimum 400px
    height = max(400, top_n * 28)
    # Dynamic left margin based on longest skill name
    max_name_len = data["skill_name"].str.len().max() if not data.empty else 20
    left_margin = max(160, int(max_name_len * 7.5))

    fig = px.bar(
        data, x=y_col, y="skill_name",
        orientation="h",
        color="category",
        color_discrete_map=COLOR_MAP,
        title=title,
        labels={"skill_name": "", y_col: "Số việc làm" if y_col == "count" else "% Việc làm", "category": "Danh mục"},
    )
    fig.update_layout(
        height=height,
        yaxis={"categoryorder": "total ascending", "tickfont": {"size": 13}},
        xaxis={"title_font": {"size": 13}},
        margin={"l": left_margin, "r": 20, "t": 50, "b": 40},
        legend={"orientation": "v", "x": 1.01, "y": 1},
        bargap=0.15,
    )
    return fig


def skill_treemap(skills_df: pd.DataFrame):
    counts = skills_df.groupby(["category", "skill_name"]).size().reset_index(name="count")
    return px.treemap(
        counts,
        path=["category", "skill_name"],
        values="count",
        color="category",
        color_discrete_map=COLOR_MAP,
        title="Bản đồ kỹ năng theo danh mục",
    )
