import re

import pandas as pd


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b[a-zA-Z0-9]+\b", str(text).lower())


def map_standardized_title(title: str) -> str:
    t = str(title).strip().lower()
    tokens = tokenize(t)

    # ---------------------------------
    # QA Engineer
    # ---------------------------------
    if (
        "tester" in tokens
        or "qa" in tokens
        or "quality assurance" in t
    ):
        return "QA Engineer"

    # ---------------------------------
    # UI/UX Designer
    # ---------------------------------
    if (
        "ui/ux" in t
        or "ux/ui" in t
        or ("ui" in tokens and "ux" in tokens)
        or ("ui" in tokens and "designer" in tokens)
        or ("ux" in tokens and "designer" in tokens)
    ):
        return "UI/UX Designer"

    # ---------------------------------
    # Fullstack Developer
    # ---------------------------------
    if (
        "fullstack" in tokens
        or ("full" in tokens and "stack" in tokens)
    ):
        return "Fullstack Developer"

    # ---------------------------------
    # Software Engineer
    # ---------------------------------
    if t in {"software developer"}:
        return "Software Engineer"

    # ---------------------------------
    # Solution Architect
    # ---------------------------------
    if t in {"solutions architect", "it architect"}:
        return "Solution Architect"

    # ---------------------------------
    # IT Support
    # ---------------------------------
    if t in {"application support", "it technician"}:
        return "IT Support"

    # ---------------------------------
    # Solution Engineer
    # ---------------------------------
    if t in {"solutions engineer"}:
        return "Solution Engineer"

    # ---------------------------------
    # Embedded Engineer
    # ---------------------------------
    if (
        t in {"iot engineer", "embedded developer", "firmware engineer", "fpga engineer"}
        or "embedded" in tokens
    ):
        return "Embedded Engineer"

    # ---------------------------------
    # Project Manager
    # ---------------------------------
    if t in {
        "it manager",
        "project managment office",
        "project management office",
        "project management officer",
    }:
        return "Project Manager"

    # ---------------------------------
    # Game Developer
    # ---------------------------------
    if t in {"unity developer"}:
        return "Game Developer"

    # ---------------------------------
    # AI Engineer
    # ---------------------------------
    if t in {
        "computer vision engineer",
        "machine vision engineer",
        "voice engineer",
        "nlp engineer",
    }:
        return "AI Engineer"

    # ---------------------------------
    # Machine Learning Engineer
    # ---------------------------------
    if t in {"ml engineer"}:
        return "Machine Learning Engineer"

    # ---------------------------------
    # Bridge Engineer
    # ---------------------------------
    if "bridge" in tokens or "brse" in tokens:
        return "Bridge Engineer"

    # ---------------------------------
    # Presales Engineer
    # ---------------------------------
    if (
        "presales" in tokens
        or ("pre" in tokens and "sales" in tokens)
    ):
        return "Presales Engineer"

    # ---------------------------------
    # Mobile Developer
    # ---------------------------------
    if t in {"ios developer"}:
        return "Mobile Developer"

    # ---------------------------------
    # ERP Developer
    # ---------------------------------
    if t in {"odoo developer"}:
        return "ERP Developer"

    # default
    return title


SEMANTIC_MERGE_MAP: dict[str, str] = {
    "Systems Engineer": "System Engineer",
    "Low Code Developer": "Low-Code Developer",
    "AI Specialist": "AI Engineer",
    "AI/ML Engineer": "AI Engineer",
    "Cybersecurity Engineer": "Security Engineer",
    "Security Specialist": "Security Engineer",
    "Application Support Engineer": "IT Support",
    "Product Support": "IT Support",
    "IT Project Manager": "Project Manager",
    "Database Administrator": "Database Engineer",
    "Database Developer": "Database Engineer",
    "Infrastructure Architect": "Solution Architect",
    "Network Architect": "Solution Architect",
    "ERP Specialist": "ERP Consultant",
    "Software Implementer": "Implementation Specialist",
    "Software Implementation Specialist": "Implementation Specialist",
    "Implementation Engineer": "Implementation Specialist",
    "Implementation Consultant": "Implementation Specialist",
    "DevSecOps Engineer": "DevOps Engineer",
    "Infrastructure Engineer": "Cloud Engineer",
    "Data Center Engineer": "Cloud Engineer",
    "Product Owner": "Product Manager",
    "Data Governance Specialist": "Data Governance Analyst",
    "NOC Engineer": "IT Operations",
    "Functional Consultant": "ERP Consultant",
    "Hardware Technician": "Hardware Engineer",
    "Quality Assurance Engineer": "QA Engineer",
}


def normalize_titles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Áp dụng map_standardized_title() rồi SEMANTIC_MERGE_MAP
    lên cột standardized_title. Trả về DataFrame mới (không mutate input).
    """
    out = df.copy()
    out["standardized_title"] = (
        out["standardized_title"]
        .apply(map_standardized_title)
        .replace(SEMANTIC_MERGE_MAP)
    )
    return out
