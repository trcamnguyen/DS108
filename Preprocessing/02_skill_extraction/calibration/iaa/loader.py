"""Data loaders for human annotation JSON and LLM parsed CSV formats.

Supported input formats:
  1. Human annotation JSON array — annotated_skills_<name>.json
     [{id, job_title, requirement, skills: [{skill_name, label, category,
       min_years, level}]}]
     Note: may contain JavaScript NaN — handled automatically.

  2. LLM parsed CSV — few_shot_parsed.csv
     Flat CSV, one row per skill: row_id, skill_name, label, category,
     min_years, level, source_text

  3. Unified JSONL — one JSON object per line:
     {record_id, annotator, skills: [...]}
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


def _fix_nan(text: str) -> str:
    """Replace bare JavaScript NaN with null for valid JSON parsing."""
    return re.sub(r"\bNaN\b", "null", text)


def _clean_skill(s: dict) -> dict:
    """Return a skill dict with guaranteed keys and clean types."""
    min_years = s.get("min_years")
    if min_years is not None:
        try:
            min_years = int(float(min_years))
        except (TypeError, ValueError):
            min_years = None

    level = s.get("level")
    if level not in ("expert", "intermediate", "basic"):
        level = None

    return {
        "skill_name": str(s.get("skill_name") or "").strip(),
        "label": str(s.get("label") or "").strip(),
        "category": str(s.get("category") or "").strip(),
        "min_years": min_years,
        "level": level,
        "source_text": str(s.get("source_text") or "").strip(),
    }


# ─── Format 1: Human annotation JSON array ───────────────────────────────────

def load_human_json(path: str | Path, annotator_name: str) -> list[dict]:
    """Load annotated_skills_<name>.json into the unified record list."""
    text = Path(path).read_text(encoding="utf-8")
    text = _fix_nan(text)
    data = json.loads(text)

    records = []
    for item in data:
        record_id = str(item.get("id", len(records)))
        skills = [_clean_skill(s) for s in item.get("skills", [])]
        skills = [s for s in skills if s["skill_name"]]  # drop empty
        records.append(
            {"record_id": record_id, "annotator": annotator_name, "skills": skills}
        )
    return records


# ─── Format 2: LLM parsed CSV ─────────────────────────────────────────────────

def load_llm_csv(path: str | Path, annotator_name: str = "llm") -> list[dict]:
    """Load few_shot_parsed.csv (flat, one row per skill) grouped by row_id."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip().str.lower()

    id_col = "row_id" if "row_id" in df.columns else "id"
    if id_col not in df.columns:
        raise ValueError(f"CSV must have a 'row_id' or 'id' column. Found: {list(df.columns)}")

    records = []
    for row_id, group in df.groupby(id_col):
        skills = []
        for _, row in group.iterrows():
            s = _clean_skill(
                {
                    "skill_name": row.get("skill_name"),
                    "label": row.get("label"),
                    "category": row.get("category"),
                    "min_years": row.get("min_years"),
                    "level": row.get("level"),
                    "source_text": row.get("source_text"),
                }
            )
            if s["skill_name"]:
                skills.append(s)
        records.append(
            {"record_id": str(row_id), "annotator": annotator_name, "skills": skills}
        )
    return records


# ─── Format 3: Unified JSONL ──────────────────────────────────────────────────

def load_unified_jsonl(path: str | Path, annotator_name: str | None = None) -> list[dict]:
    """Load unified JSONL: one {record_id, annotator, skills} object per line."""
    records = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = _fix_nan(line.strip())
        if not line:
            continue
        item = json.loads(line)
        ann = annotator_name or item.get("annotator", "unknown")
        skills_raw = item.get("skills", [])
        if isinstance(skills_raw, dict):
            skills_raw = skills_raw.get("skills", [])
        skills = [_clean_skill(s) for s in skills_raw if s.get("skill_name")]
        records.append(
            {
                "record_id": str(item.get("record_id", len(records))),
                "annotator": ann,
                "skills": skills,
            }
        )
    return records


# ─── Auto-detect loader ───────────────────────────────────────────────────────

def _detect_and_load(path: str | Path, annotator_name: str) -> list[dict]:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".csv":
        return load_llm_csv(p, annotator_name)
    if suffix == ".json":
        return load_human_json(p, annotator_name)
    if suffix in (".jsonl", ".ndjson"):
        # Try unified JSONL first; if missing 'annotator' key use provided name
        return load_unified_jsonl(p, annotator_name)
    raise ValueError(f"Unsupported file format: {suffix}")


def load_annotations(
    input_path: str | Path | None = None,
    human_a_path: str | Path | None = None,
    human_b_path: str | Path | None = None,
    llm_path: str | Path | None = None,
) -> dict[str, list[dict]]:
    """Load all annotations into a dict keyed by annotator name.

    Accepts either:
      - input_path: unified JSONL with annotator field per record, OR
      - separate paths: human_a_path, [human_b_path], [llm_path]

    Returns: {"human_a": [...], "human_b": [...], "llm": [...]}
    """
    result: dict[str, list[dict]] = {}

    if input_path is not None:
        # Unified JSONL: records from multiple annotators in one file
        records = load_unified_jsonl(input_path)
        for rec in records:
            ann = rec["annotator"]
            result.setdefault(ann, []).append(rec)
        return result

    if human_a_path is not None:
        result["human_a"] = _detect_and_load(human_a_path, "human_a")
    if human_b_path is not None:
        result["human_b"] = _detect_and_load(human_b_path, "human_b")
    if llm_path is not None:
        result["llm"] = _detect_and_load(llm_path, "llm")

    if not result:
        raise ValueError("No annotation files provided.")

    return result


def group_by_record_id(
    annotations: dict[str, list[dict]],
) -> dict[str, dict[str, list[dict]]]:
    """Reorganise into {record_id: {annotator: [skills]}}."""
    grouped: dict[str, dict[str, list[dict]]] = {}
    for annotator, records in annotations.items():
        for rec in records:
            rid = rec["record_id"]
            grouped.setdefault(rid, {})[annotator] = rec["skills"]
    return grouped
