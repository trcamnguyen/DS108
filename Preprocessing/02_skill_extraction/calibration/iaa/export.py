"""Error case collection and CSV export."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .normalization import normalize_category


def _error_type_for_unmatched(
    ann_a_name: str, ann_b_name: str, which_side: str
) -> str:
    """Determine error type for skills that appear in only one annotator.

    which_side: 'only_a' or 'only_b'
    """
    if ann_a_name == "llm":
        return "llm_hallucination" if which_side == "only_a" else "llm_omission"
    if ann_b_name == "llm":
        return "llm_omission" if which_side == "only_a" else "llm_hallucination"
    # Human vs human
    return "extraction_disagreement"


def collect_errors(
    record_data: list[dict],
    ann_a_name: str,
    ann_b_name: str,
) -> list[dict]:
    """Collect all disagreement cases from one pair of annotators.

    record_data: list of {record_id, skills_a, skills_b, matched, only_a, only_b}
    """
    errors: list[dict] = []

    for rec in record_data:
        rid = rec["record_id"]
        matched = rec["matched"]
        only_a = rec["only_a"]
        only_b = rec["only_b"]

        # Skills present in A but not matched in B
        for sa in only_a:
            errors.append(
                {
                    "record_id": rid,
                    "annotator_a": ann_a_name,
                    "annotator_b": ann_b_name,
                    "skill_name_a": sa.get("skill_name", ""),
                    "skill_name_b": "",
                    "label_a": sa.get("label", ""),
                    "label_b": "",
                    "category_a": sa.get("category", ""),
                    "category_b": "",
                    "error_type": _error_type_for_unmatched(ann_a_name, ann_b_name, "only_a"),
                    "source_text_a": sa.get("source_text", ""),
                    "source_text_b": "",
                }
            )

        # Skills present in B but not matched in A
        for sb in only_b:
            errors.append(
                {
                    "record_id": rid,
                    "annotator_a": ann_a_name,
                    "annotator_b": ann_b_name,
                    "skill_name_a": "",
                    "skill_name_b": sb.get("skill_name", ""),
                    "label_a": "",
                    "label_b": sb.get("label", ""),
                    "category_a": "",
                    "category_b": sb.get("category", ""),
                    "error_type": _error_type_for_unmatched(ann_a_name, ann_b_name, "only_b"),
                    "source_text_a": "",
                    "source_text_b": sb.get("source_text", ""),
                }
            )

        # Matched pairs — check for label / category disagreement
        for sa, sb in matched:
            label_a = sa.get("label", "")
            label_b = sb.get("label", "")
            cat_a = normalize_category(sa.get("category", ""))
            cat_b = normalize_category(sb.get("category", ""))

            label_diff = label_a != label_b
            cat_diff = cat_a != cat_b

            if label_diff and cat_diff:
                etype = "both_disagreement"
            elif label_diff:
                etype = "label_disagreement"
            elif cat_diff:
                etype = "category_disagreement"
            else:
                continue  # full agreement — not an error

            errors.append(
                {
                    "record_id": rid,
                    "annotator_a": ann_a_name,
                    "annotator_b": ann_b_name,
                    "skill_name_a": sa.get("skill_name", ""),
                    "skill_name_b": sb.get("skill_name", ""),
                    "label_a": label_a,
                    "label_b": label_b,
                    "category_a": sa.get("category", ""),
                    "category_b": sb.get("category", ""),
                    "error_type": etype,
                    "source_text_a": sa.get("source_text", ""),
                    "source_text_b": sb.get("source_text", ""),
                }
            )

    return errors


def export_error_cases(error_cases: list[dict], output_path: str | Path) -> None:
    """Write error cases to CSV."""
    if not error_cases:
        return
    df = pd.DataFrame(
        error_cases,
        columns=[
            "record_id",
            "annotator_a",
            "annotator_b",
            "skill_name_a",
            "skill_name_b",
            "label_a",
            "label_b",
            "category_a",
            "category_b",
            "error_type",
            "source_text_a",
            "source_text_b",
        ],
    )
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
