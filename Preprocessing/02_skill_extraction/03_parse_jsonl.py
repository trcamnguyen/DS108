"""
DS108 — Parse full_raw.jsonl → full_parsed.csv
================================================
Input  : Preprocessing/02_skill_extraction/output_full/full_raw.jsonl
Output : Preprocessing/02_skill_extraction/output_full/full_parsed.csv
         Preprocessing/02_skill_extraction/output_full/full_errors.jsonl

Chạy độc lập sau khi 02_full_extraction.py đã hoàn thành (hoặc giữa chừng).
Logic parse y hệt _flush_parsed_csv trong 02_full_extraction.py.
"""

import re
import json
import logging
import pandas as pd
from pathlib import Path
from pydantic import BaseModel, ValidationError
from typing import List, Optional, Literal

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ─── PYDANTIC SCHEMA ──────────────────────────────────────────────────────────

class Skill(BaseModel):
    skill_name: str
    label: Literal["required_skill", "preferred_skill"]
    category: str
    min_years: Optional[int] = None
    level: Optional[Literal["expert", "intermediate", "basic"]] = None
    source_text: str

class SkillExtractionOutput(BaseModel):
    skills: List[Skill]

# ─── JSON REPAIR ──────────────────────────────────────────────────────────────

def _repair_truncated_json(raw_text: str) -> Optional[dict]:
    cleaned = re.sub(r"^```(?:json)?\n?", "", raw_text.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\n?```$", "", cleaned.strip())
    no_trailing_comma = re.sub(r',(\s*[}\]])', r'\1', cleaned)
    try:
        return json.loads(no_trailing_comma)
    except json.JSONDecodeError:
        pass
    matches = list(re.finditer(r'\n[ \t]{4}\}', cleaned))
    for match in reversed(matches):
        candidate = cleaned[:match.end()] + "\n  ]\n}"
        candidate = re.sub(r',(\s*[}\]])', r'\1', candidate)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None

_VALID_LABELS = {"required_skill", "preferred_skill"}
_VALID_LEVELS = {"expert", "intermediate", "basic", None}

def _coerce_skills(data: dict) -> dict:
    fixed = []
    for s in data.get("skills", []):
        s = dict(s)
        if s.get("label") not in _VALID_LABELS:
            s["label"] = "required_skill"
        if s.get("level") not in _VALID_LEVELS:
            s["level"] = None
        if s.get("min_years") is not None:
            try:
                s["min_years"] = int(round(float(s["min_years"])))
            except (ValueError, TypeError):
                s["min_years"] = None
        fixed.append(s)
    return {"skills": fixed}

# ─── PARSER ───────────────────────────────────────────────────────────────────

def parse_raw_entry(entry: dict) -> tuple[list[dict], dict | None]:
    """
    Trả về (skill_rows, error_entry).
    - skill_rows: list các dict flat (1 dòng / skill), rỗng nếu thất bại.
    - error_entry: dict nếu parse thất bại, None nếu thành công.
    """
    row_id    = entry.get("row_id", "")
    raw_text  = entry.get("raw_text")
    source    = entry.get("source", "")
    std_title = entry.get("standardized_title", "")
    url       = entry.get("url", "")

    # Không có raw_text → không thể parse
    if not raw_text:
        return [], {**entry}

    # Có error nhưng vẫn có raw_text → thử parse lại (e.g. min_years float)
    # Nếu parse thành công bên dưới, error được bỏ qua

    # Thử parse thẳng
    try:
        cleaned = re.sub(r"^```(?:json)?\n?", "", raw_text.strip(), flags=re.IGNORECASE)
        cleaned = re.sub(r"\n?```$", "", cleaned.strip())
        validated = SkillExtractionOutput(**json.loads(cleaned))
        skills = validated.model_dump()["skills"]
    except Exception:
        # Thử repair
        repaired = _repair_truncated_json(raw_text)
        if not repaired:
            return [], {**entry, "parse_error": "json_repair_failed"}
        try:
            validated = SkillExtractionOutput(**_coerce_skills(repaired))
            skills = validated.model_dump()["skills"]
            log.warning(f"Repaired JSON for row_id={row_id}")
        except (ValidationError, Exception) as e:
            return [], {**entry, "parse_error": str(e)}

    skill_rows = [
        {
            "row_id":             row_id,
            "source":             source,
            "standardized_title": std_title,
            "url":                url,
            **skill,
        }
        for skill in skills
    ]
    return skill_rows, None


def parse_jsonl(raw_file: Path, parsed_file: Path, error_file: Path) -> None:
    if not raw_file.exists():
        log.error(f"Không tìm thấy file: {raw_file}")
        return

    all_skill_rows: list[dict] = []
    all_errors:     list[dict] = []
    total = 0

    with open(raw_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                log.warning(f"Malformed JSONL line skipped: {e}")
                continue

            total += 1
            skill_rows, error_entry = parse_raw_entry(entry)
            all_skill_rows.extend(skill_rows)
            if error_entry is not None:
                all_errors.append(error_entry)

    log.info(f"Processed {total} entries → {len(all_skill_rows)} skills | {len(all_errors)} errors")

    if all_skill_rows:
        pd.DataFrame(all_skill_rows).to_csv(parsed_file, index=False, encoding="utf-8-sig")
        log.info(f"Wrote {len(all_skill_rows)} skill rows → {parsed_file.name}")

    if all_errors:
        with open(error_file, "w", encoding="utf-8") as f:
            for e in all_errors:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        log.info(f"Wrote {len(all_errors)} error entries → {error_file.name}")


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    OUT_DIR     = Path(__file__).parent / "output_full"
    RAW_FILE    = OUT_DIR / "full_raw.jsonl"
    PARSED_FILE = OUT_DIR / "full_parsed.csv"
    ERROR_FILE  = OUT_DIR / "full_errors.jsonl"

    parse_jsonl(RAW_FILE, PARSED_FILE, ERROR_FILE)
