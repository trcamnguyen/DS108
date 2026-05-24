"""
DS108 — Full Dataset Skill Extraction
======================================
Input  : data/interim/01-standardized_title.csv (3,181 rows — TopCV + ITViec)
Output : Preprocessing/02_skill_extraction/output_full/
  - full_raw.jsonl     : raw API responses (dùng làm checkpoint để resume)
  - full_parsed.csv    : skills dạng flat (1 row / skill), kèm metadata
  - full_errors.jsonl  : các row thất bại sau 3 retry

Logic y hệt calibration/02_llm_skill_extraction.py:
  - Cùng model, temperature, max_tokens, retry
  - Cùng prompt + few-shot examples
  - Cùng Pydantic schema, JSON repair, coerce logic
  - Thêm: resume từ checkpoint, metadata (source, standardized_title, url)
  - Thêm: Gemini context caching — system prompt + few-shot tạo 1 lần, tái dùng
            qua 3,181 calls (~39M input tokens tĩnh tiết kiệm được)
"""

import os
import re
import json
import time
import logging
import pandas as pd
from pathlib import Path
from pydantic import BaseModel, ValidationError
from typing import List, Optional, Literal

# ─── ROOT & ENV ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent.parent

env_path = ROOT / ".env"
if env_path.exists():
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"'))

PROJECT    = os.environ.get("GOOGLE_CLOUD_PROJECT", "your-project-id")
LOCATION   = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1").strip('"')
MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-2.5-pro")

if "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
    creds_path = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
    if not Path(creds_path).is_absolute():
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(ROOT / creds_path)

from google import genai
from google.genai import types

# ─── CONFIG ───────────────────────────────────────────────────────────────────
TEMPERATURE = 0.0
MAX_TOKENS  = 32768
RETRY_LIMIT = 3
RETRY_DELAY = 5      # giây; nhân đôi mỗi lần retry (exponential backoff)
CACHE_TTL   = "86400s"  # 24 giờ — đủ cho toàn bộ 1 run (~4-5h)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger(__name__)

# ─── PYDANTIC SCHEMAS ─────────────────────────────────────────────────────────

class Skill(BaseModel):
    skill_name: str
    label: Literal["required_skill", "preferred_skill"]
    category: str
    min_years: Optional[int] = None
    level: Optional[Literal["expert", "intermediate", "basic"]] = None
    source_text: str

class SkillExtractionOutput(BaseModel):
    skills: List[Skill]

# ─── LOAD PROMPT ──────────────────────────────────────────────────────────────
_PROMPT_PATH = Path(__file__).parent / "prompt" / "prompt_skill_extraction.txt"
try:
    prompt_content = _PROMPT_PATH.read_text(encoding="utf-8")
except FileNotFoundError:
    raise FileNotFoundError(f"Không tìm thấy file prompt tại {_PROMPT_PATH}")

_parts = prompt_content.split("===FEW_SHOT===")
if len(_parts) != 2:
    raise ValueError("prompt_skill_extraction.txt phải có đúng một dấu ===FEW_SHOT===")

SYSTEM_PROMPT     = _parts[0].replace("===SYSTEM_PROMPT===", "").strip()
FEW_SHOT_EXAMPLES = json.loads(_parts[1].strip())

# ─── GLOBALS ──────────────────────────────────────────────────────────────────
CLIENT         = None
CACHED_CONTENT = None   # None = fallback về no-cache mode

# ─── MODEL & CACHE INIT ───────────────────────────────────────────────────────

def init_model(project_id: str, location: str):
    global CLIENT
    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not creds_path:
        creds_path = str(ROOT / "credentials" / "service-account.json")
        log.info(f"Using default credentials path: {creds_path}")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(creds_path)
    CLIENT = genai.Client(vertexai=True, project=project_id, location=location)
    _init_cache()


def _init_cache() -> None:
    """
    Tạo Gemini CachedContent chứa system prompt + few-shot examples.
    Cấu trúc: mỗi example là 1 cặp user/model turn trong conversation history.
    Nếu tạo cache thất bại, fallback về no-cache (vẫn chạy được, chỉ tốn hơn).
    """
    global CACHED_CONTENT

    # Build few-shot conversation history cho cached content:
    #   user:  preamble + EXAMPLE 1 requirement
    #   model: EXAMPLE 1 output JSON
    #   user:  EXAMPLE 2 requirement
    #   model: EXAMPLE 2 output JSON
    #   ... (tất cả examples)
    few_shot_contents = []

    preamble = (
        "Dưới đây là các ví dụ minh họa cách áp dụng Codebook.\n"
        "Đọc kỹ từng ví dụ, sau đó annotate input cuối cùng theo đúng schema.\n\n"
        f"{'─'*50}\n"
        f"EXAMPLE 1\n"
        f"{'─'*50}\n"
        f"REQUIREMENT:\n{FEW_SHOT_EXAMPLES[0]['requirement']}"
    )
    few_shot_contents.append(
        types.Content(role="user", parts=[types.Part(text=preamble)])
    )
    few_shot_contents.append(
        types.Content(role="model", parts=[types.Part(
            text=json.dumps(FEW_SHOT_EXAMPLES[0].get("output", {}), ensure_ascii=False, indent=2)
        )])
    )

    for i, ex in enumerate(FEW_SHOT_EXAMPLES[1:], 2):
        few_shot_contents.append(
            types.Content(role="user", parts=[types.Part(
                text=f"{'─'*50}\nEXAMPLE {i}\n{'─'*50}\nREQUIREMENT:\n{ex['requirement']}"
            )])
        )
        few_shot_contents.append(
            types.Content(role="model", parts=[types.Part(
                text=json.dumps(ex.get("output", {}), ensure_ascii=False, indent=2)
            )])
        )

    try:
        CACHED_CONTENT = CLIENT.caches.create(
            model=MODEL_NAME,
            config=types.CreateCachedContentConfig(
                system_instruction=SYSTEM_PROMPT,
                contents=few_shot_contents,
                ttl=CACHE_TTL,
                display_name="ds108_skill_extraction_v4"
            )
        )
        log.info(f"Context cache created: {CACHED_CONTENT.name} (TTL={CACHE_TTL})")
    except Exception as e:
        log.warning(f"Context cache creation failed — falling back to no-cache mode: {e}")
        CACHED_CONTENT = None

# ─── PROMPT BUILDER ───────────────────────────────────────────────────────────

def build_user_prompt(requirement: str) -> str:
    """
    Khi dùng context cache: chỉ gửi phần requirement thực tế.
    System prompt + few-shot đã nằm trong cached content.

    Khi fallback no-cache: gửi toàn bộ (system prompt được truyền qua
    system_instruction trong GenerateContentConfig, few-shot nhúng vào đây).
    """
    if CACHED_CONTENT is not None:
        return (
            f"{'═'*50}\n"
            "INPUT CẦN ANNOTATE\n"
            f"{'═'*50}\n"
            f"REQUIREMENT:\n{requirement}\n\n"
            "Trả về JSON theo đúng schema. Không có text nào khác ngoài JSON."
        )

    # Fallback: rebuild full few-shot prompt
    parts_list = [
        "Dưới đây là các ví dụ minh họa cách áp dụng Codebook.",
        "Đọc kỹ từng ví dụ, sau đó annotate input cuối cùng theo đúng schema.\n"
    ]
    for i, ex in enumerate(FEW_SHOT_EXAMPLES, 1):
        parts_list.append(f"{'─'*50}")
        parts_list.append(f"EXAMPLE {i}")
        parts_list.append(f"{'─'*50}")
        parts_list.append(f"REQUIREMENT:\n{ex['requirement']}\n")
        parts_list.append(f"OUTPUT:\n{json.dumps(ex.get('output', {}), ensure_ascii=False, indent=2)}\n")
    parts_list.append(f"{'═'*50}")
    parts_list.append("INPUT CẦN ANNOTATE")
    parts_list.append(f"{'═'*50}")
    parts_list.append(f"REQUIREMENT:\n{requirement}\n")
    parts_list.append("Trả về JSON theo đúng schema. Không có text nào khác ngoài JSON.")
    return "\n".join(parts_list)

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
        fixed.append(s)
    return {"skills": fixed}

# ─── API CALLER ───────────────────────────────────────────────────────────────

def _call_gemini_api(user_prompt: str) -> dict:
    if CLIENT is None:
        raise RuntimeError("Client chưa được khởi tạo. Gọi init_model() trước.")

    if CACHED_CONTENT is not None:
        config = types.GenerateContentConfig(
            temperature=TEMPERATURE,
            max_output_tokens=MAX_TOKENS,
            cached_content=CACHED_CONTENT.name,
            response_mime_type="application/json"
        )
    else:
        # Fallback: không có cache, gửi system_instruction mỗi lần
        config = types.GenerateContentConfig(
            temperature=TEMPERATURE,
            max_output_tokens=MAX_TOKENS,
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json"
        )

    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            response = CLIENT.models.generate_content(
                model=MODEL_NAME,
                contents=user_prompt,
                config=config
            )
            if not response.candidates:
                raise ValueError("No candidates returned from Gemini API")
            raw_text = getattr(response, "text", None)
            if not raw_text:
                raise ValueError("Empty response text from Gemini API")

            cleaned = re.sub(r"^```(?:json)?\n?", "", raw_text.strip(), flags=re.IGNORECASE)
            cleaned = re.sub(r"\n?```$", "", cleaned.strip())
            parsed_json = json.loads(cleaned)
            validated = SkillExtractionOutput(**parsed_json)
            return {"raw_text": raw_text, "parsed": validated.model_dump(), "error": None}

        except (json.JSONDecodeError, ValidationError) as e:
            log.warning(f"Attempt {attempt}/{RETRY_LIMIT}: Parse/Validation error — {e}")
            raw = locals().get("raw_text", "") or ""
            if raw:
                repaired = _repair_truncated_json(raw)
                if repaired:
                    try:
                        validated = SkillExtractionOutput(**repaired)
                        log.info(f"Attempt {attempt}: Repaired truncated JSON successfully")
                        return {"raw_text": raw, "parsed": validated.model_dump(), "error": None}
                    except ValidationError:
                        try:
                            validated = SkillExtractionOutput(**_coerce_skills(repaired))
                            log.info(f"Attempt {attempt}: Repaired+coerced JSON successfully")
                            return {"raw_text": raw, "parsed": validated.model_dump(), "error": None}
                        except ValidationError:
                            pass
            if attempt == RETRY_LIMIT:
                return {"raw_text": raw, "parsed": None, "error": f"Parse/ValidationError: {str(e)}"}
            time.sleep(RETRY_DELAY * (2 ** (attempt - 1)))

        except Exception as e:
            log.warning(f"Attempt {attempt}/{RETRY_LIMIT}: API error — {e}")
            if attempt == RETRY_LIMIT:
                return {"raw_text": None, "parsed": None, "error": str(e)}
            time.sleep(RETRY_DELAY * (2 ** (attempt - 1)))


def annotate_row(requirement: str) -> dict:
    prompt = build_user_prompt(requirement)
    return _call_gemini_api(prompt)

# ─── PIPELINE ─────────────────────────────────────────────────────────────────

def _load_processed_ids(raw_file: Path) -> set:
    """Đọc row_id đã xử lý từ checkpoint file để resume."""
    processed = set()
    if raw_file.exists():
        with open(raw_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    processed.add(entry["row_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
        log.info(f"Resume mode: {len(processed)} rows already processed in {raw_file.name}")
    return processed


def run_full_extraction(
    csv_path: str,
    project_id: str,
    location: str,
    output_dir: str = "output_full",
    delay: float    = 1.5
):
    init_model(project_id, location)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    raw_file    = out / "full_raw.jsonl"
    parsed_file = out / "full_parsed.csv"
    error_file  = out / "full_errors.jsonl"

    df = pd.read_csv(csv_path)
    log.info(
        f"Dataset loaded: {len(df)} rows — "
        f"TopCV: {(df['source']=='topcv').sum()}, ITViec: {(df['source']=='itviec').sum()}"
    )

    processed_ids = _load_processed_ids(raw_file)
    remaining = df[~df.index.isin(processed_ids)]
    log.info(f"To process: {len(remaining)} rows (already done: {len(processed_ids)})")

    cache_mode = "WITH cache" if CACHED_CONTENT is not None else "NO cache (fallback)"
    log.info(f"Running skill extraction [{cache_mode}]...")

    extracted_rows, errors = [], []
    total = len(remaining)

    for i, (_, row) in enumerate(remaining.iterrows(), 1):
        row_id    = row.name
        req       = str(row.get("requirement", "") or "")
        source    = str(row.get("source", ""))
        std_title = str(row.get("standardized_title", "") or "")
        url       = str(row.get("url", "") or "")

        if not req.strip() or req.strip().lower() in ("nan", "none"):
            log.warning(f"[{i}/{total}] Row {row_id} ({source}): empty requirement — skip")
            raw_entry = {
                "row_id": row_id, "raw_text": None, "error": "empty_requirement",
                "source": source, "standardized_title": std_title, "url": url
            }
            with open(raw_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(raw_entry, ensure_ascii=False) + "\n")
            errors.append({**raw_entry, "req_snippet": ""})
            continue

        if i % 50 == 0 or i == 1:
            log.info(f"[{i}/{total}] row_id={row_id} ({source}) — {std_title[:40]}")

        result = annotate_row(req)

        raw_entry = {
            "row_id": row_id,
            "raw_text": result["raw_text"],
            "error": result["error"],
            "source": source,
            "standardized_title": std_title,
            "url": url
        }
        with open(raw_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(raw_entry, ensure_ascii=False) + "\n")

        if result["parsed"] and "skills" in result["parsed"]:
            for skill in result["parsed"]["skills"]:
                extracted_rows.append({
                    "row_id": row_id,
                    "source": source,
                    "standardized_title": std_title,
                    "url": url,
                    **skill
                })
        else:
            errors.append({**raw_entry, "req_snippet": req[:300]})

        time.sleep(delay)

    _flush_parsed_csv(raw_file, df, parsed_file)

    if errors:
        with open(error_file, "a", encoding="utf-8") as f:
            for e in errors:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

    log.info(
        f"Done. Skills extracted this run: {len(extracted_rows)} | "
        f"Errors this run: {len(errors)} | Error log: {error_file}"
    )


def _flush_parsed_csv(raw_file: Path, df: pd.DataFrame, parsed_file: Path):
    """Rebuild full_parsed.csv từ full_raw.jsonl để luôn sync với checkpoint."""
    if not raw_file.exists():
        return

    url_map   = dict(zip(df.index, df.get("url", pd.Series(dtype=str))))
    src_map   = dict(zip(df.index, df.get("source", pd.Series(dtype=str))))
    title_map = dict(zip(df.index, df.get("standardized_title", pd.Series(dtype=str))))

    records = []
    with open(raw_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            row_id   = entry.get("row_id")
            raw_text = entry.get("raw_text")
            if not raw_text or entry.get("error"):
                continue

            try:
                cleaned = re.sub(r"^```(?:json)?\n?", "", raw_text.strip(), flags=re.IGNORECASE)
                cleaned = re.sub(r"\n?```$", "", cleaned.strip())
                validated = SkillExtractionOutput(**json.loads(cleaned))
                skills = validated.model_dump()["skills"]
            except Exception:
                repaired = _repair_truncated_json(raw_text)
                if not repaired:
                    continue
                try:
                    validated = SkillExtractionOutput(**_coerce_skills(repaired))
                    skills = validated.model_dump()["skills"]
                except Exception:
                    continue

            for skill in skills:
                records.append({
                    "row_id": row_id,
                    "source": entry.get("source") or src_map.get(row_id, ""),
                    "standardized_title": entry.get("standardized_title") or title_map.get(row_id, ""),
                    "url": entry.get("url") or url_map.get(row_id, ""),
                    **skill
                })

    if records:
        pd.DataFrame(records).to_csv(parsed_file, index=False, encoding="utf-8-sig")
        log.info(f"Wrote {len(records)} skill rows to {parsed_file.name}")


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    CSV_PATH   = ROOT / "data" / "interim" / "01-standardized_title.csv"
    OUTPUT_DIR = Path(__file__).parent / "output_full"

    if PROJECT == "your-project-id":
        log.warning("PROJECT chưa được cấu hình. Kiểm tra lại .env (GOOGLE_CLOUD_PROJECT)")

    run_full_extraction(
        csv_path   = str(CSV_PATH),
        project_id = PROJECT,
        location   = LOCATION,
        output_dir = str(OUTPUT_DIR),
        delay      = 1.5
    )
