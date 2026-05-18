"""
DS108 — Skill Annotation Prompt System for Gemini 2.5 Pro
==========================================================
Version  : v2.0
"""

import os
import re
import json
import time
import logging
import pandas as pd
import sys
from pathlib import Path
from pydantic import BaseModel, ValidationError
from typing import List, Optional, Literal
from google.oauth2 import service_account
from google import genai

# --- Cấu hình ROOT và load .env giống 01_process_job_title.py ---
ROOT = Path(__file__).parent.parent.parent.parent

env_path = ROOT / ".env"
if env_path.exists():
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"'))

PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "your-project-id")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1").strip('"')
MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-2.5-pro")

if "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
    creds_path = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
    # Xử lý path tương đối từ thư mục ROOT
    if not Path(creds_path).is_absolute():
        CREDS = ROOT / creds_path
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(CREDS)

from google import genai
from google.genai import types

# ─── CONFIG ───────────────────────────────────────────────────────────────────
TEMPERATURE = 0.0       # Bắt buộc = 0 để đảm bảo reproducibility
MAX_TOKENS  = 16384     # Tăng từ 8192 để tránh truncation với JD nhiều skills
RETRY_LIMIT = 3
RETRY_DELAY = 5         # seconds, nhân đôi mỗi lần retry (exponential backoff)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# PYDANTIC SCHEMAS
# ═══════════════════════════════════════════════════════════════════════════════

class Skill(BaseModel):
    skill_name: str
    label: Literal["required_skill", "preferred_skill"]
    category: str
    min_years: Optional[int] = None
    level: Optional[Literal["expert", "intermediate", "basic"]] = None
    source_text: str

class SkillExtractionOutput(BaseModel):
    skills: List[Skill]

# ═══════════════════════════════════════════════════════════════════════════════
# READ PROMPT FROM FILE
# ═══════════════════════════════════════════════════════════════════════════════
prompt_file_path = os.path.join(os.path.dirname(__file__), "..", "prompt", "prompt_skill_extraction.txt")
try:
    with open(prompt_file_path, "r", encoding="utf-8") as f:
        prompt_content = f.read()
except FileNotFoundError:
    raise FileNotFoundError(f"Không tìm thấy file prompt tại {prompt_file_path}")

parts = prompt_content.split("===FEW_SHOT===")
if len(parts) != 2:
    raise ValueError("File prompt_skill_extraction.txt không đúng format (cần có ===SYSTEM_PROMPT=== và ===FEW_SHOT===)")

SYSTEM_PROMPT = parts[0].replace("===SYSTEM_PROMPT===", "").strip()
FEW_SHOT_EXAMPLES = json.loads(parts[1].strip())


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBALS & PROMPT BUILDERS
# ═══════════════════════════════════════════════════════════════════════════════
CLIENT = None

def init_model(project_id: str, location: str):
    global CLIENT

    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not creds_path:
        creds_path = str(ROOT / "credentials" / "service-account.json")
        log.info(f"Using default credentials path: {creds_path}")
        
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(creds_path)

    CLIENT = genai.Client(
        vertexai=True,
        project=project_id,
        location=location
    )

# def init_model(project_id: str, location: str):
#     global CLIENT
#     CLIENT = genai.Client(vertexai=True, project=project_id, location=location)

def build_few_shot_prompt(requirement: str) -> str:
    parts_list = [
        "Dưới đây là 5 ví dụ minh họa cách áp dụng Codebook.",
        "Đọc kỹ từng ví dụ, sau đó annotate input cuối cùng theo đúng schema.\n"
    ]

    for i, ex in enumerate(FEW_SHOT_EXAMPLES, 1):
        parts_list.append(f"{'─'*50}")
        parts_list.append(f"EXAMPLE {i}")
        parts_list.append(f"{'─'*50}")
        parts_list.append(f"REQUIREMENT:\n{ex['requirement']}\n")
        output_clean = {k: v for k, v in ex.get('output', {}).items()}
        parts_list.append(f"OUTPUT:\n{json.dumps(output_clean, ensure_ascii=False, indent=2)}\n")

    parts_list.append(f"{'═'*50}")
    parts_list.append("INPUT CẦN ANNOTATE")
    parts_list.append(f"{'═'*50}")
    parts_list.append(f"REQUIREMENT:\n{requirement}\n")
    parts_list.append("Trả về JSON theo đúng schema. Không có text nào khác ngoài JSON.")

    return "\n".join(parts_list)


# ═══════════════════════════════════════════════════════════════════════════════
# API CALLER
# ═══════════════════════════════════════════════════════════════════════════════

def _repair_truncated_json(raw_text: str) -> Optional[dict]:
    """Recover partial JSON by trimming to the last complete skill object."""
    cleaned = re.sub(r"^```(?:json)?\n?", "", raw_text.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\n?```$", "", cleaned.strip())

    # Tìm tất cả vị trí kết thúc một skill object (dòng "    }")
    matches = list(re.finditer(r'\n\s{4}\}', cleaned))
    for match in reversed(matches):
        candidate = cleaned[:match.end()] + "\n  ]\n}"
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def _call_gemini_api(user_prompt: str) -> dict:
    if CLIENT is None:
        raise RuntimeError("Client chưa được khởi tạo. Hãy gọi init_model() trước.")

    config = types.GenerateContentConfig(
        temperature=TEMPERATURE,
        max_output_tokens=MAX_TOKENS,
        system_instruction=SYSTEM_PROMPT
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
            
            # Làm sạch JSON (xóa ```json và ``` ở cuối)
            cleaned_text = re.sub(r"^```(?:json)?\n?", "", raw_text.strip(), flags=re.IGNORECASE)
            cleaned_text = re.sub(r"\n?```$", "", cleaned_text.strip())

            # Cố gắng parse JSON
            parsed_json = json.loads(cleaned_text)
            
            # Validate schema bằng Pydantic
            validated_data = SkillExtractionOutput(**parsed_json)

            return {"raw_text": raw_text, "parsed": validated_data.model_dump(), "error": None}

        except (json.JSONDecodeError, ValidationError) as e:
            log.warning(f"Attempt {attempt}/{RETRY_LIMIT}: Parsing/Validation error — {e}")
            # Thử repair JSON bị truncate trước khi retry
            raw = locals().get("raw_text", "") or ""
            if raw:
                repaired = _repair_truncated_json(raw)
                if repaired:
                    try:
                        validated_data = SkillExtractionOutput(**repaired)
                        log.info(f"Attempt {attempt}: Repaired truncated JSON successfully")
                        return {"raw_text": raw, "parsed": validated_data.model_dump(), "error": None}
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


def annotate_few_shot(requirement: str) -> dict:
    prompt = build_few_shot_prompt(requirement)
    result = _call_gemini_api(prompt)
    result["mode"] = "few_shot"
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_annotation_pipeline(
    csv_path: str,
    project_id: str,
    location: str,
    output_dir: str  = "output",
    delay: float     = 1.5    
):
    init_model(project_id, location)
    
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    log.info(f"Dataset: {len(df)} records for skill extraction")

    extracted_rows, errors = [], []

    def _process(row):
        req = str(row.get("requirement", ""))
        row_id = row.name  # Sử dụng index của DataFrame làm ID thứ tự

        result = annotate_few_shot(req)

        raw_entry = {
            "row_id": row_id,
            "mode": "few_shot",
            "raw_text": result["raw_text"],
            "error": result["error"]
        }
        raw_file = out / "few_shot_raw.jsonl"
        with open(raw_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(raw_entry, ensure_ascii=False) + "\n")

        if result["parsed"] and "skills" in result["parsed"]:
            for skill in result["parsed"]["skills"]:
                flat_row = {
                    "row_id": row_id,
                    "mode": "few_shot",
                    **skill
                }
                extracted_rows.append(flat_row)
        else:
            errors.append({**raw_entry, "req_snippet": req[:300]})

    log.info(f"Running skill extraction on {len(df)} samples...")
    for _, row in df.iterrows():
        _process(row)
        time.sleep(delay)

    if extracted_rows:
        pd.DataFrame(extracted_rows).to_csv(out / "few_shot_parsed.csv", index=False, encoding="utf-8-sig")
    if errors:
        with open(out / "annotation_errors.jsonl", "w", encoding="utf-8") as f:
            for e in errors:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

    log.info(
        f"Complete. Extracted skills: {len(extracted_rows)} | Errors: {len(errors)}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Input là calibration dataset
    base_dir = Path(__file__).parent
    CSV_PATH = base_dir / "calibration_dataset.csv"

    if PROJECT == "your-project-id":
        log.warning("Bạn đang dùng PROJECT mặc định. Hãy kiểm tra lại cấu hình .env (GOOGLE_CLOUD_PROJECT)")

    # Lưu file output llm_skill_extraction vào folder calibration/output
    run_annotation_pipeline(
        csv_path        = str(CSV_PATH),
        project_id      = PROJECT,
        location        = LOCATION,
        output_dir      = str(base_dir / "output"),
        delay           = 1.5
    )

