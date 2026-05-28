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
  - Thêm: resume từ checkpoint (row_id = url, stable qua re-sort)
  - Thêm: Gemini context caching — system prompt + few-shot tạo 1 lần (~39M tokens tiết kiệm)
  - Thêm: ThreadPoolExecutor — 4 workers song song, token bucket 50 RPM
           ~3.5h sequential → ~50 phút concurrent
"""

import os
import re
import json
import time
import queue
import logging
import threading
import pandas as pd
from pathlib import Path
from pydantic import BaseModel, ValidationError
from typing import List, Optional, Literal
from concurrent.futures import ThreadPoolExecutor, as_completed

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
MAX_TOKENS  = 8192
RETRY_LIMIT = 3
RETRY_DELAY = 5        # giây; nhân đôi mỗi lần retry (exponential backoff)
CACHE_TTL   = "86400s" # 24 giờ — đủ cho toàn bộ 1 run
WORKERS     = 2      # concurrent API workers
TARGET_RPM  = 25       # token bucket — giảm xuống để tránh resource exhausted

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger(__name__)

# ─── PYDANTIC SCHEMAS ─────────────────────────────────────────────────────────

class Skill(BaseModel):
    skill_name: str
    label: Literal["required_skill", "preferred_skill"]
    category: str
    min_years: Optional[float] = None
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
_rate_limiter  = None   # _TokenBucket, khởi tạo trong run_full_extraction

# ─── RATE LIMITER ─────────────────────────────────────────────────────────────

class _TokenBucket:
    """
    Token bucket đơn giản: đảm bảo tối đa rpm request/phút.
    Thread-safe: mỗi worker acquire() trước khi gọi API.
    """
    def __init__(self, rpm: int):
        self._interval = 60.0 / rpm
        self._lock     = threading.Lock()
        self._last     = 0.0

    def acquire(self):
        with self._lock:
            now  = time.monotonic()
            wait = self._interval - (now - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()

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
                display_name=f"ds108_skill_extraction_v4_{MODEL_NAME.replace('.', '_')}"
            )
        )
        log.info(f"Context cache created: {CACHED_CONTENT.name} (TTL={CACHE_TTL})")
    except Exception as e:
        log.warning(f"Context cache creation failed — falling back to no-cache mode: {e}")
        CACHED_CONTENT = None

# ─── PROMPT BUILDER ───────────────────────────────────────────────────────────

def build_user_prompt(requirement: str) -> str:
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
        result = json.loads(no_trailing_comma)
        if isinstance(result, list):
            result = {"skills": result}
        return result
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

# ─── API CALLER ───────────────────────────────────────────────────────────────

def _call_gemini_api(user_prompt: str) -> dict:
    if CLIENT is None:
        raise RuntimeError("Client chưa được khởi tạo. Gọi init_model() trước.")

    if CACHED_CONTENT is not None:
        config = types.GenerateContentConfig(
            temperature=TEMPERATURE,
            max_output_tokens=MAX_TOKENS,
            cached_content=CACHED_CONTENT.name,
            response_mime_type="application/json",
            thinking_config=types.ThinkingConfig(thinking_budget=128)
        )
    else:
        config = types.GenerateContentConfig(
            temperature=TEMPERATURE,
            max_output_tokens=MAX_TOKENS,
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            thinking_config=types.ThinkingConfig(thinking_budget=128)
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
            if isinstance(parsed_json, list):
                parsed_json = {"skills": parsed_json}
            try:
                validated = SkillExtractionOutput(**parsed_json)
            except ValidationError:
                # Coerce trước khi retry API (e.g. min_years=0.5 float)
                validated = SkillExtractionOutput(**_coerce_skills(parsed_json))
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

# ─── WORKER TASK ──────────────────────────────────────────────────────────────

def _worker_task(row_data: dict) -> dict:
    """
    Chạy trong thread pool (producer).
    Acquire token → gọi API → trả kết quả về main thread qua queue.
    Không làm bất kỳ file I/O nào — đó là việc của writer thread.
    """
    _rate_limiter.acquire()
    result = annotate_row(row_data["req"])

    raw_entry = {
        "row_id":             row_data["row_id"],
        "raw_text":           result["raw_text"],
        "error":              result["error"],
        "source":             row_data["source"],
        "standardized_title": row_data["std_title"],
        "url":                row_data["url"],
    }

    skill_rows = None
    if result["parsed"] and "skills" in result["parsed"]:
        skill_rows = [
            {
                "row_id":             row_data["row_id"],
                "source":             row_data["source"],
                "standardized_title": row_data["std_title"],
                "url":                row_data["url"],
                **skill,
            }
            for skill in result["parsed"]["skills"]
        ]

    return {
        "raw_entry":   raw_entry,
        "skill_rows":  skill_rows,
        "req_snippet": row_data["req"][:300],
    }

# ─── PIPELINE ─────────────────────────────────────────────────────────────────

def _load_processed_ids(raw_file: Path) -> set:
    """Đọc row_id đã xử lý THÀNH CÔNG từ checkpoint để resume. Bỏ qua entries có lỗi."""
    processed = set()
    error_ids  = set()
    if raw_file.exists():
        with open(raw_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    rid = entry["row_id"]
                    if entry.get("error"):
                        error_ids.add(rid)
                    else:
                        processed.add(rid)
                except (json.JSONDecodeError, KeyError):
                    continue
        # Chỉ skip row đã thành công; row lỗi sẽ được retry
        retry_ids = error_ids - processed
        log.info(
            f"Resume mode: {len(processed)} success | "
            f"{len(retry_ids)} errors → will retry"
        )
    return processed


def run_full_extraction(
    csv_path: str,
    project_id: str,
    location: str,
    output_dir: str = "output_full",
):
    global _rate_limiter
    _rate_limiter = _TokenBucket(TARGET_RPM)
    init_model(project_id, location)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    raw_file    = out / "full_raw.jsonl"
    parsed_file = out / "full_parsed.csv"
    error_file  = out / "full_errors.jsonl"
    log_file    = out / "extraction.log"

    # Ghi toàn bộ log ra file (append) để debug sau
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(file_handler)

    df = pd.read_csv(csv_path)
    log.info(
        f"Dataset loaded: {len(df)} rows — "
        f"TopCV: {(df['source']=='topcv').sum()}, ITViec: {(df['source']=='itviec').sum()}"
    )

    processed_ids = _load_processed_ids(raw_file)
    remaining = df[~df["url"].isin(processed_ids)]
    log.info(f"To process: {len(remaining)} rows (already done: {len(processed_ids)})")

    # Tách empty-req ra trước: ghi thẳng vào checkpoint, không submit worker
    rows_to_process = []
    for _, row in remaining.iterrows():
        url       = str(row.get("url", "") or "")
        req       = str(row.get("requirement", "") or "")
        source    = str(row.get("source", ""))
        std_title = str(row.get("standardized_title", "") or "")

        if not req.strip() or req.strip().lower() in ("nan", "none"):
            log.warning(f"Empty requirement — skipping {url}")
            entry = {
                "row_id": url, "raw_text": None, "error": "empty_requirement",
                "source": source, "standardized_title": std_title, "url": url,
            }
            with open(raw_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            continue

        rows_to_process.append({
            "row_id":    url,
            "req":       req,
            "source":    source,
            "std_title": std_title,
            "url":       url,
        })

    total = len(rows_to_process)
    cache_mode = "WITH cache" if CACHED_CONTENT is not None else "NO cache (fallback)"
    log.info(
        f"Submitting {total} tasks | {WORKERS} workers | {TARGET_RPM} RPM | {cache_mode}"
    )
    print(f"\n{'─'*70}")
    print(f"  Tasks: {total} | Workers: {WORKERS} | RPM: {TARGET_RPM} | {cache_mode}")
    print(f"{'─'*70}")

    # ── Producer-Consumer ──────────────────────────────────────────────────────
    # Workers (producers): gọi API song song, push kết quả vào write_q
    # Writer thread (consumer): pop từ write_q, ghi file tuần tự — không race condition
    extracted_rows: list = []
    error_count:    int  = 0
    write_q: queue.Queue = queue.Queue()

    def _writer_loop():
        nonlocal error_count
        while True:
            item = write_q.get()
            if item is None:
                write_q.task_done()
                break

            raw_entry  = item["raw_entry"]
            skill_rows = item["skill_rows"]

            with open(raw_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(raw_entry, ensure_ascii=False) + "\n")

            if skill_rows:
                extracted_rows.extend(skill_rows)
            else:
                # Ghi lỗi ra file ngay lập tức — không gom vào memory
                debug_entry = {
                    **raw_entry,
                    "req_snippet": item["req_snippet"],
                }
                with open(error_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(debug_entry, ensure_ascii=False) + "\n")
                error_count += 1
                log.warning(
                    f"Error logged [{error_count}]: {raw_entry.get('row_id', '')} — "
                    f"{raw_entry.get('error', 'no skills extracted')}"
                )

            write_q.task_done()

    writer = threading.Thread(target=_writer_loop, daemon=True)
    writer.start()

    start_time = time.time()

    def _fmt_time(seconds: float) -> str:
        h, r = divmod(int(seconds), 3600)
        m, s = divmod(r, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(_worker_task, rd): rd for rd in rows_to_process}

        done_count = 0
        for future in as_completed(futures):
            done_count += 1
            try:
                result = future.result()
            except Exception as e:
                rd = futures[future]
                log.error(f"Unexpected worker error for {rd['row_id']}: {e}")
                result = {
                    "raw_entry": {
                        "row_id": rd["row_id"], "raw_text": None, "error": str(e),
                        "source": rd["source"], "standardized_title": rd["std_title"],
                        "url": rd["url"],
                    },
                    "skill_rows":  None,
                    "req_snippet": rd["req"][:300],
                }

            write_q.put(result)

            if done_count % 50 == 0 or done_count == total:
                elapsed = time.time() - start_time
                rate    = done_count / elapsed * 60 if elapsed > 0 else 0
                pct     = done_count / total * 100 if total > 0 else 0
                eta_sec = (total - done_count) / (done_count / elapsed) if done_count > 0 else 0
                log.info(
                    f"Progress: {done_count}/{total} ({pct:.1f}%) | "
                    f"{rate:.1f} rows/min | ETA {_fmt_time(eta_sec)} | errors {error_count}"
                )

    write_q.put(None)   # báo writer dừng
    writer.join()       # đợi writer flush hết queue

    _flush_parsed_csv(raw_file, parsed_file)

    total_elapsed = time.time() - start_time
    print(f"\n{'─'*70}")
    print(f"  Done: {total} tasks in {_fmt_time(total_elapsed)}")
    print(f"  Skills extracted : {len(extracted_rows)}")
    print(f"  Errors           : {error_count}")
    print(f"  Avg rate         : {total / total_elapsed * 60:.1f} rows/min")
    print(f"  Error log        : {error_file}")
    print(f"  Run log          : {log_file}")
    print(f"{'─'*70}\n")
    log.info(
        f"Done. Skills extracted: {len(extracted_rows)} | "
        f"Errors: {error_count} | Error log: {error_file} | Run log: {log_file}"
    )
    logging.getLogger().removeHandler(file_handler)
    file_handler.close()


def _flush_parsed_csv(raw_file: Path, parsed_file: Path):
    """Rebuild full_parsed.csv từ full_raw.jsonl để luôn sync với checkpoint."""
    if not raw_file.exists():
        return

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

            raw_text = entry.get("raw_text")
            if not raw_text or entry.get("error"):
                continue

            try:
                cleaned = re.sub(r"^```(?:json)?\n?", "", raw_text.strip(), flags=re.IGNORECASE)
                cleaned = re.sub(r"\n?```$", "", cleaned.strip())
                data = json.loads(cleaned)
                if isinstance(data, list):
                    data = {"skills": data}
                validated = SkillExtractionOutput(**data)
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
                    "row_id":             entry.get("row_id"),
                    "source":             entry.get("source", ""),
                    "standardized_title": entry.get("standardized_title", ""),
                    "url":                entry.get("url", ""),
                    **skill,
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

    log.info(f"Model: {MODEL_NAME} | Output: {OUTPUT_DIR}")

    run_full_extraction(
        csv_path   = str(CSV_PATH),
        project_id = PROJECT,
        location   = LOCATION,
        output_dir = str(OUTPUT_DIR),
    )
