"""
Xử lý cột job_title + job_description từ topcv_merged.csv bằng Gemini qua Vertex AI.
Kết quả lưu vào Preprocess/output/job_title_full.json (checkpoint sau mỗi batch).

Yêu cầu: credentials/service-account.json + các biến trong .env
"""
import csv
import json
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent        # 01-standardize_title/
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent  # DS108/DS108/

# --- load .env ---
for line in (REPO_ROOT / ".env").read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"'))

PROJECT = os.environ["GOOGLE_CLOUD_PROJECT"]
LOCATION = os.environ["GOOGLE_CLOUD_LOCATION"].strip('"')
MODEL = os.environ["GEMINI_MODEL"]

CREDS = REPO_ROOT / os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(CREDS)

from google import genai  # noqa: E402
# pyrefly: ignore [missing-import]
from google.genai import types  # noqa: E402

PROMPT_FILE = ROOT / "prompt" / "prompt_job_title.txt"
OUTPUT_DIR = ROOT / "output"
BATCH_SIZE = 10

# ---------------------------------------------------------------------------

def load_jobs(csv_path: Path) -> list[dict]:
    jobs = []
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        for idx, row in enumerate(csv.DictReader(f)):
            t = row["job_title"].strip()
            d = row.get("job_description", "").strip()
            if t:
                jobs.append({"id": idx, "title": t, "description": d})
    return jobs

def parse_response(raw: str) -> list[dict]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()
    return json.loads(raw)

def load_checkpoint(output_file: Path) -> dict:
    if output_file.exists():
        try:
            data = json.loads(output_file.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return {item["id"]: item for item in data if "id" in item}
        except Exception:
            pass
    return {}

def process_jobs(client: genai.Client, system_prompt: str, jobs: list[dict], output_file: Path):
    total = len(jobs)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    results_dict = load_checkpoint(output_file)
    done_ids = set(results_dict.keys())
    remaining = [j for j in jobs if j["id"] not in done_ids]
    
    if done_ids:
        print(f"Resumed: {len(done_ids)} already done, {len(remaining)} remaining.\n")
        
    for i in range(0, len(remaining), BATCH_SIZE):
        batch = remaining[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        
        payload = [{"id": j["id"], "title": j["title"], "description": j["description"]} for j in batch]
        batch_map = {j["id"]: j for j in batch}
        
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=json.dumps(payload, ensure_ascii=False),
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0,
                ),
            )
            parsed = parse_response(response.text)
            
            for item in parsed:
                if "id" not in item and "original" in item:
                    # Phục hồi id dựa trên original title
                    for b_id, b_val in batch_map.items():
                        if b_val["title"] == item["original"]:
                            item["id"] = b_id
                            break

                if "id" in item:
                    item_id = item["id"]
                    # Phục hồi original title nếu Gemini không trả về
                    if "original" not in item and item_id in batch_map:
                        item["original"] = batch_map[item_id]["title"]
                        
                    results_dict[item_id] = item
                    print(f"  [{item.get('standardized_title', '?')}] {item.get('original', '')}")
                else:
                    print(f"  [WARNING] Missing 'id' in response item: {item}", file=sys.stderr)
                    
        except Exception as e:
            print(f"  [ERROR batch {batch_num}] {e}", file=sys.stderr)
            for j in batch:
                results_dict[j["id"]] = {"id": j["id"], "original": j["title"], "error": str(e)}
                
        done_so_far = len(results_dict)
        pct = done_so_far / total * 100
        print(f"[{pct:5.1f}%] batch {batch_num} done ({done_so_far}/{total})")
        
        # checkpoint sau mỗi batch, lưu file sắp xếp theo id
        sorted_results = [results_dict[k] for k in sorted(results_dict.keys())]
        output_file.write_text(json.dumps(sorted_results, ensure_ascii=False, indent=2), encoding="utf-8")
        
    return [results_dict[k] for k in sorted(results_dict.keys())]

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LLM job title standardization")
    parser.add_argument(
        "--dataset", choices=["topcv", "itviec"], default="topcv",
        help="Dataset to process (default: topcv). "
             "Note: if resuming TopCV from old checkpoint, rename "
             "output/job_title_full.json → output/topcv_job_title_full.json first."
    )
    args = parser.parse_args()

    CSV_FILE = OUTPUT_DIR / f"00-{args.dataset}_filtered.csv"
    OUTPUT_FILE = OUTPUT_DIR / f"{args.dataset}_job_title_full.json"

    jobs = load_jobs(CSV_FILE)
    system_prompt = PROMPT_FILE.read_text(encoding="utf-8")

    client = genai.Client(vertexai=True, project=PROJECT, location=LOCATION)
    print(f"Dataset: {args.dataset} | Total jobs: {len(jobs)}, batch size: {BATCH_SIZE}\n")

    results = process_jobs(client, system_prompt, jobs, OUTPUT_FILE)
    print(f"\nDone. {len(results)} records → {OUTPUT_FILE}")