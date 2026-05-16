"""
Test: chạy thử 20 random sample cố định (seed=42) từ topcv_merged.csv.
Kết quả lưu vào Preprocess/output/job_title_test.json.
Yêu cầu: credentials/service-account.json + các biến trong .env
"""

import csv
import json
import os
import random
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).parent.parent

# --- load .env ---
for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"'))

PROJECT  = os.environ["GOOGLE_CLOUD_PROJECT"]
LOCATION = os.environ["GOOGLE_CLOUD_LOCATION"].strip('"')
MODEL    = os.environ["GEMINI_MODEL"]
CREDS    = ROOT / os.environ["GOOGLE_APPLICATION_CREDENTIALS"]

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(CREDS)

from google import genai          # noqa: E402
# pyrefly: ignore [missing-import]
from google.genai import types    # noqa: E402

PROMPT_FILE  = Path(__file__).parent / "prompt" / "job_title.txt"
CSV_FILE     = ROOT / "Crawl" / "data" / "topcv_merged.csv"
OUTPUT_DIR   = Path(__file__).parent / "output"
OUTPUT_FILE  = OUTPUT_DIR / "job_title_test.json"

BATCH_SIZE  = 10
TEST_SAMPLE = 20
TEST_SEED   = 42


# ---------------------------------------------------------------------------

def load_titles(csv_path: Path, limit: int | None = None) -> list[str]:
    titles = []
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            t = row["job_title"].strip()
            if t:
                titles.append(t)
            if limit and len(titles) >= limit:
                break
    return titles


def parse_response(raw: str) -> list[dict]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    return json.loads(raw)


def load_checkpoint(output_file: Path) -> list[dict]:
    if output_file.exists():
        try:
            data = json.loads(output_file.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
        except Exception:
            pass
    return []


def process_titles(client: genai.Client, system_prompt: str, titles: list[str], output_file: Path):
    total   = len(titles)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results   = load_checkpoint(output_file)
    done_set  = {item["original"] for item in results if "original" in item}
    remaining = [t for t in titles if t not in done_set]

    if done_set:
        print(f"Resumed: {len(done_set)} already done, {len(remaining)} remaining.\n")

    for i in range(0, len(remaining), BATCH_SIZE):
        batch     = remaining[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        done_so_far = len(done_set) + min(i + BATCH_SIZE, len(remaining))
        pct       = done_so_far / total * 100

        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=json.dumps(batch, ensure_ascii=False),
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0,
                ),
            )
            parsed = parse_response(response.text)
            results.extend(parsed)

            for item in parsed:
                role = item.get("core_role", "?")
                print(f"  [{role}] {item.get('original', '')}")

        except Exception as e:
            print(f"  [ERROR batch {batch_num}] {e}", file=sys.stderr)
            for t in batch:
                results.append({"original": t, "error": str(e)})

        print(f"[{pct:5.1f}%] batch {batch_num} done  ({done_so_far}/{total})")

        # checkpoint sau mỗi batch
        output_file.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    return results


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    all_titles    = load_titles(CSV_FILE)
    rng           = random.Random(TEST_SEED)
    titles        = rng.sample(all_titles, TEST_SAMPLE)
    system_prompt = PROMPT_FILE.read_text(encoding="utf-8")
    client        = genai.Client(vertexai=True, project=PROJECT, location=LOCATION)

    print(f"Test sample: {TEST_SAMPLE} titles (seed={TEST_SEED}), batch size: {BATCH_SIZE}\n")
    results = process_titles(client, system_prompt, titles, OUTPUT_FILE)
    print(f"\nDone. {len(results)} records → {OUTPUT_FILE}")
