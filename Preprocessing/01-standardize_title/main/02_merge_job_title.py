"""
Merge kết quả LLM standardization vào CSV gốc bằng id.

Input:
    data/raw/00-topcv_raw.csv
    01-standardize_title/output/job_title_full.json

Output:
    01-standardize_title/output/01-topcv_llm_standardized.csv
"""

import csv
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).parent.parent

CSV_IN  = ROOT.parent.parent / "data" / "raw" / "00-topcv_raw.csv"
JSON_IN = ROOT / "output" / "job_title_full.json"
CSV_OUT = ROOT / "output" / "01-topcv_llm_standardized.csv"

NEW_COLS = [
    "is_valid_job",
    "standardized_title",
]

# -------------------------------------------------------------------


def load_json_map(json_path: Path):
    data = json.loads(
        json_path.read_text(encoding="utf-8")
    )

    result = {}

    for item in data:
        if "id" in item:
            result[item["id"]] = item

    return result


def merge(
    csv_in: Path,
    json_in: Path,
    csv_out: Path,
):
    result_map = load_json_map(json_in)

    matched = 0
    missing = 0
    empty = 0

    with open(csv_in, encoding="utf-8-sig", newline="") as fin, \
         open(csv_out, "w", encoding="utf-8-sig", newline="") as fout:

        reader = csv.DictReader(fin)

        writer = csv.DictWriter(
            fout,
            fieldnames=reader.fieldnames + NEW_COLS,
        )

        writer.writeheader()

        for idx, row in enumerate(reader):
            title = row["job_title"].strip()

            if not title:
                row.update({
                    "is_valid_job": "",
                    "standardized_title": "",
                })

                empty += 1

            else:
                hit = result_map.get(idx)

                if not hit or "error" in hit:
                    row.update({
                        "is_valid_job": "",
                        "standardized_title": "",
                    })

                    missing += 1

                else:
                    row["is_valid_job"] = hit.get(
                        "is_valid_job", ""
                    )

                    row["standardized_title"] = hit.get(
                        "standardized_title", ""
                    )

                    matched += 1

            writer.writerow(row)

    # ---------------------------------------------------------------

    total = matched + missing + empty

    print(f"CSV rows    : {total}")
    print(f"Matched     : {matched}")
    print(f"Missing     : {missing}")
    print(f"Empty title : {empty}")

    print(f"\nSaved → {csv_out}")


# -------------------------------------------------------------------

if __name__ == "__main__":
    merge(
        csv_in=CSV_IN,
        json_in=JSON_IN,
        csv_out=CSV_OUT,
    )