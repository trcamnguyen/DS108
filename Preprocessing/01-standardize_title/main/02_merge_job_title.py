"""
Merge kết quả LLM standardization vào CSV gốc bằng id.
"""

import csv
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).parent.parent

NEW_COLS = [
    "standardized_title",
]

# -------------------------------------------------------------------


def load_json_map(json_path: Path, csv_path: Path | None = None):
    """Load JSON map keyed by url (id field).

    Nếu JSON cũ dùng numeric id, truyền csv_path để migrate tự động:
    đọc filtered CSV, build idx→url map, rewrite JSON với url làm key.
    """
    data = json.loads(json_path.read_text(encoding="utf-8"))

    # Phát hiện JSON cũ: id là số nguyên
    if data and isinstance(data[0].get("id"), int):
        if csv_path is None:
            raise ValueError(
                "JSON dùng numeric id — truyền csv_path để migrate sang url-based key."
            )
        print("  [migrate] JSON dùng numeric id → converting to url-based key...")
        idx_to_url = {}
        with open(csv_path, encoding="utf-8-sig", newline="") as f:
            for i, row in enumerate(csv.DictReader(f)):
                url = row.get("url", "").strip()
                if url:
                    idx_to_url[i] = url
        for item in data:
            old_id = item.get("id")
            if isinstance(old_id, int) and old_id in idx_to_url:
                item["id"] = idx_to_url[old_id]
        json_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"  [migrate] Done → {json_path}")

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
    result_map = load_json_map(json_in, csv_path=csv_in)

    matched = 0
    missing = 0
    empty = 0
    invalid = 0

    with open(csv_in, encoding="utf-8-sig", newline="") as fin, \
         open(csv_out, "w", encoding="utf-8-sig", newline="") as fout:

        reader = csv.DictReader(fin)

        writer = csv.DictWriter(
            fout,
            fieldnames=reader.fieldnames + NEW_COLS,
        )

        writer.writeheader()

        for row in reader:
            title = row["job_title"].strip()

            if not title:
                empty += 1
                continue

            else:
                hit = result_map.get(row.get("url", "").strip())

                if not hit or "error" in hit:
                    row["standardized_title"] = ""

                    missing += 1
                    writer.writerow(row)

                else:
                    is_valid = hit.get("is_valid_job", True)

                    if is_valid is False:
                        invalid += 1
                        continue

                    row["standardized_title"] = hit.get("standardized_title", "")

                    matched += 1
                    writer.writerow(row)

    # ---------------------------------------------------------------

    total = matched + missing + empty + invalid

    print(f"CSV rows    : {total}")
    print(f"Matched     : {matched}")
    print(f"Missing     : {missing}")
    print(f"Empty title : {empty}")
    print(f"Invalid     : {invalid}  ← dropped")

    print(f"\nSaved → {csv_out}")


# -------------------------------------------------------------------

def merge_patch(
    csv_in: Path,
    json_in: Path,
    csv_out: Path,
    recrawled_urls: set,
):
    """Cập nhật is_valid_job + standardized_title CHỈ cho các row có URL trong
    recrawled_urls. Các row khác giữ nguyên giá trị hiện tại."""
    result_map = load_json_map(json_in)

    patched = 0
    unchanged = 0

    rows = []
    fieldnames = None

    with open(csv_in, encoding="utf-8-sig", newline="") as fin:
        reader = csv.DictReader(fin)
        fieldnames = reader.fieldnames

        for idx, row in enumerate(reader):
            url = row.get("url", "")
            if url in recrawled_urls:
                hit = result_map.get(idx)
                if hit and "error" not in hit:
                    row["is_valid_job"]      = hit.get("is_valid_job", "")
                    row["standardized_title"] = hit.get("standardized_title", "")
                    patched += 1
                # nếu không có kết quả LLM mới → giữ nguyên
            else:
                unchanged += 1
            rows.append(row)

    with open(csv_out, "w", encoding="utf-8-sig", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Patched     : {patched}")
    print(f"Unchanged   : {unchanged}")
    print(f"\nSaved → {csv_out}")


# -------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Merge LLM results into dataset CSV")
    parser.add_argument(
        "--dataset", choices=["topcv", "itviec", "all"], default="all",
        help="Dataset to merge (default: all)",
    )
    parser.add_argument(
        "--recrawled-only", action="store_true",
        help="Chỉ patch is_valid_job + standardized_title cho brand_recrawled rows, "
             "không đụng các row khác trong 01-*_llm_standardized.csv.",
    )
    args = parser.parse_args()

    datasets = ["topcv", "itviec"] if args.dataset == "all" else [args.dataset]

    if args.recrawled_only:
        import pandas as pd
        REPO_ROOT = ROOT.parent.parent
        for dataset in datasets:
            print(f"\n[{dataset.upper()}] recrawled-only patch")
            RAW_CSV = REPO_ROOT / "data" / "raw" / f"00-{dataset}_raw.csv"
            raw_df  = pd.read_csv(RAW_CSV, encoding="utf-8-sig")
            recrawled_urls = set(
                raw_df.loc[raw_df["brand_recrawled"] == True, "url"].dropna()
            )
            print(f"brand_recrawled URLs : {len(recrawled_urls)}")

            CSV_IN  = ROOT / "output" / f"01-{dataset}_llm_standardized.csv"
            JSON_IN = ROOT / "output" / f"{dataset}_job_title_full.json"

            merge_patch(
                csv_in=CSV_IN,
                json_in=JSON_IN,
                csv_out=CSV_IN,       # patch in-place
                recrawled_urls=recrawled_urls,
            )
    else:
        for dataset in datasets:
            print(f"\n[{dataset.upper()}]")
            CSV_IN  = ROOT / "output" / f"00-{dataset}_filtered.csv"
            JSON_IN = ROOT / "output" / f"{dataset}_job_title_full.json"
            CSV_OUT = ROOT / "output" / f"01-{dataset}_llm_standardized.csv"

            merge(
                csv_in=CSV_IN,
                json_in=JSON_IN,
                csv_out=CSV_OUT,
            )