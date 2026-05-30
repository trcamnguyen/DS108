"""
Pipeline xử lý dữ liệu: đọc jobs.parquet → áp dụng 4 modules → lưu jobs_cleaned.parquet.

Usage:
    python Preprocessing/03_preprocessing/pipeline.py
"""
from pathlib import Path
import pandas as pd

from salary_processor import process_salary
from location_processor import process_location
from fields_extractor import process_fields
from level_normalizer import process_level

_ROOT = Path(__file__).resolve().parent.parent.parent
INPUT = _ROOT / "data" / "interim" / "02-skill_extracted" / "jobs.parquet"
OUTPUT = _ROOT / "data" / "processed" / "jobs_cleaned.parquet"


def run(input_path: Path = INPUT, output_path: Path = OUTPUT) -> pd.DataFrame:
    df = pd.read_parquet(input_path)
    print(f"Loaded {len(df)} rows from {input_path.name}")

    df = process_salary(df)
    print("salary_processor done")

    df = process_location(df)
    print("location_processor done")

    df = process_fields(df)
    print("fields_extractor done")

    df = process_level(df)
    print("level_normalizer done")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Saved to {output_path}  ({len(df)} rows, {len(df.columns)} cols)")
    return df


if __name__ == "__main__":
    run()
