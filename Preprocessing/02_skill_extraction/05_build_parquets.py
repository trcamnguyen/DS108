"""Build jobs.parquet and skills.parquet from raw and skill-extraction outputs.

Usage (run from project root):
    python Preprocessing/02_skill_extraction/05_build_parquets.py
    python Preprocessing/02_skill_extraction/05_build_parquets.py \
        --raw-csv data/interim/01-standardized_title.csv \
        --skills-csv Preprocessing/02_skill_extraction/outputs/annotations_with_canonical.csv \
        --output-dir data/interim/02-skill_extracted
"""

import argparse
import os

import pandas as pd


SKILLS_KEEP_COLS = [
    "job_id",
    "skill_name",
    "final_canonical",
    "label",
    "category",
    "level",
    "min_years",
    "source_text",
]


def build_tables(raw_csv: str, skills_csv: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # ── jobs ────────────────────────────────────────────────────────────────
    print(f"Reading raw data from {raw_csv} ...")
    raw = pd.read_csv(raw_csv, dtype=str)
    print(f"  Raw rows: {len(raw)}")

    raw = raw.drop_duplicates(subset="url", keep="last").reset_index(drop=True)
    print(f"  After URL dedup: {len(raw)}")

    raw.insert(0, "job_id", range(1, len(raw) + 1))

    raw = raw.rename(columns={
        "required_skills": "platform_required_skills",
        "preferred_skills": "platform_preferred_skills",
        "level": "job_level",
    })

    assert raw["job_id"].is_unique and raw["job_id"].notna().all(), \
        "job_id not unique or contains nulls"

    jobs_path = os.path.join(output_dir, "jobs.parquet")
    raw.to_parquet(jobs_path, index=False)
    print(f"  Saved {jobs_path}: {len(raw)} rows")

    # ── skills ───────────────────────────────────────────────────────────────
    print(f"\nReading skills data from {skills_csv} ...")
    skills = pd.read_csv(skills_csv, dtype={"level": str})
    print(f"  Skills rows: {len(skills)}")

    # Drop the old job_id (generated in extraction step) and re-derive from URL
    if "job_id" in skills.columns:
        skills = skills.drop(columns=["job_id"])

    url_to_job_id = raw.set_index("url")["job_id"]
    skills["job_id"] = skills["url"].map(url_to_job_id)

    unmatched = skills["job_id"].isna().sum()
    if unmatched > 0:
        print(f"  WARNING: {unmatched} skill rows have no matching URL in jobs — dropping")
        skills = skills[skills["job_id"].notna()]

    skills["job_id"] = skills["job_id"].astype(int)

    missing_jobs = set(skills["job_id"]) - set(raw["job_id"])
    assert not missing_jobs, f"job_ids in skills missing from jobs: {missing_jobs}"

    available = [c for c in SKILLS_KEEP_COLS if c in skills.columns]
    missing_cols = set(SKILLS_KEEP_COLS) - set(available)
    if missing_cols:
        print(f"  WARNING: expected columns not found in skills CSV: {missing_cols}")
    skills = skills[available]

    # Convert to nullable integer (handles float64 NaN → pd.NA)
    skills["min_years"] = pd.to_numeric(skills["min_years"], errors="coerce").convert_dtypes()

    skills_path = os.path.join(output_dir, "skills.parquet")
    skills.to_parquet(skills_path, index=False)
    print(f"  Saved {skills_path}: {len(skills)} rows, {skills['job_id'].nunique()} unique jobs")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build jobs.parquet and skills.parquet")
    parser.add_argument(
        "--raw-csv",
        default="data/interim/01-standardized_title.csv",
    )
    parser.add_argument(
        "--skills-csv",
        default="Preprocessing/02_skill_extraction/outputs/annotations_with_canonical.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="data/interim/02-skill_extracted",
    )
    args = parser.parse_args()
    build_tables(args.raw_csv, args.skills_csv, args.output_dir)


if __name__ == "__main__":
    main()
