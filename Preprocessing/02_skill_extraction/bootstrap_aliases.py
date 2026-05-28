"""
DS108 — Bootstrap Aliases via Gemini 2.5 Pro
==============================================
Input  : outputs/skill_distribution_after.csv
Output : outputs/aliases.yaml

Pipeline:
  1. Đọc skill_distribution_after.csv, filter total_count >= min_count
  2. Format + sort (category ASC, count DESC), chia batch
  3. Gọi Gemini 2.5 Pro với prompt_taxonomy.txt
  4. Validate output (canonical/variant phải có trong input, no circular, no dup)
  5. Merge tất cả batch → ghi aliases.yaml với metadata block

Usage:
    python bootstrap_aliases.py
    python bootstrap_aliases.py --input outputs/skill_distribution_after.csv \
                                --output outputs/aliases.yaml \
                                --min-count 5 --batch-size 200
"""

import os
import re
import yaml
import argparse
import logging
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict

# ─── ROOT & ENV ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent.parent

env_path = ROOT / ".env"
if env_path.exists():
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"'))

PROJECT   = os.environ.get("GOOGLE_CLOUD_PROJECT", "your-project-id")
LOCATION  = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1").strip('"')
MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-2.5-pro")

if "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
    creds_path = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
    if not Path(creds_path).is_absolute():
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(ROOT / creds_path)

from google import genai
from google.genai import types

# ─── CONFIG ───────────────────────────────────────────────────────────────────
TEMPERATURE     = 0.2
MAX_TOKENS      = 8192
HERE            = Path(__file__).parent
PROMPT_PATH     = HERE / "prompt" / "prompt_taxonomy.txt"

SYSTEM_PROMPT = (
    "Bạn là expert IT skill taxonomy. Nhiệm vụ: nhóm skill cùng concept thành alias groups. "
    "Output YAML THUẦN TÚY — không markdown fence, không giải thích, không preamble."
)

_LOG_DIR = HERE / "outputs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = _LOG_DIR / "bootstrap_aliases.log"

_fmt = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
_console = logging.StreamHandler()
_console.setFormatter(_fmt)
_file_handler = logging.FileHandler(_LOG_FILE, encoding="utf-8", mode="a")
_file_handler.setFormatter(_fmt)

logging.basicConfig(level=logging.INFO, handlers=[_console, _file_handler])
log = logging.getLogger(__name__)

# ─── LOAD PROMPT TEMPLATE ─────────────────────────────────────────────────────

def load_prompt_template() -> str:
    """Đọc prompt_taxonomy.txt, chuẩn bị để dùng làm user prompt template."""
    raw = PROMPT_PATH.read_text(encoding="utf-8")
    # Thay placeholder cứng bằng {skill_list} để format sau
    placeholder = "[paste 100 skill từ skills_for_alias.txt vào đây]"
    if placeholder in raw:
        return raw.replace(placeholder, "{skill_list}")
    # Fallback: nếu không có placeholder, append skill list vào cuối
    log.warning("Không tìm thấy placeholder trong prompt_taxonomy.txt — append skill list vào cuối.")
    return raw.rstrip() + "\n\n{skill_list}\n\nOutput YAML aliases:"

PROMPT_TEMPLATE = load_prompt_template()

# ─── CLIENT INIT ──────────────────────────────────────────────────────────────

def init_client() -> genai.Client:
    creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not creds:
        default = str(ROOT / "credentials" / "service-account.json")
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = default
        log.info(f"Using default credentials: {default}")
    return genai.Client(vertexai=True, project=PROJECT, location=LOCATION)

# ─── PREPARE INPUT ────────────────────────────────────────────────────────────

def load_skills(csv_path: Path, min_count: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8")
    required_cols = {"final_canonical", "dominant_category", "total_count"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV thiếu cột: {missing}")
    df["final_canonical"] = df["final_canonical"].astype(str).str.strip()
    df["dominant_category"] = df["dominant_category"].astype(str).str.strip()
    df["total_count"] = pd.to_numeric(df["total_count"], errors="coerce").fillna(0).astype(int)
    filtered = df[df["total_count"] >= min_count].copy()
    # Sort: category ASC, count DESC
    filtered = filtered.sort_values(
        ["dominant_category", "total_count"], ascending=[True, False]
    ).reset_index(drop=True)
    log.info(f"Input rows: {len(df)} | After filter (count >= {min_count}): {len(filtered)}")
    return filtered


def format_skill_list(df: pd.DataFrame) -> list[str]:
    """Trả về list chuỗi 1 dòng/skill: '{name} | count={n} | category={cat}'"""
    lines = []
    for _, row in df.iterrows():
        lines.append(
            f"{row['final_canonical']} | count={row['total_count']} | category={row['dominant_category']}"
        )
    return lines


def make_batches(lines: list[str], batch_size: int) -> list[list[str]]:
    return [lines[i : i + batch_size] for i in range(0, len(lines), batch_size)]

# ─── API CALL ─────────────────────────────────────────────────────────────────

def call_gemini(client: genai.Client, skill_lines: list[str], batch_idx: int) -> list[dict]:
    """Gọi Gemini 1 batch → parse YAML → return list alias groups."""
    skill_list_str = "\n".join(skill_lines)
    user_prompt = PROMPT_TEMPLATE.format(skill_list=skill_list_str)

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=TEMPERATURE,
                max_output_tokens=MAX_TOKENS,
            ),
        )
    except Exception as e:
        log.error(f"  [Batch {batch_idx}] API call failed: {e}")
        return []

    raw = response.text.strip()

    # Defensive: strip markdown fences nếu Gemini vẫn output
    if raw.startswith("```"):
        lines = raw.split("\n")
        # bỏ dòng đầu (```yaml hoặc ```) và dòng cuối ```)
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines).strip()

    try:
        parsed = yaml.safe_load(raw)
        if not isinstance(parsed, dict):
            log.warning(f"  [Batch {batch_idx}] YAML parse OK nhưng root không phải dict. Skipping.")
            return []
        groups = parsed.get("aliases", [])
        if not isinstance(groups, list):
            log.warning(f"  [Batch {batch_idx}] 'aliases' không phải list. Skipping.")
            return []
        log.info(f"  [Batch {batch_idx}] OK — {len(groups)} alias groups")
        return groups
    except yaml.YAMLError as e:
        log.warning(f"  [Batch {batch_idx}] YAML parse failed: {e}")
        log.warning(f"  Raw output (first 500 chars): {raw[:500]}")
        return []

# ─── VALIDATE ─────────────────────────────────────────────────────────────────

def validate_groups(
    groups: list[dict],
    valid_names: set[str],
    name_to_count: dict[str, int],
    batch_idx: int,
) -> list[dict]:
    """
    Validate alias groups:
    1. Canonical phải trong valid_names; nếu không → auto-promote variant có count cao nhất
    2. Mỗi variant phải trong valid_names
    3. Group cần >= 1 variant hợp lệ sau promote
    """
    clean = []
    for g in groups:
        if not isinstance(g, dict):
            continue
        canonical = str(g.get("canonical", "")).strip().lower()
        if not canonical:
            continue

        raw_variants = g.get("variants", [])
        if not isinstance(raw_variants, list):
            continue

        # Lọc variants hợp lệ trước (không phụ thuộc vào canonical)
        valid_variants = []
        for v in raw_variants:
            v_str = str(v).strip().lower()
            if not v_str:
                continue
            if v_str not in valid_names:
                log.warning(f"  [Batch {batch_idx}] DROP variant '{v_str}' (không có trong input)")
                continue
            valid_variants.append(v_str)

        # Nếu canonical không có trong input → auto-promote
        if canonical not in valid_names:
            if not valid_variants:
                log.warning(
                    f"  [Batch {batch_idx}] DROP group '{canonical}': "
                    f"canonical hallucinated + không có variant hợp lệ"
                )
                continue
            # Promote variant có count cao nhất lên làm canonical
            promoted = max(valid_variants, key=lambda v: name_to_count.get(v, 0))
            log.warning(
                f"  [Batch {batch_idx}] PROMOTE '{promoted}' → canonical "
                f"(thay '{canonical}' hallucinated, count={name_to_count.get(promoted, 0)})"
            )
            canonical = promoted
            valid_variants = [v for v in valid_variants if v != canonical]

        else:
            # Bỏ variant trùng canonical
            valid_variants = [v for v in valid_variants if v != canonical]

        if not valid_variants:
            log.warning(f"  [Batch {batch_idx}] DROP group '{canonical}': không còn variant hợp lệ")
            continue

        clean.append({"canonical": canonical, "variants": valid_variants})
    return clean

# ─── MERGE ────────────────────────────────────────────────────────────────────

def merge_groups(all_groups: list[dict], name_to_count: dict[str, int]) -> list[dict]:
    """
    Merge tất cả batch:
    - Gộp group cùng canonical (union variants)
    - No circular: variant không được là canonical của group khác
      → nếu conflict, ưu tiên canonical có count cao hơn
    - No duplicate: variant chỉ thuộc 1 canonical
    - Sort canonical alphabetically
    """
    # Step 1: gộp group cùng canonical
    merged: dict[str, set] = defaultdict(set)
    for g in all_groups:
        c = g["canonical"]
        for v in g["variants"]:
            merged[c].add(v)

    # Step 2: resolve circular — variant không được là canonical
    canonical_set = set(merged.keys())
    for c in list(merged.keys()):
        bad = merged[c] & canonical_set  # variants that are also canonicals
        for b in bad:
            # so sánh count, giữ canonical có count cao hơn
            count_c = name_to_count.get(c, 0)
            count_b = name_to_count.get(b, 0)
            if count_c >= count_b:
                log.warning(f"  CIRCULAR: '{b}' là cả canonical và variant của '{c}' — drop khỏi variants của '{c}'")
                merged[c].discard(b)
            else:
                log.warning(f"  CIRCULAR: '{b}' là cả canonical và variant của '{c}' — drop canonical '{c}' thay vào đó")
                # move tất cả variants của c sang b (nếu chưa có)
                merged[b].update(merged[c] - {b})
                del merged[c]
                break

    # Step 3: no duplicate variant — variant chỉ thuộc 1 canonical
    variant_owner: dict[str, str] = {}
    for c in sorted(merged.keys(), key=lambda x: -name_to_count.get(x, 0)):
        clean_variants = set()
        for v in merged[c]:
            if v in variant_owner:
                log.warning(
                    f"  DEDUP: variant '{v}' đã thuộc '{variant_owner[v]}', "
                    f"drop khỏi '{c}'"
                )
            else:
                variant_owner[v] = c
                clean_variants.add(v)
        merged[c] = clean_variants

    # Step 4: drop group không còn variant
    result = []
    for c in sorted(merged.keys()):
        if merged[c]:
            result.append({"canonical": c, "variants": sorted(merged[c])})
    return result

# ─── SANITY CHECK ─────────────────────────────────────────────────────────────

def print_sanity(
    df_filtered: pd.DataFrame,
    n_input_total: int,
    min_count: int,
    groups: list[dict],
    batch_sizes: list[int],
) -> None:
    n_filtered = len(df_filtered)
    n_other = int((df_filtered["dominant_category"].str.contains("Other")).sum())
    n_batches = len(batch_sizes)
    n_variants_total = sum(len(g["variants"]) for g in groups)
    n_skills_in_input = n_filtered
    pct = f"{n_variants_total / n_skills_in_input * 100:.1f}%" if n_skills_in_input else "0%"

    # Cross-category groups
    canonical_to_cat = dict(zip(df_filtered["final_canonical"].str.lower(), df_filtered["dominant_category"]))
    cross_cat = 0
    for g in groups:
        cats = {canonical_to_cat.get(g["canonical"], "?")}
        for v in g["variants"]:
            cats.add(canonical_to_cat.get(v, "?"))
        if len(cats) > 1:
            cross_cat += 1

    batch_str = " + ".join(str(s) for s in batch_sizes)
    print()
    print("=" * 60)
    print("SANITY CHECK")
    print("=" * 60)
    print(f"Input skills (after filter count >= {min_count}): {n_filtered}")
    print(f"  Including <Category> Other: {n_other} skills")
    print(f"Batches: {n_batches} ({batch_str})")
    print(f"Alias groups generated: {len(groups)}")
    print(f"Variants mapped: {n_variants_total} ({pct} of input)")
    print(f"Groups with cross-category variants: {cross_cat}  ← cần review thủ công")
    print()
    # Top 20 by variant count
    top = sorted(groups, key=lambda g: len(g["variants"]), reverse=True)[:20]
    if top:
        print("Top alias groups by variant count:")
        for g in top:
            variants_preview = " | ".join(g["variants"][:5])
            more = f" + {len(g['variants'])-5} more" if len(g["variants"]) > 5 else ""
            print(f"  {g['canonical']} ({len(g['variants'])} variants): {variants_preview}{more}")
    print("=" * 60)
    print()

# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main(
    input_path: Path,
    output_path: Path,
    min_count: int,
    batch_size: int,
) -> None:
    log.info("=" * 60)
    log.info(f"RUN START  input={input_path}  min_count={min_count}  batch_size={batch_size}")
    log.info(f"Log file → {_LOG_FILE}")
    log.info("=" * 60)

    # 1. Load & filter
    df_all = pd.read_csv(input_path, encoding="utf-8")
    n_input_total = len(df_all)
    df_filtered = load_skills(input_path, min_count)
    skill_lines = format_skill_list(df_filtered)
    batches = make_batches(skill_lines, batch_size)

    valid_names = set(df_filtered["final_canonical"].str.strip().str.lower())
    name_to_count = dict(
        zip(df_filtered["final_canonical"].str.lower(), df_filtered["total_count"])
    )

    log.info(f"Model: {MODEL_NAME} | Batches: {len(batches)} | batch_size: {batch_size}")

    # 2. Init client
    client = init_client()

    # 3. Call API per batch + validate
    all_groups: list[dict] = []
    batch_sizes: list[int] = []
    for i, batch in enumerate(batches, 1):
        log.info(f"Batch {i}/{len(batches)} — {len(batch)} skills")
        raw_groups = call_gemini(client, batch, i)
        validated = validate_groups(raw_groups, valid_names, name_to_count, i)
        all_groups.extend(validated)
        batch_sizes.append(len(batch))

    # 4. Merge + dedupe
    final_groups = merge_groups(all_groups, name_to_count)

    # 5. Write output
    n_variants_total = sum(len(g["variants"]) for g in final_groups)
    metadata = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        "model": MODEL_NAME,
        "input_file": str(input_path),
        "filter_rule": f"total_count >= {min_count} (including <Category> Other)",
        "n_input_skills": len(df_filtered),
        "n_batches": len(batches),
        "batch_size": batch_size,
        "n_alias_groups": len(final_groups),
        "n_variants_total": n_variants_total,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        yaml.dump({"metadata": metadata}, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
        f.write("\n")
        yaml.dump({"aliases": final_groups}, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

    log.info(f"Output written → {output_path}")

    # 6. Sanity check
    print_sanity(df_filtered, n_input_total, min_count, final_groups, batch_sizes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bootstrap aliases.yaml via Gemini 2.5 Pro")
    parser.add_argument(
        "--input",
        default=str(HERE / "outputs" / "skill_distribution_after.csv"),
        help="Path to skill_distribution_after.csv",
    )
    parser.add_argument(
        "--output",
        default=str(HERE / "outputs" / "aliases.yaml"),
        help="Output path for aliases.yaml",
    )
    parser.add_argument(
        "--min-count", type=int, default=5,
        help="Filter skill có total_count >= N (default: 5)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=200,
        help="Số skill mỗi batch (default: 200)",
    )
    args = parser.parse_args()
    main(
        input_path=Path(args.input),
        output_path=Path(args.output),
        min_count=args.min_count,
        batch_size=args.batch_size,
    )
