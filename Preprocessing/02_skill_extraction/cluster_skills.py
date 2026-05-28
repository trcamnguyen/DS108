#!/usr/bin/env python3
"""
cluster_skills.py — DS108 Skill Canonicalization Pipeline
==========================================================
Post-processing step: takes raw extracted skill mentions from LLM and
canonicalizes them using multilingual embeddings + agglomerative clustering.

Usage:
    python Preprocessing/02_skill_extraction/cluster_skills.py `
        --input Preprocessing/02_skill_extraction/output_full/full_parsed.csv `
        --output-dir Preprocessing/02_skill_extraction/outputs/ `
        --aliases Preprocessing/02_skill_extraction/outputs/aliases.yaml `
        --save-aliased Preprocessing/02_skill_extraction/outputs/full_parsed_aliased.csv `
        --min-keep-count 5 `
        --short-threshold 6 `
        --role-block Preprocessing/02_skill_extraction/role_block.txt


Dependencies:
    python >= 3.10, sentence-transformers >= 2.7, scikit-learn >= 1.3,
    pandas >= 2.0, numpy >= 1.24
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sentence-transformers import guard (§ spec)
# ---------------------------------------------------------------------------
try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:  # noqa: F841
    raise ImportError(
        "sentence-transformers is not installed. "
        "Install it with: pip install sentence-transformers>=2.7"
    ) from exc

try:
    from sklearn.cluster import AgglomerativeClustering
except ImportError as exc:  # noqa: F841
    raise ImportError(
        "scikit-learn is not installed. "
        "Install it with: pip install scikit-learn>=1.3"
    ) from exc

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MATCHING_VERSION = "v3_embedding_clustering"

REQUIRED_COLUMNS_BASE = {"skill_name", "label", "category"}

# Vietnamese prefix descriptors to strip (§3 — apply one-time, not loop)
PREFIX_DESCRIPTORS = [
    "có kinh nghiệm với ",  # longer variants first
    "có kinh nghiệm ",
    "hiểu biết về ",
    "kiến thức về ",
    "kỹ năng ",
    "khả năng ",
    "kĩ năng ",
    "kinh nghiệm ",
    "am hiểu ",
]

# Sort by descending length to match the longest prefix first
PREFIX_DESCRIPTORS.sort(key=len, reverse=True)

ACTION_ORDER = {"REVIEW_SPLIT": 0, "MERGE_REVIEW": 1, "MERGE": 2, "IGNORE": 3, "N/A": 4}


# ---------------------------------------------------------------------------
# §3  Pre-normalization
# ---------------------------------------------------------------------------

def _normalize_dots(s: str) -> str:
    """
    Handle '.' according to spec §3.1 rule 4:
    - Keep '.' if it is between 2 letters AND total string length < 6 chars.
    - Otherwise replace '.' with space.
    - Strip trailing '.'.
    """
    result = []
    for i, ch in enumerate(s):
        if ch == ".":
            left_is_letter = i > 0 and s[i - 1].isalpha()
            right_is_letter = i < len(s) - 1 and s[i + 1].isalpha()
            if left_is_letter and right_is_letter and len(s) < 6:
                result.append(ch)  # keep (e.g. "a.i")
            elif i == len(s) - 1:
                pass  # strip trailing dot
            else:
                result.append(" ")
        else:
            result.append(ch)
    # Strip trailing dot that survived (edge case)
    joined = "".join(result).rstrip(".")
    return joined


def normalize_skill(raw: str) -> str:
    """Apply §3 pre-normalization pipeline to a single skill_name string."""
    # 1. Lowercase
    s = raw.lower()

    # 2. Strip whitespace
    s = s.strip()

    # 3. Strip prefix descriptor (one-time)
    for prefix in PREFIX_DESCRIPTORS:
        if s.startswith(prefix):
            candidate = s[len(prefix):]
            if candidate.strip():  # non-empty after strip → apply
                s = candidate.strip()
            break  # apply at most one prefix

    # 4a. Replace '/' and '_' → space
    s = s.replace("/", " ").replace("_", " ")

    # 4b. Replace '-' between alphanumeric chars → space
    #     e.g. "fine-tuning" → "fine tuning", but "c++" unchanged
    s = re.sub(r"(?<=[A-Za-z0-9])-(?=[A-Za-z0-9])", " ", s)

    # 4c. Handle '.' according to position and length rules
    s = _normalize_dots(s)

    # 5. Collapse multiple spaces → single space, strip
    s = re.sub(r" +", " ", s).strip()

    return s


# ---------------------------------------------------------------------------
# §3.5  Alias helpers
# ---------------------------------------------------------------------------

def load_alias_map(aliases_path: Path) -> dict[str, str]:
    """Load aliases.yaml → flat dict {variant: canonical}."""
    with aliases_path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    alias_map: dict[str, str] = {}
    for group in data.get("aliases", []):
        canonical = str(group.get("canonical", "")).strip()
        if not canonical:
            continue
        for variant in group.get("variants", []):
            v = str(variant).strip()
            if v and v != canonical:
                alias_map[v] = canonical
    return alias_map


def load_role_block(path: Path) -> frozenset[str]:
    """Load role_block.txt — one role per line, returns normalized frozenset. Lines starting with '#' are ignored."""
    roles: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                roles.add(normalize_skill(line))
    return frozenset(roles)


def resolve_transitive(alias_map: dict[str, str], max_iter: int = 5) -> dict[str, str]:
    """Resolve chains A→B, B→C into A→C. Mutates and returns alias_map."""
    for _ in range(max_iter):
        changed = False
        for variant, canonical in list(alias_map.items()):
            if canonical in alias_map:
                alias_map[variant] = alias_map[canonical]
                changed = True
        if not changed:
            break
    return alias_map


def detect_circular(alias_map: dict[str, str]) -> None:
    """Raise ValueError if any variant maps to itself after transitive resolution."""
    bad = [v for v, c in alias_map.items() if v == c]
    if bad:
        raise ValueError(f"Circular alias detected: {bad[:5]}")


# ---------------------------------------------------------------------------
# §5  Embedding & cache
# ---------------------------------------------------------------------------

def _compute_cache_key(skill_strings: list[str], model_name: str) -> str:
    """sha256 of sorted skill strings + model name."""
    payload = json.dumps(sorted(skill_strings) + [model_name], ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_or_compute_embeddings(
    skill_strings: list[str],
    model_name: str,
    cache_dir: Path,
) -> np.ndarray:
    """Load cached embeddings if key matches; otherwise compute and cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    emb_path = cache_dir / "embeddings.npy"
    key_path = cache_dir / "embeddings_key.txt"

    current_key = _compute_cache_key(skill_strings, model_name)

    if emb_path.exists() and key_path.exists():
        cached_key = key_path.read_text(encoding="utf-8").strip()
        if cached_key == current_key:
            logger.info("Loading cached embeddings from %s", emb_path)
            return np.load(str(emb_path))
        else:
            logger.info("Cache key mismatch — recomputing embeddings.")

    logger.info("Loading model: %s", model_name)
    model = SentenceTransformer(model_name)
    logger.info("Encoding %d skill strings (batch_size=64)...", len(skill_strings))
    embeddings = model.encode(
        skill_strings,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=64,
    )
    np.save(str(emb_path), embeddings)
    key_path.write_text(current_key, encoding="utf-8")
    logger.info("Embeddings saved to %s", emb_path)
    return embeddings


# ---------------------------------------------------------------------------
# §7  Suggest action for cross-category clusters
# ---------------------------------------------------------------------------

def suggest_action(total_count: int, category_distribution: dict[str, int]) -> str:
    if total_count < 5:
        return "IGNORE"
    sorted_counts = sorted(category_distribution.values(), reverse=True)
    dominant_pct = sorted_counts[0] / total_count
    if dominant_pct >= 0.90:
        return "MERGE"
    elif dominant_pct >= 0.70:
        return "MERGE_REVIEW"
    else:
        return "REVIEW_SPLIT"


# ---------------------------------------------------------------------------
# §8  Canonical name selection
# ---------------------------------------------------------------------------

def _is_ascii_only(s: str) -> bool:
    try:
        s.encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def pick_canonical(members: list[dict]) -> str:
    """
    Select canonical name from a list of member dicts with keys:
    skill_normalized, total_count.
    Priority: ASCII members → highest count → shorter name → alphabetical.
    """
    ascii_pool = [m for m in members if _is_ascii_only(m["skill_normalized"])]
    pool = ascii_pool if ascii_pool else members

    best = sorted(
        pool,
        key=lambda m: (-m["total_count"], len(m["skill_normalized"]), m["skill_normalized"]),
    )[0]
    return best["skill_normalized"]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    input_path: Path,
    output_dir: Path,
    cache_dir: Path,
    distance_threshold: float,
    min_keep_count: int,
    apply_overrides_path: Path | None,
    seed: int,
    job_id_col: str = "job_id",
    short_threshold: int = 5,
    aliases_path: Path | None = None,
    save_aliased_path: Path | None = None,
    role_block_path: Path | None = None,
) -> None:
    t_start = time.time()

    # Fix seeds
    random.seed(seed)
    np.random.seed(seed)

    output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # §2  Load & filter input
    # -----------------------------------------------------------------------
    logger.info("Reading input: %s", input_path)
    df_raw = pd.read_csv(input_path, dtype=str)

    required_cols = REQUIRED_COLUMNS_BASE | {job_id_col}
    missing_cols = required_cols - set(df_raw.columns)
    if missing_cols:
        raise ValueError(f"Input CSV missing required columns: {missing_cols}")
    # Normalize job_id column name to 'job_id' internally
    if job_id_col != "job_id":
        df_raw = df_raw.rename(columns={job_id_col: "job_id"})

    total_input_rows = len(df_raw)

    # Track dropped rows
    drop_reasons: list[tuple[int, str]] = []

    mask_skill_null = df_raw["skill_name"].isna() | (df_raw["skill_name"].str.strip() == "")
    mask_cat_null = df_raw["category"].isna()

    n_drop_skill = mask_skill_null.sum()
    n_drop_cat = (~mask_skill_null & mask_cat_null).sum()

    drop_reasons.append((int(n_drop_skill), "skill_name null or empty"))
    drop_reasons.append((int(n_drop_cat), "category null"))

    df = df_raw[~mask_skill_null & ~mask_cat_null].copy()
    df["skill_name"] = df["skill_name"].str.strip()
    df["category"] = df["category"].str.strip()

    # -----------------------------------------------------------------------
    # §2.3  Sanity check input
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("INPUT SANITY CHECK")
    print("=" * 60)
    print(f"Total rows read       : {total_input_rows}")
    print(f"Dropped (skill null)  : {n_drop_skill}")
    print(f"Dropped (category null): {n_drop_cat}")
    print(f"Rows after filter     : {len(df)}")
    print(f"Unique job_ids        : {df['job_id'].nunique()}")
    print(f"Unique skill_names    : {df['skill_name'].nunique()}")
    print("\nCategory distribution (mention count | unique skills):")
    for cat, grp in df.groupby("category", sort=False):
        print(f"  {cat:<40} {len(grp):>6} mentions | {grp['skill_name'].nunique():>5} unique skills")
    print("=" * 60 + "\n")

    # -----------------------------------------------------------------------
    # §3  Pre-normalization
    # -----------------------------------------------------------------------
    logger.info("Step 1/6 — Pre-normalization (%d rows)...", len(df))
    tqdm.pandas(desc="  normalize_skill", leave=False)
    df["skill_normalized"] = df["skill_name"].progress_apply(normalize_skill)

    # -----------------------------------------------------------------------
    # §3.5  Apply aliases (optional)
    # -----------------------------------------------------------------------
    alias_stats: dict[str, Any] | None = None
    if aliases_path is not None:
        logger.info("Loading aliases from %s", aliases_path)
        alias_map = load_alias_map(aliases_path)
        alias_map_norm = {
            normalize_skill(k): normalize_skill(v)
            for k, v in alias_map.items()
        }
        alias_map_norm = resolve_transitive(alias_map_norm)
        detect_circular(alias_map_norm)

        n_groups = 0
        n_variants = len(alias_map_norm)
        with aliases_path.open(encoding="utf-8") as _f:
            _d = yaml.safe_load(_f)
            n_groups = len(_d.get("aliases", []))

        distinct_before = df["skill_normalized"].nunique()
        df["skill_aliased"] = df["skill_normalized"].map(
            lambda s: alias_map_norm.get(s, s)
        )
        rows_changed = int((df["skill_aliased"] != df["skill_normalized"]).sum())
        distinct_after = df["skill_aliased"].nunique()

        if save_aliased_path is not None:
            save_aliased_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(save_aliased_path, index=False)
            logger.info("Saved aliased dataset → %s", save_aliased_path)

        # Replace skill_normalized so downstream is unchanged
        df["skill_normalized"] = df["skill_aliased"]
        df.drop(columns=["skill_aliased"], inplace=True)

        pct = rows_changed / len(df) * 100 if len(df) else 0.0
        alias_stats = {
            "aliases_file": str(aliases_path),
            "n_alias_groups": n_groups,
            "n_variants_total": n_variants,
            "rows_changed": rows_changed,
            "pct_changed": round(pct, 1),
            "distinct_skills_before": distinct_before,
            "distinct_skills_after": distinct_after,
        }

        print("\n" + "=" * 60)
        print("APPLY ALIASES")
        print("=" * 60)
        print(f"  Loaded aliases.yaml: {n_groups} groups, {n_variants} variants")
        print(f"  Applied to dataset: {len(df)} rows")
        print(f"    - Rows changed   : {rows_changed} ({pct:.1f}%)")
        print(f"    - Rows unchanged : {len(df) - rows_changed}")
        print(f"  Distinct skill values:")
        print(f"    - Before aliases : {distinct_before}")
        print(f"    - After aliases  : {distinct_after} (giảm {distinct_before - distinct_after})")
        print("=" * 60 + "\n")

    # -----------------------------------------------------------------------
    # §4  Aggregation by (skill_normalized, category)
    # -----------------------------------------------------------------------
    _agg_groups = list(df.groupby(["skill_normalized", "category"], sort=False))
    logger.info("Step 2/6 — Aggregation (%d unique skill×category pairs)...", len(_agg_groups))

    agg_records = []
    for (skill_norm, category), grp in tqdm(_agg_groups, desc="  aggregate", unit="skill", leave=False):
        agg_records.append(
            {
                "skill_normalized": skill_norm,
                "category": category,
                "total_count": len(grp),
                "n_required": int((grp["label"] == "required_skill").sum()),
                "n_preferred": int((grp["label"] == "preferred_skill").sum()),
                "raw_variants": grp["skill_name"].tolist(),
                "job_ids": grp["job_id"].tolist(),
            }
        )

    agg_df = pd.DataFrame(agg_records)
    total_mentions_before = int(agg_df["total_count"].sum())
    input_unique_skills = len(agg_df)

    logger.info(
        "Aggregation done: %d unique (skill_normalized, category) pairs, %d total mentions.",
        input_unique_skills,
        total_mentions_before,
    )

    # -----------------------------------------------------------------------
    # §4.5  Short-string bypass — separate before embedding
    # -----------------------------------------------------------------------
    short_mask = agg_df["skill_normalized"].str.len() <= short_threshold
    short_agg_df = agg_df[short_mask].copy().reset_index(drop=True)
    long_agg_df = agg_df[~short_mask].copy().reset_index(drop=True)

    n_short_skills = len(short_agg_df)
    n_long_skills = len(long_agg_df)

    # -----------------------------------------------------------------------
    # §5  Embedding — long pool only
    # -----------------------------------------------------------------------
    long_skill_strings = long_agg_df["skill_normalized"].tolist()
    logger.info("Step 3/6 — Embedding (%d long skills)...", len(long_skill_strings))
    embeddings = load_or_compute_embeddings(long_skill_strings, MODEL_NAME, cache_dir)

    # -----------------------------------------------------------------------
    # §6  Clustering — long pool only
    # -----------------------------------------------------------------------
    logger.info(
        "Step 4/6 — Clustering (threshold=%.3f, metric=cosine, linkage=average)...",
        distance_threshold,
    )
    clusterer = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="cosine",
        linkage="average",
    )
    cluster_labels = clusterer.fit_predict(embeddings)
    long_agg_df["cluster_id"] = cluster_labels
    n_long_clusters = int(cluster_labels.max()) + 1
    logger.info("Clustering done: %d clusters from %d long skills.", n_long_clusters, n_long_skills)

    # -----------------------------------------------------------------------
    # §4.5.4  Short-string exact-match consolidation
    # -----------------------------------------------------------------------
    # Group short strings by skill_normalized (exact match, case-insensitive after normalize).
    # Each unique skill_normalized = 1 cluster. Cross-category handled normally.
    short_cluster_id_start = n_long_clusters
    short_norm_to_cid: dict[str, int] = {}
    next_short_cid = short_cluster_id_start
    short_cluster_ids = []
    for norm in short_agg_df["skill_normalized"]:
        if norm not in short_norm_to_cid:
            short_norm_to_cid[norm] = next_short_cid
            next_short_cid += 1
        short_cluster_ids.append(short_norm_to_cid[norm])
    short_agg_df["cluster_id"] = short_cluster_ids
    n_short_clusters = len(short_norm_to_cid)

    # Console output §4.5.8
    print(f"[Short bypass] {n_short_skills} skills (<={short_threshold} chars) -> {n_short_clusters} exact-match clusters")
    print(f"[Embedding]    {n_long_skills} skills (>{short_threshold} chars) -> {n_long_clusters} semantic clusters")
    print(f"[Total]        {n_long_clusters + n_short_clusters} clusters")

    # Recombine
    agg_df = pd.concat([long_agg_df, short_agg_df], ignore_index=True)
    n_clusters = n_long_clusters + n_short_clusters
    logger.info("Total clusters: %d (%d semantic + %d exact-match short).", n_clusters, n_long_clusters, n_short_clusters)

    # -----------------------------------------------------------------------
    # Build cluster-level summary
    # -----------------------------------------------------------------------
    logger.info("Step 5/6 — Building cluster summaries (%d clusters)...", n_clusters)
    cluster_rows = []
    for cid, grp in tqdm(agg_df.groupby("cluster_id", sort=False), total=n_clusters, desc="  build clusters", unit="cluster", leave=False):
        cat_dist: dict[str, int] = {}
        members_meta: list[dict] = []
        all_raw_variants: list[str] = []
        tot_req = 0
        tot_pref = 0

        for _, row in grp.iterrows():
            cat = row["category"]
            cnt = int(row["total_count"])
            cat_dist[cat] = cat_dist.get(cat, 0) + cnt
            tot_req += int(row["n_required"])
            tot_pref += int(row["n_preferred"])
            members_meta.append(
                {"skill_normalized": row["skill_normalized"], "total_count": cnt, "category": cat}
            )
            all_raw_variants.extend(row["raw_variants"])

        total_count = sum(cat_dist.values())
        dominant_category = max(cat_dist, key=cat_dist.__getitem__)
        dominant_pct = cat_dist[dominant_category] / total_count if total_count > 0 else 1.0
        is_cross = len(cat_dist) > 1

        auto_canonical = pick_canonical(members_meta)
        n_members = len(members_meta)

        if is_cross:
            action = suggest_action(total_count, cat_dist)
        else:
            action = "N/A"

        # Long-tail bucket §9
        applied_other = n_members == 1 and total_count < min_keep_count
        if applied_other:
            final_canonical_auto = f"{dominant_category} Other"
        else:
            final_canonical_auto = auto_canonical

        cluster_rows.append(
            {
                "cluster_id": int(cid),
                "auto_canonical": auto_canonical,
                "override_canonical": "",
                "final_canonical": final_canonical_auto,
                "n_members": n_members,
                "total_count": total_count,
                "n_required": tot_req,
                "n_preferred": tot_pref,
                "dominant_category": dominant_category,
                "dominant_category_pct": round(dominant_pct, 4),
                "is_cross_category": is_cross,
                "categories_involved": ", ".join(sorted(cat_dist.keys())),
                "category_distribution": json.dumps(cat_dist, ensure_ascii=False),
                "suggested_action": action,
                "all_members": "|".join(m["skill_normalized"] for m in members_meta),
                "all_members_raw": "|".join(all_raw_variants),
                "applied_other_bucket": applied_other,
                # internal helpers
                "_members_meta": members_meta,
                "_cat_dist": cat_dist,
            }
        )

    clusters_df = pd.DataFrame(cluster_rows)

    # -----------------------------------------------------------------------
    # §11.5 Warn on auto_canonical collision
    # -----------------------------------------------------------------------
    canonical_counts = clusters_df["auto_canonical"].value_counts()
    collisions = canonical_counts[canonical_counts > 1]
    if len(collisions) > 0:
        logger.warning(
            "auto_canonical collision detected for %d names (same name in ≥2 clusters): %s",
            len(collisions),
            collisions.index.tolist()[:20],
        )

    # -----------------------------------------------------------------------
    # §-- Apply overrides if requested
    # -----------------------------------------------------------------------
    if apply_overrides_path is not None:
        logger.info("Applying overrides from %s", apply_overrides_path)
        overrides_df = pd.read_csv(apply_overrides_path, dtype=str).fillna("")
        # Build override lookup: (raw_skill_name, category) → override_canonical
        override_map: dict[tuple[str, str], str] = {}
        for _, row in overrides_df.iterrows():
            ov = str(row.get("override_canonical", "")).strip()
            if ov:
                override_map[(str(row["raw_skill_name"]).strip(), str(row["category"]).strip())] = ov

        # Propagate overrides: for each (skill_normalized, category) check its raw_variants
        # Build a per-cluster override: cluster_id → set of override canonicals
        cluster_overrides: dict[int, set[str]] = {}
        for _, agg_row in agg_df.iterrows():
            cid = int(agg_row["cluster_id"])
            cat = str(agg_row["category"]).strip()
            for raw_v in agg_row["raw_variants"]:
                key = (str(raw_v).strip(), cat)
                if key in override_map:
                    cluster_overrides.setdefault(cid, set()).add(override_map[key])

        def _resolve_final(cluster_row: pd.Series) -> str:
            cid = int(cluster_row["cluster_id"])
            ovs = cluster_overrides.get(cid, set())
            if len(ovs) == 1:
                return ovs.pop()
            elif len(ovs) > 1:
                logger.warning(
                    "Cluster %d has conflicting overrides: %s — keeping auto_canonical.",
                    cid,
                    ovs,
                )
                return cluster_row["auto_canonical"]
            return cluster_row["final_canonical"]

        clusters_df["override_canonical"] = clusters_df["cluster_id"].map(
            lambda cid: "|".join(cluster_overrides.get(int(cid), set())) if cluster_overrides.get(int(cid)) else ""
        )
        clusters_df["final_canonical"] = clusters_df.apply(_resolve_final, axis=1)
    # end override block

    # -----------------------------------------------------------------------
    # Build cluster_id → final_canonical lookup
    # -----------------------------------------------------------------------
    cid_to_final: dict[int, str] = dict(
        zip(clusters_df["cluster_id"], clusters_df["final_canonical"])
    )
    cid_to_auto: dict[int, str] = dict(
        zip(clusters_df["cluster_id"], clusters_df["auto_canonical"])
    )
    cid_to_override: dict[int, str] = dict(
        zip(clusters_df["cluster_id"], clusters_df["override_canonical"])
    )

    # -----------------------------------------------------------------------
    # Build canonical_mapping (§10.2) — one row per (raw_skill_name, category)
    # -----------------------------------------------------------------------
    logger.info("Step 6/6 — Building canonical mapping (%d skill×cat pairs)...", len(agg_df))
    mapping_records = []
    for _, agg_row in tqdm(agg_df.iterrows(), total=len(agg_df), desc="  build mapping", unit="skill", leave=False):
        cid = int(agg_row["cluster_id"])
        for raw_v in agg_row["raw_variants"]:
            raw_v = str(raw_v)
            # raw_count = count of this specific raw_skill_name within this (norm, cat)
            raw_count = agg_row["raw_variants"].count(raw_v)
            mapping_records.append(
                {
                    "raw_skill_name": raw_v,
                    "category": agg_row["category"],
                    "skill_normalized": agg_row["skill_normalized"],
                    "cluster_id": cid,
                    "auto_canonical": cid_to_auto[cid],
                    "override_canonical": cid_to_override.get(cid, ""),
                    "final_canonical": cid_to_final[cid],
                    "raw_count": raw_count,
                }
            )

    mapping_df = pd.DataFrame(mapping_records)
    # Deduplicate (raw_skill_name, category) — aggregate raw_count
    mapping_df = (
        mapping_df.groupby(["raw_skill_name", "category"], sort=False)
        .agg(
            skill_normalized=("skill_normalized", "first"),
            cluster_id=("cluster_id", "first"),
            auto_canonical=("auto_canonical", "first"),
            override_canonical=("override_canonical", "first"),
            final_canonical=("final_canonical", "first"),
            raw_count=("raw_count", "sum"),
        )
        .reset_index()
    )

    # -----------------------------------------------------------------------
    # §11  Sanity checks
    # -----------------------------------------------------------------------
    logger.info("Running sanity checks...")

    # 1. Mention conservation
    total_mentions_after = int(clusters_df["total_count"].sum())
    assert total_mentions_before == total_mentions_after, (
        f"SANITY FAIL: mention count changed! Before={total_mentions_before}, After={total_mentions_after}"
    )
    logger.info("[OK]Mention conservation: %d mentions preserved.", total_mentions_before)

    # 2. Mapping completeness: each (raw_skill_name, category) has exactly 1 row
    input_pairs = set(zip(df["skill_name"].str.strip(), df["category"].str.strip()))
    mapping_pairs = set(zip(mapping_df["raw_skill_name"].str.strip(), mapping_df["category"].str.strip()))
    missing_pairs = input_pairs - mapping_pairs
    extra_pairs = mapping_pairs - input_pairs
    assert not missing_pairs, (
        f"SANITY FAIL: {len(missing_pairs)} (raw_skill_name, category) pairs missing from mapping."
    )
    assert not extra_pairs, (
        f"SANITY FAIL: {len(extra_pairs)} unexpected pairs in mapping."
    )
    logger.info("[OK]Mapping completeness: all %d (raw_skill, category) pairs mapped.", len(mapping_pairs))

    # 3. No final_canonical null or empty
    bad_canonical = clusters_df[
        clusters_df["final_canonical"].isna() | (clusters_df["final_canonical"].str.strip() == "")
    ]
    assert len(bad_canonical) == 0, (
        f"SANITY FAIL: {len(bad_canonical)} clusters have null/empty final_canonical."
    )
    logger.info("[OK]No null final_canonical.")

    # 4. Category preservation (checked when building annotations_with_canonical)

    # -----------------------------------------------------------------------
    # §10.5  annotations_with_canonical.csv
    # -----------------------------------------------------------------------
    logger.info("Building annotations_with_canonical...")
    df_out = df.copy()
    df_out["skill_normalized"] = df["skill_normalized"]

    # Map final_canonical by joining on (skill_name, category)
    merge_key = mapping_df[["raw_skill_name", "category", "final_canonical", "skill_normalized"]].rename(
        columns={"raw_skill_name": "skill_name", "skill_normalized": "_skill_norm_check"}
    )
    df_out = df_out.merge(merge_key, on=["skill_name", "category"], how="left")
    # Prefer the skill_normalized we already computed over the one from mapping_df
    # (they should be identical, but keep our computed one as authoritative)
    df_out["skill_normalized"] = df_out["skill_normalized"].fillna(df_out["_skill_norm_check"])
    df_out.drop(columns=["_skill_norm_check"], inplace=True, errors="ignore")

    # Sanity check §11.4: category preserved
    assert (df_out["category"] == df["category"].values).all(), (
        "SANITY FAIL: category column changed after join!"
    )
    logger.info("[OK]Category preservation confirmed.")

    # -----------------------------------------------------------------------
    # Drop internal columns from clusters_df before saving
    # -----------------------------------------------------------------------
    clusters_df_out = clusters_df.drop(columns=["_members_meta", "_cat_dist"], errors="ignore")

    # Sort clusters by total_count desc
    clusters_df_out = clusters_df_out.sort_values("total_count", ascending=False).reset_index(drop=True)

    # -----------------------------------------------------------------------
    # §-- Role-block filter — drop clusters whose final_canonical is blocked
    # -----------------------------------------------------------------------
    role_block_stats: dict[str, Any] | None = None
    if role_block_path is not None:
        role_block = load_role_block(role_block_path)
        logger.info("Loaded role_block: %d entries from %s", len(role_block), role_block_path)

        blocked_mask = clusters_df_out["final_canonical"].apply(normalize_skill).isin(role_block)
        blocked_cids: set[int] = set(clusters_df_out.loc[blocked_mask, "cluster_id"])
        n_blocked_clusters = int(blocked_mask.sum())
        n_blocked_mentions = int(clusters_df_out.loc[blocked_mask, "total_count"].sum())

        clusters_df_out = clusters_df_out[~blocked_mask].reset_index(drop=True)
        mapping_df = mapping_df[~mapping_df["cluster_id"].isin(blocked_cids)].reset_index(drop=True)
        df_out = df_out[
            ~df_out["final_canonical"].fillna("").apply(normalize_skill).isin(role_block)
        ].reset_index(drop=True)

        role_block_stats = {
            "role_block_file": str(role_block_path),
            "n_blocked_entries": len(role_block),
            "n_blocked_clusters": n_blocked_clusters,
            "n_blocked_mentions": n_blocked_mentions,
        }

        print("\n" + "=" * 60)
        print("ROLE-BLOCK FILTER")
        print("=" * 60)
        print(f"  Loaded role_block : {len(role_block)} entries")
        print(f"  Dropped clusters  : {n_blocked_clusters}")
        print(f"  Dropped mentions  : {n_blocked_mentions}")
        print("=" * 60 + "\n")
        logger.info(
            "Role-block filter: %d clusters dropped (%d mentions).",
            n_blocked_clusters, n_blocked_mentions,
        )

    # -----------------------------------------------------------------------
    # §10.3  skill_distribution_after.csv
    # -----------------------------------------------------------------------
    logger.info("Building skill_distribution_after...")
    dist_records = []
    for final_can, grp in clusters_df_out.groupby("final_canonical", sort=False):
        total_cnt = int(grp["total_count"].sum())
        n_req = int(grp["n_required"].sum())
        n_pref = int(grp["n_preferred"].sum())
        dominant_cat = grp.loc[grp["total_count"].idxmax(), "dominant_category"]

        # Collect raw variants for sample (up to 5)
        all_raw = []
        for raw_str in grp["all_members_raw"]:
            all_raw.extend([v for v in raw_str.split("|") if v])

        n_raw_variants = len(set(all_raw))
        sample_vars = "|".join(list(dict.fromkeys(all_raw))[:5])

        dist_records.append(
            {
                "final_canonical": final_can,
                "dominant_category": dominant_cat,
                "total_count": total_cnt,
                "n_raw_variants": n_raw_variants,
                "n_required": n_req,
                "n_preferred": n_pref,
                "sample_variants": sample_vars,
            }
        )

    dist_df = pd.DataFrame(dist_records).sort_values("total_count", ascending=False).reset_index(drop=True)

    # -----------------------------------------------------------------------
    # §10.4  cross_category_review.csv
    # -----------------------------------------------------------------------
    cross_df = clusters_df_out[clusters_df_out["is_cross_category"] == True].copy()  # noqa: E712
    cross_df["_action_order"] = cross_df["suggested_action"].map(ACTION_ORDER).fillna(99)
    cross_df = cross_df.sort_values(
        ["_action_order", "total_count"], ascending=[True, False]
    ).drop(columns=["_action_order"])
    cross_df = cross_df.reset_index(drop=True)

    # -----------------------------------------------------------------------
    # §7.6  Cross-category summary for console & report
    # -----------------------------------------------------------------------
    action_breakdown: dict[str, int] = {"MERGE": 0, "MERGE_REVIEW": 0, "REVIEW_SPLIT": 0, "IGNORE": 0}
    for _, row in cross_df.iterrows():
        act = str(row["suggested_action"])
        if act in action_breakdown:
            action_breakdown[act] += 1

    n_review_split = action_breakdown["REVIEW_SPLIT"]
    n_merge_review = action_breakdown["MERGE_REVIEW"]
    cross_warning = ""
    if n_review_split > 0:
        cross_warning = (
            f"{n_review_split} cluster(s) marked REVIEW_SPLIT — "
            "manual review recommended. See cross_category_review.csv."
        )
        logger.warning(cross_warning)
    if n_merge_review > 0:
        logger.warning(
            "%d cluster(s) marked MERGE_REVIEW — review recommended.", n_merge_review
        )

    cross_category_summary: dict[str, Any] = {
        "total_cross_category_clusters": len(cross_df),
        "action_breakdown": action_breakdown,
        "warning": cross_warning,
    }

    # -----------------------------------------------------------------------
    # Console cross-category summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("CROSS-CATEGORY SUMMARY")
    print("=" * 60)
    print(f"[Cross-category] {len(cross_df)} cross-cat clusters detected")
    print(f"  ->{action_breakdown['MERGE']} auto-merged (dominant category >= 90%)")
    print(f"  ->{action_breakdown['MERGE_REVIEW']} auto-merged with warning (dominant 70-90%)")
    print(f"  ->{action_breakdown['REVIEW_SPLIT']} require manual review (see cross_category_review.csv)")
    print("=" * 60 + "\n")

    # -----------------------------------------------------------------------
    # §10.6  clustering_report.json
    # -----------------------------------------------------------------------
    runtime = round(time.time() - t_start, 2)
    config_hash = hashlib.sha256(
        json.dumps(
            {
                "model": MODEL_NAME,
                "distance_threshold": distance_threshold,
                "min_keep_count": min_keep_count,
                "seed": seed,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()

    report: dict[str, Any] = {
        "matching_version": MATCHING_VERSION,
        "model": MODEL_NAME,
        "distance_threshold": distance_threshold,
        "linkage": "average",
        "min_keep_count": min_keep_count,
        "short_string_bypass": {
            "threshold": short_threshold,
            "n_short_skills_separated": n_short_skills,
            "n_short_clusters_created": n_short_clusters,
            "n_long_skills_clustered": n_long_skills,
            "rationale": f"Short strings (<={short_threshold} chars) routed to exact-match consolidation to avoid catch-all clusters from noisy embeddings.",
        },
        "input_rows": int(len(df)),
        "input_unique_skills": int(input_unique_skills),
        "n_clusters": int(n_clusters),
        "n_clusters_cross_category": int(len(cross_df)),
        "cross_category_summary": cross_category_summary,
        "n_clusters_in_other_bucket": int(clusters_df_out["applied_other_bucket"].sum()),
        "skills_after_canonical": int(dist_df["final_canonical"].nunique()),
        "total_mentions_preserved": int(total_mentions_before),
        "runtime_seconds": runtime,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config_hash": f"sha256-{config_hash[:16]}",
        "alias_application": alias_stats,
        "role_block": role_block_stats,
    }

    # -----------------------------------------------------------------------
    # Write outputs
    # -----------------------------------------------------------------------
    logger.info("Writing output files to %s...", output_dir)

    # 1. clusters_review.csv
    out_cols_clusters = [
        "cluster_id", "auto_canonical", "override_canonical", "final_canonical",
        "n_members", "total_count", "n_required", "n_preferred",
        "dominant_category", "dominant_category_pct", "is_cross_category",
        "categories_involved", "category_distribution", "suggested_action",
        "all_members", "all_members_raw", "applied_other_bucket",
    ]
    clusters_df_out[out_cols_clusters].to_csv(output_dir / "clusters_review.csv", index=False)
    logger.info("Wrote clusters_review.csv (%d rows)", len(clusters_df_out))

    # 2. canonical_mapping.csv
    out_cols_mapping = [
        "raw_skill_name", "category", "skill_normalized",
        "cluster_id", "auto_canonical", "override_canonical", "final_canonical", "raw_count",
    ]
    mapping_df[out_cols_mapping].to_csv(output_dir / "canonical_mapping.csv", index=False)
    logger.info("Wrote canonical_mapping.csv (%d rows)", len(mapping_df))

    # 3. skill_distribution_after.csv
    dist_df.to_csv(output_dir / "skill_distribution_after.csv", index=False)
    logger.info("Wrote skill_distribution_after.csv (%d rows)", len(dist_df))

    # 4. cross_category_review.csv
    cross_out_cols = [c for c in out_cols_clusters if c in cross_df.columns]
    cross_df[cross_out_cols].to_csv(output_dir / "cross_category_review.csv", index=False)
    logger.info("Wrote cross_category_review.csv (%d rows)", len(cross_df))

    # 5. annotations_with_canonical.csv
    df_out.to_csv(output_dir / "annotations_with_canonical.csv", index=False)
    logger.info("Wrote annotations_with_canonical.csv (%d rows)", len(df_out))

    # 6. clustering_report.json
    report_path = output_dir / "clustering_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote clustering_report.json")

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("RUN COMPLETE")
    print("=" * 60)
    print(f"Input rows            : {len(df)}")
    print(f"Unique (norm, cat)    : {input_unique_skills}")
    print(f"Clusters formed       : {n_clusters}")
    print(f"Cross-cat clusters    : {len(cross_df)}")
    print(f"Other-bucket clusters : {int(clusters_df_out['applied_other_bucket'].sum())}")
    print(f"Canonical skills      : {dist_df['final_canonical'].nunique()}")
    print(f"Mentions preserved    : {total_mentions_before}")
    print(f"Runtime               : {runtime}s")
    print(f"Outputs in            : {output_dir.resolve()}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DS108 Skill Canonicalization — multilingual embedding + agglomerative clustering",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/skills_raw.csv"),
        help="Path to input CSV file with skill mentions.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/"),
        help="Directory for output files.",
    )
    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=0.15,
        help="Cosine distance threshold for agglomerative clustering.",
    )
    parser.add_argument(
        "--min-keep-count",
        type=int,
        default=2,
        help="Minimum mention count to keep a singleton cluster (else → Other bucket).",
    )
    parser.add_argument(
        "--apply-overrides",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to canonical_mapping.csv with override_canonical filled in by user.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(__file__).parent / "cache",
        help="Directory to store/load cached embeddings.",
    )
    parser.add_argument(
        "--short-threshold",
        type=int,
        default=5,
        help="Skills with normalized length <= this value bypass embedding and use exact-match clustering (see spec §4.5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--job-id-col",
        type=str,
        default="row_id",
        help="Column name to use as job identifier (e.g. 'row_id' if input lacks 'job_id').",
    )
    parser.add_argument(
        "--aliases",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to aliases.yaml (optional). Apply alias normalization before clustering.",
    )
    parser.add_argument(
        "--save-aliased",
        type=Path,
        default=None,
        metavar="PATH",
        help="If set, save intermediate full_parsed_aliased.csv for audit.",
    )
    parser.add_argument(
        "--role-block",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to a text file (one role per line) whose canonical names are dropped from all outputs after clustering.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if not args.input.exists():
        logger.error("Input file not found: %s", args.input)
        sys.exit(1)

    if args.apply_overrides is not None and not args.apply_overrides.exists():
        logger.error("Override file not found: %s", args.apply_overrides)
        sys.exit(1)

    run_pipeline(
        input_path=args.input,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        distance_threshold=args.distance_threshold,
        min_keep_count=args.min_keep_count,
        apply_overrides_path=args.apply_overrides,
        seed=args.seed,
        job_id_col=args.job_id_col,
        short_threshold=args.short_threshold,
        aliases_path=args.aliases,
        save_aliased_path=args.save_aliased,
        role_block_path=args.role_block,
    )


if __name__ == "__main__":
    main()
