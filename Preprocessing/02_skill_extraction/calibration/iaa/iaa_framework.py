"""IAA Framework — main orchestration and CLI entry point.

Usage (unified JSONL):
    python -m iaa.iaa_framework --input annotations.jsonl --output-dir ./iaa_results

Usage (separate files — current project structure):
    python -m iaa.iaa_framework \\
        --human-a annotated_skills_cnguyen.json \\
        --human-b annotated_skills_human_b.json \\
        --llm output/few_shot_parsed.csv \\
        --output-dir ./iaa_results

If --human-b is omitted, computes only llm_vs_human_a (no Fleiss' Kappa).
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

# Allow running as a script directly (python iaa/iaa_framework.py ...)
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    __package__ = "iaa"

from .bootstrap import bootstrap_f1_ci, bootstrap_kappa_ci, bootstrap_fleiss_ci
from .export import collect_errors, export_error_cases
from .loader import load_annotations, group_by_record_id
from .matching import greedy_bipartite_match, three_way_match
from .metrics import (
    compute_macro_f1,
    cohen_kappa_label,
    cohen_kappa_category,
    fleiss_kappa,
)
from .visualization import (
    plot_label_confusion,
    plot_category_confusion,
    plot_error_distribution,
)

# ─── Acceptance thresholds (from codebook IAA protocol) ───────────────────────

T1_THRESHOLDS: dict[str, float] = {
    "llm_vs_human_a": 0.82,
    "llm_vs_human_b": 0.82,
    "human_a_vs_human_b": 0.85,
}
T2_THRESHOLDS: dict[str, float] = {
    "llm_vs_human_a": 0.70,
    "llm_vs_human_b": 0.70,
    "human_a_vs_human_b": 0.75,
    "fleiss_kappa_all": 0.70,
}
T3_THRESHOLDS: dict[str, float] = {
    "llm_vs_human_a": 0.60,
    "human_a_vs_human_b": 0.65,
}
BOOTSTRAP_CI_LOWER_MIN = 0.60  # applies to all kappa metrics


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _round(v: float, n: int = 4) -> float:
    return round(v, n) if not math.isnan(v) else v


def _safe_ci(ci: tuple[float, float]) -> list[float]:
    return [_round(ci[0]), _round(ci[1])]


def _kappa_status(kappa: float, ci: tuple[float, float], threshold: float) -> str:
    if math.isnan(kappa):
        return "N/A"
    ci_lower_ok = math.isnan(ci[0]) or ci[0] >= BOOTSTRAP_CI_LOWER_MIN
    return "PASS" if kappa >= threshold and ci_lower_ok else "FAIL"


def _f1_status(f1: float, threshold: float) -> str:
    if math.isnan(f1):
        return "N/A"
    return "PASS" if f1 >= threshold else "FAIL"


# ─── Core pipeline ────────────────────────────────────────────────────────────

def run_iaa(
    input_path: str | Path | None = None,
    output_dir: str | Path = "./iaa_results",
    human_a_path: str | Path | None = None,
    human_b_path: str | Path | None = None,
    llm_path: str | Path | None = None,
    codebook_version: str = "v1.3",
) -> dict:
    """Run the full IAA pipeline and write results to output_dir.

    Returns the iaa_report dict.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Load and group annotations
    annotations = load_annotations(input_path, human_a_path, human_b_path, llm_path)
    records_map = group_by_record_id(annotations)
    record_ids = sorted(records_map.keys())
    n_records = len(record_ids)

    has_human_b = "human_b" in annotations
    has_llm = "llm" in annotations

    # Determine which annotator pairs to compute
    pairs: list[tuple[str, str]] = []
    if has_llm and "human_a" in annotations:
        pairs.append(("llm", "human_a"))
    if has_llm and has_human_b:
        pairs.append(("llm", "human_b"))
    if "human_a" in annotations and has_human_b:
        pairs.append(("human_a", "human_b"))

    t1_results: dict[str, dict] = {}
    t2_results: dict[str, dict] = {}
    t3_results: dict[str, dict] = {}
    all_errors: list[dict] = []
    pair_matched_data: dict[str, list[tuple]] = {}

    # 2. Per-pair computation
    for ann_a, ann_b in pairs:
        pair_key = f"{ann_a}_vs_{ann_b}"

        record_data: list[dict] = []
        for rid in record_ids:
            sa = records_map[rid].get(ann_a, [])
            sb = records_map[rid].get(ann_b, [])
            matched, only_a, only_b = greedy_bipartite_match(sa, sb)
            record_data.append(
                {
                    "record_id": rid,
                    "skills_a": sa,
                    "skills_b": sb,
                    "matched": matched,
                    "only_a": only_a,
                    "only_b": only_b,
                }
            )

        all_matched = [m for r in record_data for m in r["matched"]]
        pair_matched_data[pair_key] = all_matched

        # T1: Macro F1
        f1_res = compute_macro_f1(record_data)
        ci_f1 = bootstrap_f1_ci(f1_res["per_record_stats"])
        f1_thresh = T1_THRESHOLDS.get(pair_key, 0.82)
        t1_results[pair_key] = {
            "precision": _round(f1_res["precision"]),
            "recall": _round(f1_res["recall"]),
            "f1": _round(f1_res["f1"]),
            "ci_95": _safe_ci(ci_f1),
            "status": _f1_status(f1_res["f1"], f1_thresh),
        }

        # T2: Cohen's Kappa — label
        if len(all_matched) >= 2:
            k_label = cohen_kappa_label(all_matched)
            ci_label = bootstrap_kappa_ci(all_matched, cohen_kappa_label)
            k_thresh = T2_THRESHOLDS.get(pair_key, 0.70)
            t2_results[pair_key] = {
                "kappa": _round(k_label),
                "ci_95": _safe_ci(ci_label),
                "status": _kappa_status(k_label, ci_label, k_thresh),
            }

        # T3: Cohen's Kappa — category (only for selected pairs)
        if pair_key in T3_THRESHOLDS and len(all_matched) >= 2:
            k_cat = cohen_kappa_category(all_matched)
            ci_cat = bootstrap_kappa_ci(all_matched, cohen_kappa_category)
            k_cat_thresh = T3_THRESHOLDS[pair_key]
            t3_results[pair_key] = {
                "kappa": _round(k_cat),
                "ci_95": _safe_ci(ci_cat),
                "status": _kappa_status(k_cat, ci_cat, k_cat_thresh),
            }

        # Errors
        errors = collect_errors(record_data, ann_a, ann_b)
        all_errors.extend(errors)

    # 3. Fleiss' Kappa (requires all 3 annotators)
    if "human_a" in annotations and has_human_b and has_llm:
        all_triples = []
        for rid in record_ids:
            sa = records_map[rid].get("human_a", [])
            sb = records_map[rid].get("human_b", [])
            sl = records_map[rid].get("llm", [])
            all_triples.extend(three_way_match(sa, sb, sl))

        if len(all_triples) >= 2:
            fk = fleiss_kappa(all_triples, "label")
            ci_fk = bootstrap_fleiss_ci(all_triples, "label")
            t2_results["fleiss_kappa_all"] = {
                "kappa": _round(fk),
                "ci_95": _safe_ci(ci_fk),
                "status": _kappa_status(fk, ci_fk, T2_THRESHOLDS["fleiss_kappa_all"]),
            }

    # 4. Visualizations
    primary_pair = "llm_vs_human_a"
    primary_matched = pair_matched_data.get(primary_pair, [])

    if primary_matched:
        plot_label_confusion(
            primary_matched, out / "label_confusion_llm_vs_a.png"
        )
        plot_category_confusion(
            primary_matched, out / "category_confusion_llm_vs_a.png"
        )

    if all_errors:
        plot_error_distribution(all_errors, out / "error_distribution.png")
        export_error_cases(all_errors, out / "error_cases.csv")

    # 5. Build report
    all_results = (
        list(t1_results.values())
        + list(t2_results.values())
        + list(t3_results.values())
    )
    valid_results = [r for r in all_results if r.get("status") not in (None, "N/A")]
    overall_pass = bool(valid_results) and all(
        r["status"] == "PASS" for r in valid_results
    )

    report = {
        "config": {
            "n_records": n_records,
            "bootstrap_iter": 1000,
            "bootstrap_seed": 42,
            "fuzzy_threshold": 0.80,
            "codebook_version": codebook_version,
            "matching_version": "v2_token_jaccard_hybrid",
        },
        "t1_extraction": t1_results,
        "t2_label": t2_results,
        "t3_category": t3_results,
        "overall_status": "PASS" if overall_pass else "FAIL",
        "n_error_cases": len(all_errors),
    }

    report_path = out / "iaa_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    _print_summary(report)
    print(f"\nResults saved to: {out.resolve()}")
    return report


# ─── Console summary ──────────────────────────────────────────────────────────

def _print_summary(report: dict) -> None:
    def _row(name: str, d: dict) -> str:
        if "f1" in d:
            score = f"F1={d['f1']:.3f}"
        elif "kappa" in d:
            score = f"κ={d['kappa']:.3f}"
        else:
            return ""
        ci = d.get("ci_95", [float("nan"), float("nan")])
        ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if not math.isnan(ci[0]) else "[N/A]"
        status = d.get("status", "?")
        symbol = "✓" if status == "PASS" else ("~" if status == "N/A" else "✗")
        return f"  {symbol} {name:<30} {score}  95%CI {ci_str}  {status}"

    print("\n" + "=" * 65)
    print("IAA REPORT")
    print("=" * 65)

    print("\n[T1] Extraction F1")
    for k, v in report["t1_extraction"].items():
        print(_row(k, v))

    print("\n[T2] Label Kappa")
    for k, v in report["t2_label"].items():
        print(_row(k, v))

    print("\n[T3] Category Kappa")
    for k, v in report["t3_category"].items():
        print(_row(k, v))

    status = report["overall_status"]
    symbol = "PASS ✓" if status == "PASS" else "FAIL ✗"
    print(f"\nOverall: {symbol}  |  Error cases: {report['n_error_cases']}")
    print("=" * 65)


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="DS108 IAA Framework — compute Inter-Annotator Agreement metrics"
    )
    parser.add_argument(
        "--input",
        help="Unified JSONL annotation file (one {record_id, annotator, skills} per line)",
    )
    parser.add_argument(
        "--human-a",
        help="Human annotator A file (.json array or unified .jsonl)",
    )
    parser.add_argument(
        "--human-b",
        help="Human annotator B file (.json array or unified .jsonl) [optional]",
    )
    parser.add_argument(
        "--llm",
        help="LLM annotation file (.csv from few_shot_parsed or .jsonl)",
    )
    parser.add_argument(
        "--output-dir",
        default="./iaa_results",
        help="Output directory for report and plots (default: ./iaa_results)",
    )
    parser.add_argument(
        "--codebook-version",
        default="v1.3",
        help="Codebook version tag to embed in report (default: v1.3)",
    )

    args = parser.parse_args()

    if not args.input and not args.human_a and not args.llm:
        parser.error("Provide either --input or at least one of --human-a / --llm.")

    run_iaa(
        input_path=args.input,
        output_dir=args.output_dir,
        human_a_path=args.human_a,
        human_b_path=args.human_b,
        llm_path=args.llm,
        codebook_version=args.codebook_version,
    )
