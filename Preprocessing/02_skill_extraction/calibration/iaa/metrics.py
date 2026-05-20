"""Extraction F1, Cohen's Kappa, and Fleiss' Kappa metrics."""

from __future__ import annotations

import math
import numpy as np
from sklearn.metrics import cohen_kappa_score

from .normalization import normalize_category
from .matching import greedy_bipartite_match


# ─── T1: Extraction F1 ────────────────────────────────────────────────────────

def _record_prf(n_matched: int, n_a: int, n_b: int) -> tuple[float, float, float]:
    """Precision, recall, F1 for a single record."""
    p = n_matched / n_b if n_b > 0 else 0.0
    r = n_matched / n_a if n_a > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def compute_macro_f1(record_data: list[dict]) -> dict:
    """Macro-averaged precision, recall, F1 over all records.

    Each element of record_data must have keys: skills_a, skills_b.
    Uses pre-computed 'matched' key when present to skip re-matching.
    Returns dict with keys: precision, recall, f1, per_record_stats
    """
    ps, rs, f1s = [], [], []
    per_record_stats = []  # (n_matched, n_a, n_b) for bootstrap

    for rec in record_data:
        if "matched" in rec:
            matched = rec["matched"]
        else:
            matched, _, _ = greedy_bipartite_match(rec["skills_a"], rec["skills_b"])
        n_m = len(matched)
        n_a = len(rec["skills_a"])
        n_b = len(rec["skills_b"])
        p, r, f1 = _record_prf(n_m, n_a, n_b)
        ps.append(p)
        rs.append(r)
        f1s.append(f1)
        per_record_stats.append((n_m, n_a, n_b))

    return {
        "precision": float(np.mean(ps)),
        "recall": float(np.mean(rs)),
        "f1": float(np.mean(f1s)),
        "per_record_stats": per_record_stats,
    }


# ─── T2: Label Kappa ──────────────────────────────────────────────────────────

def cohen_kappa_label(matched_pairs: list[tuple]) -> float:
    """Cohen's Kappa for binary label agreement on matched pairs."""
    if len(matched_pairs) < 2:
        return float("nan")

    labels_a = [p[0].get("label", "") for p in matched_pairs]
    labels_b = [p[1].get("label", "") for p in matched_pairs]

    if labels_a == labels_b:
        return 1.0
    if len(set(labels_a + labels_b)) < 2:
        return 1.0

    try:
        return float(cohen_kappa_score(labels_a, labels_b))
    except ValueError:
        return float("nan")


def fleiss_kappa(triples: list[tuple], field: str = "label") -> float:
    """Fleiss' Kappa for 3 raters (human_a, human_b, llm) on matched triples.

    Formula (Fleiss 1971):
        P_bar  = [Σ_i Σ_j n_ij² - n·k] / [n·k·(k-1)]
        p_j    = Σ_i n_ij / (n·k)
        P_e    = Σ_j p_j²
        κ      = (P_bar - P_e) / (1 - P_e)
    """
    if len(triples) < 2:
        return float("nan")

    k = 3  # raters

    # Collect all category values
    all_vals: set[str] = set()
    for sa, sb, sl in triples:
        for s in (sa, sb, sl):
            v = s.get(field, "")
            if field == "category":
                v = normalize_category(v)
            all_vals.add(v)

    cats = sorted(all_vals)
    if len(cats) < 2:
        return 1.0

    cat_idx = {c: i for i, c in enumerate(cats)}
    n = len(triples)
    J = len(cats)

    # Build n_ij: n_ij[i][j] = # raters who assigned item i to category j
    n_ij = np.zeros((n, J), dtype=float)
    for i, (sa, sb, sl) in enumerate(triples):
        for s in (sa, sb, sl):
            v = s.get(field, "")
            if field == "category":
                v = normalize_category(v)
            if v in cat_idx:
                n_ij[i, cat_idx[v]] += 1

    P_i = (1.0 / (k * (k - 1))) * (np.sum(n_ij**2, axis=1) - k)
    P_bar = float(np.mean(P_i))

    p_j = np.sum(n_ij, axis=0) / (n * k)
    P_e = float(np.sum(p_j**2))

    if P_e >= 1.0:
        return 1.0

    return float((P_bar - P_e) / (1.0 - P_e))


# ─── T3: Category Kappa ───────────────────────────────────────────────────────

def cohen_kappa_category(matched_pairs: list[tuple]) -> float:
    """Cohen's Kappa for multi-class category agreement on matched pairs."""
    if len(matched_pairs) < 2:
        return float("nan")

    cats_a = [normalize_category(p[0].get("category", "")) for p in matched_pairs]
    cats_b = [normalize_category(p[1].get("category", "")) for p in matched_pairs]

    if cats_a == cats_b:
        return 1.0
    if len(set(cats_a + cats_b)) < 2:
        return 1.0

    try:
        return float(cohen_kappa_score(cats_a, cats_b))
    except ValueError:
        return float("nan")
