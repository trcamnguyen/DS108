"""Bootstrap confidence interval utilities (seed=42, 1000 iterations)."""

from __future__ import annotations

import math
import numpy as np

BOOTSTRAP_ITER = 1000
BOOTSTRAP_SEED = 42


def bootstrap_f1_ci(
    per_record_stats: list[tuple[int, int, int]],
    n_iter: int = BOOTSTRAP_ITER,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """95% CI for macro F1 by resampling records.

    Args:
        per_record_stats: list of (n_matched, n_skills_a, n_skills_b) per record
    """
    if len(per_record_stats) < 2:
        return (float("nan"), float("nan"))

    rng = np.random.default_rng(seed)
    n = len(per_record_stats)
    f1_scores = []

    for _ in range(n_iter):
        indices = rng.choice(n, size=n, replace=True)
        f1s = []
        for idx in indices:
            n_m, n_a, n_b = per_record_stats[idx]
            p = n_m / n_b if n_b > 0 else 0.0
            r = n_m / n_a if n_a > 0 else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            f1s.append(f1)
        f1_scores.append(float(np.mean(f1s)))

    return (
        float(np.percentile(f1_scores, 2.5)),
        float(np.percentile(f1_scores, 97.5)),
    )


def bootstrap_kappa_ci(
    matched_pairs: list,
    kappa_fn,
    n_iter: int = BOOTSTRAP_ITER,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """95% CI for Kappa by resampling matched pairs."""
    if len(matched_pairs) < 2:
        return (float("nan"), float("nan"))

    rng = np.random.default_rng(seed)
    n = len(matched_pairs)
    scores = []

    for _ in range(n_iter):
        indices = rng.choice(n, size=n, replace=True)
        sample = [matched_pairs[idx] for idx in indices]
        try:
            score = kappa_fn(sample)
            if not math.isnan(score):
                scores.append(score)
        except Exception:
            continue

    if len(scores) < 10:
        return (float("nan"), float("nan"))

    return (
        float(np.percentile(scores, 2.5)),
        float(np.percentile(scores, 97.5)),
    )


def bootstrap_fleiss_ci(
    triples: list[tuple],
    field: str = "label",
    n_iter: int = BOOTSTRAP_ITER,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """95% CI for Fleiss' Kappa by resampling triples."""
    from .metrics import fleiss_kappa

    if len(triples) < 2:
        return (float("nan"), float("nan"))

    rng = np.random.default_rng(seed)
    n = len(triples)
    scores = []

    for _ in range(n_iter):
        indices = rng.choice(n, size=n, replace=True)
        sample = [triples[idx] for idx in indices]
        try:
            score = fleiss_kappa(sample, field)
            if not math.isnan(score):
                scores.append(score)
        except Exception:
            continue

    if len(scores) < 10:
        return (float("nan"), float("nan"))

    return (
        float(np.percentile(scores, 2.5)),
        float(np.percentile(scores, 97.5)),
    )
