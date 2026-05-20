"""Fuzzy skill matching and bipartite assignment."""

from difflib import SequenceMatcher
from .normalization import normalize_skill

FUZZY_THRESHOLD = 0.80
JACCARD_THRESHOLD = 0.50


def is_match(a: str, b: str) -> bool:
    """Hybrid matching: Token Jaccard (primary) + SequenceMatcher (fallback).

    Input: 2 skill names đã qua normalize().
    """
    tokens_a = set(a.split())
    tokens_b = set(b.split())

    if tokens_a and tokens_b:
        intersection = len(tokens_a & tokens_b)
        union = len(tokens_a | tokens_b)
        jaccard = intersection / union
        if jaccard >= JACCARD_THRESHOLD:
            return True

    return SequenceMatcher(None, a, b).ratio() >= FUZZY_THRESHOLD


def greedy_bipartite_match(
    skills_a: list[dict], skills_b: list[dict]
) -> tuple[list, list, list]:
    """Greedy bipartite matching — each skill in B matched to at most 1 in A.

    Strategy: collect all (similarity, i, j) pairs above threshold, sort
    descending, then assign greedily so no index is reused.

    Returns:
        matched_pairs: list of (skill_a, skill_b) tuples
        only_a:        skills in A with no match in B
        only_b:        skills in B with no match in A
    """
    if not skills_a or not skills_b:
        return [], list(skills_a), list(skills_b)

    candidates = []
    for i, sa in enumerate(skills_a):
        for j, sb in enumerate(skills_b):
            na = normalize_skill(sa["skill_name"])
            nb = normalize_skill(sb["skill_name"])
            if is_match(na, nb):
                candidates.append((1.0, i, j))

    candidates.sort(key=lambda x: -x[0])

    used_a: set[int] = set()
    used_b: set[int] = set()
    matched_pairs = []

    for _, i, j in candidates:
        if i not in used_a and j not in used_b:
            matched_pairs.append((skills_a[i], skills_b[j]))
            used_a.add(i)
            used_b.add(j)

    only_a = [skills_a[i] for i in range(len(skills_a)) if i not in used_a]
    only_b = [skills_b[j] for j in range(len(skills_b)) if j not in used_b]

    return matched_pairs, only_a, only_b


def three_way_match(
    skills_a: list[dict], skills_b: list[dict], skills_llm: list[dict]
) -> list[tuple]:
    """Find (skill_a, skill_b, skill_llm) triples matched across all 3 annotators.

    Strategy:
        1. Match A vs B via greedy bipartite.
        2. For each (A, B) pair, find the first available LLM match.
    """
    ab_pairs, _, _ = greedy_bipartite_match(skills_a, skills_b)

    llm_used: set[int] = set()
    triples = []

    for sa, sb in ab_pairs:
        na = normalize_skill(sa["skill_name"])
        best_j = -1

        for j, sl in enumerate(skills_llm):
            if j in llm_used:
                continue
            if is_match(na, normalize_skill(sl["skill_name"])):
                best_j = j
                break

        if best_j >= 0:
            triples.append((sa, sb, skills_llm[best_j]))
            llm_used.add(best_j)

    return triples
