"""Skill name and category normalization utilities."""

import re

# Characters removed in Step 2 of skill normalization
_CHARS_TO_REMOVE = frozenset(" .'-_/")

# Suffixes stripped in Step 3
_SUFFIXES = ["js", "framework", "library"]


def normalize_skill(s: str) -> str:
    """Normalize skill name for fuzzy matching.

    Step 1: lowercase + strip whitespace
    Step 2: remove chars in {space, '.', '-', '_', '/'}
    Step 3: strip suffixes ['js', 'framework', 'library']

    Examples:
        "ReactJS"    -> "react"
        "react.js"   -> "react"
        "Vue.js"     -> "vue"
        "ASP.NET"    -> "aspnet"
        "Docker"     -> "docker"
    """
    s = s.lower().strip()
    
    # step 2: remove punctuation chars (KHÔNG remove space)
    for ch in ['.', '-', '_', '/']:   # ← bỏ ' ' khỏi list
        s = s.replace(ch, '')
    
    # step 3: collapse multiple spaces (do "CI/CD" → "CICD" có thể tạo space dư)
    s = ' '.join(s.split())
    
    # step 4: strip suffixes (chỉ áp dụng cho single-token, không tách token nội bộ)
    suffixes = ['js', 'framework', 'library']
    tokens = s.split()
    if len(tokens) == 1:
        for suffix in suffixes:
            if tokens[0].endswith(suffix) and len(tokens[0]) > len(suffix):
                tokens[0] = tokens[0][:-len(suffix)]
                break
    s = ' '.join(tokens)
    
    return s

def normalize_category(cat: str) -> str:
    """Normalize category name by collapsing whitespace around slashes.

    Handles variants like "AI / ML / Data" -> "AI/ML/Data".
    Preserves non-standard categories (e.g. "Testing & QA") as-is so
    disagreements on non-standard categories are still captured.
    """
    if not cat:
        return "Other"
    return re.sub(r"\s*/\s*", "/", cat.strip())
