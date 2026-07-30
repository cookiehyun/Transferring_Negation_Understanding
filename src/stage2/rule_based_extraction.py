"""
Reimplementation of the rule-based negation-scope parser described in
"Seeing What's Not There" (Aggarwal et al., ICLR 2026), Appendix A.1 --
no code was released, so this follows their described algorithm exactly:
pre-negators (forward scope, from negator to next clause boundary) and
post-negators (backward scope, from previous clause boundary to the negator),
where clause boundaries are punctuation and contrasting conjunctions, but
NOT coordinating/subordinating conjunctions (per their Table 7/8 and
Algorithm 1 description).

Same interface as content_aware_correction.extract_negated_concepts, so it
can be swapped in as a drop-in alternative extractor for a fair A/B
comparison against the LLM-based extractor.
"""

import re

# Table 7 (paper): pre-negators, forward-scoped
PRE_NEGATORS = [
    "does not", "doesn't", "don't", "didn't", "hasn't", "haven't",
    "lack of", "lacking", "lacks", "lack",
    "without", "absence of", "devoid of", "empty of",
    "no one", "noone", "nowhere", "neither", "nor", "none",
    "never", "nothing", "not", "no",
]

# Table 8 (paper): post-negators, backward-scoped, triggered by copular verb + negator
POST_NEGATOR_PATTERN = re.compile(
    r"\b(is|are|was|were)\s+(nowhere|not|absent|never|lacking|missing|neither|nor)\b",
    re.IGNORECASE,
)

# clause boundaries that DO terminate scope
TERMINATING_CONJ = ["but", "although", "though", "however", "nevertheless",
                     "yet", "even so", "while", "whereas"]

_conj_pattern = "|".join(re.escape(c) for c in TERMINATING_CONJ)
BOUNDARY_RE = re.compile(rf"([,.;])|(\b(?:{_conj_pattern})\b)", re.IGNORECASE)

# sorted longest-first so e.g. "does not" matches before "not"
_PRE_NEGATORS_SORTED = sorted(PRE_NEGATORS, key=len, reverse=True)


def rule_based_extract_one(caption):
    """Returns the negated-scope phrase, or None if no negator found."""
    low = caption.lower()

    best_match = None
    for neg in _PRE_NEGATORS_SORTED:
        pattern = r"\b" + re.escape(neg) + r"\b"
        m = re.search(pattern, low)
        if m and (best_match is None or m.start() < best_match[0]):
            best_match = (m.start(), m.end(), "pre")

    pm = POST_NEGATOR_PATTERN.search(low)
    if pm and (best_match is None or pm.start() < best_match[0]):
        best_match = (pm.start(), pm.end(), "post")

    if best_match is None:
        return None

    start, end, kind = best_match

    if kind == "pre":
        rest = caption[end:]
        bmatch = BOUNDARY_RE.search(rest)
        scope = rest[:bmatch.start()] if bmatch else rest
    else:
        before = caption[:start]
        last_boundary_end = 0
        for bm in BOUNDARY_RE.finditer(before):
            last_boundary_end = bm.end()
        scope = before[last_boundary_end:]

    scope = scope.strip(" ,.;")
    return scope if scope else None


def extract_negated_concepts(captions, **kwargs):
    """Same interface as the LLM extractor (accepts/ignores llm_model,
    llm_tokenizer kwargs so it can be swapped in without changing call sites)."""
    return [rule_based_extract_one(c) for c in captions]