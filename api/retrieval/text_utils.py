"""
Clinical text utilities shared by the BM25 and Dense retrievers.

- Light clinical tokenizer (lower-case, abbreviation expansion, simple stemming)
- Stop-word removal that keeps negations ("no", "denies", "without") because they
  flip the meaning of a clinical claim.
"""

import re
from typing import List

# Common brand-to-generic medication mappings
BRAND_TO_GENERIC = {
    "tylenol": "acetaminophen",
    "lipitor": "atorvastatin",
    "keppra": "levetiracetam",
    "zoloft": "sertraline",
    "lasix": "furosemide",
    "glucophage": "metformin",
}

# Common clinical abbreviations -> expansion (so "HTN" in the EHR matches "hypertension" in a claim)
# TODO [Checkpoint 3C]: Integrate UMLS Consumer Health Vocabulary (CHV) / lay language mapping
# to expand patient lay terms (e.g., 'water pill' -> 'furosemide/diuretic', 'sugar problems' -> 'diabetes').
ABBREVIATIONS = {
    "htn": "hypertension",
    "dm": "diabetes mellitus",
    "t2dm": "type 2 diabetes mellitus",
    "t1dm": "type 1 diabetes mellitus",
    "chf": "congestive heart failure",
    "cad": "coronary artery disease",
    "copd": "chronic obstructive pulmonary disease",
    "gerd": "gastroesophageal reflux disease",
    "ckd": "chronic kidney disease",
    "mi": "myocardial infarction",
    "afib": "atrial fibrillation",
    "uti": "urinary tract infection",
    "dvt": "deep vein thrombosis",
    "pcp": "primary care physician",
    "ct": "computed tomography",
    "mri": "magnetic resonance imaging",
    "ekg": "electrocardiogram",
    "ecg": "electrocardiogram",
    "bp": "blood pressure",
    "hr": "heart rate",
    "sob": "shortness of breath",
    "hx": "history",
    "dcis": "ductal carcinoma in situ",
    "ra": "rheumatoid arthritis",
    "oa": "osteoarthritis",
    "bmi": "body mass index",
    **BRAND_TO_GENERIC,
}

STOPWORDS = {
    "a", "an", "the", "of", "to", "and", "or", "in", "on", "at", "for", "with", "is", "are", "was",
    "were", "be", "been", "being", "has", "have", "had", "his", "her", "their", "this", "that",
    "these", "those", "as", "by", "from", "it", "its", "patient", "patients", "he", "she", "they",
    "also", "which", "who", "than", "then", "so", "such", "any", "per", "into", "over", "about",
}

# NOTE: negation words are intentionally NOT in STOPWORDS.
NEGATIONS = {"no", "not", "denies", "denied", "without", "negative", "never", "none"}

# number(+unit) tokens first so '9.1', '200mg', '12%' stay single tokens
_UNITS = r"mg|mcg|ml|iu|units|kg|cm|mmhg|u/l|g|%"
_TOKEN_RE = re.compile(r"\d+(?:\.\d+)?(?:" + _UNITS + r")?|[a-z][a-z0-9]*(?:[-'][a-z0-9]+)*")

_UNIT_ALIASES = [
    (r"\bmilligrams?\b", "mg"),
    (r"\bmicrograms?\b", "mcg"),
    (r"\bmilliliters?\b", "ml"),
    (r"\bgrams?\b", "g"),
    (r"\bpercent\b", "%"),
    (r"\bunits?\b", "units"),
]
_GLUE_RE = re.compile(r"(\d(?:\.\d+)?)\s*(mg|mcg|ml|iu|units|g|%)(?![a-z])")
_DOSE_RE = re.compile(r"(\d+(?:\.\d+)?)(mg|mcg|ml|iu|units|g|%)")


def normalize_units(text: str) -> str:
    """'200 milligrams' / '200 mg' / '200mg' -> '200mg'; '9.1 percent' -> '9.1%'."""
    text = text.lower()
    for pat, rep in _UNIT_ALIASES:
        text = re.sub(pat, rep, text)
    return _GLUE_RE.sub(r"\1\2", text)


def extract_doses(text: str) -> list:
    """[(value, unit)] for every quantity that has a unit, e.g. [(200.0, 'mg')]."""
    return [(float(v), u) for v, u in _DOSE_RE.findall(normalize_units(text))]


def dose_agreement(claim: str, doc: str) -> str:
    """
    'match'    - claim and doc share a unit and at least one value is equal
    'mismatch' - they share a unit but NO value is equal (e.g. aspirin 25 mg vs 200 mg)
    'n/a'      - claim has no quantity, or doc has no quantity in the same unit
    """
    c, d = extract_doses(claim), extract_doses(doc)
    shared_units = {u for _, u in c} & {u for _, u in d}
    if not shared_units:
        return "n/a"
    for u in shared_units:
        if {v for v, uu in c if uu == u} & {v for v, uu in d if uu == u}:
            return "match"
    return "mismatch"


def _stem(tok: str) -> str:
    """Very small suffix stripper (enough to unify infection/infections, prescribed/prescribe)."""
    if len(tok) <= 4 or tok[0].isdigit():
        return tok
    for suf in ("ations", "ation", "ings", "ing", "edly", "ed", "ies", "es", "s"):
        if tok.endswith(suf) and len(tok) - len(suf) >= 3:
            base = tok[: -len(suf)]
            return base + "y" if suf == "ies" else base
    return tok


def expand_abbreviations(text: str) -> str:
    def repl(m):
        w = m.group(0)
        return ABBREVIATIONS.get(w.lower(), w)

    return re.sub(r"\b[A-Za-z0-9]+\b", repl, text)


def tokenize(text: str, remove_stopwords: bool = True, stem: bool = True) -> List[str]:
    """Tokenize clinical text for lexical retrieval."""
    if not text:
        return []
    text = normalize_units(expand_abbreviations(text))
    toks = _TOKEN_RE.findall(text)
    out = []
    for t in toks:
        if remove_stopwords and t in STOPWORDS:
            continue
        out.append(_stem(t) if stem else t)
    return out


def content_terms(text: str) -> set:
    """Set of stemmed content words (no stop-words, no negations) used for silver labelling."""
    return {t for t in tokenize(text) if t not in NEGATIONS}


def extract_dose(text: str):
    """Convenience single-dose extractor returning (value, unit) or (None, None)."""
    doses = extract_doses(text)
    return doses[0] if doses else (None, None)


tokenize_clinical = tokenize

