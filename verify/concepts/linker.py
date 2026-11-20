from dataclasses import dataclass
from typing import List, Optional, Tuple, Protocol, Dict
import logging
import re
from verify.concepts.hierarchy import ConceptHierarchy
from api.retrieval.text_utils import BRAND_TO_GENERIC

logger = logging.getLogger("ConceptLinker")

@dataclass
class Mention:
    text: str
    cui: str
    semantic_type: Optional[str]
    span: Tuple[int, int]
    canonical_name: str = ""

class ConceptLinker(Protocol):
    def link(self, text: str) -> List[Mention]:
        """Extracts medical mentions and links them to CUIs."""
        ...

# Layman terms and clinical aliases mapped to canonical concept names
LAYMAN_LEXICON: Dict[str, str] = {
    "water pill": "furosemide",
    "water pills": "furosemide",
    "fluid pill": "furosemide",
    "sugar trouble": "type_2_diabetes",
    "blood sugar": "type_2_diabetes",
    "high sugar": "type_2_diabetes",
    "high blood pressure": "hypertension",
    "blood pressure": "hypertension",
    "high bp": "hypertension",
    "heart failure": "heart_failure",
    "fluid in lungs": "heart_failure",
    "heart flutter": "atrial_fibrillation",
    "irregular heartbeat": "atrial_fibrillation",
    "breathing trouble": "asthma",
    "wheezing": "asthma",
    "kidney failure": "chronic_kidney_disease",
    "kidney trouble": "chronic_kidney_disease",
    "acid reflux": "gerd",
    "heartburn": "gerd",
    "blood thinner": "aspirin",
    "pain reliever": "ibuprofen",
    "steroid": "prednisone",
}

class CuratedDictionaryLinker:
    """
    Offline dictionary linker utilizing brand-to-generic mappings,
    clinical synonyms, and curated hierarchy CUIs.
    """

    def __init__(self, hierarchy: Optional[ConceptHierarchy] = None):
        self.hierarchy = hierarchy or ConceptHierarchy(backend="curated")
        self._build_lexicon()

    def _build_lexicon(self):
        self.term_to_canonical: Dict[str, str] = {}
        # 1. Add layman lexicon
        for term, canon in LAYMAN_LEXICON.items():
            self.term_to_canonical[term.lower()] = canon

        # 2. Add brand-to-generic mappings from text_utils
        for brand, generic in BRAND_TO_GENERIC.items():
            self.term_to_canonical[brand.lower()] = generic.lower()
            self.term_to_canonical[generic.lower()] = generic.lower()

        # 3. Add all concepts and parents from hierarchy
        for concept in self.hierarchy.concept_to_cui.keys():
            self.term_to_canonical[concept.lower().replace("_", " ")] = concept
            self.term_to_canonical[concept.lower()] = concept

        for parent in self.hierarchy.parents_map.keys():
            self.term_to_canonical[parent.lower().replace("_", " ")] = parent
            self.term_to_canonical[parent.lower()] = parent

        # Sort terms by length descending to match longest phrases first
        self.sorted_terms = [t for t in sorted(self.term_to_canonical.keys(), key=len, reverse=True) if t]
        # Pre-compile single combined regex for O(1) pass + memoize link() calls
        self._compiled_pattern = (
            re.compile(r"\b(" + "|".join(re.escape(t) for t in self.sorted_terms) + r")\b")
            if self.sorted_terms
            else None
        )
        self._link_cache: Dict[str, List[Mention]] = {}

    def link(self, text: str) -> List[Mention]:
        if not text:
            return []
        if text in self._link_cache:
            return self._link_cache[text]

        mentions: List[Mention] = []
        if self._compiled_pattern is None:
            return mentions

        text_lower = text.lower()
        occupied_spans: List[Tuple[int, int]] = []

        for match in self._compiled_pattern.finditer(text_lower):
            term = match.group(1)
            start, end = match.span()
            if any(start < o_end and end > o_start for o_start, o_end in occupied_spans):
                continue

            canonical = self.term_to_canonical[term]
            cui = self.hierarchy.resolve_cui(canonical)
            ancestors = set(self.hierarchy.get_ancestors(canonical, max_hops=3))
            med_ancestors = {
                "antibiotic", "antihypertensive", "cardiovascular_agent",
                "lipid_lowering_agent", "antidiabetic_agent", "nsaid", "analgesic",
                "diuretic", "bronchodilator", "corticosteroid", "anticonvulsant",
                "ssri", "ppi", "statin", "ace_inhibitor", "arb", "sulfonylurea",
                "beta2_agonist", "antiplatelet", "biguanide", "loop_diuretic",
                "thiazide_diuretic", "cephalosporin", "penicillin", "beta_lactam_antibiotic",
                "calcium_channel_blocker", "beta_blocker", "anticoagulant",
                "antithrombotic", "opioid_analgesic", "insulin_therapy", "bisphosphonate",
            }
            if "allergy" in canonical or "drug_allergy" in ancestors:
                sem_type = "T046"
            elif canonical in med_ancestors or bool(ancestors & med_ancestors) or canonical in BRAND_TO_GENERIC.values():
                sem_type = "T121"
            else:
                sem_type = "T047"

            mentions.append(
                Mention(
                    text=text[start:end],
                    cui=cui,
                    semantic_type=sem_type,
                    span=(start, end),
                    canonical_name=canonical,
                )
            )
            occupied_spans.append((start, end))

        res = sorted(mentions, key=lambda m: m.span[0])
        if len(self._link_cache) < 20000:
            self._link_cache[text] = res
        return res


class ScispaCyLinker:
    """Optional ScispaCy UMLS Linker if installed."""
    def __init__(self):
        try:
            import spacy
            import scispacy
            from scispacy.linking import EntityLinker
            self.nlp = spacy.load("en_core_sci_sm")
            self.nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
            logger.info("UMLS backend active (scispaCy)")
        except Exception as e:
            raise ImportError(f"scispaCy UMLS linker could not be loaded: {e}")

    def link(self, text: str) -> List[Mention]:
        doc = self.nlp(text)
        mentions = []
        for ent in doc.ents:
            cui = "CUI_UNKNOWN"
            if ent._.kb_ents:
                cui = ent._.kb_ents[0][0]
            mentions.append(
                Mention(
                    text=ent.text,
                    cui=cui,
                    semantic_type=None,
                    span=(ent.start_char, ent.end_char),
                    canonical_name=ent.text.lower(),
                )
            )
        return mentions


def get_concept_linker(backend: str = "curated") -> ConceptLinker:
    """
    Factory to instantiate concept linker based on config backend.
    """
    if backend.lower() == "umls":
        try:
            return ScispaCyLinker()
        except Exception as e:
            logger.warning(f"UMLS backend failed ({e}). Falling back to curated hierarchy.")
            logger.info("Curated hierarchy backend active")
            return CuratedDictionaryLinker()
    else:
        logger.info("Curated hierarchy backend active")
        return CuratedDictionaryLinker()
