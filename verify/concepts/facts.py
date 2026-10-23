from dataclasses import dataclass
from typing import Optional, List, Tuple
import re
from verify.concepts.attr_type import AttrType, infer_attr_type
from verify.concepts.linker import ConceptLinker, CuratedDictionaryLinker
from api.retrieval.text_utils import extract_dose, BRAND_TO_GENERIC

@dataclass(frozen=True)
class TypedFact:
    record_id: str
    source: str # "EHR" | "TRANSCRIPT"
    attr_type: AttrType
    key_cui: str # normalized to ingredient-level CUI
    value: str # e.g., "10mg", "active", "present", "negative"
    unit: Optional[str]
    polarity: bool # True = positive/active, False = negated/denied
    event_type: str # "START" | "CHANGE" | "STOP" | "RESOLVE" | "ASSERTION"
    timestamp: Optional[str] = None

    @property
    def key(self) -> Tuple[str, AttrType]:
        """Unique state tracking key: (key_cui, attr_type)."""
        return (self.key_cui, self.attr_type)

_NEGATION_PATTERNS = [
    r"\bno\b",
    r"\bnot\b",
    r"\bdenies\b",
    r"\bdenied\b",
    r"\bwithout\b",
    r"\bnegative\s+for\b",
    r"\bnever\b",
]

_STOP_PATTERNS = [
    r"\bdiscontinued\b",
    r"\bstopped\b",
    r"\bceased\b",
    r"\bhold\b",
    r"\bwithdrawn\b",
]

_RESOLVE_PATTERNS = [
    r"\bresolved\b",
    r"\bcured\b",
    r"\bcleared\b",
    r"\bstatus\s+post\b",
    r"\brecovered\b",
]

_CHANGE_PATTERNS = [
    r"\bincreased\s+to\b",
    r"\bdecreased\s+to\b",
    r"\bswitched\s+to\b",
    r"\bchanged\s+to\b",
    r"\btitrated\s+to\b",
]

def extract_facts(
    record_id: str,
    text: str,
    source: str = "EHR",
    event_type: Optional[str] = None,
    timestamp: Optional[str] = None,
    linker: Optional[ConceptLinker] = None,
) -> List[TypedFact]:
    """
    Rule-based fact extraction producing TypedFact instances from clinical text.
    Normalizes medications to ingredient level.
    """
    if not text:
        return []

    if linker is None:
        linker = CuratedDictionaryLinker()

    mentions = linker.link(text)
    if not mentions:
        return []

    text_lower = text.lower()

    # Determine event type from text if not provided
    inferred_event = event_type
    if not inferred_event or inferred_event == "ASSERTION":
        if any(re.search(p, text_lower) for p in _STOP_PATTERNS):
            inferred_event = "STOP"
        elif any(re.search(p, text_lower) for p in _RESOLVE_PATTERNS):
            inferred_event = "RESOLVE"
        elif any(re.search(p, text_lower) for p in _CHANGE_PATTERNS):
            inferred_event = "CHANGE"
        else:
            inferred_event = "START"

    facts: List[TypedFact] = []
    dose_val, unit_val = extract_dose(text)

    for m in mentions:
        # Determine polarity
        # Look for negation cue within 5 words preceding mention
        mention_start = m.span[0]
        preceding_window = text_lower[max(0, mention_start - 40):mention_start]
        is_negated = any(re.search(p, preceding_window) for p in _NEGATION_PATTERNS)
        polarity = not is_negated

        # Determine AttrType
        attr = infer_attr_type(text, semantic_type=m.semantic_type, concept_id=m.canonical_name)

        # Ingredient normalization for medications
        canonical = m.canonical_name.lower()
        if canonical in BRAND_TO_GENERIC:
            canonical = BRAND_TO_GENERIC[canonical].lower()

        # Resolve CUI for normalized canonical concept
        norm_cui = linker.hierarchy.resolve_cui(canonical) if hasattr(linker, "hierarchy") else m.cui

        # Value
        if attr == AttrType.MEDICATION and dose_val:
            dose_str = str(int(dose_val)) if int(dose_val) == dose_val else str(dose_val)
            value_str = f"{dose_str}{unit_val}" if unit_val else dose_str
        elif inferred_event == "STOP":
            value_str = "discontinued"
        elif inferred_event == "RESOLVE":
            value_str = "resolved"
        elif not polarity:
            value_str = "negative"
        else:
            value_str = "active"

        facts.append(
            TypedFact(
                record_id=record_id,
                source=source,
                attr_type=attr,
                key_cui=norm_cui,
                value=value_str,
                unit=unit_val,
                polarity=polarity,
                event_type=inferred_event,
                timestamp=timestamp,
            )
        )

    return facts
