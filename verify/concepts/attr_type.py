from enum import Enum
from pathlib import Path
from typing import Optional, Dict, List, Any
import yaml

class AttrType(str, Enum):
    MEDICATION = "MEDICATION"
    CHRONIC_DX = "CHRONIC_DX"
    ACUTE_DX = "ACUTE_DX"
    ALLERGY = "ALLERGY"
    LAB = "LAB"
    PROCEDURE = "PROCEDURE"
    OTHER = "OTHER"

_CONFIG_PATH = Path(__file__).parent / "attr_overrides.yaml"
_CACHED_CONFIG: Optional[Dict[str, Any]] = None

def _load_attr_config() -> Dict[str, Any]:
    global _CACHED_CONFIG
    if _CACHED_CONFIG is None:
        if _CONFIG_PATH.exists():
            with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
                _CACHED_CONFIG = yaml.safe_load(f) or {}
        else:
            _CACHED_CONFIG = {}
    return _CACHED_CONFIG

def infer_attr_type(
    text: str,
    semantic_type: Optional[str] = None,
    concept_id: Optional[str] = None,
) -> AttrType:
    """
    Infers the clinical attribute type using semantic types, overrides, and cue words.
    """
    cfg = _load_attr_config()
    text_lower = text.lower().strip()

    # 1. Exact match override
    overrides = cfg.get("overrides", {})
    if text_lower in overrides:
        return AttrType(overrides[text_lower])
    if concept_id and concept_id in overrides:
        return AttrType(overrides[concept_id])

    # 2. Semantic type mapping (e.g., UMLS TUI)
    if semantic_type:
        st_map = cfg.get("semantic_types", {})
        if semantic_type in st_map:
            return AttrType(st_map[semantic_type])

    # 3. Cue word patterns
    cue_words = cfg.get("cue_words", {})
    # Priority order for cue words: ALLERGY > LAB > PROCEDURE > ACUTE_DX > CHRONIC_DX > MEDICATION
    for category in ["ALLERGY", "LAB", "PROCEDURE", "ACUTE_DX", "CHRONIC_DX", "MEDICATION"]:
        keywords = cue_words.get(category, [])
        for kw in keywords:
            if kw.lower() in text_lower:
                return AttrType(category)

    # 4. Fallback check for numbers with clinical units (often LAB or MED)
    if any(u in text_lower for u in ["mg", "mcg", "tablet", "capsule", "units", "bid", "daily"]):
        return AttrType.MEDICATION

    return AttrType.OTHER
