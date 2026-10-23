"""
Stage 5: Medical Concept Linking & Curated Ontology Layer.
Links surface medical terms to CUIs and navigates clinical hierarchies (curated concept layer).
"""

from typing import List, Dict, Any, Optional
from verify.concepts.hierarchy import ConceptHierarchy
from verify.concepts.linker import CuratedDictionaryLinker, Mention
from verify.concepts.attr_type import AttrType, infer_attr_type

class Stage5ConceptService:
    """
    Stage 5 Service: Medical Concept Linking & Curated Concept Layer.
    Resolves clinical terms to CUIs, checks subclass/subsumption relations, and assigns AttrType.
    """

    def __init__(self, backend: str = "curated"):
        self.hierarchy = ConceptHierarchy(backend=backend)
        self.linker = CuratedDictionaryLinker(hierarchy=self.hierarchy)

    def link(self, text: str) -> List[Mention]:
        """Links clinical entities in text to CUIs."""
        return self.linker.link(text)

    def get_attr_type(self, text: str) -> AttrType:
        """Infers attribute category (MEDICATION, CHRONIC_DX, ALLERGY, LAB, etc.)."""
        return infer_attr_type(text)

    def is_subclass(self, child_cui: str, parent_cui: str) -> bool:
        """Checks IS_A / subclass hierarchy entailment."""
        return self.hierarchy.is_subclass(child_cui, parent_cui)

def get_stage5_concept_service() -> Stage5ConceptService:
    return Stage5ConceptService()
