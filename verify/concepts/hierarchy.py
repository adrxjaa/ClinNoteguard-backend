from pathlib import Path
from typing import Dict, List, Optional, Set
import json
import logging

logger = logging.getLogger("ConceptHierarchy")

class ConceptHierarchy:
    """
    Concept hierarchy query interface for clinical knowledge graphs.
    Supports offline curated JSON hierarchy and optional UMLS MRREL files.
    """

    def __init__(
        self,
        backend: str = "curated",
        hierarchy_json_path: Optional[str] = None,
        mrrel_path: Optional[str] = None,
    ):
        self.backend = backend.lower()
        self.parents_map: Dict[str, List[str]] = {}
        self.cui_to_concept: Dict[str, str] = {}
        self.concept_to_cui: Dict[str, str] = {}

        if self.backend == "umls" and mrrel_path and Path(mrrel_path).exists():
            self._load_umls_mrrel(mrrel_path)
            logger.info("UMLS backend active (MRREL)")
        else:
            self.backend = "curated"
            json_path = hierarchy_path = (
                Path(hierarchy_json_path)
                if hierarchy_json_path
                else Path(__file__).parent / "concept_hierarchy.json"
            )
            self._load_curated_hierarchy(json_path)
            logger.info("Curated hierarchy backend active")

    def _load_curated_hierarchy(self, json_path: Path):
        if not json_path.exists():
            logger.warning(f"Hierarchy JSON not found at {json_path}")
            return
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.parents_map = data.get("parents", {})
        self.concept_to_cui = data.get("cui_map", {})
        self.cui_to_concept = {v: k for k, v in self.concept_to_cui.items()}

    def _load_umls_mrrel(self, mrrel_path: str):
        # Optional MRREL parser if local UMLS installation is provided
        logger.info(f"Parsing UMLS MRREL file at {mrrel_path}")
        with open(mrrel_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("|")
                if len(parts) >= 8:
                    cui1, rel, cui2, rela = parts[0], parts[3], parts[4], parts[7]
                    if rel in ["PAR", "RB", "SY"] or rela in ["isa", "is_a"]:
                        self.parents_map.setdefault(cui1, []).append(cui2)

    def resolve_cui(self, identifier: str) -> str:
        """Returns standard CUI for a concept string or itself if already CUI."""
        clean = identifier.lower().replace(" ", "_").strip()
        return self.concept_to_cui.get(clean, identifier)

    def resolve_concept_name(self, cui_or_name: str) -> str:
        """Returns readable name for CUI if known."""
        return self.cui_to_concept.get(cui_or_name, cui_or_name)

    def get_ancestors(
        self,
        concept: str,
        max_hops: int = 2,
        rel_types: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Traverses hierarchy graph up to max_hops to find broader concepts.
        """
        key = concept.lower().replace(" ", "_").strip()
        # Also check if it's a CUI that maps to a concept
        if key in self.cui_to_concept:
            key = self.cui_to_concept[key]

        ancestors: Set[str] = set()
        frontier: Set[str] = {key}

        for _ in range(max_hops):
            next_frontier: Set[str] = set()
            for curr in frontier:
                parents = self.parents_map.get(curr, [])
                for p in parents:
                    if p not in ancestors and p != key:
                        ancestors.add(p)
                        next_frontier.add(p)
            frontier = next_frontier
            if not frontier:
                break

        return sorted(list(ancestors))

    def is_a(self, child: str, parent: str, max_hops: int = 2) -> bool:
        """Checks if child concept is a subclass/member of parent concept."""
        child_key = child.lower().replace(" ", "_").strip()
        parent_key = parent.lower().replace(" ", "_").strip()
        ancestors = self.get_ancestors(child_key, max_hops=max_hops)
        return parent_key in ancestors or child_key == parent_key
