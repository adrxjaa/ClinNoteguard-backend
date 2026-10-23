"""
Stage 6: Temporal Validity Engine & Multi-Relational Evidence Graph Builder.
Resolves temporal status (CURRENT vs SUPERSEDED) and constructs EvidenceGraph instances.
"""

from typing import List, Dict, Any, Optional
from verify.graph.builder import GraphBuilder, EvidenceGraph
from verify.graph.validity import resolve_state, StateStatus
from verify.graph.transcript_store import TranscriptStore

class Stage6ValidityGraphService:
    """
    Stage 6 Service: Temporal Validity & Evidence Graph Construction.
    Builds structured graph representations combining claims, candidate records,
    concept ontology nodes, and temporal decay features.
    """

    def __init__(self):
        self.builder = GraphBuilder()
        self.transcript_store = TranscriptStore()

    def build_evidence_graph(
        self,
        claim: Dict[str, Any],
        candidates: List[Dict[str, Any]],
        encounter_time: str,
        validity_mode: str = "typed",
    ) -> EvidenceGraph:
        """
        Constructs a complete multi-relational EvidenceGraph for a claim.
        """
        return self.builder.build_graph(
            claim=claim,
            candidates=candidates,
            encounter_time=encounter_time,
            cfg={"validity_mode": validity_mode},
        )

def get_stage6_validity_service() -> Stage6ValidityGraphService:
    return Stage6ValidityGraphService()
